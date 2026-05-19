#!/usr/bin/env python3
"""Add genomic coordinates (chr_pos) to existing usage_{sps}.npz prediction files.

Iterates the same dataset that was used during evaluation (no model / GPU needed)
to reconstruct the exact per-observation accumulation order, then writes a
``chr_pos`` string array (format "chrN:position") into each existing NPZ.

The ``chr_pos`` array is parallel to the existing ``cond_ids`` / ``pred`` / ``true``
arrays, so ``chr_pos[i]`` is the genomic location of observation ``i``.

Usage
-----
    python scripts/add_usage_coords.py \\
        --data-config   /path/to/data_config.json \\
        --work-dir      /path/to/132kb_human_mouse_rat_rabbit_opossum \\
        --eval-species  human mouse rat rabbit opossum \\
        --pred-dirs     union gtf usage intersect intersect_usage \\
        [--min-coverage 10] [--max-sites 1024] [--batch-size 8]

Notes
-----
- Uses ``sampled_windows_{sps}.bed`` if present in the pred dir (matches the
  original evaluation run); falls back to ``test_bed`` / ``val_bed`` from the
  data config.
- ``--min-coverage`` and ``--max-sites`` must match the values used during the
  original evaluation run (both default to the same values as evaluate_splice.py).
- Skips NPZ files that already contain a ``chr_pos`` key unless ``--overwrite``
  is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add chr_pos coordinates to existing usage NPZ prediction files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data-config", required=True,
                   help="Path to data config JSON (same file used for evaluate_splice.py).")
    p.add_argument("--work-dir", required=True,
                   help="Model work directory that contains the preds_* subdirectories "
                        "(e.g. .../132kb_human_mouse_rat_rabbit_opossum).")
    p.add_argument("--eval-species", nargs="+", required=True,
                   help="Species to process (e.g. human mouse rat).")
    p.add_argument("--pred-dirs", nargs="+",
                   default=["gtf", "usage", "union", "intersect", "intersect_usage"],
                   help="Prediction subset keys to process (default: all five).")
    p.add_argument("--min-coverage", type=int, default=10,
                   help="Min (Alpha+Beta) coverage used when building the usage index "
                        "(must match the original evaluation run, default: 10).")
    p.add_argument("--max-sites", type=int, default=1024,
                   help="Max usage sites per window (must match original run, default: 1024).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default: 0 = main process, safest for order reproducibility).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-add chr_pos even if it already exists in the NPZ.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without modifying any files.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data config loading (mirrors evaluate_splice.py)
# ---------------------------------------------------------------------------

def load_data_config(path: Path, species_list: list[str]) -> dict[str, dict]:
    from alphagenome_pytorch.utils.paths import expand_paths_in_dict
    with open(path) as f:
        cfg = json.load(f)
    path_keys = {"genome", "annotation_parquet", "usage_parquet",
                 "train_bed", "val_bed", "test_bed"}
    cfg = expand_paths_in_dict(cfg, path_keys)
    species_dict = cfg.get("species", cfg)
    specs = {}
    for sps in species_list:
        if sps not in species_dict:
            sys.exit(f"Species '{sps}' not found in data config {path}")
        specs[sps] = species_dict[sps].copy()
        specs[sps]["name"] = sps
    return specs


# ---------------------------------------------------------------------------
# Core: accumulate chr_pos without running the model
# ---------------------------------------------------------------------------

def accumulate_coords(
    dataset,
    present_cond_ids: set[int],
    batch_size: int,
    num_workers: int,
) -> dict[int, list[str]]:
    """Iterate dataset, collecting chr_pos strings for each condition in the same
    order they were accumulated during the original prediction run.

    Args:
        dataset: SpliceSiteDataset (or Subset thereof).
        present_cond_ids: Set of condition indices actually present in the NPZ.
            Conditions not in this set are skipped to avoid allocating memory
            for conditions that weren't stored during the original run.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.

    Returns:
        dict mapping condition_idx -> list of "chr:pos" strings in accumulation order.
    """
    import torch
    from torch.utils.data import DataLoader
    from alphagenome_pytorch.extensions.finetuning.splice_datasets import collate_splice
    from tqdm import tqdm

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_splice,
    )

    # Resolve chrom_names from the dataset (handle Subset wrapper)
    _ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    chrom_names: list[str] = _ds.chrom_names

    coords_per_cond: dict[int, list[str]] = {}

    for batch in tqdm(loader, desc="  Accumulating coords", unit="batch"):
        if "usage_positions" not in batch:
            continue

        positions    = batch["usage_positions"].numpy()  # (B, max_sites)
        usage_mask   = batch["usage_mask"].numpy()       # (B, max_sites, n_cond)
        window_starts = batch["window_start"].numpy()    # (B,)
        chrom_idxs   = batch["chrom_idx"].numpy()        # (B,)

        B            = positions.shape[0]
        n_data_cond  = usage_mask.shape[2]

        for i in range(B):
            valid = positions[i] != -1
            if not valid.any():
                continue
            valid_pos  = positions[i][valid]
            valid_mask = usage_mask[i][valid]              # (k, n_cond)
            win_start  = int(window_starts[i])
            chrom      = chrom_names[int(chrom_idxs[i])]
            genomic_pos = valid_pos + win_start            # absolute 0-based

            for data_c in range(n_data_cond):
                if data_c not in present_cond_ids:
                    continue
                obs = valid_mask[:, data_c]
                if not obs.any():
                    continue
                coords_per_cond.setdefault(data_c, []).extend(
                    f"{chrom}:{gp}" for gp in genomic_pos[obs]
                )

    return coords_per_cond


def build_chr_pos_array(
    npz_cond_ids: np.ndarray,
    coords_per_cond: dict[int, list[str]],
) -> np.ndarray | None:
    """Build a chr_pos string array aligned to npz_cond_ids.

    Within each condition block in the NPZ the observations are in the same
    order as they were accumulated (batch-major, site-minor).  The
    coords_per_cond lists have the same order, so we step through them with
    per-condition cursors.

    Returns None if coord counts don't match (mismatch → skip this file).
    """
    # Verify counts match before writing anything
    for cid in np.unique(npz_cond_ids):
        expected = int((npz_cond_ids == cid).sum())
        actual   = len(coords_per_cond.get(int(cid), []))
        if expected != actual:
            print(f"  [WARN] Condition {cid}: NPZ has {expected} entries, "
                  f"re-iterated dataset produced {actual}. Skipping this file.")
            return None

    chr_pos = np.empty(len(npz_cond_ids), dtype=object)
    cursors: dict[int, int] = {}
    for i, cid in enumerate(npz_cond_ids.tolist()):
        cid = int(cid)
        pos = cursors.get(cid, 0)
        chr_pos[i] = coords_per_cond[cid][pos]
        cursors[cid] = pos + 1

    return chr_pos.astype(str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        sys.exit(f"Work directory not found: {work_dir}")

    specs = load_data_config(Path(args.data_config), args.eval_species)

    from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
        SpliceSiteAnnotation,
        SpliceSiteDataset,
        SpliceSiteUsageIndex,
    )

    for sps, spec in specs.items():
        print(f"\n{'='*60}")
        print(f"  {sps.upper()}")
        print(f"{'='*60}")

        if not spec.get("usage_parquet"):
            print(f"  No usage_parquet for {sps}, skipping.")
            continue

        # Build annotation and usage index once per species
        print(f"  Loading annotation …")
        annotation = SpliceSiteAnnotation(spec["annotation_parquet"])

        print(f"  Loading usage index (min_coverage={args.min_coverage}) …")
        usage_index = SpliceSiteUsageIndex(
            spec["usage_parquet"],
            min_coverage=args.min_coverage,
            usage_coord_base=0,
            observed_conditions_only=False,
        )

        for pd_key in args.pred_dirs:
            pred_dir = work_dir / f"preds_{pd_key}" / sps
            npz_path = pred_dir / f"usage_{sps}.npz"

            print(f"\n  [{pd_key}] {npz_path}")

            if not npz_path.exists():
                print(f"    NPZ not found, skipping.")
                continue

            # Load existing NPZ
            data = np.load(npz_path)

            if "chr_pos" in data.files and not args.overwrite:
                print(f"    chr_pos already present. Use --overwrite to replace.")
                continue

            if "cond_ids" not in data.files or data["cond_ids"].size == 0:
                print(f"    No cond_ids / empty NPZ, skipping.")
                continue

            present_cond_ids = set(data["cond_ids"].astype(int).tolist())
            print(f"    {data['cond_ids'].size:,} observations, "
                  f"{len(present_cond_ids)} conditions.")

            # Pick BED file: prefer sampled_windows from the pred dir
            sampled_bed = pred_dir / f"sampled_windows_{sps}.bed"
            if sampled_bed.exists():
                bed_file = str(sampled_bed)
                print(f"    Using sampled windows BED: {sampled_bed.name}")
            else:
                bed_file = spec.get("test_bed") or spec.get("val_bed")
                if not bed_file:
                    print(f"    No BED file found for {sps}/{pd_key}, skipping.")
                    continue
                print(f"    Using data-config BED: {Path(bed_file).name}")

            # Build dataset
            print(f"    Building dataset …")
            dataset = SpliceSiteDataset(
                genome=spec["genome"],
                bed_file=bed_file,
                annotation=annotation,
                usage_index=usage_index,
                sequence_length=131_072,
                organism_index=0,          # not used (no inference)
                max_sites=args.max_sites,
            )
            print(f"    {len(dataset):,} windows")

            if args.dry_run:
                print(f"    [dry-run] Would iterate dataset and patch NPZ.")
                continue

            # Accumulate coordinates
            coords_per_cond = accumulate_coords(
                dataset,
                present_cond_ids=present_cond_ids,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            # Build chr_pos array
            chr_pos = build_chr_pos_array(data["cond_ids"], coords_per_cond)
            if chr_pos is None:
                continue  # mismatch warning already printed

            # Write updated NPZ (atomic: write to tmp then replace).
            # np.savez_compressed appends .npz when the path doesn't end in
            # .npz already, so use a tmp name that already ends in .npz.
            tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
            npz_kwargs = {k: data[k] for k in data.files if k != "chr_pos"}
            npz_kwargs["chr_pos"] = chr_pos
            np.savez_compressed(tmp_path, **npz_kwargs)
            tmp_path.replace(npz_path)
            print(f"    Wrote chr_pos ({chr_pos.size:,} entries) → {npz_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
