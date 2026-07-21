#!/usr/bin/env python3
"""Generate model predictions (splice classification + usage) for targeted
genomic region(s), without needing a test BED.

Two modes:

1. Prediction-only (no ground truth): pass --genome and --organism /
   --organism-index directly.
2. Prediction + evaluation against ground truth: pass --data-config (same
   format as evaluate_splice.py) and --organism (a species name present in
   that config). Genome, annotation and usage parquet paths are then taken
   from the data config, ground-truth splice classes/usage are looked up for
   the requested region(s), and AUPRC / Pearson r metrics are computed and
   saved alongside the predictions — exactly like evaluate_splice.py, but
   restricted to the region(s) you ask for instead of a sampled test split.

Multiple organisms, each with their own regions, can be run in one invocation
via --organisms (plural) paired positionally with --regions: the i-th
--regions entry (a BED file path or a single 'chrom:start-end' string) is
used for the i-th --organisms entry. This requires --data-config, since each
organism's genome is resolved by name from it.

Loads the model the same way as evaluate_splice.py (LoRA + per-organism
classification/usage heads reconstructed from a fine-tuned checkpoint
directory), via `load_model_for_inference`.

Examples
--------
    # Prediction only, human (organism matched by name in checkpoint config)
    python scripts/predict_region.py \\
        --checkpoint /path/to/132kb_lora \\
        --genome /path/to/Homo_sapiens.fa \\
        --organism human \\
        --regions chr17:41196312-41277500 \\
        --output-dir results/region_preds

    # Prediction + evaluation against ground truth, with cross-species usage
    python scripts/predict_region.py \\
        --checkpoint /path/to/132kb_lora \\
        --data-config /path/to/data_config.json \\
        --organism mouse \\
        --regions chr11:69000000-69050000 \\
        --observed-conditions-only \\
        --cross-species-usage \\
        --plot \\
        --output-dir results/region_preds

    # Multiple organisms, each with their own test BED (genome/annotation/usage
    # resolved per-organism from the data config)
    python scripts/predict_region.py \\
        --checkpoint /path/to/132kb_lora \\
        --data-config /path/to/data_config.json \\
        --organisms human mouse \\
        --regions test_human.bed test_mouse.bed \\
        --output-dir results/region_preds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SPLICE_CLASS_NAMES = ["Donor+", "Acceptor+", "Donor-", "Acceptor-"]
BACKGROUND_CLASS = 4
CLASS_LABELS = {0: "donor +", 1: "acceptor +", 2: "donor -", 3: "acceptor -", 4: "no splice site"}
CLASS_COLORS = {0: "#ff7f00", 1: "#33a02c", 2: "#fdbf6f", 3: "#b2df8a", 4: "#1f78b4"}

SEQ_LEN_DEFAULT = 131_072

# `evaluate_splice.py` lives next to this script and is imported (not run) to
# reuse its data-config loading / species-matching / metrics helpers instead
# of duplicating them.
sys.path.insert(0, str(Path(__file__).parent))
import evaluate_splice as ev  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict splice classification/usage for targeted region(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", required=True,
                    help="Fine-tuned checkpoint directory (best_model.pth + config.json) or .pth file "
                         "(config.json expected in the same directory unless --model-config is given).")
    p.add_argument("--model-config", default=None,
                    help="Optional explicit path to the checkpoint's config.json/.yaml. Needed if the "
                         "config file next to best_model.pth isn't named 'config.json'/'config.yaml' "
                         "(e.g. 'config_0-5.json').")

    p.add_argument("--regions", nargs="+", default=None,
                    help="Region source(s). Each entry is either a path to a BED file (chrom, start, "
                         "end, optional name columns) or a single 'chrom:start-end' string. In "
                         "single-organism mode, all entries are pooled together. In multi-organism "
                         "mode (--organisms), must have exactly one entry per organism, paired by "
                         "position (e.g. --organisms human mouse --regions test_human.bed test_mouse.bed).")
    p.add_argument("--regions-bed", default=None,
                    help="Single BED file with chrom, start, end columns (optionally a 4th name "
                         "column). Equivalent to a single --regions entry; kept for backward "
                         "compatibility. Cannot be combined with --organisms (use --regions instead).")

    p.add_argument("--genome", default=None,
                    help="Reference genome FASTA. Required unless --data-config is given "
                         "(in which case the genome path is taken from the matched species).")
    p.add_argument("--organism", default=None,
                    help="Species name. With --data-config, must match a species key in that config "
                         "(ground-truth evaluation mode). Without --data-config, matches a 'name' entry "
                         "in the checkpoint's species_specs (prediction-only mode).")
    p.add_argument("--organism-index", type=int, default=None,
                    help="Organism index directly, bypassing name matching (prediction-only mode only).")
    p.add_argument("--organisms", nargs="+", default=None,
                    help="Multiple species names, paired positionally with --regions (one region "
                         "source per organism). Requires --data-config, which resolves each "
                         "organism's genome/annotation/usage paths by name.")

    p.add_argument("--data-config", default=None,
                    help="Path to data config JSON (species genome/annotation/usage paths, as used by "
                         "evaluate_splice.py). When given, ground-truth splice classes and usage values "
                         "are looked up for the requested region(s) and metrics (AUPRC, Pearson r) are "
                         "computed and saved alongside the predictions.")
    p.add_argument("--min-coverage", type=int, default=10,
                    help="Min (Alpha+Beta) coverage for usage ground truth (default: 10). Only used with --data-config.")
    p.add_argument(
        "--observed-conditions-only", action="store_true",
        help="When set, only evaluate usage on conditions observed in the data (and 0s only if no "
             "condition in tissue is observed). Otherwise, evaluate on all conditions. Only used with --data-config.",
    )
    p.add_argument(
        "--cross-species-usage", action="store_true",
        help="When set, also evaluate usage predictions from all model species' usage heads against the "
             "target species' ground-truth conditions (if metadata allows matching). Only used with --data-config.",
    )

    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True, help="Directory for output parquet/npz/plots.")
    p.add_argument("--tile-stride", type=int, default=None,
                    help="Stride (bp) for tiling windows across regions larger than the model's sequence "
                         "length. Defaults to the sequence length (non-overlapping tiles).")
    p.add_argument("--skip-usage", action="store_true", help="Skip usage-head inference even if available.")
    p.add_argument("--plot", action="store_true", help="Save a per-region track plot of classification probabilities.")

    args = p.parse_args()

    if args.organisms is not None:
        if args.organism is not None or args.organism_index is not None:
            p.error("--organisms cannot be combined with --organism / --organism-index")
        if not args.data_config:
            p.error("--data-config is required when using --organisms (each organism's genome is "
                     "resolved from it by name)")
        if args.regions_bed:
            p.error("--regions-bed cannot be combined with --organisms; pass one BED file (or region "
                     "string) per organism via --regions instead")
        if not args.regions:
            p.error("--regions is required when using --organisms: pass one entry (a BED file path or "
                     "a single region string) per organism, paired by position")
        if len(args.regions) != len(args.organisms):
            p.error(f"--regions must have exactly one entry per --organisms entry "
                     f"(got {len(args.regions)} regions for {len(args.organisms)} organisms)")
    else:
        if not args.regions and not args.regions_bed:
            p.error("one of --regions or --regions-bed is required")
        if args.regions and args.regions_bed:
            p.error("--regions and --regions-bed are mutually exclusive")
        if args.data_config is not None:
            if not args.organism:
                p.error("--organism is required when using --data-config")
        else:
            if not args.genome:
                p.error("--genome is required unless --data-config is given")
            if not args.organism and args.organism_index is None:
                p.error("one of --organism or --organism-index is required unless --data-config is given")

    return args


# ---------------------------------------------------------------------------
# Region parsing
# ---------------------------------------------------------------------------

def parse_region_str(region: str) -> tuple[str, int, int]:
    if ":" not in region:
        raise ValueError(f"Invalid region '{region}'; expected 'chrom:start-end'")
    chrom, rest = region.split(":", 1)
    if "-" in rest:
        start_s, end_s = rest.split("-")
    elif ":" in rest:
        start_s, end_s = rest.split(":")
    else:
        raise ValueError(f"Invalid region '{region}'; expected 'chrom:start-end'")
    return chrom, int(start_s.replace("_", "")), int(end_s.replace("_", ""))


def load_bed_regions(bed_path: str) -> list[tuple[str, int, int, str]]:
    out = []
    with open(bed_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            name = parts[3] if len(parts) > 3 else f"{chrom}:{start}-{end}"
            out.append((chrom, start, end, name))
    return out


def load_regions_from_source(source: str) -> list[tuple[str, int, int, str]]:
    """A region source is either a path to a BED file or a single 'chrom:start-end' string."""
    if os.path.isfile(source):
        return load_bed_regions(source)
    chrom, start, end = parse_region_str(source)
    return [(chrom, start, end, f"{chrom}:{start}-{end}")]


def build_jobs(args: argparse.Namespace) -> list[tuple[str | int, list[tuple[str, int, int, str]]]]:
    """Returns [(organism_arg, regions), ...]. organism_arg is a species name (str) or organism index (int)."""
    if args.organisms:
        return [
            (organism_name, load_regions_from_source(region_source))
            for organism_name, region_source in zip(args.organisms, args.regions)
        ]

    organism_arg: str | int = args.organism if args.organism is not None else args.organism_index
    sources = args.regions if args.regions else [args.regions_bed]
    regions: list[tuple[str, int, int, str]] = []
    for src in sources:
        regions.extend(load_regions_from_source(src))
    return [(organism_arg, regions)]


def make_windows(chrom: str, start: int, end: int, seq_len: int, stride: int) -> list[tuple[int, int]]:
    """Tile [start, end) into seq_len windows. A region smaller than seq_len is centered."""
    if end - start <= seq_len:
        center = (start + end) // 2
        win_start = center - seq_len // 2
        return [(win_start, win_start + seq_len)]

    windows = []
    win_start = start
    while win_start < end:
        windows.append((win_start, win_start + seq_len))
        if win_start + seq_len >= end:
            break
        win_start += stride
    # Make sure the final window covers the region's end.
    last_start, last_end = windows[-1]
    if last_end < end:
        windows.append((end - seq_len, end))
    return windows


# ---------------------------------------------------------------------------
# Organism / ground-truth resolution
# ---------------------------------------------------------------------------

def resolve_organism_index_simple(cfg: dict | None, organism_arg: str | int) -> tuple[int, dict | None]:
    """Prediction-only mode: resolve organism_index from an --organism name or --organism-index int."""
    if isinstance(organism_arg, int):
        return organism_arg, None
    if cfg is None or "species_specs" not in cfg:
        sys.exit(
            "--organism was given but the checkpoint has no config/species_specs to match "
            "names against; use --organism-index instead."
        )
    for spec in cfg["species_specs"]:
        if spec.get("name", "").lower() == organism_arg.lower():
            return spec["organism_index"], spec
    available = [s.get("name") for s in cfg["species_specs"]]
    sys.exit(f"Organism '{organism_arg}' not found in checkpoint config. Available: {available}")


def resolve_organism_with_data_config(
    model_cfg: dict, data_config_path: str, organism_name: str,
) -> tuple[dict, int]:
    """Ground-truth mode: load the data config, match organism_name to the model, return (spec, organism_index).

    `spec` gets `organism_index` and `cross_species` fields set, matching evaluate_splice.py's main().
    """
    data_cfg_path = Path(data_config_path)
    if not data_cfg_path.exists():
        sys.exit(f"Data config not found: {data_cfg_path}")

    species_specs = ev.load_data_config(data_cfg_path, [organism_name])
    spec = species_specs[0]

    match_result = ev.match_species_to_model(model_cfg, spec)
    if match_result is None:
        spec["organism_index"] = 0
        spec["cross_species"] = True
        print(f"  '{spec['name']}' not in model training data -> cross-species evaluation (organism_index 0 for classification)")
    else:
        org_idx, model_spec = match_result
        spec["organism_index"] = org_idx
        spec["cross_species"] = False
        model_species = ev.extract_species_from_path(model_spec["annotation_parquet"])
        print(f"  Matched '{spec['name']}' -> organism_index {org_idx} ({model_species})")

    return spec, spec["organism_index"]


def build_usage_head_configs(
    model_cfg: dict, spec: dict, cross_species_usage: bool,
) -> dict[int, tuple[dict[int, int], str]]:
    """Mirror evaluate_splice.py's usage-head matching: returns
    {usage_org_idx: (condition_mapping data_idx->model_idx, source_species_name)}.
    """
    configs: dict[int, tuple[dict[int, int], str]] = {}
    if not spec.get("usage_parquet"):
        return configs

    org_idx = spec["organism_index"]

    if spec.get("cross_species", False):
        cross_mappings = ev.match_cross_species_usage_conditions(model_cfg, spec)
        for usage_org_idx, condition_mapping in cross_mappings.items():
            source_spec = next(
                (s for s in model_cfg.get("species_specs", []) if s["organism_index"] == usage_org_idx), None,
            )
            source_species = ev.extract_species_from_path(source_spec.get("annotation_parquet", "")) if source_spec else None
            source_name = next((k for k, v in ev.SPECIES_NAME_MAP.items() if v == source_species), f"org{usage_org_idx}")
            configs[usage_org_idx] = (condition_mapping, source_name)
        return configs

    data_condition_indices, model_to_data_map = ev.match_usage_conditions(model_cfg, spec)
    if model_to_data_map is not None:
        condition_mapping = {v: k for k, v in model_to_data_map.items()}
        configs[org_idx] = (condition_mapping, spec["name"])

    if cross_species_usage:
        cross_mappings = ev.match_cross_species_usage_conditions(model_cfg, spec)
        for other_org_idx, other_mapping in cross_mappings.items():
            if other_org_idx == org_idx:
                continue
            source_spec = next(
                (s for s in model_cfg.get("species_specs", []) if s["organism_index"] == other_org_idx), None,
            )
            source_species = ev.extract_species_from_path(source_spec.get("annotation_parquet", "")) if source_spec else None
            source_name = next((k for k, v in ev.SPECIES_NAME_MAP.items() if v == source_species), f"org{other_org_idx}")
            configs[other_org_idx] = (other_mapping, source_name)

    return configs


def resolve_checkpoint_and_config(checkpoint_arg: str, model_config_arg: str | None) -> tuple[Path, Path]:
    """Resolve (best_model.pth path, config path), same convention as evaluate_splice.py's
    resolve_checkpoint(), but allows --model-config to point at a non-standard config
    filename (e.g. 'config_0-5.json') instead of requiring 'config.json' next to the checkpoint.
    """
    p = Path(checkpoint_arg)
    if p.is_dir():
        pth = p / "best_model.pth"
        default_cfg = p / "config.json"
    else:
        pth = p
        default_cfg = p.parent / "config.json"

    if not pth.exists():
        sys.exit(f"Checkpoint not found: {pth}")

    cfg_path = Path(model_config_arg) if model_config_arg else default_cfg
    if not cfg_path.exists():
        hint = "" if model_config_arg else " (pass --model-config if it isn't named config.json)"
        sys.exit(f"Model config not found: {cfg_path}{hint}")

    return pth, cfg_path


def load_condition_labels(species_spec: dict | None) -> dict[int, str] | None:
    """Invert {name: idx} condition_labels from a species' usage.json metadata, if present."""
    if not species_spec or not species_spec.get("usage_parquet"):
        return None
    meta_path = Path(species_spec["usage_parquet"]).with_suffix(".json")
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    condition_labels = meta.get("condition_labels", {})
    return {v: k for k, v in condition_labels.items()}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_window(
    model,
    fasta,
    chrom: str,
    win_start: int,
    win_end: int,
    organism_index: int,
    device: torch.device,
    usage_head_configs: dict[int, tuple[dict[int, int], str]] | None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Run the model on one window.

    Returns:
        cls_probs: (S, 5) classification probabilities.
        usage_preds_by_head: {usage_org_idx: (S, T_model)} predictions from each
            requested usage head (sharing the same embeddings, computed once).
    """
    from alphagenome_pytorch.utils.sequence import sequence_to_onehot

    seq_len = win_end - win_start
    chrom_len = len(fasta[chrom])
    fetch_start, fetch_end = max(win_start, 0), min(win_end, chrom_len)
    seq_str = str(fasta[chrom][fetch_start:fetch_end])

    seq_np = sequence_to_onehot(seq_str).astype(np.float32)  # (L, 4)
    pad_left = max(0, -win_start)
    pad_right = seq_len - seq_np.shape[0] - pad_left
    if pad_left or pad_right:
        seq_np = np.pad(seq_np, ((pad_left, max(pad_right, 0)), (0, 0)), mode="constant")
    seq_np = seq_np[:seq_len]

    seq_t = torch.from_numpy(seq_np).unsqueeze(0).to(device)
    org_t = torch.full((1,), organism_index, dtype=torch.long, device=device)

    usage_preds_by_head: dict[int, np.ndarray] = {}
    with torch.no_grad():
        out = model.predict(seq_t, org_t, resolutions=(1,))
        cls_probs = out["splice_sites_classification"]["probs"][0].cpu().numpy()  # (S, 5)

        if usage_head_configs:
            emb_out = model.predict(
                seq_t, org_t, resolutions=(1,), channels_last=False, embeddings_only=True,
            )
            emb_1bp = emb_out["embeddings_1bp"]  # (1, C, S)
            for usage_org_idx in usage_head_configs:
                head = model.splice_sites_usage_head[str(usage_org_idx)]
                usage_org_idx_t = torch.tensor([usage_org_idx], dtype=torch.long, device=device)
                usage_preds = head(emb_1bp, usage_org_idx_t, channels_last=True)["predictions"]
                usage_preds_by_head[usage_org_idx] = usage_preds.float()[0].cpu().numpy()  # (S, T)

    return cls_probs, usage_preds_by_head


# ---------------------------------------------------------------------------
# Ground-truth lookup
# ---------------------------------------------------------------------------

def ground_truth_classes(annotation, chrom: str, start: int, end: int) -> np.ndarray:
    """Full per-position true-class array over [start, end), default = background."""
    true_classes = np.full(end - start, BACKGROUND_CLASS, dtype=np.int64)
    gt_positions, gt_classes = annotation.query(chrom, start, end)
    true_classes[gt_positions - start] = gt_classes
    return true_classes


def accumulate_usage_ground_truth(
    usage_preds: np.ndarray,   # (S, T_model)
    positions: np.ndarray,     # (S,) absolute genomic positions, contiguous, matches usage_preds rows
    chrom: str,
    usage_index,
    condition_mapping: dict[int, int],  # data_idx -> model_idx
    acc: dict,
) -> None:
    """Accumulate (pos, pred, true) triples per data-condition index into `acc`."""
    site_positions, values_list, masks_list = usage_index.query(chrom, positions)
    if not site_positions:
        return

    pos_to_row = {int(p): i for i, p in enumerate(positions)}
    for site_pos, vals, mask in zip(site_positions, values_list, masks_list):
        row = pos_to_row.get(int(site_pos))
        if row is None:
            continue
        for data_c, model_c in condition_mapping.items():
            if not mask[data_c]:
                continue
            entry = acc.setdefault(data_c, {"pos": [], "pred": [], "true": []})
            entry["pos"].append(int(site_pos))
            entry["pred"].append(float(usage_preds[row, model_c]))
            entry["true"].append(float(vals[data_c]))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_region_predictions(
    out_dir: Path,
    region_name: str,
    chrom: str,
    positions: np.ndarray,
    cls_probs: np.ndarray,
    usage_preds: np.ndarray | None,
    condition_labels: dict[int, str] | None,
    true_classes: np.ndarray | None = None,
) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_class = np.argmax(cls_probs, axis=-1)

    data = {
        "Chromosome": chrom,
        "Position": positions,
        "Donor+": cls_probs[:, 0],
        "Acceptor+": cls_probs[:, 1],
        "Donor-": cls_probs[:, 2],
        "Acceptor-": cls_probs[:, 3],
        "NoSplice": cls_probs[:, 4],
        "PredClass": pred_class,
    }
    if true_classes is not None:
        data["TrueClass"] = true_classes
    df = pd.DataFrame(data)

    if usage_preds is not None:
        # Only attach usage at positions predicted as splice sites, to keep the
        # table a manageable width (usage heads can have 100s of conditions).
        site_mask = pred_class != BACKGROUND_CLASS
        n_cond = usage_preds.shape[1]
        cond_names = [condition_labels.get(c, f"cond_{c}") for c in range(n_cond)] if condition_labels else [f"cond_{c}" for c in range(n_cond)]
        usage_df = pd.DataFrame(usage_preds[site_mask], columns=cond_names)
        usage_df.insert(0, "Position", positions[site_mask])
        usage_path = out_dir / f"usage_{region_name}.parquet"
        usage_df.to_parquet(usage_path)

        # Full per-position, per-condition array for programmatic access.
        np.savez_compressed(
            out_dir / f"usage_full_{region_name}.npz",
            positions=positions, usage=usage_preds.astype(np.float32),
            condition_names=np.array(cond_names),
        )

    parquet_path = out_dir / f"predictions_{region_name}.parquet"
    df.to_parquet(parquet_path)
    print(f"  Saved: {parquet_path}" + (f"  usage_{region_name}.parquet" if usage_preds is not None else ""))


def save_usage_ground_truth(
    out_dir: Path,
    region_name: str,
    chrom: str,
    usage_acc_by_source: dict[str, dict],
    data_condition_names: dict[int, str] | None,
) -> None:
    """Long-format predicted-vs-true usage table (one row per observed site x condition x source)."""
    import pandas as pd

    rows = []
    for source_name, acc in usage_acc_by_source.items():
        for data_c, entry in acc.items():
            cond_name = data_condition_names.get(data_c, f"cond_{data_c}") if data_condition_names else f"cond_{data_c}"
            for pos, pred, true in zip(entry["pos"], entry["pred"], entry["true"]):
                rows.append({
                    "Chromosome": chrom, "Position": pos, "Source": source_name,
                    "ConditionIdx": data_c, "Condition": cond_name,
                    "Predicted": pred, "True": true,
                })

    if not rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out_dir / f"usage_ground_truth_{region_name}.parquet"
    df.to_parquet(path)
    print(f"  Saved: {path}")


def save_region_metrics(
    out_dir: Path,
    region_name: str,
    cls_metrics: dict | None,
    usage_metrics_by_source: dict[str, dict],
) -> None:
    metrics = {"classification": cls_metrics, "usage_by_source": usage_metrics_by_source}
    path = out_dir / f"metrics_{region_name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {path}")

    if cls_metrics:
        print(f"    Classification: binary AUPRC={cls_metrics['binary_auprc']:.4f}  "
              f"mean per-class AUPRC={cls_metrics['mean_splice_class_auprc']:.4f}  "
              f"(n={cls_metrics['n_positions']:,}, pos_rate={cls_metrics['positive_rate']:.4%})")
    for source, um in usage_metrics_by_source.items():
        if um.get("usage_n_conditions_evaluated", 0) > 0:
            print(f"    Usage [{source}]: mean r={um['usage_mean_pearson_r']:.4f}  "
                  f"median r={um['usage_median_pearson_r']:.4f}  "
                  f"({um['usage_n_conditions_evaluated']}/{um['usage_n_conditions_total']} conditions, "
                  f"{um['usage_n_observations']:,} obs)")
        else:
            print(f"    Usage [{source}]: no observed ground-truth conditions in this region")


def plot_region_track(
    out_dir: Path,
    region_name: str,
    positions: np.ndarray,
    cls_probs: np.ndarray,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    for c in range(4):
        ax.plot(positions, cls_probs[:, c], label=CLASS_LABELS[c], color=CLASS_COLORS[c], linewidth=0.8)
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title(region_name)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / f"track_{region_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-organism processing
# ---------------------------------------------------------------------------

def process_organism(
    model,
    cfg: dict | None,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    organism_arg: str | int,
    regions: list[tuple[str, int, int, str]],
) -> None:
    import pyfaidx

    annotation = None
    usage_index = None
    data_condition_names: dict[int, str] | None = None
    condition_mapping_by_head: dict[int, tuple[dict[int, int], str]] = {}

    if args.data_config is not None:
        spec, organism_index = resolve_organism_with_data_config(cfg, args.data_config, organism_arg)
        genome_path = spec["genome"]
        species_spec_for_labels = spec
        organism_label = spec["name"]

        print(f"[{spec['name']}] Loading annotation from {spec['annotation_parquet']} ...")
        annotation = ev.SpliceSiteAnnotation(spec["annotation_parquet"])

        if spec.get("usage_parquet") and not args.skip_usage:
            print(f"[{spec['name']}] Loading usage index from {spec['usage_parquet']} ...")
            usage_index = ev.SpliceSiteUsageIndex(
                spec["usage_parquet"],
                min_coverage=args.min_coverage,
                usage_coord_base=0,
                observed_conditions_only=args.observed_conditions_only,
            )
            data_condition_names = {v: k for k, v in usage_index.condition_labels.items()}
            condition_mapping_by_head = build_usage_head_configs(cfg, spec, args.cross_species_usage)
            for usage_org_idx, (mapping, source_name) in condition_mapping_by_head.items():
                print(f"[{spec['name']}] Will evaluate usage with '{source_name}' head "
                      f"(organism_index {usage_org_idx}, {len(mapping)} matched conditions)")
    else:
        genome_path = args.genome
        organism_index, species_spec_for_labels = resolve_organism_index_simple(cfg, organism_arg)
        organism_label = species_spec_for_labels["name"] if species_spec_for_labels else str(organism_index)
        print(f"Using organism_index={organism_index}" + (f" ('{organism_label}')" if species_spec_for_labels else ""))

        if not args.skip_usage and getattr(model, "splice_sites_usage_head", None) is not None:
            head = model.splice_sites_usage_head.get(str(organism_index))
            if head is not None:
                condition_mapping_by_head[organism_index] = (None, organism_label)
                print(f"  Usage head found for organism {organism_index}")
            else:
                print(f"  No usage head trained for organism {organism_index}; skipping usage predictions")

    usage_head_configs = {
        org_idx: cfg_tuple for org_idx, cfg_tuple in condition_mapping_by_head.items()
        if getattr(model, "splice_sites_usage_head", None) is not None and str(org_idx) in model.splice_sites_usage_head
    }
    if not usage_head_configs:
        usage_head_configs = None

    condition_labels = load_condition_labels(species_spec_for_labels)

    seq_len = (cfg or {}).get("sequence_length", SEQ_LEN_DEFAULT)
    stride = args.tile_stride or seq_len

    fasta = pyfaidx.Fasta(genome_path, as_raw=True, sequence_always_upper=True)

    print(f"[{organism_label}] Processing {len(regions)} region(s)")

    for chrom, start, end, name in regions:
        safe_name = f"{organism_label}_{name}".replace(":", "_").replace("-", "_")
        print(f"[{organism_label}] [{name}] chrom={chrom} start={start} end={end}")
        windows = make_windows(chrom, start, end, seq_len, stride)

        all_positions, all_cls_probs = [], []
        all_usage_by_head: dict[int, list[np.ndarray]] = {org_idx: [] for org_idx in (usage_head_configs or {})}
        for win_start, win_end in windows:
            cls_probs, usage_preds_by_head = predict_window(
                model, fasta, chrom, win_start, win_end, organism_index, device, usage_head_configs,
            )
            positions = np.arange(win_start, win_end)
            in_region = (positions >= start) & (positions < end)
            all_positions.append(positions[in_region])
            all_cls_probs.append(cls_probs[in_region])
            for org_idx, preds in usage_preds_by_head.items():
                all_usage_by_head[org_idx].append(preds[in_region])

        positions = np.concatenate(all_positions)
        cls_probs = np.concatenate(all_cls_probs, axis=0)
        usage_by_head = {
            org_idx: np.concatenate(chunks, axis=0) for org_idx, chunks in all_usage_by_head.items()
        }

        # De-duplicate positions from overlapping tiles (keep first occurrence).
        _, uniq_idx = np.unique(positions, return_index=True)
        uniq_idx.sort()
        positions = positions[uniq_idx]
        cls_probs = cls_probs[uniq_idx]
        usage_by_head = {org_idx: preds[uniq_idx] for org_idx, preds in usage_by_head.items()}

        # Ground truth (only in --data-config mode).
        true_classes = None
        cls_metrics = None
        usage_metrics_by_source: dict[str, dict] = {}
        if annotation is not None:
            true_classes = ground_truth_classes(annotation, chrom, start, end)
            cls_metrics = ev.compute_classification_metrics(cls_probs, true_classes)

        usage_acc_by_source: dict[str, dict] = {}
        if usage_index is not None:
            for usage_org_idx, (condition_mapping, source_name) in condition_mapping_by_head.items():
                if usage_org_idx not in usage_by_head:
                    continue
                acc: dict = {}
                accumulate_usage_ground_truth(
                    usage_by_head[usage_org_idx], positions, chrom, usage_index, condition_mapping, acc,
                )
                usage_metrics_by_source[source_name] = ev.compute_usage_metrics(acc)
                usage_acc_by_source[source_name] = acc

        # For saving, use the eval organism's own head predictions when available,
        # otherwise the first (only) requested head.
        usage_preds_to_save = usage_by_head.get(organism_index) if usage_by_head else None
        if usage_preds_to_save is None and usage_by_head:
            usage_preds_to_save = next(iter(usage_by_head.values()))

        save_region_predictions(
            out_dir, safe_name, chrom, positions, cls_probs, usage_preds_to_save, condition_labels, true_classes,
        )
        if cls_metrics is not None or usage_metrics_by_source:
            save_region_metrics(out_dir, safe_name, cls_metrics, usage_metrics_by_source)
        if usage_acc_by_source:
            save_usage_ground_truth(out_dir, safe_name, chrom, usage_acc_by_source, data_condition_names)
        if args.plot:
            plot_region_track(out_dir, safe_name, positions, cls_probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyfaidx  # noqa: F401
    except ImportError:
        sys.exit("pyfaidx is required: pip install pyfaidx")

    device = torch.device(args.device)

    pth_path, cfg_path = resolve_checkpoint_and_config(args.checkpoint, args.model_config)
    print(f"Checkpoint   : {pth_path}")
    print(f"Model config : {cfg_path}")
    cfg = ev.load_config(cfg_path)
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    model = ev.build_model(cfg, ckpt, device)
    model.eval()

    jobs = build_jobs(args)
    print(f"Processing {len(jobs)} organism job(s)")

    for organism_arg, regions in jobs:
        process_organism(model, cfg, args, device, out_dir, organism_arg, regions)

    print("Done.")


if __name__ == "__main__":
    main()
