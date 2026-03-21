#!/usr/bin/env python3
"""Evaluate splice classification accuracy split by annotation source.

Tags each annotated splice site into one of three groups:
  - "intersect"  : present in BOTH the GTF-derived parquet AND usage.parquet
  - "gtf_only"   : present in GTF parquet but NOT in usage.parquet
  - "usage_only" : present in usage.parquet but NOT in GTF parquet

Then runs the **pretrained** model and reports per-class accuracy and AUPRC
separately for each source group.

Usage
-----
    python scripts/evaluate_splice_by_source.py \\
        --pretrained-weights /path/to/model_fold_0.safetensors \\
        --bed /path/to/valid.bed \\
        --annotation /path/to/splice_sites.parquet \\
        --gtf-annotation /path/to/gtf_splice_sites.parquet \\
        --usage-parquet /path/to/usage.parquet \\
        --genome /path/to/genome.fa \\
        --output-dir /path/to/out \\
        [--organism-index 0] \\
        [--n-windows 20] \\
        [--device cuda]
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch


CLASS_LABELS = {0: "Donor+", 1: "Acceptor+", 2: "Donor-", 3: "Acceptor-"}
BACKGROUND_CLASS = 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained-weights", required=True,
                   help="Path to pretrained weights (.safetensors or .pth)")
    p.add_argument("--bed", required=True)
    p.add_argument("--annotation", required=True,
                   help="Combined splice_sites.parquet (GTF + usage union)")
    p.add_argument("--gtf-annotation", required=True,
                   help="GTF-only splice_sites parquet (sites derived solely from GTF)")
    p.add_argument("--usage-parquet", required=True,
                   help="usage.parquet (Spliser output) used to tag usage-derived sites")
    p.add_argument("--genome", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--organism-index", type=int, default=0,
                   help="Organism index (0=human, 1=mouse, default: 0)")
    p.add_argument("--n-windows", type=int, default=20,
                   help="Number of windows to evaluate (default: 20)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_pretrained_model(weights_path: str, device: torch.device):
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy

    model = AlphaGenome(dtype_policy=DtypePolicy.mixed_precision())
    print(f"Loading pretrained weights from {weights_path} …")
    # safetensors or torch checkpoint
    path = Path(weights_path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path), device="cpu")
        model.load_state_dict(sd, strict=False)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def build_usage_position_set(usage_parquet: str) -> set[tuple[str, int]]:
    """Return set of (chrom, 0-based position) present in usage.parquet."""
    import pandas as pd
    usage_json = Path(usage_parquet).with_suffix(".json")
    with open(usage_json) as f:
        meta = json.load(f)
    none_class = meta["class_labels"].get("None", 4)

    df = pd.read_parquet(usage_parquet)
    df = df[df["Label"] != none_class]
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Position"]   = df["Position"].astype(int)  # already 0-based in usage.parquet
    return set(zip(df["Chromosome"], df["Position"]))


def build_gtf_position_set(gtf_parquet: str) -> set[tuple[str, int]]:
    """Return set of (chrom, 0-based position) present in a GTF-derived parquet."""
    import pandas as pd
    df = pd.read_parquet(gtf_parquet)
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Position"]   = df["Position"].astype(int)
    return set(zip(df["Chromosome"], df["Position"]))


# Source-flag constants
SRC_INTERSECT  = 0  # in both GTF and usage
SRC_GTF_ONLY   = 1  # in GTF only
SRC_USAGE_ONLY = 2  # in usage only
SRC_BACKGROUND = 3  # unannotated

SOURCE_LABELS = {
    SRC_INTERSECT:  "intersect",
    SRC_GTF_ONLY:   "gtf_only",
    SRC_USAGE_ONLY: "usage_only",
    SRC_BACKGROUND: "background",
}


def build_annotation_index(
    annotation_parquet: str,
    gtf_positions: set[tuple[str, int]],
    usage_positions: set[tuple[str, int]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return per-chrom (positions, classes, source_flag) arrays.

    source_flag:
      SRC_INTERSECT  (0) – pos in both GTF and usage
      SRC_GTF_ONLY   (1) – pos in GTF only
      SRC_USAGE_ONLY (2) – pos in usage only
    """
    import pandas as pd
    SITE_TYPE_TO_CLASS = {"Donor+": 0, "Acceptor+": 1, "Donor-": 2, "Acceptor-": 3}

    df = pd.read_parquet(annotation_parquet)
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Position"]   = df["Position"].astype(int)

    result = {}
    for chrom, grp in df.groupby("Chromosome"):
        grp = grp.sort_values("Position")
        pos = grp["Position"].to_numpy(dtype=np.int64)
        cls = np.array([SITE_TYPE_TO_CLASS.get(st, BACKGROUND_CLASS)
                        for st in grp["SiteType"]], dtype=np.int64)
        src = np.empty(len(pos), dtype=np.int8)
        for i, p in enumerate(pos):
            key = (str(chrom), int(p))
            in_gtf   = key in gtf_positions
            in_usage = key in usage_positions
            if in_gtf and in_usage:
                src[i] = SRC_INTERSECT
            elif in_gtf:
                src[i] = SRC_GTF_ONLY
            else:
                src[i] = SRC_USAGE_ONLY  # in usage (since annotation is the union)
        result[str(chrom)] = (pos, cls, src)
    return result


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def report(label: str, probs_list: list, true_list: list) -> dict:
    if not probs_list:
        print(f"  {label}: no sites")
        return {}
    probs = np.concatenate(probs_list, axis=0)   # (N, 5)
    trues = np.concatenate(true_list,  axis=0)   # (N,)
    n = len(trues)

    pred_cls = probs.argmax(axis=1)
    # Correct = predicted class matches true class (excluding background which shouldn't appear)
    correct = (pred_cls == trues).sum()
    acc = correct / n

    print(f"\n  [{label}]  n={n:,}  accuracy={acc:.4f}")
    auprcs = {}
    for c, name in CLASS_LABELS.items():
        y_t = (trues == c).astype(np.int32)
        if y_t.sum() == 0:
            continue
        ap = auprc(y_t, probs[:, c])
        n_pos = int(y_t.sum())
        auprcs[name] = ap
        print(f"    {name:<14} AUPRC={ap:.4f}  n={n_pos:,}")

    return {"n": n, "accuracy": float(acc), "per_class_auprc": auprcs}


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model …")
    model = load_pretrained_model(args.pretrained_weights, device)
    seq_len = 131_072

    print("Building usage position set …")
    usage_positions = build_usage_position_set(args.usage_parquet)
    print(f"  {len(usage_positions):,} unique positions in usage.parquet")

    print("Building GTF position set …")
    gtf_positions = build_gtf_position_set(args.gtf_annotation)
    print(f"  {len(gtf_positions):,} unique positions in GTF parquet")

    print("Loading annotation …")
    ann = build_annotation_index(args.annotation, gtf_positions, usage_positions)
    n_intersect  = sum((src == SRC_INTERSECT).sum()  for _, _, src in ann.values())
    n_gtf_only   = sum((src == SRC_GTF_ONLY).sum()   for _, _, src in ann.values())
    n_usage_only = sum((src == SRC_USAGE_ONLY).sum() for _, _, src in ann.values())
    print(f"  Intersect sites  : {n_intersect:,}")
    print(f"  GTF-only sites   : {n_gtf_only:,}")
    print(f"  Usage-only sites : {n_usage_only:,}")

    try:
        import pyfaidx
        fasta = pyfaidx.Fasta(args.genome, as_raw=True, sequence_always_upper=True)
    except ImportError:
        sys.exit("pyfaidx required: pip install pyfaidx")

    from alphagenome_pytorch.utils.sequence import sequence_to_onehot

    # Read BED
    half = seq_len // 2
    windows: list[tuple[str, int, int]] = []
    with open(args.bed) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3 or line.startswith("#"):
                continue
            chrom  = parts[0]
            center = (int(parts[1]) + int(parts[2])) // 2
            windows.append((chrom, center - half, center + half))

    if args.n_windows:
        windows = windows[: args.n_windows]
    print(f"\nEvaluating {len(windows)} windows …")

    # Accumulators keyed by source flag
    probs_acc: dict[int, list] = {SRC_INTERSECT: [], SRC_GTF_ONLY: [], SRC_USAGE_ONLY: [], SRC_BACKGROUND: []}
    true_acc:  dict[int, list] = {SRC_INTERSECT: [], SRC_GTF_ONLY: [], SRC_USAGE_ONLY: [], SRC_BACKGROUND: []}

    org_t = torch.full((1,), args.organism_index, dtype=torch.long, device=device)

    for win_idx, (chrom, win_start, win_end) in enumerate(windows):
        if (win_idx + 1) % 10 == 0 or win_idx == 0:
            print(f"  Window {win_idx+1}/{len(windows)} …", flush=True)

        try:
            seq_str = str(fasta[chrom][win_start:win_end])
        except Exception as e:
            print(f"  Skipping {chrom}:{win_start}-{win_end}: {e}")
            continue

        seq_t = torch.from_numpy(
            sequence_to_onehot(seq_str).astype(np.float32)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            preds  = model.predict(seq_t, org_t, resolutions=(1,))
            probs  = preds["splice_sites_classification"]["probs"][0].float().cpu().numpy()  # (S, 5)

        if chrom not in ann:
            continue
        pos_arr, cls_arr, src_arr = ann[chrom]
        lo = int(np.searchsorted(pos_arr, win_start, side="left"))
        hi = int(np.searchsorted(pos_arr, win_end,   side="left"))

        # Build per-position label and source arrays for the full window
        labels = np.full(seq_len, BACKGROUND_CLASS, dtype=np.int64)
        sources = np.full(seq_len, SRC_BACKGROUND, dtype=np.int8)
        for abs_pos, true_cls, src in zip(pos_arr[lo:hi], cls_arr[lo:hi], src_arr[lo:hi]):
            rel = int(abs_pos) - win_start
            if 0 <= rel < seq_len:
                labels[rel]  = true_cls
                sources[rel] = src

        # Accumulate by source
        for src_val in (SRC_INTERSECT, SRC_GTF_ONLY, SRC_USAGE_ONLY, SRC_BACKGROUND):
            mask = sources == src_val
            if mask.any():
                probs_acc[src_val].append(probs[mask])            # (k, 5)
                true_acc[src_val].append(labels[mask])            # (k,)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  RESULTS BY ANNOTATION SOURCE")
    print("=" * 62)
    results = {}
    results["intersect"]  = report("Intersect",   probs_acc[SRC_INTERSECT],  true_acc[SRC_INTERSECT])
    results["gtf_only"]   = report("GTF-only",    probs_acc[SRC_GTF_ONLY],   true_acc[SRC_GTF_ONLY])
    results["usage_only"] = report("Usage-only",  probs_acc[SRC_USAGE_ONLY], true_acc[SRC_USAGE_ONLY])
    results["background"] = report("Background",  probs_acc[SRC_BACKGROUND], true_acc[SRC_BACKGROUND])

    # ── Plots ─────────────────────────────────────────────────────────────────
    import matplotlib
    import seaborn as sns
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc as sklearn_auc

    CLASS_COLORS = {0: "#ff7f00", 1: "#33a02c", 2: "#fdbf6f", 3: "#b2df8a"}

    for src_key, src_label in [(SRC_INTERSECT, "intersect"), (SRC_GTF_ONLY, "gtf_only"), (SRC_USAGE_ONLY, "usage_only")]:
        if not probs_acc[src_key]:
            continue
        probs_all = np.concatenate(probs_acc[src_key], axis=0)
        trues_all  = np.concatenate(true_acc[src_key],  axis=0)

        fig, ax = plt.subplots(figsize=(5, 4))
        for c, name in CLASS_LABELS.items():
            y_t = (trues_all == c).astype(np.int32)
            if y_t.sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_t, probs_all[:, c])
            ap = sklearn_auc(rec, prec)
            ax.plot(rec, prec, label=f"{name} (AUC={ap:.3f})", color=CLASS_COLORS[c])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve – {src_label}")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fn = out_dir / f"pr_curve_{src_label}.png"
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        print(f"\nSaved: {fn}")

    # ── JSON summary ──────────────────────────────────────────────────────────
    out_json = out_dir / "metrics_by_source.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: None)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
