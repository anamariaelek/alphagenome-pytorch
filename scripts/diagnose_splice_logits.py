#!/usr/bin/env python3
"""Diagnose splice-site logit alignment.

For a handful of windows from a BED file, runs the fine-tuned model and
plots per-position logit profiles centred on each annotated splice site.
If the predicted peak is systematically offset from position 0, that reveals
the mismatch between the model's coordinate convention and the annotation.

Usage
-----
    python scripts/diagnose_splice_logits.py \\
        --checkpoint /path/to/run_dir \\
        --bed /path/to/valid.bed \\
        --annotation /path/to/splice_sites.parquet \\
        --genome /path/to/genome.fa \\
        --output-dir /path/to/out \\
        [--n-windows 3] \\
        [--window-bp 5] \\
        [--device cuda]
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint dir (best_model.pth + config.json) or .pth file")
    p.add_argument("--bed", required=True, help="BED file for windows")
    p.add_argument("--annotation", required=True, help="splice_sites.parquet")
    p.add_argument("--genome", required=True, help="Reference genome FASTA")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-windows", type=int, default=3,
                   help="Number of windows to inspect (default: 3)")
    p.add_argument("--window-bp", type=int, default=5,
                   help="bp to plot either side of each annotated site (default: 5)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-sites-per-window", type=int, default=40,
                   help="Max annotated sites to include per window")
    return p.parse_args()


# ── Model loading (mirrors evaluate_splice.py) ────────────────────────────────

def load_model(checkpoint_arg: str, device: torch.device):
    """Return (model, cfg) reconstructed from checkpoint dir."""
    p = Path(checkpoint_arg)
    pth  = p / "best_model.pth" if p.is_dir() else p
    cfg_path = (p if p.is_dir() else p.parent) / "config.json"
    for f in (pth, cfg_path):
        if not f.exists():
            sys.exit(f"Not found: {f}")

    with open(cfg_path) as f:
        cfg = json.load(f)

    ckpt = torch.load(pth, map_location="cpu", weights_only=False)

    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.extensions.finetuning.heads import (
        create_splice_classification_finetuning_head,
    )
    from alphagenome_pytorch.extensions.finetuning.transfer import (
        load_trunk, remove_all_heads, prepare_for_transfer, TransferConfig,
    )

    species_specs = cfg["species_specs"]
    num_organisms = max(s["organism_index"] for s in species_specs) + 1
    dtype_str = cfg.get("dtype", "bfloat16")
    dtype_policy = (DtypePolicy.full_float32() if dtype_str == "float32"
                    else DtypePolicy.mixed_precision())

    model = AlphaGenome(dtype_policy=dtype_policy)
    model = load_trunk(model, cfg["pretrained_weights"], exclude_heads=True)
    model = remove_all_heads(model)

    if cfg.get("mode") == "lora" and cfg.get("lora_rank", 0) > 0:
        lora_targets = [t.strip() for t in cfg["lora_targets"].split(",")]
        lora_cfg = TransferConfig(mode="lora", lora_targets=lora_targets,
                                  lora_rank=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"])
        model = prepare_for_transfer(model, lora_cfg)

    cls_head = create_splice_classification_finetuning_head(num_organisms=num_organisms)
    model.splice_sites_classification_head = cls_head
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


# ── Main ──────────────────────────────────────────────────────────────────────

CLASS_LABELS = {0: "Donor+", 1: "Acceptor+", 2: "Donor-", 3: "Acceptor-", 4: "Background"}
CLASS_COLORS  = {0: "#ff7f00", 1: "#33a02c",  2: "#fdbf6f", 3: "#b2df8a",  4: "#1f78b4"}


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model …")
    model, cfg = load_model(args.checkpoint, device)
    seq_len = cfg.get("sequence_length", 131_072)

    # Load annotation
    import pandas as pd
    print(f"Loading annotation: {args.annotation}")
    ann_df = pd.read_parquet(args.annotation)
    ann_df["Chromosome"] = ann_df["Chromosome"].astype(str)
    ann_df["Position"]   = ann_df["Position"].astype(int)
    # Build per-chromosome lookup for fast range queries
    ann_by_chrom: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, grp in ann_df.groupby("Chromosome"):
        grp = grp.sort_values("Position")
        site_type_to_class = {"Donor+": 0, "Acceptor+": 1, "Donor-": 2, "Acceptor-": 3}
        ann_by_chrom[str(chrom)] = (
            grp["Position"].to_numpy(dtype=np.int64),
            np.array([site_type_to_class.get(st, 4) for st in grp["SiteType"]], dtype=np.int64),
        )

    # Load genome
    try:
        import pyfaidx
        fasta = pyfaidx.Fasta(args.genome, as_raw=True, sequence_always_upper=True)
    except ImportError:
        sys.exit("pyfaidx required: pip install pyfaidx")

    from alphagenome_pytorch.utils.sequence import sequence_to_onehot

    # Read BED
    windows: list[tuple[str, int, int]] = []
    half = seq_len // 2
    with open(args.bed) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            center = (int(parts[1]) + int(parts[2])) // 2
            windows.append((chrom, center - half, center + half))

    windows = windows[: args.n_windows]
    print(f"Inspecting {len(windows)} windows …")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_rows = []

    for win_idx, (chrom, win_start, win_end) in enumerate(windows):
        print(f"\nWindow {win_idx}: {chrom}:{win_start}-{win_end}")

        # One-hot encode sequence
        try:
            seq_str = str(fasta[chrom][win_start:win_end])
        except (KeyError, Exception) as e:
            print(f"  Skipping — FASTA fetch failed: {e}")
            continue

        seq_np = sequence_to_onehot(seq_str).astype(np.float32)  # (S, 4)
        seq_t  = torch.from_numpy(seq_np).unsqueeze(0).to(device)  # (1, S, 4)
        org_t  = torch.zeros(1, dtype=torch.long, device=device)   # human = 0

        # Run model
        with torch.no_grad():
            out = model.predict(seq_t, org_t, resolutions=(1,),
                                channels_last=False, embeddings_only=True)
            emb = out["embeddings_1bp"]  # (1, C, S)
            result = model.splice_sites_classification_head(emb, org_t, channels_last=True)
            logits = result["logits"][0].float().cpu().numpy()   # (S, 5)
            probs  = result["probs"][0].float().cpu().numpy()    # (S, 5)

        # Get annotated sites in this window
        if chrom not in ann_by_chrom:
            print(f"  No annotation for chrom '{chrom}'")
            continue
        pos_arr, cls_arr = ann_by_chrom[chrom]
        lo = int(np.searchsorted(pos_arr, win_start, side="left"))
        hi = int(np.searchsorted(pos_arr, win_end,   side="left"))
        site_pos = pos_arr[lo:hi]
        site_cls = cls_arr[lo:hi]

        if len(site_pos) == 0:
            print("  No annotated sites in window — skipping")
            continue

        # Cap to max_sites_per_window
        if len(site_pos) > args.max_sites_per_window:
            site_pos = site_pos[:args.max_sites_per_window]
            site_cls = site_cls[:args.max_sites_per_window]

        print(f"  Annotated sites: {len(site_pos)}")

        W = args.window_bp

        # Collect only misclassified sites for plotting
        wrong_indices = []
        for i in range(len(site_pos)):
            rel_pos = int(site_pos[i]) - win_start
            peak_cls = int(np.argmax(probs[rel_pos]))
            if peak_cls != int(site_cls[i]):
                wrong_indices.append(i)

        print(f"  Misclassified sites: {len(wrong_indices)} / {len(site_pos)}")
        if not wrong_indices:
            print("  All sites correctly predicted — skipping plots for this window")
            continue

        fig_rows = min(len(wrong_indices), 8)
        fig, axes = plt.subplots(fig_rows, 1, figsize=(10, 2.5 * fig_rows), squeeze=False)

        for plot_i, i in enumerate(wrong_indices[:fig_rows]):
            ax = axes[plot_i, 0]
            abs_pos  = int(site_pos[i])
            true_cls = int(site_cls[i])
            rel_pos  = abs_pos - win_start   # 0-based index into logits

            lo2 = max(0, rel_pos - W)
            hi2 = min(seq_len, rel_pos + W + 1)
            offsets = np.arange(lo2, hi2) - rel_pos

            for c in range(5):
                ax.plot(offsets, probs[lo2:hi2, c],
                        label=CLASS_LABELS[c], color=CLASS_COLORS[c],
                        alpha=0.85, lw=1.5)

            ax.axvline(0, color="black", lw=1.2, ls="--", label="annotated pos")
            peak_cls = int(np.argmax(probs[rel_pos]))
            peak_offset = int(np.argmax(probs[lo2:hi2, true_cls])) + lo2 - rel_pos

            ax.set_title(
                f"Site {i} (WRONG): {chrom}:{abs_pos}  true={CLASS_LABELS[true_cls]}  "
                f"pred_at_site={CLASS_LABELS[peak_cls]}  "
                f"peak_offset_for_true_cls={peak_offset:+d}",
                fontsize=8,
            )
            ax.set_xlabel("Offset from annotated position (bp)")
            ax.set_ylabel("Probability")
            ax.legend(loc="upper right", fontsize=6, ncol=5)
            ax.grid(True, alpha=0.3)

            summary_rows.append({
                "window": win_idx,
                "chrom": chrom,
                "abs_pos": abs_pos,
                "true_cls": CLASS_LABELS[true_cls],
                "pred_cls_at_site": CLASS_LABELS[peak_cls],
                "peak_offset_for_true_cls": peak_offset,
                "logit_true_cls_at_site": float(logits[rel_pos, true_cls]),
                "prob_true_cls_at_site":  float(probs[rel_pos, true_cls]),
            })

        fig.suptitle(f"Window {win_idx}: {chrom}:{win_start}-{win_end}", fontsize=10)
        fig.tight_layout()
        fn = out_dir / f"logit_profiles_window{win_idx}.png"
        fig.savefig(fn, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fn}")

    # Summary CSV
    if summary_rows:
        import pandas as pd
        summary = pd.DataFrame(summary_rows)
        csv_path = out_dir / "offset_summary.csv"
        summary.to_csv(csv_path, index=False)
        print(f"\nOffset summary saved to {csv_path}")
        print(summary[["true_cls", "peak_offset_for_true_cls", "pred_cls_at_site"]].to_string())

        # Histogram of peak offsets per class
        fig, ax = plt.subplots(figsize=(7, 4))
        for cls_name in ["Donor+", "Acceptor+", "Donor-", "Acceptor-"]:
            sub = summary[summary["true_cls"] == cls_name]["peak_offset_for_true_cls"]
            if len(sub):
                ax.hist(sub, bins=range(-args.window_bp, args.window_bp + 2),
                        alpha=0.6, label=cls_name)
        ax.axvline(0, color="black", lw=1.5, ls="--")
        ax.set_xlabel("Peak offset from annotated position (bp)")
        ax.set_ylabel("Count")
        ax.set_title("Systematic offset between annotation and predicted peak")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "offset_histogram.png", dpi=120)
        plt.close(fig)
        print(f"Saved: {out_dir}/offset_histogram.png")


if __name__ == "__main__":
    main()
