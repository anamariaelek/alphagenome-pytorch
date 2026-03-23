#!/usr/bin/env python3
"""Evaluate splice-site model performance on a test split.

Computes per-organism:
  - AUPRC for splice site classification
      * Binary: any splice site (Donor/Acceptor) vs. background
      * Per-class one-vs-rest: Donor+, Acceptor+, Donor-, Acceptor-
  - Pearson r for splice site usage (per condition, then averaged)

The simplest invocation reads all data paths from a config.json saved
alongside the training checkpoint (produced by finetune_splice.py).
Provide ``--bed`` to override the default BED files (e.g. to use a
held-out test split instead of the validation set).

Examples
--------
    # Evaluate on validation BEDs taken from config.json
    python scripts/evaluate_splice.py \\
        --checkpoint /path/to/20260318_114448_132kb_lora \\
        --device cuda

    # Override with explicit test BEDs (one per organism, in organism index order)
    python scripts/evaluate_splice.py \\
        --checkpoint /path/to/20260318_114448_132kb_lora \\
        --bed /data/human_test.bed /data/mouse_test.bed \\
        --device cuda

    # Save results and plots to a directory
    python scripts/evaluate_splice.py \\
        --checkpoint /path/to/20260318_114448_132kb_lora \\
        --bed /data/human_test.bed /data/mouse_test.bed \\
        --output-dir results/eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class _Timer:
    """Simple context manager and utility for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        print(f"  ⏱  {self.name:<45s} {self.elapsed:>7.2f}s")
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds to a readable string."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{seconds/60:.2f}m"
        else:
            return f"{seconds/3600:.2f}h"


class _Tee:
    """Mirror writes to both the original stream and an open log file."""

    def __init__(self, stream, log_path: Path):
        self._stream = stream
        self._log = open(log_path, "a", buffering=1)  # line-buffered

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def fileno(self) -> int:          # needed by some C libs
        return self._stream.fileno()

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        self._log.close()

# Splice class labels (must match SpliceSiteAnnotation)
SPLICE_CLASS_NAMES = ["Donor+", "Acceptor+", "Donor-", "Acceptor-"]
BACKGROUND_CLASS = 4
ORGANISM_NAMES = {0: "human", 1: "mouse"}

# Visual labels and colors matching predict_splicing_windows.py
CLASS_LABELS = {0: 'donor +', 1: 'acceptor +', 2: 'donor -', 3: 'acceptor -', 4: 'no splice site'}
CLASS_COLORS  = {0: '#ff7f00', 1: '#33a02c', 2: '#fdbf6f', 3: '#b2df8a', 4: '#1f78b4'}

# Source-flag constants (for by-source AUPRC analysis)
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


# ---------------------------------------------------------------------------
# Source-annotation helpers
# ---------------------------------------------------------------------------

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
    df["Position"]   = df["Position"].astype(int)
    return set(zip(df["Chromosome"], df["Position"]))


def build_gtf_position_set(gtf_parquet: str) -> set[tuple[str, int]]:
    """Return set of (chrom, 0-based position) present in a GTF-derived parquet."""
    import pandas as pd
    df = pd.read_parquet(gtf_parquet)
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Position"]   = df["Position"].astype(int)
    return set(zip(df["Chromosome"], df["Position"]))


def build_source_arrays(
    annotation_parquet: str,
    gtf_positions: set[tuple[str, int]],
    usage_positions: set[tuple[str, int]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return per-chrom (sorted_positions, src_flags) for source tagging."""
    import pandas as pd
    df = pd.read_parquet(annotation_parquet)
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Position"]   = df["Position"].astype(int)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, grp in df.groupby("Chromosome"):
        grp = grp.sort_values("Position")
        pos = grp["Position"].to_numpy(dtype=np.int64)
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
                src[i] = SRC_USAGE_ONLY
        result[str(chrom)] = (pos, src)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate splice AUPRC and usage Pearson r on a test split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory (contains best_model.pth + config.json) "
             "or directly to the .pth file. The companion config.json must be in the "
             "same directory.",
    )
    parser.add_argument(
        "--bed",
        nargs="*",
        default=None,
        help="BED file(s) for evaluation, one per organism, in the same order as "
             "config.json species_specs. If omitted, val_bed from config.json is used.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-coverage", type=int, default=10,
                        help="Min (Alpha+Beta) coverage for usage targets (default: 10)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save results JSON and plots.")
    parser.add_argument(
        "--skip-predictions", action="store_true",
        help="Skip inference; load saved predictions from --output-dir instead.",
    )
    parser.add_argument(
        "--gtf-annotation", nargs="*", default=None,
        help="GTF-only splice-site parquets, one per organism (same order as "
             "species_specs). Enables per-source evaluation.",
    )
    parser.add_argument(
        "--per-condition", action="store_true",
        help="Evaluate and plot usage separately per condition.",
    )
    parser.add_argument(
        "--per-tissue", action="store_true",
        help="Evaluate and plot usage separately per tissue (from usage metadata).",
    )
    parser.add_argument(
        "--per-source", action="store_true",
        help="Evaluate and plot separately for GTF-only vs usage-only sites.",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip all plotting (metrics only, faster).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint / config loading
# ---------------------------------------------------------------------------

def resolve_checkpoint(checkpoint_arg: str) -> tuple[Path, Path]:
    """Return (pth_path, config_path) from a directory or .pth path."""
    p = Path(checkpoint_arg)
    if p.is_dir():
        pth = p / "best_model.pth"
        cfg = p / "config.json"
    else:
        pth = p
        cfg = p.parent / "config.json"

    if not pth.exists():
        sys.exit(f"Checkpoint not found: {pth}")
    if not cfg.exists():
        sys.exit(f"config.json not found: {cfg}")
    return pth, cfg


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

def build_model(cfg: dict, ckpt: dict, device: torch.device) -> nn.Module:
    """Reconstruct the AlphaGenome model with LoRA and splice heads from a checkpoint.

    Mirrors the model construction logic in finetune_splice.py:create_model().
    """
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.extensions.finetuning.heads import (
        create_splice_classification_finetuning_head,
    )
    from alphagenome_pytorch.extensions.finetuning.transfer import (
        load_trunk,
        remove_all_heads,
        prepare_for_transfer,
        TransferConfig,
    )

    species_specs = cfg["species_specs"]
    num_organisms = max(s["organism_index"] for s in species_specs) + 1

    dtype_str = cfg.get("dtype", "bfloat16")
    dtype_policy = (
        DtypePolicy.full_float32() if dtype_str == "float32"
        else DtypePolicy.mixed_precision()
    )

    print(f"Building model (mode={cfg['mode']}, dtype={dtype_str}, "
          f"num_organisms={num_organisms})...")

    model = AlphaGenome(dtype_policy=dtype_policy)

    # Load pretrained trunk weights (base model, no heads)
    model = load_trunk(model, cfg["pretrained_weights"], exclude_heads=True)

    # Remove all heads so we can attach fresh fine-tuned ones
    model = remove_all_heads(model)

    # Apply LoRA adapters (must happen *before* loading model_state_dict
    # so the parameter names match)
    if cfg["mode"] == "lora" and cfg.get("lora_rank", 0) > 0:
        lora_targets = [t.strip() for t in cfg["lora_targets"].split(",")]
        print(f"  Applying LoRA: rank={cfg['lora_rank']}, alpha={cfg['lora_alpha']}, "
              f"targets={lora_targets}")
        lora_cfg = TransferConfig(
            mode="lora",
            lora_targets=lora_targets,
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
        )
        model = prepare_for_transfer(model, lora_cfg)

    # Attach the fine-tuned classification head
    cls_head = create_splice_classification_finetuning_head(num_organisms=num_organisms)
    model.splice_sites_classification_head = cls_head

    # Load fine-tuned weights (LoRA adapters + classification head)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded model_state_dict (epoch {ckpt.get('epoch', '?')})")

    model.to(device).eval()
    return model


def build_usage_heads(
    cfg: dict,
    ckpt: dict,
    device: torch.device,
) -> dict[int, nn.Module]:
    """Reconstruct per-organism usage heads and load their weights."""
    from alphagenome_pytorch.extensions.finetuning.heads import (
        create_splice_usage_finetuning_head,
    )

    species_specs = cfg["species_specs"]
    num_organisms = max(s["organism_index"] for s in species_specs) + 1
    species_n_conditions: dict[int, int] = {
        int(k): v for k, v in cfg.get("species_n_conditions", {}).items()
    }
    usage_heads_state_dicts: dict[str, dict] = ckpt.get("usage_heads_state_dicts", {})

    usage_heads: dict[int, nn.Module] = {}
    for org_idx, n_cond in species_n_conditions.items():
        if n_cond <= 0:
            continue
        head = create_splice_usage_finetuning_head(
            n_conditions=n_cond,
            num_organisms=num_organisms,
        )
        sd_key = str(org_idx)
        if sd_key in usage_heads_state_dicts:
            head.load_state_dict(usage_heads_state_dicts[sd_key])
            print(f"  Loaded usage head for organism {org_idx} ({n_cond} conditions)")
        else:
            print(f"  Warning: no saved weights for usage head organism {org_idx}, using random init",
                  file=sys.stderr)
        head.to(device).eval()
        usage_heads[org_idx] = head

    return usage_heads


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def collect_predictions(
    model: nn.Module,
    usage_heads: dict[int, nn.Module],
    loader,
    device: torch.device,
    organism_index: int,
    windows: list | None = None,
    source_arrays: dict | None = None,
    seq_len: int = 131_072,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray | None, dict]:
    """Run inference for one organism.

    Returns
    -------
    cls_probs               : (N_positions, 5)  float32
    cls_labels              : (N_positions,)    int64
    usage_per_cond          : dict  condition_idx -> {'pred': list[float], 'true': list[float]}
    src_tags                : (N_positions,)    int8  or None if source tracking disabled
    usage_per_cond_by_source: dict  'gtf_only'/'usage_only' -> cond_idx -> {'pred': list[float], 'true': list[float]}
    """
    all_cls_probs: list[np.ndarray] = []
    all_cls_labels: list[np.ndarray] = []
    usage_per_cond: dict[int, dict[str, list[float]]] = {}
    usage_per_cond_by_source: dict = {}  # {source_name: {cond_idx: {pred, true}}}
    all_src_tags: list[np.ndarray] | None = (
        [] if (source_arrays is not None and windows is not None) else None
    )
    win_ptr = 0

    # Usage head for this organism (may be None if not available)
    usage_head = usage_heads.get(organism_index)

    for batch in tqdm(loader, desc="  Inference", unit="batch"):
        seq = batch["sequence"].to(device)
        org_idx_t = batch["organism_index"].to(device)

        with torch.no_grad():
            if usage_head is not None:
                # Get 1bp NCL embeddings then call each head manually.
                # channels_last=False -> embeddings are (B, C, S) as expected by head.forward()
                out = model.predict(
                    seq, org_idx_t,
                    resolutions=(1,),
                    channels_last=False,
                    embeddings_only=True,
                )
                emb_1bp = out["embeddings_1bp"]  # (B, C, S)

                cls_probs = model.splice_sites_classification_head(
                    emb_1bp, org_idx_t, channels_last=True
                )["probs"].float()  # (B, S, 5)

                # Only use the usage head for this organism
                usage_preds = usage_head(
                    emb_1bp, org_idx_t, channels_last=True
                )["predictions"].float()  # (B, S, n_cond)

            else:
                # No usage head: run full forward, get classification only
                preds = model.predict(seq, org_idx_t, resolutions=(1,))
                cls_probs = preds["splice_sites_classification"]["probs"]  # (B, S, 5)
                usage_preds = None

        all_cls_probs.append(cls_probs.cpu().numpy().reshape(-1, 5))
        all_cls_labels.append(batch["classification_labels"].numpy().reshape(-1))

        if all_src_tags is not None:
            B = batch["sequence"].shape[0]
            for chrom, wstart, wend in windows[win_ptr:win_ptr + B]:
                src_arr = np.full(seq_len, SRC_BACKGROUND, dtype=np.int8)
                if chrom in source_arrays:
                    pos_arr, src_flags = source_arrays[chrom]
                    lo = int(np.searchsorted(pos_arr, wstart, side="left"))
                    hi = int(np.searchsorted(pos_arr, wend,   side="left"))
                    for abs_pos, sf in zip(pos_arr[lo:hi], src_flags[lo:hi]):
                        rel = int(abs_pos) - wstart
                        if 0 <= rel < seq_len:
                            src_arr[rel] = sf
                all_src_tags.append(src_arr)
            win_ptr += B

        if usage_preds is not None and "usage_positions" in batch:
            _accumulate_usage(
                usage_preds.cpu().numpy(),
                batch["usage_positions"].numpy(),
                batch["usage_values"].numpy(),
                batch["usage_mask"].numpy(),
                usage_per_cond,
            )
            # Also accumulate by source if available
            if all_src_tags is not None and len(all_src_tags) > 0:
                # Extract source tags for this batch's usage sites
                B = batch["sequence"].shape[0]
                src_batch = all_src_tags[-B:] if B > 0 else []
                if src_batch:
                    _accumulate_usage_by_source(
                        usage_preds.cpu().numpy(),
                        batch["usage_positions"].numpy(),
                        batch["usage_values"].numpy(),
                        batch["usage_mask"].numpy(),
                        src_batch,
                        usage_per_cond_by_source,
                    )

    return (
        np.concatenate(all_cls_probs, axis=0),
        np.concatenate(all_cls_labels, axis=0),
        usage_per_cond,
        np.concatenate(all_src_tags) if all_src_tags is not None else None,
        usage_per_cond_by_source if usage_per_cond_by_source else {},
    )


def _accumulate_usage(
    usage_preds: np.ndarray,    # (B, S, T)
    positions: np.ndarray,      # (B, max_sites)  -1 padded
    values: np.ndarray,         # (B, max_sites, n_cond)
    mask: np.ndarray,           # (B, max_sites, n_cond) bool
    acc: dict,
) -> None:
    B = positions.shape[0]
    n_cond = values.shape[2]
    for i in range(B):
        valid = positions[i] != -1
        if not valid.any():
            continue
        valid_pos = positions[i][valid]
        valid_vals = values[i][valid]       # (k, n_cond)
        valid_mask = mask[i][valid]         # (k, n_cond) bool
        valid_preds = usage_preds[i, valid_pos, :]  # (k, T)

        for c in range(n_cond):
            obs = valid_mask[:, c]
            if not obs.any():
                continue
            entry = acc.setdefault(c, {"pred": [], "true": []})
            entry["pred"].extend(valid_preds[obs, c].tolist())
            entry["true"].extend(valid_vals[obs, c].tolist())


def _accumulate_usage_by_source(
    usage_preds: np.ndarray,    # (B, S, T)
    positions: np.ndarray,      # (B, max_sites)  -1 padded
    values: np.ndarray,         # (B, max_sites, n_cond)
    mask: np.ndarray,           # (B, max_sites, n_cond) bool
    src_tags_batch: list,       # list of src_arr (one per batch sample)
    acc: dict,
) -> None:
    """Accumulate usage predictions by source (GTF-only vs usage-only)."""
    B = positions.shape[0]
    n_cond = values.shape[2]
    for i in range(B):
        valid = positions[i] != -1
        if not valid.any():
            continue
        valid_pos = positions[i][valid]
        valid_vals = values[i][valid]       # (k, n_cond)
        valid_mask = mask[i][valid]         # (k, n_cond) bool
        valid_preds = usage_preds[i, valid_pos, :]  # (k, T)
        
        # Get source tags for this batch's positions
        if i < len(src_tags_batch):
            src_arr = src_tags_batch[i]
            valid_src = src_arr[valid_pos]  # (k,) source tags for valid positions
        else:
            continue

        for c in range(n_cond):
            obs = valid_mask[:, c]
            if not obs.any():
                continue
            
            # Separate by source: GTF-only (1) vs usage-only (2)
            gtf_only_mask = valid_src[obs] == SRC_GTF_ONLY
            usage_only_mask = valid_src[obs] == SRC_USAGE_ONLY
            
            for src_name, src_mask in [("gtf_only", gtf_only_mask), ("usage_only", usage_only_mask)]:
                if not src_mask.any():
                    continue
                src_entry = acc.setdefault(src_name, {})
                cond_entry = src_entry.setdefault(c, {"pred": [], "true": []})
                cond_entry["pred"].extend(valid_preds[obs][src_mask, c].tolist())
                cond_entry["true"].extend(valid_vals[obs][src_mask, c].tolist())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_class_auprc(c: int, labels: np.ndarray, probs: np.ndarray) -> tuple[int, str, float, int]:
    """Compute AUPRC for a single class (for parallelization)."""
    from sklearn.metrics import average_precision_score
    y_c = (labels == c).astype(np.int32)
    n_pos = int(y_c.sum())
    if n_pos == 0:
        return c, SPLICE_CLASS_NAMES[c], float("nan"), 0
    ap = float(average_precision_score(y_c, probs[:, c]))
    return c, SPLICE_CLASS_NAMES[c], ap, n_pos


def compute_classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        sys.exit("scikit-learn is required. Install: pip install scikit-learn")

    y_binary = (labels != BACKGROUND_CLASS).astype(np.int32)
    s_binary = 1.0 - probs[:, BACKGROUND_CLASS]
    binary_auprc = float(average_precision_score(y_binary, s_binary))
    pos_rate = float(y_binary.mean())

    # Parallel per-class AUPRC computation
    per_class_auprc: dict[str, float] = {}
    per_class_n: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_compute_class_auprc, c, labels, probs)
            for c in range(len(SPLICE_CLASS_NAMES))
        ]
        for future in futures:
            c, name, ap, n_pos = future.result()
            if not np.isnan(ap):
                per_class_auprc[name] = ap
                per_class_n[name] = n_pos

    mean_per_class = (
        float(np.mean(list(per_class_auprc.values()))) if per_class_auprc else float("nan")
    )

    return {
        "binary_auprc": binary_auprc,
        "positive_rate": pos_rate,
        "per_class_auprc": per_class_auprc,
        "per_class_n_positives": per_class_n,
        "mean_splice_class_auprc": mean_per_class,
        "n_positions": int(len(labels)),
    }


def _compute_condition_pearsonr(cond_data: tuple[int, dict]) -> float:
    """Compute Pearson r for a single condition (for parallelization)."""
    from scipy.stats import pearsonr
    cond_idx, data = cond_data
    pred = np.array(data["pred"], dtype=np.float32)
    true = np.array(data["true"], dtype=np.float32)
    if len(pred) < 2 or pred.std() < 1e-8 or true.std() < 1e-8:
        return float("nan")
    r, _ = pearsonr(pred, true)
    return float(r)


def compute_usage_metrics(usage_per_cond: dict) -> dict:
    try:
        from scipy.stats import pearsonr
    except ImportError:
        sys.exit("scipy is required. Install: pip install scipy")

    n_obs_total = sum(len(data["pred"]) for data in usage_per_cond.values())
    
    # Parallel Pearson r computation per condition
    rs: list[float] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_compute_condition_pearsonr, (cond_idx, data))
            for cond_idx, data in usage_per_cond.items()
        ]
        for future in futures:
            r = future.result()
            if not np.isnan(r):
                rs.append(r)

    if not rs:
        return {
            "usage_mean_pearson_r": float("nan"),
            "usage_median_pearson_r": float("nan"),
            "usage_n_conditions_evaluated": 0,
            "usage_n_conditions_total": len(usage_per_cond),
            "usage_n_observations": n_obs_total,
        }

    return {
        "usage_mean_pearson_r": float(np.mean(rs)),
        "usage_median_pearson_r": float(np.median(rs)),
        "usage_n_conditions_evaluated": len(rs),
        "usage_n_conditions_total": len(usage_per_cond),
        "usage_n_observations": n_obs_total,
    }


def _compute_source_usage_metrics(src_name: str, src_conds_dict: dict) -> tuple[str, dict]:
    """Compute usage metrics for a single source (for parallelization)."""
    from scipy.stats import pearsonr
    
    rs: list[float] = []
    n_obs_total = 0
    
    for data in src_conds_dict.values():
        pred = np.array(data["pred"], dtype=np.float32)
        true = np.array(data["true"], dtype=np.float32)
        n_obs_total += len(pred)
        if len(pred) < 2 or pred.std() < 1e-8 or true.std() < 1e-8:
            continue
        r, _ = pearsonr(pred, true)
        rs.append(float(r))
    
    if rs:
        result = {
            "usage_mean_pearson_r": float(np.mean(rs)),
            "usage_median_pearson_r": float(np.median(rs)),
            "usage_n_conditions_evaluated": len(rs),
            "usage_n_conditions_total": len(src_conds_dict),
            "usage_n_observations": n_obs_total,
        }
    else:
        result = {
            "usage_mean_pearson_r": float("nan"),
            "usage_median_pearson_r": float("nan"),
            "usage_n_conditions_evaluated": 0,
            "usage_n_conditions_total": len(src_conds_dict),
            "usage_n_observations": n_obs_total,
        }
    
    return src_name, result


def compute_usage_metrics_by_source(usage_by_source: dict) -> dict:
    """Compute usage metrics separately for GTF-only and usage-only sites (parallelized)."""
    results = {}
    
    # Parallel computation per source
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for src_name in ("gtf_only", "usage_only"):
            if src_name in usage_by_source:
                futures.append(
                    executor.submit(_compute_source_usage_metrics, src_name, usage_by_source[src_name])
                )
        
        for future in futures:
            src_name, src_result = future.result()
            results[src_name] = src_result
    
    return results


def print_usage_metrics(org_name: str, usage_m: dict) -> None:
    print(f"\n{org_name} – Usage Prediction Metrics:")
    print(f"  Mean Pearson r (across conditions): {usage_m['usage_mean_pearson_r']:.4f}")
    print(f"  Median Pearson r (across conditions): {usage_m['usage_median_pearson_r']:.4f}")
    print(
        f"  Conditions evaluated: {usage_m['usage_n_conditions_evaluated']} / {usage_m['usage_n_conditions_total']}"
    )
    print(f"  Total observations: {usage_m['usage_n_observations']:,}")


def print_usage_metrics_by_source(org_name: str, usage_by_source_m: dict) -> None:
    """Print usage metrics separately for GTF-only and usage-only sites."""
    print(f"\n{org_name} – Usage Prediction Metrics by Source:")
    
    for src_name in ("gtf_only", "usage_only"):
        if src_name not in usage_by_source_m:
            continue
        
        m = usage_by_source_m[src_name]
        src_label = "GTF-only" if src_name == "gtf_only" else "Usage-only"
        print(f"\n  {src_label}:")
        print(f"    Mean Pearson r: {m['usage_mean_pearson_r']:.4f}")
        print(f"    Median Pearson r: {m['usage_median_pearson_r']:.4f}")
        print(f"    Conditions evaluated: {m['usage_n_conditions_evaluated']} / {m['usage_n_conditions_total']}")
        print(f"    Total observations: {m['usage_n_observations']:,}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pr_curves(
    cls_probs: np.ndarray,
    cls_labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Precision-recall curves for the 4 splice-site classes (one-vs-rest)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=(5, 4))
    n_sites = len(cls_labels)
    for c in range(4):
        y_true = (cls_labels == c).astype(np.int32)
        n_pos = y_true.sum()
        if n_pos == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true, cls_probs[:, c])
        pr_auc = float(average_precision_score(y_true, cls_probs[:, c]))
        ax.plot(recall, precision,
                label=f"{CLASS_LABELS[c]} (AUC={pr_auc:.3f}, n={n_pos:,})",
                color=CLASS_COLORS[c])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve – {title} (n={n_sites:,})")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_usage_density(
    usage_per_cond: dict,
    title: str,
    output_path: Path,
) -> None:
    """Hexbin density plot of predicted vs true splice-site usage.

    Adapted from ``plot_sse_density`` in predict_splicing_windows.py.
    Pools all conditions into a single scatter; shows marginal histograms and
    the overall Pearson r.
    """
    if not usage_per_cond:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_all: list[float] = []
    pred_all: list[float] = []
    for data in usage_per_cond.values():
        true_all.extend(data["true"])
        pred_all.extend(data["pred"])

    if len(true_all) < 2:
        return

    true_arr = np.array(true_all, dtype=np.float32)
    pred_arr = np.array(pred_all, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(4.8, 4))

    hb = ax.hexbin(true_arr, pred_arr, gridsize=25, cmap="magma_r", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count", pad=0.02)

    num_points = len(true_arr)
    if true_arr.std() > 1e-8 and pred_arr.std() > 1e-8:
        corr = float(np.corrcoef(true_arr, pred_arr)[0, 1])
        ax.text(0.05, 0.95, f"r = {corr:.3f}\nn = {num_points:,}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top")

    # Marginal histograms (matching plot_sse_density style)
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.2], sharex=ax)
    ax_histx.hist(true_arr, bins=30, color="gray", alpha=0.7)
    ax_histx.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    ax_histy = ax.inset_axes([1.05, 0, 0.2, 1], sharey=ax)
    ax_histy.hist(pred_arr, bins=30, orientation="horizontal", color="gray", alpha=0.7)
    ax_histy.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    ax.set_xlabel("True Usage")
    ax.set_ylabel("Predicted Usage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_usage_per_condition(
    usage_per_cond: dict,
    out_dir: Path,
    org_name: str,
) -> None:
    """One hexbin plot per condition."""
    if not usage_per_cond:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cond_idx, data in sorted(usage_per_cond.items()):
        true_vals = np.array(data["true"], dtype=np.float32)
        pred_vals = np.array(data["pred"], dtype=np.float32)

        if len(true_vals) < 2:
            continue

        fig, ax = plt.subplots(figsize=(4.8, 4))
        hb = ax.hexbin(true_vals, pred_vals, gridsize=25, cmap="magma_r", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count", pad=0.02)

        num_points = len(true_vals)
        if true_vals.std() > 1e-8 and pred_vals.std() > 1e-8:
            corr = float(np.corrcoef(true_vals, pred_vals)[0, 1])
            ax.text(0.05, 0.95, f"r = {corr:.3f}\nn = {num_points:,}",
                    transform=ax.transAxes, fontsize=10, verticalalignment="top")

        ax.set_xlabel("True Usage")
        ax.set_ylabel("Predicted Usage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Usage Condition {cond_idx}")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fn = out_dir / f"usage_per_condition_{org_name}_cond_{cond_idx}.png"
        fig.savefig(fn, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fn}")


def plot_usage_per_tissue(
    usage_per_cond: dict,
    usage_metadata_path: Path,
    out_dir: Path,
    org_name: str,
) -> None:
    """Group conditions by tissue and plot aggregated usage."""
    if not usage_per_cond:
        return
    
    if not usage_metadata_path.exists():
        print(f"  Warning: Usage metadata not found at {usage_metadata_path}, skipping per-tissue plots")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load tissue mapping from metadata
    try:
        with open(usage_metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load usage metadata: {e}, skipping per-tissue plots")
        return

    # Extract tissue mapping if available
    tissue_map = metadata.get("tissue_map", metadata.get("condition_tissue", {}))
    if not tissue_map:
        print(f"  Warning: No tissue mapping in metadata, skipping per-tissue plots")
        return

    # Group conditions by tissue
    tissue_data: dict[str, dict[str, list]] = {}
    for cond_idx, data in usage_per_cond.items():
        cond_idx_str = str(cond_idx)
        tissue = tissue_map.get(cond_idx_str, f"unknown_{cond_idx}")
        if tissue not in tissue_data:
            tissue_data[tissue] = {"pred": [], "true": []}
        tissue_data[tissue]["pred"].extend(data["pred"])
        tissue_data[tissue]["true"].extend(data["true"])

    # Plot per tissue
    for tissue, data in sorted(tissue_data.items()):
        true_vals = np.array(data["true"], dtype=np.float32)
        pred_vals = np.array(data["pred"], dtype=np.float32)

        if len(true_vals) < 2:
            continue

        fig, ax = plt.subplots(figsize=(4.8, 4))
        hb = ax.hexbin(true_vals, pred_vals, gridsize=30, cmap="magma_r", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count", pad=0.02)

        num_points = len(true_vals)
        if true_vals.std() > 1e-8 and pred_vals.std() > 1e-8:
            corr = float(np.corrcoef(true_vals, pred_vals)[0, 1])
            ax.text(0.05, 0.95, f"r = {corr:.3f}\nn = {num_points:,}",
                    transform=ax.transAxes, fontsize=10, verticalalignment="top")

        ax.set_xlabel("True Usage")
        ax.set_ylabel("Predicted Usage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Usage Tissue: {tissue}")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fn = out_dir / f"usage_per_tissue_{org_name}_{tissue}.png"
        fig.savefig(fn, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fn}")


def plot_usage_by_source(
    usage_by_source: dict,
    out_dir: Path,
    org_name: str,
) -> None:
    """Plot aggregated usage separately for GTF-only and usage-only sites."""
    if not usage_by_source:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for src_name in ("gtf_only", "usage_only"):
        if src_name not in usage_by_source:
            continue

        src_data_per_cond = usage_by_source[src_name]
        if not src_data_per_cond:
            continue

        # Aggregate across conditions
        true_all: list[float] = []
        pred_all: list[float] = []
        for data in src_data_per_cond.values():
            true_all.extend(data["true"])
            pred_all.extend(data["pred"])

        if len(true_all) < 2:
            continue

        true_arr = np.array(true_all, dtype=np.float32)
        pred_arr = np.array(pred_all, dtype=np.float32)

        fig, ax = plt.subplots(figsize=(4.8, 4))
        hb = ax.hexbin(true_arr, pred_arr, gridsize=25, cmap="magma_r", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count", pad=0.02)

        num_points = len(true_arr)
        if true_arr.std() > 1e-8 and pred_arr.std() > 1e-8:
            corr = float(np.corrcoef(true_arr, pred_arr)[0, 1])
            ax.text(0.05, 0.95, f"r = {corr:.3f}\nn = {num_points:,}",
                    transform=ax.transAxes, fontsize=10, verticalalignment="top")

        # Marginal histograms
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.2], sharex=ax)
        ax_histx.hist(true_arr, bins=30, color="gray", alpha=0.7)
        ax_histx.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        ax_histy = ax.inset_axes([1.05, 0, 0.2, 1], sharey=ax)
        ax_histy.hist(pred_arr, bins=30, orientation="horizontal", color="gray", alpha=0.7)
        ax_histy.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        ax.set_xlabel("True Usage")
        ax.set_ylabel("Predicted Usage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        src_label = "GTF-only" if src_name == "gtf_only" else "Usage-only"
        ax.set_title(f"Usage {src_label}")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fn = out_dir / f"usage_by_source_{org_name}_{src_name}.png"
        fig.savefig(fn, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fn}")


# ---------------------------------------------------------------------------
# Save / load predictions
# ---------------------------------------------------------------------------

def save_predictions(
    out_dir: Path,
    org_name: str,
    cls_probs: np.ndarray,
    cls_labels: np.ndarray,
    usage_per_cond: dict,
    src_tags: np.ndarray | None = None,
    usage_by_source: dict | None = None,
) -> None:
    """Persist prediction arrays to disk for later re-plotting / re-analysis as Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_kwargs: dict = dict(
        cls_probs  = cls_probs.astype(np.float32),
        cls_labels = cls_labels.astype(np.int64),
    )
    if src_tags is not None:
        npz_kwargs["src_tags"] = src_tags.astype(np.int8)
    npz_path = out_dir / f"predictions_{org_name}.npz"
    np.savez_compressed(npz_path, **npz_kwargs)

    usage_path = out_dir / f"usage_{org_name}.json"
    import json
    with open(usage_path, "w") as f:
        json.dump({str(k): v for k, v in usage_per_cond.items()}, f)
    
    if usage_by_source:
        usage_by_src_path = out_dir / f"usage_by_source_{org_name}.json"
        with open(usage_by_src_path, "w") as f:
            # Convert int keys to strings for JSON serialization
            serializable = {}
            for src_name, cond_dict in usage_by_source.items():
                serializable[src_name] = {str(k): v for k, v in cond_dict.items()}
            json.dump(serializable, f)
    
    print(f"  Saved: {npz_path}  {usage_path}")


def load_predictions(
    out_dir: Path,
    org_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict, dict]:
    """Load predictions saved by save_predictions."""
    npz_path = out_dir / f"predictions_{org_name}.npz"
    if not npz_path.exists():
        sys.exit(f"Prediction file not found: {npz_path}  (run without --skip-predictions first)")
    data       = np.load(npz_path)
    cls_probs  = data["cls_probs"]
    cls_labels = data["cls_labels"]
    src_tags   = data["src_tags"] if "src_tags" in data else None

    usage_path = out_dir / f"usage_{org_name}.json"
    usage_per_cond: dict = {}
    if usage_path.exists():
        with open(usage_path) as f:
            usage_per_cond = {int(k): v for k, v in json.load(f).items()}
    
    usage_by_src_path = out_dir / f"usage_by_source_{org_name}.json"
    usage_by_source: dict = {}
    if usage_by_src_path.exists():
        with open(usage_by_src_path) as f:
            data_by_src = json.load(f)
            for src_name, cond_dict in data_by_src.items():
                usage_by_source[src_name] = {int(k): v for k, v in cond_dict.items()}
    
    print(f"  Loaded predictions from {npz_path}")
    return cls_probs, cls_labels, src_tags, usage_per_cond, usage_by_source


# ---------------------------------------------------------------------------
# Per-source metrics
# ---------------------------------------------------------------------------

def _compute_source_metrics_single(
    src_val: int,
    cls_probs: np.ndarray,
    cls_labels: np.ndarray,
    src_tags: np.ndarray,
    true_bg_mask: np.ndarray,
    bg_probs: np.ndarray,
) -> tuple[str, dict]:
    """Compute metrics for a single source (for parallelization)."""
    from sklearn.metrics import average_precision_score

    src_name = SOURCE_LABELS[src_val]
    ann_mask = src_tags == src_val
    if not ann_mask.any():
        return src_name, {}

    # Binary AUPRC
    mask = ann_mask | true_bg_mask
    y_bin = ann_mask[mask].astype(np.int32)
    s_bin = 1.0 - cls_probs[mask, BACKGROUND_CLASS]
    binary_ap = (
        float(average_precision_score(y_bin, s_bin))
        if y_bin.sum() > 0 else float("nan")
    )

    # Per-class AUPRC
    ann_probs = cls_probs[ann_mask]
    ann_labels = cls_labels[ann_mask]
    per_class: dict[str, float] = {}
    per_class_n: dict[str, int] = {}
    
    for c, name in enumerate(SPLICE_CLASS_NAMES):
        y_c_ann = (ann_labels == c).astype(np.int32)
        if y_c_ann.sum() == 0:
            continue
        # Build combined scores once (reuse bg_probs)
        y_c_combined = np.concatenate([y_c_ann, np.zeros(len(bg_probs), dtype=np.int32)])
        s_c_combined = np.concatenate([ann_probs[:, c], bg_probs[:, c]])
        per_class[name] = float(average_precision_score(y_c_combined, s_c_combined))
        per_class_n[name] = int(y_c_ann.sum())

    mean_pc = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    
    return src_name, {
        "n_sites": int(ann_mask.sum()),
        "n_background_sites": int(true_bg_mask.sum()),
        "binary_auprc_vs_bg": binary_ap,
        "per_class_auprc": per_class,
        "per_class_n_positives": per_class_n,
        "mean_per_class_auprc": mean_pc,
    }


def compute_source_metrics(
    cls_probs: np.ndarray,
    cls_labels: np.ndarray,
    src_tags: np.ndarray,
) -> dict:
    """AUPRC per annotation source group (parallelized).

    For each source (intersect, gtf_only, usage_only):
      - Binary AUPRC: sites of that source vs background
      - Per-class AUPRC (one-vs-rest) among annotated sites of that source
    """
    true_bg_mask = cls_labels == BACKGROUND_CLASS
    bg_probs = cls_probs[true_bg_mask]

    results: dict[str, dict] = {}
    
    # Parallel computation of each source
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                _compute_source_metrics_single,
                src_val,
                cls_probs,
                cls_labels,
                src_tags,
                true_bg_mask,
                bg_probs,
            )
            for src_val in (SRC_INTERSECT, SRC_GTF_ONLY, SRC_USAGE_ONLY)
        ]
        for future in futures:
            src_name, src_result = future.result()
            if src_result:
                results[src_name] = src_result

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics(org_name: str, cls_m: dict, usage_m: dict | None) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {org_name.upper()} RESULTS")
    print(sep)
    print(f"  Positions evaluated : {cls_m['n_positions']:>12,}")
    print(f"  Positive rate       : {cls_m['positive_rate']:>12.4%}  (splice / all)")
    print(f"\n  Classification AUPRC")
    print(f"  {'Binary (splice vs background)':<38s} {cls_m['binary_auprc']:.4f}")
    print(f"  {'Mean per-class AUPRC':<38s} {cls_m['mean_splice_class_auprc']:.4f}")
    for name in SPLICE_CLASS_NAMES:
        if name in cls_m["per_class_auprc"]:
            auprc = cls_m["per_class_auprc"][name]
            n = cls_m["per_class_n_positives"][name]
            print(f"    {name:<36s} {auprc:.4f}  (n={n:,})")

    if usage_m and usage_m.get("usage_n_conditions_evaluated", 0) > 0:
        print(
            f"\n  Usage Pearson r  (evaluated on "
            f"{usage_m['usage_n_conditions_evaluated']} / "
            f"{usage_m['usage_n_conditions_total']} conditions, "
            f"{usage_m['usage_n_observations']:,} observations)"
        )
        print(f"  {'Mean r':<38s} {usage_m['usage_mean_pearson_r']:.4f}")
        print(f"  {'Median r':<38s} {usage_m['usage_median_pearson_r']:.4f}")
    elif usage_m:
        print("\n  Usage: no valid conditions found (too few observations per condition)")


def print_source_metrics(org_name: str, source_m: dict) -> None:
    if not source_m:
        return
    print(f"\n  [{org_name.upper()}] BY-SOURCE CLASSIFICATION AUPRC")
    for src_name, m in source_m.items():
        n       = m.get("n_sites", 0)
        n_bg    = m.get("n_background_sites", 0)
        bin_ap  = m.get("binary_auprc_vs_bg", float("nan"))
        mean_pc = m.get("mean_per_class_auprc", float("nan"))
        print(f"\n  Source: {src_name}  (n={n:,} sites, {n_bg:,} background)")
        print(f"    {'Binary AUPRC (vs background)':<38s} {bin_ap:.4f}")
        print(f"    {'Mean per-class AUPRC':<38s} {mean_pc:.4f}")
        for cls_name, ap in m.get("per_class_auprc", {}).items():
            n_pos = m.get("per_class_n_positives", {}).get(cls_name, "?")
            print(f"      {cls_name:<36s} {ap:.4f}  (n={n_pos:,})")


def plot_pr_curves_by_source(
    cls_probs: np.ndarray,
    cls_labels: np.ndarray,
    src_tags: np.ndarray,
    org_name: str,
    out_dir: Path,
) -> None:
    """One PR-curve plot per annotation source group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    for src_val in (SRC_INTERSECT, SRC_GTF_ONLY, SRC_USAGE_ONLY):
        src_name = SOURCE_LABELS[src_val]
        mask = src_tags == src_val
        if not mask.any():
            continue
        probs_s  = cls_probs[mask]
        labels_s = cls_labels[mask]
        n_sites = len(labels_s)

        fig, ax = plt.subplots(figsize=(5, 4))
        for c in range(4):
            y_true = (labels_s == c).astype(np.int32)
            n_pos = y_true.sum()
            if n_pos == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_true, probs_s[:, c])
            ap = float(average_precision_score(y_true, probs_s[:, c]))
            ax.plot(recall, precision,
                    label=f"{CLASS_LABELS[c]} (AUC={ap:.3f}, n={n_pos:,})",
                    color=CLASS_COLORS[c])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve \u2013 {org_name} / {src_name} (n={n_sites:,})")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fn = out_dir / f"pr_curve_{org_name}_{src_name}.png"
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fn}")


# ---------------------------------------------------------------------------
# Parallel plotting
# ---------------------------------------------------------------------------

def execute_plots_parallel(plot_tasks: list[tuple]) -> None:
    """Execute plot tasks in parallel using ThreadPoolExecutor.
    
    Each task is a tuple: (callable, args, kwargs)
    """
    if not plot_tasks:
        return
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(func, *args, **kwargs)
            for func, args, kwargs in plot_tasks
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  Warning: Plot generation failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    main_start = time.perf_counter()
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Set up log file (tee stdout + stderr to eval.log) -------------------
    log_path = out_dir / "eval.log"
    _tee_out = _Tee(sys.stdout, log_path)
    _tee_err = _Tee(sys.stderr, log_path)
    sys.stdout = _tee_out  # type: ignore[assignment]
    sys.stderr = _tee_err  # type: ignore[assignment]
    print(f"=== evaluate_splice  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Log: {log_path}")

    # -- Resolve checkpoint and config ---------------------------------------
    pth_path, cfg_path = resolve_checkpoint(args.checkpoint)
    cfg = load_config(cfg_path)
    print(f"Checkpoint : {pth_path}")
    print(f"Config     : {cfg_path}")

    # -- Resolve organisms / BED files ---------------------------------------
    species_specs = sorted(cfg["species_specs"], key=lambda s: s["organism_index"])

    if args.bed is not None and not args.skip_predictions:
        if len(args.bed) != len(species_specs):
            sys.exit(
                f"--bed: expected {len(species_specs)} file(s) "
                f"(one per organism), got {len(args.bed)}"
            )
        bed_files = args.bed
        print(f"BED files  : {bed_files}  (user-supplied)")
    else:
        bed_files = [s["val_bed"] for s in species_specs]
        print(f"BED files  : {bed_files}  (val_bed from config)")

    if args.gtf_annotation is not None and len(args.gtf_annotation) != len(species_specs):
        sys.exit(
            f"--gtf-annotation: expected {len(species_specs)} file(s), "
            f"got {len(args.gtf_annotation)}"
        )

    device = torch.device(args.device)

    if not args.skip_predictions:
        with _Timer("Loading checkpoint"):
            ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
        with _Timer("Building model"):
            model = build_model(cfg, ckpt, device)
        with _Timer("Building usage heads"):
            usage_heads = build_usage_heads(cfg, ckpt, device)

        from torch.utils.data import DataLoader
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteAnnotation,
            SpliceSiteDataset,
            SpliceSiteUsageIndex,
            collate_splice,
        )

    all_results: dict[str, dict] = {}

    for i, (spec, bed_file) in enumerate(zip(species_specs, bed_files)):
        org_idx: int = spec["organism_index"]
        org_name: str = spec.get("name", ORGANISM_NAMES.get(org_idx, f"organism_{org_idx}"))

        print(f"\n{'='*62}")
        print(f"  {org_name.upper()}")
        print(f"{'='*62}")

        if args.skip_predictions:
            print(f"[{org_name}] Loading saved predictions …")
            cls_probs, cls_labels, src_tags, usage_per_cond, usage_by_source = load_predictions(out_dir, org_name)
        else:
            # -- Optionally build per-source annotation index ----------------
            source_arrays = None
            if args.gtf_annotation is not None:
                gtf_path   = args.gtf_annotation[i]
                usage_path = spec.get("usage_parquet")
                if usage_path:
                    print(f"[{org_name}] Building source annotation …")
                    gtf_positions   = build_gtf_position_set(gtf_path)
                    usage_positions = build_usage_position_set(usage_path)
                    source_arrays   = build_source_arrays(
                        spec["annotation_parquet"], gtf_positions, usage_positions
                    )
                    n_i = sum((s == SRC_INTERSECT).sum()  for _, s in source_arrays.values())
                    n_g = sum((s == SRC_GTF_ONLY).sum()   for _, s in source_arrays.values())
                    n_u = sum((s == SRC_USAGE_ONLY).sum() for _, s in source_arrays.values())
                    print(f"  Intersect: {n_i:,}  GTF-only: {n_g:,}  Usage-only: {n_u:,}")
                else:
                    print(
                        f"[{org_name}] Warning: no usage_parquet in config, "
                        "skipping source tagging.",
                        file=sys.stderr,
                    )

            # -- Dataset & loader --------------------------------------------
            seq_len = cfg.get("sequence_length", 131_072)
            print(f"[{org_name}] Loading annotation from {spec['annotation_parquet']} …")
            annotation = SpliceSiteAnnotation(spec["annotation_parquet"])

            usage_index: SpliceSiteUsageIndex | None = None
            if spec.get("usage_parquet"):
                print(f"[{org_name}] Loading usage index from {spec['usage_parquet']} …")
                usage_index = SpliceSiteUsageIndex(
                    spec["usage_parquet"],
                    min_coverage=args.min_coverage,
                )

            print(f"[{org_name}] Building dataset from {bed_file} …")
            dataset = SpliceSiteDataset(
                genome=spec["genome"],
                bed_file=bed_file,
                annotation=annotation,
                usage_index=usage_index,
                sequence_length=seq_len,
                organism_index=org_idx,
                max_sites=cfg.get("max_sites", 1024),
            )
            print(f"[{org_name}] {len(dataset):,} windows to evaluate")

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_splice,
            )

            # Build windows list for source tagging (same order as dataset)
            windows = None
            if source_arrays is not None:
                half = seq_len // 2
                windows = []
                with open(bed_file) as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) < 3 or line.startswith("#"):
                            continue
                        chrom  = parts[0]
                        center = (int(parts[1]) + int(parts[2])) // 2
                        windows.append((chrom, center - half, center + half))

            # -- Inference ---------------------------------------------------
            print(f"[{org_name}] Running inference …")
            with _Timer(f"Inference ({org_name})"):
                cls_probs, cls_labels, usage_per_cond, src_tags, usage_by_source = collect_predictions(
                    model=model,
                    usage_heads=usage_heads,
                    loader=loader,
                    device=device,
                    organism_index=org_idx,
                    windows=windows,
                    source_arrays=source_arrays,
                    seq_len=seq_len,
                )

            with _Timer(f"Saving predictions ({org_name})"):
                save_predictions(out_dir, org_name, cls_probs, cls_labels, usage_per_cond, src_tags, usage_by_source)

        # -- Metrics ---------------------------------------------------------
        print(f"[{org_name}] Computing metrics …")
        with _Timer(f"Classification metrics ({org_name})"):
            cls_m = compute_classification_metrics(cls_probs, cls_labels)
        with _Timer(f"Usage metrics ({org_name})"):
            usage_m = compute_usage_metrics(usage_per_cond) if usage_per_cond else None
        with _Timer(f"Usage metrics by source ({org_name})"):
            usage_by_source_m = compute_usage_metrics_by_source(usage_by_source) if (args.per_source and usage_by_source) else {}
        with _Timer(f"Source metrics ({org_name})"):
            source_m = compute_source_metrics(cls_probs, cls_labels, src_tags) if (args.per_source and src_tags is not None) else {}
        print_metrics(org_name, cls_m, usage_m)
        if source_m:
            print_source_metrics(org_name, source_m)
        if usage_by_source_m:
            print_usage_metrics_by_source(org_name, usage_by_source_m)

        # -- Plots (parallel) ------------------------------------------------
        if not args.skip_plots:
            plot_tasks = []
            
            # PR curves
            plot_tasks.append((
                plot_pr_curves,
                (cls_probs, cls_labels, org_name, out_dir / f"pr_curve_{org_name}.png"),
                {}
            ))
            
            # Usage plots
            if usage_per_cond:
                plot_tasks.append((
                    plot_usage_density,
                    (usage_per_cond, org_name, out_dir / f"usage_density_{org_name}.png"),
                    {}
                ))
                if args.per_condition:
                    plot_tasks.append((
                        plot_usage_per_condition,
                        (usage_per_cond, out_dir, org_name),
                        {}
                    ))
                if args.per_tissue and spec.get("usage_parquet"):
                    usage_json = Path(spec["usage_parquet"]).with_suffix(".json")
                    plot_tasks.append((
                        plot_usage_per_tissue,
                        (usage_per_cond, usage_json, out_dir, org_name),
                        {}
                    ))
            
            # By-source plots
            if args.per_source and src_tags is not None:
                plot_tasks.append((
                    plot_pr_curves_by_source,
                    (cls_probs, cls_labels, src_tags, org_name, out_dir),
                    {}
                ))
            if args.per_source and usage_by_source:
                plot_tasks.append((
                    plot_usage_by_source,
                    (usage_by_source, out_dir, org_name),
                    {}
                ))
            
            if plot_tasks:
                print(f"[{org_name}] Generating {len(plot_tasks)} plot(s) in parallel …")
                with _Timer(f"Plotting ({org_name})"):
                    execute_plots_parallel(plot_tasks)

        all_results[org_name] = {
            **cls_m,
            **(usage_m or {}),
            **(({"by_source": source_m}) if source_m else {}),
            **(({"usage_by_source": usage_by_source_m}) if usage_by_source_m else {}),
        }

    # -- Multi-species macro-average -----------------------------------------
    if len(species_specs) > 1:
        keys_to_avg = [
            "binary_auprc", "mean_splice_class_auprc",
            "usage_mean_pearson_r", "usage_median_pearson_r",
        ]
        print(f"\n{'='*62}")
        print("  MACRO-AVERAGE ACROSS SPECIES")
        print(f"{'='*62}")
        for k in keys_to_avg:
            vals = [v[k] for v in all_results.values()
                    if k in v and not np.isnan(v.get(k, float("nan")))]
            if vals:
                print(f"  {k:<38s} {np.mean(vals):.4f}")

    # -- Save JSON -----------------------------------------------------------
    with _Timer("Saving JSON results"):
        out_path = out_dir / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=lambda x: None)
    
    # -- Overall timing --------------------------------------------------
    total_time = time.perf_counter() - main_start
    print(f"\n{'='*62}")
    print(f"  TOTAL TIME: {_Timer.format_time(total_time)}")
    print(f"{'='*62}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
