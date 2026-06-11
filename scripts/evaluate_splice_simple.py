#!/usr/bin/env python3
"""Simple splice-site model evaluation script.

Computes per-species:
  - AUPRC per class (Donor+, Acceptor+, Donor-, Acceptor-) and binary splice vs. background
  - Pearson r per tissue for splice-site usage (same-species head only)

Usage:
    python scripts/evaluate_splice_simple.py \\
        --checkpoint /path/to/run_dir \\
        --data-config /path/to/data_config.json \\
        --eval-species human mouse rat \\
        --output-dir /path/to/results

The data config JSON has the format:
    {
      "species": {
        "human": {
          "genome": "/path/to/hg38.fa",
          "annotation_parquet": "/path/to/splice_sites.parquet",
          "usage_parquet": "/path/to/usage.parquet",
          "test_bed": "/path/to/test.bed"
        },
        ...
      }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from alphagenome_pytorch.utils.paths import expand_paths_in_dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLICE_CLASS_NAMES = ["Donor+", "Acceptor+", "Donor-", "Acceptor-"]
BACKGROUND_CLASS = 4

SPECIES_NAME_MAP = {
    "human": "Homo_sapiens",
    "mouse": "Mus_musculus",
    "rat": "Rattus_norvegicus",
    "rabbit": "Oryctolagus_cuniculus",
    "opossum": "Monodelphis_domestica",
    "macaque": "Macaca_mulatta",
    "chicken": "Gallus_gallus",
}

TISSUE_COLORS = {
    "Brain": "#3399cc",
    "Midbrain": "#34b3e6",
    "Cerebellum": "#34ccff",
    "Heart": "#cc0100",
    "Kidney": "#cc9900",
    "Liver": "#339900",
    "Ovary": "#cc329a",
    "Testis": "#ff6600",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_simple")
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simple splice evaluation: per-class AUPRC + per-tissue usage Pearson r",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint directory (contains best_model.pth + config.json) or .pth path")
    p.add_argument("--model-config", default=None,
                   help="Optional model config.json path (defaults to config.json in checkpoint dir)")
    p.add_argument("--data-config", required=True,
                   help="Data config JSON with species genome/annotation/usage/BED paths")
    p.add_argument("--eval-species", nargs="+", required=True,
                   help="Species to evaluate, e.g. human mouse rat")
    p.add_argument("--output-dir", required=True,
                   help="Directory to save metrics JSON, tissue plots, and eval.log")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--min-coverage", type=int, default=10,
                   help="Min coverage for usage targets")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Randomly subsample this many windows per species")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-plots", action="store_true",
                   help="Skip tissue correlation plots")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def resolve_checkpoint(arg: str) -> tuple[Path, Path]:
    p = Path(arg)
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


def load_json_config(path: Path) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    path_keys = {"genome", "annotation_parquet", "usage_parquet", "train_bed", "val_bed", "test_bed", "pretrained_weights"}
    return expand_paths_in_dict(cfg, path_keys)


def load_data_config(path: Path, eval_species: list[str]) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    path_keys = {"genome", "annotation_parquet", "usage_parquet", "train_bed", "val_bed", "test_bed"}
    raw = expand_paths_in_dict(raw, path_keys)
    species_dict = raw.get("species", raw)
    specs = []
    for name in eval_species:
        if name not in species_dict:
            sys.exit(f"Species '{name}' not in data config. Available: {list(species_dict.keys())}")
        spec = species_dict[name].copy()
        spec["name"] = name
        specs.append(spec)
    return specs


def extract_species_from_path(path: str | Path) -> str | None:
    for part in reversed(Path(path).parts):
        if part in SPECIES_NAME_MAP.values():
            return part
        if "_" in part and part[0].isupper() and not part.startswith("FOLD"):
            return part
    return None


def match_species_to_model(model_cfg: dict, data_spec: dict) -> tuple[int, dict] | None:
    scientific = SPECIES_NAME_MAP.get(data_spec.get("name", ""))
    if not scientific:
        return None
    for ms in model_cfg.get("species_specs", []):
        if extract_species_from_path(ms.get("annotation_parquet", "")) == scientific:
            return ms["organism_index"], ms
    return None


def match_usage_conditions(
    model_cfg: dict,
    data_spec: dict,
    logger: logging.Logger,
) -> dict[int, int] | None:
    """Return mapping data_condition_index -> model_condition_index, or None."""
    usage_parquet = data_spec.get("usage_parquet")
    if not usage_parquet:
        return None

    data_meta_path = Path(usage_parquet).with_suffix(".json")
    if not data_meta_path.exists():
        logger.warning(f"No usage metadata at {data_meta_path}; skipping usage")
        return None

    with open(data_meta_path) as f:
        data_meta = json.load(f)
    data_labels: dict[str, int] = data_meta.get("condition_labels", {})
    if not data_labels:
        return None

    org_idx = data_spec["organism_index"]
    model_spec = next(
        (s for s in model_cfg.get("species_specs", []) if s["organism_index"] == org_idx),
        None,
    )
    if not model_spec or not model_spec.get("usage_parquet"):
        logger.info(f"Model has no usage head for organism {org_idx}; skipping usage")
        return None

    model_meta_path = Path(model_spec["usage_parquet"]).with_suffix(".json")
    if not model_meta_path.exists():
        logger.warning(f"No model usage metadata at {model_meta_path}; skipping usage")
        return None

    with open(model_meta_path) as f:
        model_meta = json.load(f)
    model_labels: dict[str, int] = model_meta.get("condition_labels", {})

    common = set(model_labels) & set(data_labels)
    if not common:
        logger.warning("No overlapping conditions between model and data; skipping usage")
        return None

    mapping = {data_labels[n]: model_labels[n] for n in common}
    logger.info(
        f"Matched {len(common)} conditions "
        f"(model: {len(model_labels)}, data: {len(data_labels)})"
    )
    return mapping


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(cfg: dict, ckpt: dict, device: torch.device, logger: logging.Logger) -> nn.Module:
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.extensions.finetuning.heads import (
        create_splice_classification_finetuning_head,
        create_splice_usage_finetuning_head,
    )
    from alphagenome_pytorch.extensions.finetuning.transfer import (
        load_trunk, remove_all_heads, prepare_for_transfer, TransferConfig,
    )

    species_specs = cfg["species_specs"]
    num_organisms = max(s["organism_index"] for s in species_specs) + 1
    dtype_str = cfg.get("dtype", "bfloat16")
    dtype_policy = (
        DtypePolicy.full_float32() if dtype_str == "float32"
        else DtypePolicy.mixed_precision()
    )
    logger.info(f"Building model (mode={cfg['mode']}, dtype={dtype_str}, num_organisms={num_organisms})")

    model = AlphaGenome(num_organisms=num_organisms, dtype_policy=dtype_policy)

    _pretrained = Path(os.path.expandvars(os.path.expanduser(cfg["pretrained_weights"])))
    if not _pretrained.exists():
        _fallback = Path(__file__).parent.parent / "checkpoints" / _pretrained.name
        if _fallback.exists():
            logger.warning(f"Pretrained weights not found at {_pretrained}; using {_fallback}")
            _pretrained = _fallback
        else:
            raise FileNotFoundError(f"Pretrained weights not found: {_pretrained}")
    model = load_trunk(model, str(_pretrained), exclude_heads=True)
    model = remove_all_heads(model)

    if cfg["mode"] == "lora" and cfg.get("lora_rank", 0) > 0:
        lora_targets = [t.strip() for t in cfg["lora_targets"].split(",")]
        logger.info(f"  LoRA: rank={cfg['lora_rank']}, alpha={cfg['lora_alpha']}, targets={lora_targets}")
        model = prepare_for_transfer(model, TransferConfig(
            mode="lora",
            lora_targets=lora_targets,
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
        ))

    model.splice_sites_classification_head = create_splice_classification_finetuning_head(
        num_organisms=num_organisms
    )

    species_n_conditions: dict[int, int] = {
        int(k): v for k, v in cfg.get("species_n_conditions", {}).items()
    }
    usage_modules = {}
    for org_idx, n_cond in species_n_conditions.items():
        if n_cond > 0:
            usage_modules[str(org_idx)] = create_splice_usage_finetuning_head(
                n_conditions=n_cond, num_organisms=num_organisms
            )
    model.splice_sites_usage_head = nn.ModuleDict(usage_modules) if usage_modules else None

    raw_sd = ckpt["model_state_dict"]
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in raw_sd.items()}
    model.load_state_dict(sd, strict=False)

    for org_key, raw_usd in ckpt.get("usage_heads_state_dicts", {}).items():
        if model.splice_sites_usage_head and org_key in model.splice_sites_usage_head:
            usd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in raw_usd.items()}
            model.splice_sites_usage_head[org_key].load_state_dict(usd)

    logger.info(f"  Loaded weights (epoch {ckpt.get('epoch', '?')})")
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: nn.Module,
    loader,
    device: torch.device,
    org_idx: int,
    condition_mapping: dict[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict]]:
    """Run inference for one species.

    Returns:
        cls_probs:     (N, 5) float32  — masked to gene body positions
        cls_labels:    (N,)   int64
        usage_per_cond: dict data_cond_idx -> {"pred", "true", "genomic_pos", "chrom_idx"}
    """
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    usage_per_cond: dict[int, dict] = {}

    usage_head: nn.Module | None = None
    if model.splice_sites_usage_head is not None and condition_mapping is not None:
        key = str(org_idx)
        usage_head = model.splice_sites_usage_head[key] if key in model.splice_sites_usage_head else None

    org_idx_t_template = torch.tensor([org_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Inference", unit="batch"):
            seq = batch["sequence"].to(device)
            org_t = batch["organism_index"].to(device)

            preds = model.predict(seq, org_t, resolutions=(1,))
            cls_probs = preds["splice_sites_classification"]["probs"]  # (B, S, 5)

            all_probs.append(cls_probs.cpu().numpy().reshape(-1, 5))
            all_labels.append(batch["classification_labels"].numpy().reshape(-1))
            if "loss_mask" in batch:
                all_masks.append(batch["loss_mask"].numpy().reshape(-1))

            if usage_head is not None and "usage_positions" in batch:
                out = model.predict(seq, org_t, resolutions=(1,), channels_last=False, embeddings_only=True)
                emb_1bp = out["embeddings_1bp"]
                B = emb_1bp.shape[0]
                usage_org_t = org_idx_t_template.expand(B)
                usage_preds = usage_head(emb_1bp, usage_org_t, channels_last=True)["predictions"].float()

                _accumulate_usage(
                    usage_preds.cpu().numpy(),
                    batch["usage_positions"].numpy(),
                    batch["usage_values"].numpy(),
                    batch["usage_mask"].numpy(),
                    usage_per_cond,
                    condition_mapping=condition_mapping,
                    window_starts=batch["window_start"].numpy() if "window_start" in batch else None,
                    chrom_idxs=batch["chrom_idx"].numpy() if "chrom_idx" in batch else None,
                )

    cls_probs_arr = np.concatenate(all_probs, axis=0)
    cls_labels_arr = np.concatenate(all_labels, axis=0)

    if all_masks:
        mask = np.concatenate(all_masks, axis=0).astype(bool)
        cls_probs_arr = cls_probs_arr[mask]
        cls_labels_arr = cls_labels_arr[mask]

    return cls_probs_arr, cls_labels_arr, usage_per_cond


def _accumulate_usage(
    usage_preds: np.ndarray,   # (B, S, T_model)
    positions: np.ndarray,     # (B, max_sites)  -1 padded
    values: np.ndarray,        # (B, max_sites, n_data_cond)
    mask: np.ndarray,          # (B, max_sites, n_data_cond) bool
    acc: dict,
    condition_mapping: dict[int, int],
    window_starts: np.ndarray | None = None,  # (B,) int64
    chrom_idxs: np.ndarray | None = None,     # (B,) int32
) -> None:
    B, n_data_cond = positions.shape[0], values.shape[2]
    for i in range(B):
        valid = positions[i] != -1
        if not valid.any():
            continue
        valid_pos = positions[i][valid]
        valid_vals = values[i][valid]       # (k, n_data_cond)
        valid_mask = mask[i][valid]         # (k, n_data_cond) bool
        valid_preds = usage_preds[i, valid_pos, :]  # (k, T_model)
        win_start = int(window_starts[i]) if window_starts is not None else 0
        chrom_idx = int(chrom_idxs[i]) if chrom_idxs is not None else -1
        genomic_pos = (valid_pos + win_start).tolist()

        for data_c in range(n_data_cond):
            if data_c not in condition_mapping:
                continue
            model_c = condition_mapping[data_c]
            obs = valid_mask[:, data_c]
            if not obs.any():
                continue
            entry = acc.setdefault(data_c, {"pred": [], "true": [], "genomic_pos": [], "chrom_idx": []})
            entry["pred"].extend(valid_preds[obs, model_c].tolist())
            entry["true"].extend(valid_vals[obs, data_c].tolist())
            entry["genomic_pos"].extend(np.array(genomic_pos)[obs].tolist())
            entry["chrom_idx"].extend([chrom_idx] * int(obs.sum()))


# ---------------------------------------------------------------------------
# Saving predictions
# ---------------------------------------------------------------------------

def save_predictions(
    out_dir: Path,
    org_name: str,
    cls_probs: np.ndarray,
    cls_labels: np.ndarray,
    usage_per_cond: dict[int, dict],
    logger: logging.Logger,
    chrom_names: list[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_path = out_dir / f"predictions_{org_name}.npz"
    np.savez_compressed(
        cls_path,
        cls_probs=cls_probs.astype(np.float32),
        cls_labels=cls_labels.astype(np.int64),
    )
    logger.info(f"  Saved classification predictions: {cls_path}")

    if not usage_per_cond:
        return

    usage_path = out_dir / f"usage_{org_name}.npz"
    cond_ids_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []
    genomic_pos_chunks: list[np.ndarray] = []
    chrom_idx_chunks: list[np.ndarray] = []
    stats_cond_ids: list[int] = []
    stats_n: list[int] = []
    stats_sum_pred: list[float] = []
    stats_sum_true: list[float] = []
    stats_sum_pred2: list[float] = []
    stats_sum_true2: list[float] = []
    stats_sum_prod: list[float] = []
    has_coords = False

    for cond_idx, data in usage_per_cond.items():
        pred = np.asarray(data["pred"], dtype=np.float32)
        true = np.asarray(data["true"], dtype=np.float32)
        if pred.size == 0 or pred.size != true.size:
            continue
        cond_ids_chunks.append(np.full(pred.size, int(cond_idx), dtype=np.int32))
        pred_chunks.append(pred)
        true_chunks.append(true)

        if "genomic_pos" in data and "chrom_idx" in data:
            genomic_pos_chunks.append(np.asarray(data["genomic_pos"], dtype=np.int64))
            chrom_idx_chunks.append(np.asarray(data["chrom_idx"], dtype=np.int32))
            has_coords = True

        stats_cond_ids.append(int(cond_idx))
        stats_n.append(int(pred.size))
        p64 = pred.astype(np.float64, copy=False)
        t64 = true.astype(np.float64, copy=False)
        stats_sum_pred.append(float(p64.sum()))
        stats_sum_true.append(float(t64.sum()))
        stats_sum_pred2.append(float((p64 * p64).sum()))
        stats_sum_true2.append(float((t64 * t64).sum()))
        stats_sum_prod.append(float((p64 * t64).sum()))

    if cond_ids_chunks:
        cond_ids_arr = np.concatenate(cond_ids_chunks, axis=0)
        pred_arr = np.concatenate(pred_chunks, axis=0)
        true_arr = np.concatenate(true_chunks, axis=0)
    else:
        cond_ids_arr = np.array([], dtype=np.int32)
        pred_arr = np.array([], dtype=np.float32)
        true_arr = np.array([], dtype=np.float32)

    npz_kwargs: dict = dict(
        cond_ids=cond_ids_arr,
        pred=pred_arr,
        true=true_arr,
        stats_cond_ids=np.array(stats_cond_ids, dtype=np.int32),
        stats_n=np.array(stats_n, dtype=np.int64),
        stats_sum_pred=np.array(stats_sum_pred, dtype=np.float64),
        stats_sum_true=np.array(stats_sum_true, dtype=np.float64),
        stats_sum_pred2=np.array(stats_sum_pred2, dtype=np.float64),
        stats_sum_true2=np.array(stats_sum_true2, dtype=np.float64),
        stats_sum_prod=np.array(stats_sum_prod, dtype=np.float64),
    )

    if has_coords and genomic_pos_chunks and chrom_names:
        all_chrom_idxs = np.concatenate(chrom_idx_chunks, axis=0)
        all_genomic_pos = np.concatenate(genomic_pos_chunks, axis=0)
        npz_kwargs["chr_pos"] = np.array(
            [f"{chrom_names[ci]}:{gp}" for ci, gp in zip(all_chrom_idxs, all_genomic_pos)],
            dtype=str,
        )

    np.savez_compressed(usage_path, **npz_kwargs)
    logger.info(f"  Saved usage predictions:        {usage_path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        sys.exit("scikit-learn required: pip install scikit-learn")

    y_binary = (labels != BACKGROUND_CLASS).astype(np.int32)
    binary_auprc = float(average_precision_score(y_binary, 1.0 - probs[:, BACKGROUND_CLASS]))

    per_class: dict[str, float] = {}
    per_class_n: dict[str, int] = {}
    for c, name in enumerate(SPLICE_CLASS_NAMES):
        y_c = (labels == c).astype(np.int32)
        n_pos = int(y_c.sum())
        if n_pos == 0:
            continue
        per_class[name] = float(average_precision_score(y_c, probs[:, c]))
        per_class_n[name] = n_pos

    mean_per_class = float(np.mean(list(per_class.values()))) if per_class else float("nan")

    return {
        "binary_auprc": binary_auprc,
        "mean_splice_class_auprc": mean_per_class,
        "per_class_auprc": per_class,
        "per_class_n_positives": per_class_n,
        "positive_rate": float(y_binary.mean()),
        "n_positions": int(len(labels)),
    }


def compute_usage_metrics_per_tissue(
    usage_per_cond: dict,
    usage_parquet: str,
    logger: logging.Logger,
) -> dict | None:
    """Compute per-condition Pearson r and group by tissue.

    Returns dict with overall pooled_r, mean_r, median_r, and per-tissue breakdown.
    """
    try:
        from scipy.stats import pearsonr
    except ImportError:
        sys.exit("scipy required: pip install scipy")

    meta_path = Path(usage_parquet).with_suffix(".json")
    if not meta_path.exists():
        logger.warning(f"No usage metadata at {meta_path}; cannot compute per-tissue metrics")
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    condition_labels: dict[str, int] = meta.get("condition_labels", {})
    idx_to_name = {v: k for k, v in condition_labels.items()}

    # Compute per-condition r
    cond_records = []
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    for cond_idx, data in usage_per_cond.items():
        pred = np.array(data["pred"], dtype=np.float32)
        true = np.array(data["true"], dtype=np.float32)
        all_pred.append(pred)
        all_true.append(true)
        if len(pred) < 2 or pred.std() < 1e-8 or true.std() < 1e-8:
            continue
        r, _ = pearsonr(pred, true)
        if not np.isnan(r):
            cond_name = idx_to_name.get(cond_idx, str(cond_idx))
            parts = cond_name.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    timepoint = int(parts[1])
                    tissue = parts[0]
                except ValueError:
                    tissue, timepoint = cond_name, 0
            else:
                tissue, timepoint = cond_name, 0
            cond_records.append({
                "cond_idx": cond_idx,
                "name": cond_name,
                "tissue": tissue,
                "timepoint": timepoint,
                "r": float(r),
                "n": len(pred),
            })

    if not cond_records:
        logger.warning("No valid per-condition correlations")
        return None

    rs = [rec["r"] for rec in cond_records]

    # Pooled r
    pooled_r = float("nan")
    if all_pred:
        pp = np.concatenate(all_pred)
        pt = np.concatenate(all_true)
        if len(pp) >= 2 and pp.std() > 1e-8 and pt.std() > 1e-8:
            pooled_r, _ = pearsonr(pp, pt)
            pooled_r = float(pooled_r)

    # Per-tissue summary
    from collections import defaultdict
    tissue_rs: dict[str, list[float]] = defaultdict(list)
    for rec in cond_records:
        tissue_rs[rec["tissue"]].append(rec["r"])

    per_tissue: dict[str, dict] = {}
    for tissue, t_rs in sorted(tissue_rs.items()):
        per_tissue[tissue] = {
            "n_conditions": len(t_rs),
            "mean_pearson_r": float(np.mean(t_rs)),
            "median_pearson_r": float(np.median(t_rs)),
            "std_pearson_r": float(np.std(t_rs)),
        }

    return {
        "pooled_pearson_r": pooled_r,
        "mean_pearson_r": float(np.mean(rs)),
        "median_pearson_r": float(np.median(rs)),
        "n_conditions_evaluated": len(rs),
        "n_observations": int(sum(rec["n"] for rec in cond_records)),
        "per_tissue": per_tissue,
        "_cond_records": cond_records,  # kept for plotting; stripped before JSON save
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PR Curve Plotting (per-class AUPRC)
# ---------------------------------------------------------------------------

def plot_pr_curves(
    cls_probs: np.ndarray,
    cls_labels: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
):
    """Precision-recall curves for the 4 splice-site classes (one-vs-rest)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    CLASS_COLORS = ['#ff7f00', '#33a02c', '#fdbf6f', '#b2df8a']
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
                label=f"{class_names[c]} (AUC={pr_auc:.3f}, n={n_pos:,})",
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

def plot_tissue_correlations(
    cond_records: list[dict],
    org_name: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    import pandas as pd

    df = pd.DataFrame(cond_records)
    if df.empty or len(df) < 2:
        return

    tissues = sorted(df["tissue"].unique())
    n_tissues = len(tissues)

    fig, ax = plt.subplots(figsize=(max(6, n_tissues * 0.9), 5))
    sns.boxplot(
        data=df, x="tissue", y="r", hue="tissue",
        order=tissues,
        palette=[TISSUE_COLORS.get(t, "#888888") for t in tissues],
        showfliers=False, legend=False, ax=ax, zorder=1,
    )
    rng = np.random.default_rng(42)
    for i, tissue in enumerate(tissues):
        td = df[df["tissue"] == tissue]
        jitter = rng.normal(i, 0.05, size=len(td))
        sizes = td["timepoint"].values * 3
        ax.scatter(
            jitter, td["r"].values,
            s=sizes, color=TISSUE_COLORS.get(tissue, "#888888"),
            alpha=0.45, edgecolors="black", linewidths=0.5, zorder=3,
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Tissue", fontsize=12)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title(f"Splice Usage Correlation by Tissue\n{org_name}", fontsize=13)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    for i, tissue in enumerate(tissues):
        n = int((df["tissue"] == tissue).sum())
        ax.text(i, 0.02, f"n={n}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir / "eval.log")
    logger.info(f"=== evaluate_splice_simple  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    logger.info(f"Output: {out_dir}")

    # Load configs
    pth_path, default_cfg_path = resolve_checkpoint(args.checkpoint)
    model_cfg_path = Path(args.model_config) if args.model_config else default_cfg_path
    if not model_cfg_path.exists():
        sys.exit(f"Model config not found: {model_cfg_path}")
    model_cfg = load_json_config(model_cfg_path)
    logger.info(f"Checkpoint:   {pth_path}")
    logger.info(f"Model config: {model_cfg_path}")

    data_cfg_path = Path(args.data_config)
    if not data_cfg_path.exists():
        sys.exit(f"Data config not found: {data_cfg_path}")
    species_specs = load_data_config(data_cfg_path, args.eval_species)
    logger.info(f"Data config:  {data_cfg_path}")

    # Match each data species to model organism_index
    for spec in species_specs:
        match = match_species_to_model(model_cfg, spec)
        if match is None:
            sys.exit(
                f"Species '{spec['name']}' not found in model training data. "
                f"For cross-species evaluation use evaluate_splice.py instead."
            )
        spec["organism_index"] = match[0]
        logger.info(f"  '{spec['name']}' → organism_index {match[0]}")

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info(f"Device: {device}")
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    model = build_model(model_cfg, ckpt, device, logger)
    del ckpt

    from torch.utils.data import DataLoader
    from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
        SpliceSiteAnnotation,
        SpliceSiteUsageIndex,
        SpliceSiteDataset,
        collate_splice,
    )

    all_results: dict[str, dict] = {}

    for spec in species_specs:
        org_name = spec["name"]
        org_idx = spec["organism_index"]
        logger.info(f"{'='*60}")
        logger.info(f"  {org_name.upper()}")
        logger.info(f"{'='*60}")

        # Annotation
        logger.info(f"Loading annotation: {spec['annotation_parquet']}")
        annotation = SpliceSiteAnnotation(spec["annotation_parquet"])

        # Usage index (same-species only)
        usage_index = None
        condition_mapping: dict[int, int] | None = None
        if spec.get("usage_parquet"):
            logger.info(f"Loading usage index: {spec['usage_parquet']}")
            usage_index = SpliceSiteUsageIndex(
                spec["usage_parquet"],
                min_coverage=args.min_coverage,
                usage_coord_base=0,
                observed_conditions_only=False,
            )
            spec_with_org = {**spec}
            condition_mapping = match_usage_conditions(model_cfg, spec_with_org, logger)

        # Dataset
        bed_file = spec.get("test_bed") or spec.get("val_bed")
        if not bed_file:
            sys.exit(f"Species '{org_name}' missing 'test_bed' and 'val_bed' in data config")
        logger.info(f"BED file: {bed_file}")

        seq_len = model_cfg.get("sequence_length", 131_072)
        dataset = SpliceSiteDataset(
            genome=spec["genome"],
            bed_file=bed_file,
            annotation=annotation,
            usage_index=usage_index,
            sequence_length=seq_len,
            organism_index=org_idx,
            max_sites=model_cfg.get("max_sites", 1024),
        )

        total = len(dataset)
        if args.max_windows and args.max_windows < total:
            import random
            from torch.utils.data import Subset
            random.seed(args.seed)
            indices = sorted(random.sample(range(total), args.max_windows))
            dataset = Subset(dataset, indices)
            logger.info(f"Sampled {len(dataset):,} / {total:,} windows (seed={args.seed})")
        else:
            logger.info(f"{total:,} windows")

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_splice,
        )

        # Inference
        cls_probs, cls_labels, usage_per_cond = run_inference(
            model, loader, device, org_idx, condition_mapping
        )
        logger.info(f"Inference done: {len(cls_labels):,} positions, {len(usage_per_cond)} usage conditions")

        # Save raw predictions into per-species subdir
        species_dir = out_dir / org_name
        species_dir.mkdir(parents=True, exist_ok=True)
        chrom_names = getattr(dataset, "chrom_names", None) or getattr(
            getattr(dataset, "dataset", None), "chrom_names", None
        )
        save_predictions(species_dir, org_name, cls_probs, cls_labels, usage_per_cond, logger, chrom_names=chrom_names)


        # Classification metrics
        cls_m = compute_classification_metrics(cls_probs, cls_labels)
        logger.info(f"  binary AUPRC:           {cls_m['binary_auprc']:.4f}")
        logger.info(f"  mean per-class AUPRC:   {cls_m['mean_splice_class_auprc']:.4f}")
        for name, ap in cls_m["per_class_auprc"].items():
            logger.info(f"    {name:<12s}  AUPRC={ap:.4f}  (n={cls_m['per_class_n_positives'][name]:,})")

        # PR curve plot (per-class)
        if not args.skip_plots:
            pr_curve_path = species_dir / f"pr_curve_{org_name}.png"
            plot_pr_curves(
                cls_probs,
                cls_labels,
                SPLICE_CLASS_NAMES,
                title=f"{org_name}",
                output_path=pr_curve_path,
            )
            logger.info(f"  Saved PR curve plot: {pr_curve_path}")

        result = {**cls_m}

        # Usage metrics per tissue
        if usage_per_cond and spec.get("usage_parquet"):
            usage_m = compute_usage_metrics_per_tissue(
                usage_per_cond, spec["usage_parquet"], logger
            )
            if usage_m:
                cond_records = usage_m.pop("_cond_records", [])
                result["usage"] = usage_m
                logger.info(f"  pooled usage Pearson r: {usage_m['pooled_pearson_r']:.4f}")
                logger.info(f"  mean condition r:       {usage_m['mean_pearson_r']:.4f}")
                logger.info(f"  median condition r:     {usage_m['median_pearson_r']:.4f}")
                logger.info(f"  conditions evaluated:   {usage_m['n_conditions_evaluated']}")
                for tissue, tm in usage_m["per_tissue"].items():
                    logger.info(
                        f"    {tissue:<15s}  n={tm['n_conditions']:>3d}  "
                        f"mean={tm['mean_pearson_r']:.3f}  median={tm['median_pearson_r']:.3f}"
                    )

                if not args.skip_plots and cond_records:
                    plot_path = species_dir / f"tissue_correlation_{org_name}.png"
                    plot_tissue_correlations(cond_records, org_name, plot_path)
                    logger.info(f"  Saved tissue plot: {plot_path}")

        all_results[org_name] = result

        # Save per-species metrics into its subdir
        species_metrics_path = species_dir / "metrics.json"
        with open(species_metrics_path, "w") as f:
            json.dump({org_name: result}, f, indent=2, default=lambda x: None)
        logger.info(f"  Saved species metrics: {species_metrics_path}")

    # Macro-average across species
    if len(species_specs) > 1:
        logger.info(f"{'='*60}")
        logger.info("  MACRO-AVERAGE")
        logger.info(f"{'='*60}")
        for key in ["binary_auprc", "mean_splice_class_auprc"]:
            vals = [v[key] for v in all_results.values() if key in v and not np.isnan(v[key])]
            if vals:
                logger.info(f"  {key:<38s} {np.mean(vals):.4f}")
        for key in ["pooled_pearson_r", "mean_pearson_r"]:
            vals = [v["usage"][key] for v in all_results.values()
                    if "usage" in v and not np.isnan(v["usage"].get(key, float("nan")))]
            if vals:
                logger.info(f"  usage {key:<33s} {np.mean(vals):.4f}")

    # Save JSON
    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None)
    logger.info(f"\nResults saved to {out_path}")

    # Release GPU
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
