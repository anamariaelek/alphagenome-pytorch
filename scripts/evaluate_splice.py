#!/usr/bin/env python3
"""Evaluate splice-site model performance on a test split.

Computes per-organism:
  - AUPRC for splice site classification
      * Binary: any splice site (Donor/Acceptor) vs. background
      * Per-class one-vs-rest: Donor+, Acceptor+, Donor-, Acceptor-
  - Pearson r for splice site usage (per condition, then averaged)

Usage requires:
  - Model checkpoint (contains best_model.pth and config.json with model architecture)
  - Data config JSON (defines species with genome paths, annotations, BED files)
  - Species to evaluate (can evaluate subsets of species)

Examples
--------
    # Evaluate mouse and rat from a human-trained model
    python scripts/evaluate_splice.py \\
        --checkpoint /path/to/132kb_human_lora \\
        --data-config /data/data_config.json \\
        --eval-species mouse rat \\
        --output-dir /results/predictions \\
        --device cuda

    # Evaluate single species with custom model config
    python scripts/evaluate_splice.py \\
        --checkpoint /path/to/132kb_lora \\
        --model-config /path/to/custom_config.json \\
        --data-config /data/data_config.json \\
        --eval-species rat \\
        --output-dir /results/predictions_rat \\
        --bed /data/human_test.bed /data/mouse_test.bed \\
        --output-dir results/eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from alphagenome_pytorch.utils.paths import expand_path, expand_paths_in_dict


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure logging to write to both file and console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("evaluate_splice")
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler (verbose, all levels)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    
    # Console handler (info and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter with timestamp
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# Splice class labels (must match SpliceSiteAnnotation)
SPLICE_CLASS_NAMES = ["Donor+", "Acceptor+", "Donor-", "Acceptor-"]
BACKGROUND_CLASS = 4
ORGANISM_NAMES = {0: "human", 1: "mouse"}

# Visual labels and colors matching predict_splicing_windows.py
CLASS_LABELS = {0: 'donor +', 1: 'acceptor +', 2: 'donor -', 3: 'acceptor -', 4: 'no splice site'}
CLASS_COLORS  = {0: '#ff7f00', 1: '#33a02c', 2: '#fdbf6f', 3: '#b2df8a', 4: '#1f78b4'}




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
    # Convert from 1-based (Spliser format) to 0-based (internal format)
    # This matches SpliceSiteUsageIndex which does: df["Position"] = df["Position"] - 1
    df["Position"] = df["Position"] - 1
    return set(zip(df["Chromosome"], df["Position"]))


def build_gtf_position_set(gtf_parquet: str) -> set[tuple[str, int]]:
    """Return set of (chrom, 0-based position) present in a GTF-derived parquet."""
    import pandas as pd
    df = pd.read_parquet(gtf_parquet)
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Position"]   = df["Position"].astype(int)
    return set(zip(df["Chromosome"], df["Position"]))





# ---------------------------------------------------------------------------
# Gene-overlap filtering
# ---------------------------------------------------------------------------

def build_gene_intervals(gene_annotation_parquet: str) -> dict[str, np.ndarray]:
    """Build merged gene intervals per chromosome from a gene annotation parquet.

    Reads transcript (or all) features and merges overlapping intervals per
    chromosome.  Returns ``{chrom: array of shape (N, 2)}`` where each row is
    a ``[start, end)`` 0-based half-open interval (sorted, non-overlapping).
    """
    import pandas as pd

    df = pd.read_parquet(
        gene_annotation_parquet,
        columns=["Chromosome", "Feature", "Start", "End"],
    )
    tx = df[df["Feature"] == "transcript"][["Chromosome", "Start", "End"]]
    if tx.empty:
        # Fallback: use all rows (some annotations lack an explicit transcript feature)
        tx = df[["Chromosome", "Start", "End"]].drop_duplicates()
    tx = tx.copy()
    tx["Chromosome"] = tx["Chromosome"].astype(str)

    result: dict[str, np.ndarray] = {}
    for chrom, grp in tx.groupby("Chromosome"):
        starts = grp["Start"].values.astype(np.int64)
        ends = grp["End"].values.astype(np.int64)
        order = np.argsort(starts)
        starts = starts[order]
        ends = ends[order]

        # Merge overlapping / adjacent intervals
        ms = [int(starts[0])]
        me = [int(ends[0])]
        for i in range(1, len(starts)):
            if starts[i] <= me[-1]:
                me[-1] = max(me[-1], int(ends[i]))
            else:
                ms.append(int(starts[i]))
                me.append(int(ends[i]))
        result[str(chrom)] = np.column_stack([ms, me])
    return result


def _window_overlaps_genes(
    chrom: str,
    win_start: int,
    win_end: int,
    gene_intervals: dict[str, np.ndarray],
) -> bool:
    """Return True if [win_start, win_end) overlaps any merged gene interval."""
    intervals = gene_intervals.get(chrom)
    if intervals is None:
        return False
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    # Overlap condition: interval.start < win_end AND interval.end > win_start
    lo = int(np.searchsorted(ends, win_start, side="right"))   # first end > win_start
    hi = int(np.searchsorted(starts, win_end, side="left"))    # first start >= win_end
    return lo < hi


def filter_bed_by_gene_overlap(
    bed_file: str,
    gene_intervals: dict[str, np.ndarray],
    output_bed: str | Path,
    sequence_length: int = 131_072,
) -> tuple[int, int]:
    """Write a filtered BED keeping only windows that overlap gene regions.

    Returns ``(n_kept, n_total)``.
    """
    half = sequence_length // 2
    output_bed = Path(output_bed)
    output_bed.parent.mkdir(parents=True, exist_ok=True)

    # Build a chromosome alias map (handle chr-prefix mismatches)
    alias: dict[str, str] = {}
    for chrom in gene_intervals:
        alias[chrom] = chrom
        if chrom.startswith("chr"):
            alias[chrom[3:]] = chrom
        else:
            alias["chr" + chrom] = chrom

    n_kept = n_total = 0
    with open(bed_file) as fin, open(output_bed, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            n_total += 1
            chrom_raw = parts[0]
            chrom = alias.get(chrom_raw, chrom_raw)
            center = (int(parts[1]) + int(parts[2])) // 2
            win_start = center - half
            win_end = center + half
            if _window_overlaps_genes(chrom, win_start, win_end, gene_intervals):
                fout.write(line)
                n_kept += 1
    return n_kept, n_total


def build_gene_overlap_mask(
    bed_file: str,
    gene_intervals: dict[str, np.ndarray],
    sequence_length: int = 131_072,
) -> np.ndarray:
    """Build a boolean mask over all per-position predictions.

    Returns a 1-D boolean array of length ``n_windows * sequence_length``
    where ``True`` means the position falls within a gene interval.
    """
    half = sequence_length // 2

    # Chromosome alias map (handle chr-prefix mismatches)
    alias: dict[str, str] = {}
    for chrom in gene_intervals:
        alias[chrom] = chrom
        if chrom.startswith("chr"):
            alias[chrom[3:]] = chrom
        else:
            alias["chr" + chrom] = chrom

    masks: list[np.ndarray] = []
    with open(bed_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            chrom_raw = parts[0]
            chrom = alias.get(chrom_raw, chrom_raw)
            center = (int(parts[1]) + int(parts[2])) // 2
            win_start = center - half

            win_mask = np.zeros(sequence_length, dtype=bool)
            intervals = gene_intervals.get(chrom)
            if intervals is not None:
                starts = intervals[:, 0]
                ends = intervals[:, 1]
                lo = int(np.searchsorted(ends, win_start, side="right"))
                hi = int(np.searchsorted(starts, win_start + sequence_length, side="left"))
                for gs, ge in zip(starts[lo:hi], ends[lo:hi]):
                    rel_start = max(0, int(gs) - win_start)
                    rel_end = min(sequence_length, int(ge) - win_start)
                    if rel_start < rel_end:
                        win_mask[rel_start:rel_end] = True
            masks.append(win_mask)

    return np.concatenate(masks)


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
             "or directly to the .pth file. The companion config.json should be in the "
             "same directory.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Optional model config.json path. If not specified, config.json in the --checkpoint "
             "directory will be used. This config defines model architecture, LoRA settings, etc.",
    )
    parser.add_argument(
        "--data-config",
        required=True,
        help="Path to data config JSON containing species definitions with genome paths, "
             "annotation parquets, usage parquets, and test/val BED files.",
    )
    parser.add_argument(
        "--eval-species",
        nargs="+",
        required=True,
        help="Species to evaluate (e.g., 'mouse' 'rat'). Must match keys in --data-config.",
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
        "--gene-overlap-annotation", nargs="*", default=None,
        help="Gene annotation parquets used only for gene-overlap filtering "
             "(Chromosome/Start/End/Feature), one per organism. When provided, "
             "only BED windows/positions overlapping gene regions are evaluated.",
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
        "--skip-plots", action="store_true",
        help="Skip all plotting (metrics only, faster).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing predictions. If not set, will skip prediction generation for species with existing predictions.",
    )
    parser.add_argument(
        "--skip-usage", action="store_true",
        help="Skip usage prediction accumulation (e.g., when evaluating a model trained on a different organism).",
    )
    parser.add_argument(
        "--max-windows", type=int, default=None,
        help="Maximum number of test windows to evaluate per species. If set, randomly samples this many windows.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling when --max-windows is used (default: 42).",
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
    """Load config and expand paths with ~ and $HOME."""
    with open(cfg_path) as f:
        cfg = json.load(f)
    
    # Expand paths in model config
    path_keys = {
        "genome", "annotation_parquet", "usage_parquet",
        "train_bed", "val_bed", "test_bed"
    }
    return expand_paths_in_dict(cfg, path_keys)


def load_data_config(data_cfg_path: Path, eval_species: list[str]) -> list[dict]:
    """Load data config and extract species_specs for requested species.
    
    Args:
        data_cfg_path: Path to data config JSON file
        eval_species: List of species names to evaluate (e.g., ['mouse', 'rat'])
    
    Returns:
        List of species_spec dicts (organism_index will be added later by matching with model)
    """
    with open(data_cfg_path) as f:
        data_cfg = json.load(f)
    
    # Expand paths in data config
    path_keys = {
        "genome", "annotation_parquet", "usage_parquet",
        "train_bed", "val_bed", "test_bed"
    }
    data_cfg = expand_paths_in_dict(data_cfg, path_keys)
    
    # Support both {"species": {...}} and direct {species_name: {...}} formats
    species_dict = data_cfg.get("species", data_cfg)
    
    species_specs = []
    for species_name in eval_species:
        if species_name not in species_dict:
            sys.exit(
                f"Species '{species_name}' not found in data config. "
                f"Available: {list(species_dict.keys())}"
            )
        
        spec = species_dict[species_name].copy()
        spec["name"] = species_name
        
        # Validate required fields
        required = ["genome", "annotation_parquet"]
        missing = [f for f in required if f not in spec]
        if missing:
            sys.exit(
                f"Species '{species_name}' missing required fields: {missing}"
            )
        
        # NOTE: organism_index is NOT set here - it's model-specific and will be
        # determined by matching annotation paths with the model's species_specs
        
        species_specs.append(spec)
    
    return species_specs

# Map common species names to scientific names
# Used to match data config species (by common name) to model species (by annotation path)
SPECIES_NAME_MAP = {
    "human": "Homo_sapiens",
    "mouse": "Mus_musculus",
    "rat": "Rattus_norvegicus",
}


def extract_species_from_path(path: str | Path) -> str | None:
    """Extract species scientific name from annotation_parquet path.
    
    Args:
        path: Path containing species name (e.g., ".../Homo_sapiens/splice_sites.parquet")
    
    Returns:
        Species scientific name (e.g., "Homo_sapiens") or None if not found
    """
    parts = Path(path).parts
    # Look for directory names matching species pattern (Genus_species)
    for part in reversed(parts):
        if part in SPECIES_NAME_MAP.values():
            return part
        # Check if it looks like a scientific name (capitalized with underscore)
        if "_" in part and part[0].isupper() and not part.startswith("FOLD"):
            return part
    return None


def match_species_to_model(
    model_cfg: dict,
    data_spec: dict,
) -> tuple[int, dict] | None:
    """Match a data species to the model's species by scientific name.
    
    Args:
        model_cfg: Model config with species_specs
        data_spec: Data species spec with common name (e.g., "mouse", "rat")
    
    Returns:
        (organism_index, matched_model_spec) or None if no match found
    """
    # Map common name to scientific name
    data_common_name = data_spec.get("name", "")
    data_scientific = SPECIES_NAME_MAP.get(data_common_name)
    
    if not data_scientific:
        return None
    
    # Match against model species by extracting from annotation paths
    for model_spec in model_cfg.get("species_specs", []):
        model_scientific = extract_species_from_path(
            model_spec.get("annotation_parquet", "")
        )
        
        if model_scientific == data_scientific:
            return model_spec["organism_index"], model_spec
    
    return None


def match_usage_conditions(
    model_cfg: dict,
    data_spec: dict,
    logger: logging.Logger | None = None,
) -> tuple[list[int] | None, dict[int, int] | None]:
    """Match trained usage conditions to data usage conditions (intersect).
    
    Args:
        model_cfg: Model config with trained conditions metadata
        data_spec: Data species spec with usage_parquet path and organism_index
        logger: Optional logger for info messages
    
    Returns:
        (data_condition_indices, model_to_data_mapping) or (None, None) if no usage
        - data_condition_indices: List of condition indices in data to evaluate
        - model_to_data_mapping: Dict mapping model output index to data condition index
    """
    if "usage_parquet" not in data_spec or not data_spec["usage_parquet"]:
        return None, None
    
    # Load data usage metadata
    data_usage_meta_path = Path(data_spec["usage_parquet"]).with_suffix(".json")
    if not data_usage_meta_path.exists():
        if logger:
            logger.warning(
                f"No usage metadata found at {data_usage_meta_path}; "
                f"cannot match conditions, will use all data conditions"
            )
        return None, None
    
    with open(data_usage_meta_path) as f:
        data_usage_meta = json.load(f)
    data_condition_labels = data_usage_meta.get("condition_labels", {})
    if not data_condition_labels:
        return None, None
    
    # Find model's species spec by organism_index (already matched in main())
    org_idx = data_spec["organism_index"]
    model_species = None
    for spec in model_cfg.get("species_specs", []):
        if spec["organism_index"] == org_idx:
            model_species = spec
            break
    
    if not model_species or "usage_parquet" not in model_species:
        # Model wasn't trained with usage for this organism
        return None, None
    
    model_usage_meta_path = Path(model_species["usage_parquet"]).with_suffix(".json")
    if not model_usage_meta_path.exists():
        if logger:
            logger.warning(
                f"No model usage metadata at {model_usage_meta_path}; "
                f"cannot match conditions"
            )
        return None, None
    
    with open(model_usage_meta_path) as f:
        model_usage_meta = json.load(f)
    model_condition_labels = model_usage_meta.get("condition_labels", {})
    
    # Find intersection: conditions present in both model and data
    model_cond_names = set(model_condition_labels.keys())
    data_cond_names = set(data_condition_labels.keys())
    common_cond_names = model_cond_names & data_cond_names
    
    if not common_cond_names:
        if logger:
            logger.warning(
                f"No overlapping condition names between model and data; "
                f"skipping usage evaluation"
            )
        return None, None
    
    # Build mapping: model condition index -> data condition index
    model_to_data = {}
    data_condition_indices = []
    
    for cond_name in sorted(common_cond_names):
        model_idx = model_condition_labels[cond_name]
        data_idx = data_condition_labels[cond_name]
        model_to_data[model_idx] = data_idx
        data_condition_indices.append(data_idx)
    
    if logger:
        logger.info(
            f"Matched {len(common_cond_names)} conditions between model and data "
            f"(model has {len(model_cond_names)}, data has {len(data_cond_names)})"
        )
    
    return data_condition_indices, model_to_data


def match_cross_species_usage_conditions(
    model_cfg: dict,
    data_spec: dict,
    logger: logging.Logger | None = None,
) -> dict[int, dict[int, int]]:
    """Match usage conditions for cross-species evaluation.
    
    For each model species with usage data, find overlapping conditions with
    the evaluation data. This enables evaluating predictions from multiple
    species-specific usage heads against the target species' conditions.
    
    Args:
        model_cfg: Model config with trained conditions metadata
        data_spec: Data species spec with usage_parquet path
        logger: Optional logger for info messages
    
    Returns:
        Dict mapping organism_index → condition_mapping (data_idx → model_idx)
        Empty dict if no usage data or no matches found.
    """
    if "usage_parquet" not in data_spec or not data_spec["usage_parquet"]:
        return {}
    
    # Load data usage metadata
    data_usage_meta_path = Path(data_spec["usage_parquet"]).with_suffix(".json")
    if not data_usage_meta_path.exists():
        if logger:
            logger.warning(
                f"No usage metadata found at {data_usage_meta_path}; "
                f"skipping cross-species usage evaluation"
            )
        return {}
    
    with open(data_usage_meta_path) as f:
        data_usage_meta = json.load(f)
    data_condition_labels = data_usage_meta.get("condition_labels", {})
    if not data_condition_labels:
        return {}
    
    data_cond_names = set(data_condition_labels.keys())
    cross_species_mappings: dict[int, dict[int, int]] = {}
    
    # Try matching against each model species' usage conditions
    for model_spec in model_cfg.get("species_specs", []):
        if "usage_parquet" not in model_spec:
            continue
            
        org_idx = model_spec["organism_index"]
        model_usage_meta_path = Path(model_spec["usage_parquet"]).with_suffix(".json")
        
        if not model_usage_meta_path.exists():
            continue
        
        with open(model_usage_meta_path) as f:
            model_usage_meta = json.load(f)
        model_condition_labels = model_usage_meta.get("condition_labels", {})
        
        if not model_condition_labels:
            continue
        
        # Find overlapping conditions
        model_cond_names = set(model_condition_labels.keys())
        common_cond_names = model_cond_names & data_cond_names
        
        if not common_cond_names:
            continue
        
        # Build mapping: data_idx -> model_idx
        condition_mapping = {}
        for cond_name in common_cond_names:
            data_idx = data_condition_labels[cond_name]
            model_idx = model_condition_labels[cond_name]
            condition_mapping[data_idx] = model_idx
        
        cross_species_mappings[org_idx] = condition_mapping
        
        if logger:
            model_species = extract_species_from_path(model_spec.get("annotation_parquet", ""))
            common_name = next((k for k, v in SPECIES_NAME_MAP.items() if v == model_species), model_species)
            logger.info(
                f"    Matched {len(common_cond_names)} conditions with {common_name} usage head "
                f"(organism_index {org_idx})"
            )
    
    return cross_species_mappings
    
    # Find intersection: conditions present in both model and data
    model_cond_names = set(model_condition_labels.keys())
    data_cond_names = set(data_condition_labels.keys())
    common_cond_names = model_cond_names & data_cond_names
    
    if not common_cond_names:
        if logger:
            logger.warning(
                f"No overlapping condition names between model and data; "
                f"skipping usage evaluation"
            )
        return None, None
    
    # Build mapping: model condition index -> data condition index
    model_to_data = {}
    data_condition_indices = []
    
    for cond_name in sorted(common_cond_names):
        model_idx = model_condition_labels[cond_name]
        data_idx = data_condition_labels[cond_name]
        model_to_data[model_idx] = data_idx
        data_condition_indices.append(data_idx)
    
    if logger:
        logger.info(
            f"Matched {len(common_cond_names)} conditions between model and data "
            f"(model has {len(model_cond_names)}, data has {len(data_cond_names)})"
        )
    
    return data_condition_indices, model_to_data


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

def build_model(cfg: dict, ckpt: dict, device: torch.device, logger: logging.Logger | None = None) -> nn.Module:
    """Reconstruct the AlphaGenome model with LoRA and splice heads from a checkpoint.

    Mirrors the model construction logic in finetune_splice.py:create_model().
    """
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.extensions.finetuning.heads import (
        create_splice_classification_finetuning_head,
        create_splice_usage_finetuning_head,
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

    msg = f"Building model (mode={cfg['mode']}, dtype={dtype_str}, num_organisms={num_organisms})..."
    if logger:
        logger.info(msg)
    else:
        print(msg)

    model = AlphaGenome(dtype_policy=dtype_policy)

    # Load pretrained trunk weights (base model, no heads)
    model = load_trunk(model, cfg["pretrained_weights"], exclude_heads=True)

    # Remove all heads so we can attach fresh fine-tuned ones
    model = remove_all_heads(model)

    # Apply LoRA adapters (must happen *before* loading model_state_dict
    # so the parameter names match)
    if cfg["mode"] == "lora" and cfg.get("lora_rank", 0) > 0:
        lora_targets = [t.strip() for t in cfg["lora_targets"].split(",")]
        msg = f"  Applying LoRA: rank={cfg['lora_rank']}, alpha={cfg['lora_alpha']}, targets={lora_targets}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
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
    
    # Attach usage heads (per-organism, stored as ModuleDict in model.splice_sites_usage_head)
    species_n_conditions: dict[int, int] = {
        int(k): v for k, v in cfg.get("species_n_conditions", {}).items()
    }
    usage_heads_modules = {}
    for org_idx, n_cond in species_n_conditions.items():
        if n_cond > 0:
            head = create_splice_usage_finetuning_head(
                n_conditions=n_cond,
                num_organisms=num_organisms,
            )
            usage_heads_modules[str(org_idx)] = head
    
    if usage_heads_modules:
        model.splice_sites_usage_head = nn.ModuleDict(usage_heads_modules)
    else:
        model.splice_sites_usage_head = None

    # Load fine-tuned weights (LoRA adapters + heads)
    # Strip _orig_mod. prefix that torch.compile adds to state-dict keys.
    raw_sd = ckpt["model_state_dict"]
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in raw_sd.items()}
    model.load_state_dict(sd, strict=False)
    
    # Load usage head weights from separate storage
    usage_heads_state_dicts: dict[str, dict] = ckpt.get("usage_heads_state_dicts", {})
    if model.splice_sites_usage_head is not None and usage_heads_state_dicts:
        for org_key, raw_usd in usage_heads_state_dicts.items():
            if org_key in model.splice_sites_usage_head:
                usd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in raw_usd.items()}
                model.splice_sites_usage_head[org_key].load_state_dict(usd)
    
    msg = f"  Loaded model_state_dict (epoch {ckpt.get('epoch', '?')})"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    model.to(device).eval()
    return model


def build_usage_heads(
    cfg: dict,
    ckpt: dict,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> dict[int, nn.Module]:
    """Extract usage heads from model.splice_sites_usage_head.
    
    **DEPRECATED**: This function is kept for backward compatibility with notebooks
    that expect a separate dict. Usage heads are now built in build_model() and
    stored in model.splice_sites_usage_head. New code should extract heads directly
    from the model instead of calling this function.
    
    For backward compatibility with notebooks that expect a separate dict.
    The actual heads are now stored in model.splice_sites_usage_head as a ModuleDict.
    """
    # This function is now deprecated - usage heads are built in build_model()
    # But we keep it for backward compatibility with existing notebooks
    
    # Build a minimal model just to extract usage heads
    # (This is inefficient but maintains API compatibility)
    temp_model = build_model(cfg, ckpt, device, logger=logger)
    
    usage_heads: dict[int, nn.Module] = {}
    if temp_model.splice_sites_usage_head is not None:
        for org_key, head in temp_model.splice_sites_usage_head.items():
            usage_heads[int(org_key)] = head
    
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
    seq_len: int = 131_072,
    skip_usage: bool = False,
    condition_mapping: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run inference for one organism.

    Args:
        condition_mapping: Optional dict mapping data condition index -> model condition index
                          for cross-species evaluation. When provided, only matched conditions
                          are accumulated.

    Returns
    -------
    cls_probs               : (N_positions, 5)  float32
    cls_labels              : (N_positions,)    int64
    usage_per_cond          : dict  condition_idx -> {'pred': list[float], 'true': list[float]}
    """
    all_cls_probs: list[np.ndarray] = []
    all_cls_labels: list[np.ndarray] = []
    usage_per_cond: dict[int, dict[str, list[float]]] = {}

    # Usage head for this organism (may be None if not available)
    usage_head = usage_heads.get(organism_index)
    
    # For usage head inference, we need the organism index of the usage head, not the data
    # Create a tensor with the usage head's organism index for all samples in batch
    usage_org_idx_t = None
    if usage_head is not None:
        # Will be broadcast to batch size when needed
        usage_org_idx_template = torch.tensor([organism_index], dtype=torch.long, device=device)

    for batch in tqdm(loader, desc="  Inference", unit="batch"):
        seq = batch["sequence"].to(device)
        org_idx_t = batch["organism_index"].to(device)

        with torch.no_grad():
            # Always use model.predict() for classification (works correctly)
            preds = model.predict(seq, org_idx_t, resolutions=(1,))
            cls_probs = preds["splice_sites_classification"]["probs"]  # (B, S, 5)
            
            if usage_head is not None:
                # For usage, get embeddings separately and call usage head
                # channels_last=False -> embeddings are (B, C, S) NCL format
                out = model.predict(
                    seq, org_idx_t,
                    resolutions=(1,),
                    channels_last=False,
                    embeddings_only=True,
                )
                emb_1bp = out["embeddings_1bp"]  # (B, C, S) NCL format

                # Create organism index tensor for usage head (use usage head's organism index, not data's)
                batch_size = emb_1bp.shape[0]
                usage_org_idx_t = usage_org_idx_template.expand(batch_size)

                # Usage head expects channels_last=True even with NCL input
                # IMPORTANT: Pass the usage head's organism index, not the data's organism index
                usage_preds = usage_head(
                    emb_1bp, usage_org_idx_t, channels_last=True
                )["predictions"].float()  # (B, S, n_cond)
            else:
                usage_preds = None

        all_cls_probs.append(cls_probs.cpu().numpy().reshape(-1, 5))
        all_cls_labels.append(batch["classification_labels"].numpy().reshape(-1))

        if usage_preds is not None and "usage_positions" in batch and not skip_usage:
            _accumulate_usage(
                usage_preds.cpu().numpy(),
                batch["usage_positions"].numpy(),
                batch["usage_values"].numpy(),
                batch["usage_mask"].numpy(),
                usage_per_cond,
                condition_mapping=condition_mapping,
            )

    return (
        np.concatenate(all_cls_probs, axis=0),
        np.concatenate(all_cls_labels, axis=0),
        usage_per_cond,
    )


def _accumulate_usage(
    usage_preds: np.ndarray,    # (B, S, T_model) where T_model = model n_conditions
    positions: np.ndarray,      # (B, max_sites)  -1 padded
    values: np.ndarray,         # (B, max_sites, n_data_cond)
    mask: np.ndarray,           # (B, max_sites, n_data_cond) bool
    acc: dict,
    condition_mapping: dict[int, int] | None = None,
) -> None:
    """Accumulate usage predictions and ground truth.
    
    Args:
        usage_preds: Model predictions (B, S, T_model)
        condition_mapping: Optional mapping from data condition index -> model condition index
                          for cross-species evaluation. Only accumulates mapped conditions.
    """
    B = positions.shape[0]
    n_data_cond = values.shape[2]
    for i in range(B):
        valid = positions[i] != -1
        if not valid.any():
            continue
        valid_pos = positions[i][valid]
        valid_vals = values[i][valid]       # (k, n_data_cond)
        valid_mask = mask[i][valid]         # (k, n_data_cond) bool
        valid_preds = usage_preds[i, valid_pos, :]  # (k, T_model)

        for data_c in range(n_data_cond):
            obs = valid_mask[:, data_c]
            if not obs.any():
                continue
            
            # Map data condition index to model condition index if needed
            if condition_mapping is not None:
                if data_c not in condition_mapping:
                    # This data condition wasn't in model training, skip
                    continue
                model_c = condition_mapping[data_c]
            else:
                # No mapping, assume 1:1 correspondence
                model_c = data_c
            
            # Extract predictions from model output index, store under data index
            entry = acc.setdefault(data_c, {"pred": [], "true": []})
            entry["pred"].extend(valid_preds[obs, model_c].tolist())
            entry["true"].extend(valid_vals[obs, data_c].tolist())





# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        sys.exit("scikit-learn is required. Install: pip install scikit-learn")

    y_binary = (labels != BACKGROUND_CLASS).astype(np.int32)
    s_binary = 1.0 - probs[:, BACKGROUND_CLASS]
    binary_auprc = float(average_precision_score(y_binary, s_binary))
    pos_rate = float(y_binary.mean())

    per_class_auprc: dict[str, float] = {}
    per_class_n: dict[str, int] = {}
    for c, name in enumerate(SPLICE_CLASS_NAMES):
        y_c = (labels == c).astype(np.int32)
        n_pos = int(y_c.sum())
        if n_pos == 0:
            continue
        ap = float(average_precision_score(y_c, probs[:, c]))
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


def compute_usage_metrics(usage_per_cond: dict) -> dict:
    try:
        from scipy.stats import pearsonr
    except ImportError:
        sys.exit("scipy is required. Install: pip install scipy")

    n_obs_total = sum(len(data["pred"]) for data in usage_per_cond.values())

    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    rs: list[float] = []
    for data in usage_per_cond.values():
        pred = np.array(data["pred"], dtype=np.float32)
        true = np.array(data["true"], dtype=np.float32)
        all_pred.append(pred)
        all_true.append(true)
        if len(pred) < 2 or pred.std() < 1e-8 or true.std() < 1e-8:
            continue
        r, _ = pearsonr(pred, true)
        rs.append(float(r))

    # Pooled (global) Pearson r across all observations
    pooled_r = float("nan")
    if all_pred:
        pool_pred = np.concatenate(all_pred)
        pool_true = np.concatenate(all_true)
        if len(pool_pred) >= 2 and pool_pred.std() > 1e-8 and pool_true.std() > 1e-8:
            pooled_r, _ = pearsonr(pool_pred, pool_true)
            pooled_r = float(pooled_r)

    if not rs:
        return {
            "usage_pearson_r": pooled_r,
            "usage_mean_pearson_r": float("nan"),
            "usage_median_pearson_r": float("nan"),
            "usage_n_conditions_evaluated": 0,
            "usage_n_conditions_total": len(usage_per_cond),
            "usage_n_observations": n_obs_total,
        }

    return {
        "usage_pearson_r": pooled_r,
        "usage_mean_pearson_r": float(np.mean(rs)),
        "usage_median_pearson_r": float(np.median(rs)),
        "usage_n_conditions_evaluated": len(rs),
        "usage_n_conditions_total": len(usage_per_cond),
        "usage_n_observations": n_obs_total,
    }


def _pearson_from_sufficient_stats(
    n: np.ndarray,
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    sum_x2: np.ndarray,
    sum_y2: np.ndarray,
    sum_xy: np.ndarray,
) -> np.ndarray:
    """Compute Pearson r from sufficient statistics (vectorized, exact)."""
    n = n.astype(np.float64, copy=False)
    mean_x = sum_x / np.maximum(n, 1.0)
    mean_y = sum_y / np.maximum(n, 1.0)
    var_x = sum_x2 - n * (mean_x * mean_x)
    var_y = sum_y2 - n * (mean_y * mean_y)
    cov_xy = sum_xy - n * (mean_x * mean_y)
    denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))

    r = np.full_like(denom, np.nan, dtype=np.float64)
    valid = (n >= 2.0) & (denom > 1e-12)
    r[valid] = cov_xy[valid] / denom[valid]
    return r


def compute_usage_metrics_from_stats(usage_stats: dict | None) -> dict:
    """Compute usage metrics from cached sufficient statistics."""
    if not usage_stats or len(usage_stats.get("n", [])) == 0:
        return {
            "usage_pearson_r": float("nan"),
            "usage_mean_pearson_r": float("nan"),
            "usage_median_pearson_r": float("nan"),
            "usage_n_conditions_evaluated": 0,
            "usage_n_conditions_total": int(len(usage_stats.get("cond_ids", []))) if usage_stats else 0,
            "usage_n_observations": 0,
        }

    n = np.asarray(usage_stats["n"], dtype=np.float64)
    sum_pred = np.asarray(usage_stats["sum_pred"], dtype=np.float64)
    sum_true = np.asarray(usage_stats["sum_true"], dtype=np.float64)
    sum_pred2 = np.asarray(usage_stats["sum_pred2"], dtype=np.float64)
    sum_true2 = np.asarray(usage_stats["sum_true2"], dtype=np.float64)
    sum_prod = np.asarray(usage_stats["sum_prod"], dtype=np.float64)

    rs = _pearson_from_sufficient_stats(n, sum_pred, sum_true, sum_pred2, sum_true2, sum_prod)
    valid_rs = rs[np.isfinite(rs)]

    # Pooled (global) Pearson r: sum sufficient statistics across all conditions
    pooled_r_arr = _pearson_from_sufficient_stats(
        np.array([n.sum()]),
        np.array([sum_pred.sum()]),
        np.array([sum_true.sum()]),
        np.array([sum_pred2.sum()]),
        np.array([sum_true2.sum()]),
        np.array([sum_prod.sum()]),
    )
    pooled_r = float(pooled_r_arr[0]) if np.isfinite(pooled_r_arr[0]) else float("nan")

    if valid_rs.size == 0:
        return {
            "usage_pearson_r": pooled_r,
            "usage_mean_pearson_r": float("nan"),
            "usage_median_pearson_r": float("nan"),
            "usage_n_conditions_evaluated": 0,
            "usage_n_conditions_total": int(n.size),
            "usage_n_observations": int(n.sum()),
        }

    return {
        "usage_pearson_r": pooled_r,
        "usage_mean_pearson_r": float(np.mean(valid_rs)),
        "usage_median_pearson_r": float(np.median(valid_rs)),
        "usage_n_conditions_evaluated": int(valid_rs.size),
        "usage_n_conditions_total": int(n.size),
        "usage_n_observations": int(n.sum()),
    }








def print_usage_metrics(org_name: str, usage_m: dict, logger: logging.Logger | None = None) -> None:
    log_func = logger.info if logger else print
    log_func(f"\n{org_name} – Usage Prediction Metrics:")
    log_func(f"  Mean Pearson r (across conditions): {usage_m['usage_mean_pearson_r']:.4f}")
    log_func(f"  Median Pearson r (across conditions): {usage_m['usage_median_pearson_r']:.4f}")
    log_func(
        f"  Conditions evaluated: {usage_m['usage_n_conditions_evaluated']} / {usage_m['usage_n_conditions_total']}"
    )
    log_func(f"  Total observations: {usage_m['usage_n_observations']:,}")


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
    logging.getLogger("evaluate_splice").info(f"  Saved: {output_path}")


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

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    hb = ax.hexbin(true_arr, pred_arr, gridsize=25, cmap="magma_r", mincnt=1)

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

    # Place colorbar below the marginal histograms
    cax = ax.inset_axes([1.28, 0, 0.04, 1])
    fig.colorbar(hb, cax=cax, label="Count")

    ax.set_xlabel("True Usage")
    ax.set_ylabel("Predicted Usage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger("evaluate_splice").info(f"  Saved: {output_path}")


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
        logging.getLogger("evaluate_splice").info(f"  Saved: {fn}")


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
        logging.getLogger("evaluate_splice").warning(f"Usage metadata not found at {usage_metadata_path}, skipping per-tissue plots")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load condition labels from metadata and build tissue map on the fly.
    # condition_labels is {"Brain_1": 0, "Cerebellum_2": 8, ...}
    # We invert to {0: "Brain_1", 8: "Cerebellum_2", ...} then split on
    # the last underscore to extract tissue name.
    try:
        with open(usage_metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        logging.getLogger("evaluate_splice").warning(f"Could not load usage metadata: {e}, skipping per-tissue plots")
        return

    condition_labels = metadata.get("condition_labels", {})
    if not condition_labels:
        logging.getLogger("evaluate_splice").warning("No condition_labels in metadata, skipping per-tissue plots")
        return

    # Invert: condition index -> label string
    idx_to_label = {int(v): k for k, v in condition_labels.items()}

    # Group conditions by tissue
    tissue_data: dict[str, dict[str, list]] = {}
    for cond_idx, data in usage_per_cond.items():
        label = idx_to_label.get(int(cond_idx), f"unknown_{cond_idx}")
        # Split on last underscore: "Brain_1" -> tissue="Brain"
        tissue = label.rsplit("_", 1)[0] if "_" in label else label
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
        logging.getLogger("evaluate_splice").info(f"  Saved: {fn}")


def plot_usage_correlation_by_tissue(
    usage_per_cond: dict,
    usage_metadata_path: Path,
    out_dir: Path,
    org_name: str,
) -> dict | None:
    """Plot boxplots of per-condition correlations grouped by tissue.
    
    Creates a single plot with boxplots showing the distribution of correlation
    values across timepoints for each tissue. Individual points are overlaid
    with size scaled to timepoint and colored by tissue.
    
    Returns:
        Dictionary with per-tissue metrics (mean, median, std) if successful,
        None otherwise.
    """
    if not usage_per_cond:
        return None
    
    if not usage_metadata_path.exists():
        logging.getLogger("evaluate_splice").warning(
            f"Usage metadata not found at {usage_metadata_path}, skipping tissue correlation boxplot"
        )
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr

    # Load condition labels from metadata
    try:
        with open(usage_metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        logging.getLogger("evaluate_splice").warning(
            f"Could not load usage metadata: {e}, skipping tissue correlation boxplot"
        )
        return None

    condition_labels = metadata.get("condition_labels", {})
    if not condition_labels:
        logging.getLogger("evaluate_splice").warning(
            "No condition_labels in metadata, skipping tissue correlation boxplot"
        )
        return None

    # Parse tissue and timepoint from condition names
    condition_info = {}
    for cond_name, cond_idx in condition_labels.items():
        parts = cond_name.rsplit('_', 1)
        if len(parts) == 2:
            tissue = parts[0]
            try:
                timepoint = int(parts[1])
                condition_info[cond_idx] = {
                    'name': cond_name,
                    'tissue': tissue,
                    'timepoint': timepoint
                }
            except ValueError:
                continue

    # Compute per-condition correlations
    condition_correlations = []
    for cond_idx, data in usage_per_cond.items():
        if len(data["pred"]) > 1 and cond_idx in condition_info:
            try:
                r, p_value = pearsonr(data["true"], data["pred"])
                condition_correlations.append({
                    'condition_idx': cond_idx,
                    'condition_name': condition_info[cond_idx]['name'],
                    'tissue': condition_info[cond_idx]['tissue'],
                    'timepoint': condition_info[cond_idx]['timepoint'],
                    'correlation': r,
                    'p_value': p_value,
                    'n_sites': len(data["pred"])
                })
            except Exception:
                continue

    if not condition_correlations:
        logging.getLogger("evaluate_splice").warning(
            "No valid condition correlations computed, skipping tissue correlation boxplot"
        )
        return None

    import pandas as pd
    corr_df = pd.DataFrame(condition_correlations)
    
    # Skip tissue plot if fewer than 2 conditions (not enough for boxplot)
    if len(corr_df) < 2:
        logging.getLogger("evaluate_splice").info(
            f"Only {len(corr_df)} valid condition(s) for tissue plot; skipping tissue correlation boxplot"
        )
        return None
    
    # Tissue colors (matching notebook)
    TISSUE_COLORS = {
        'Brain': '#3399cc',
        'Cerebellum': '#34ccff',
        'Heart': '#cc0100',
        'Kidney': '#cc9900',
        'Liver': '#339900',
        'Midbrain': '#6699cc',
        'Ovary': '#cc329a',
        'Testis': '#ff6600'
    }
    
    tissues = sorted(corr_df['tissue'].unique())
    n_tissues = len(tissues)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(6, n_tissues * 0.8), 5))
    
    # Create boxplot with lower zorder
    sns.boxplot(
        data=corr_df,
        x='tissue',
        y='correlation',
        order=tissues,
        showfliers=False,
        palette=[TISSUE_COLORS.get(t, '#888888') for t in tissues],
        ax=ax,
        zorder=1
    )
    
    # Add individual points with size scaled to timepoint and colored by tissue
    for i, tissue in enumerate(tissues):
        tissue_data = corr_df[corr_df['tissue'] == tissue]
        x_positions = np.random.default_rng(42).normal(i, 0.04, size=len(tissue_data))
        point_sizes = tissue_data['timepoint'].values * 3
        tissue_color = TISSUE_COLORS.get(tissue, '#888888')
        ax.scatter(
            x_positions,
            tissue_data['correlation'].values,
            s=point_sizes,
            color=tissue_color,
            alpha=0.4,
            edgecolors='black',
            linewidths=0.5,
            zorder=3
        )
    
    # Customize plot
    ax.set_ylim(0, 1)
    ax.set_xlabel('Tissue', fontsize=12)
    ax.set_ylabel('Pearson Correlation (r)', fontsize=12)
    ax.set_title(f'Splice Usage Correlations by Tissue\n{org_name}', fontsize=14, pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    
    # Add sample size annotations
    for i, tissue in enumerate(tissues):
        n = len(corr_df[corr_df['tissue'] == tissue])
        ax.text(i, 0.05, f'n={n}', ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    fn = out_dir / f"usage_correlation_by_tissue_{org_name}.png"
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.getLogger("evaluate_splice").info(f"  Saved: {fn}")
    
    # Log summary statistics
    logger = logging.getLogger("evaluate_splice")
    logger.info(f"\n{org_name} – Correlation Summary by Tissue:")
    summary = corr_df.groupby('tissue')['correlation'].agg(['count', 'mean', 'median', 'std'])
    for tissue in tissues:
        stats = summary.loc[tissue]
        logger.info(
            f"  {tissue:<15s} n={int(stats['count']):>3d}  "
            f"mean={stats['mean']:>5.3f}  median={stats['median']:>5.3f}  "
            f"std={stats['std']:>5.3f}"
        )
    
    # Build per-tissue metrics dictionary
    per_tissue_metrics = {}
    for tissue in tissues:
        stats = summary.loc[tissue]
        per_tissue_metrics[tissue] = {
            "n_conditions": int(stats['count']),
            "mean_pearson_r": float(stats['mean']),
            "median_pearson_r": float(stats['median']),
            "std_pearson_r": float(stats['std']),
        }
    
    return per_tissue_metrics





# ---------------------------------------------------------------------------
# Save / load predictions
# ---------------------------------------------------------------------------

def save_predictions(
    out_dir: Path,
    org_name: str,
    cls_probs: np.ndarray,
    cls_labels: np.ndarray,
    usage_per_cond: dict,
) -> None:
    """Persist prediction arrays to disk for later re-plotting / re-analysis as Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_kwargs: dict = dict(
        cls_probs  = cls_probs.astype(np.float32),
        cls_labels = cls_labels.astype(np.int64),
    )
    npz_path = out_dir / f"predictions_{org_name}.npz"
    np.savez_compressed(npz_path, **npz_kwargs)

    usage_path = out_dir / f"usage_{org_name}.npz"
    cond_ids_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []
    stats_cond_ids: list[int] = []
    stats_n: list[int] = []
    stats_sum_pred: list[float] = []
    stats_sum_true: list[float] = []
    stats_sum_pred2: list[float] = []
    stats_sum_true2: list[float] = []
    stats_sum_prod: list[float] = []

    for cond_idx, data in usage_per_cond.items():
        pred = np.asarray(data["pred"], dtype=np.float32)
        true = np.asarray(data["true"], dtype=np.float32)
        if pred.size == 0 or pred.size != true.size:
            continue
        cond_ids_chunks.append(np.full(pred.size, int(cond_idx), dtype=np.int32))
        pred_chunks.append(pred)
        true_chunks.append(true)

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

    np.savez_compressed(
        usage_path,
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

    logging.getLogger("evaluate_splice").info(f"  Saved: {npz_path}  {usage_path}")


def load_predictions(
    out_dir: Path,
    org_name: str,
    require_usage_arrays: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Load predictions saved by save_predictions."""
    npz_path = out_dir / f"predictions_{org_name}.npz"
    if not npz_path.exists():
        sys.exit(f"Prediction file not found: {npz_path}  (run without --skip-predictions first)")
    data       = np.load(npz_path)
    cls_probs  = data["cls_probs"]
    cls_labels = data["cls_labels"]

    usage_path_npz = out_dir / f"usage_{org_name}.npz"
    usage_path_json = out_dir / f"usage_{org_name}.json"
    usage_per_cond: dict = {}
    usage_stats: dict = {}
    if usage_path_npz.exists():
        u = np.load(usage_path_npz)
        if "stats_cond_ids" in u:
            usage_stats = {
                "cond_ids": u["stats_cond_ids"].astype(np.int32, copy=False),
                "n": u["stats_n"].astype(np.int64, copy=False),
                "sum_pred": u["stats_sum_pred"].astype(np.float64, copy=False),
                "sum_true": u["stats_sum_true"].astype(np.float64, copy=False),
                "sum_pred2": u["stats_sum_pred2"].astype(np.float64, copy=False),
                "sum_true2": u["stats_sum_true2"].astype(np.float64, copy=False),
                "sum_prod": u["stats_sum_prod"].astype(np.float64, copy=False),
            }
        if require_usage_arrays or not usage_stats:
            cond_ids = u["cond_ids"].astype(np.int32, copy=False)
            pred = u["pred"].astype(np.float32, copy=False)
            true = u["true"].astype(np.float32, copy=False)
            if cond_ids.size > 0:
                order = np.argsort(cond_ids, kind="stable")
                cond_sorted = cond_ids[order]
                pred_sorted = pred[order]
                true_sorted = true[order]
                uniq, starts, counts = np.unique(cond_sorted, return_index=True, return_counts=True)
                for c, s, k in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
                    usage_per_cond[int(c)] = {
                        "pred": pred_sorted[s:s + k],
                        "true": true_sorted[s:s + k],
                    }
    elif usage_path_json.exists():
        with open(usage_path_json) as f:
            usage_per_cond = {int(k): v for k, v in json.load(f).items()}
    

    
    logging.getLogger("evaluate_splice").info(f"  Loaded predictions from {npz_path}")
    return cls_probs, cls_labels, usage_stats, usage_per_cond


# ---------------------------------------------------------------------------
# Per-source metrics
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics(org_name: str, cls_m: dict, usage_m: dict | None, logger: logging.Logger | None = None) -> None:
    sep = "=" * 62
    log_func = logger.info if logger else print
    log_func(sep)
    log_func(f"  {org_name.upper()} RESULTS")
    log_func(sep)
    
    # Only print classification metrics if cls_m is not empty
    if cls_m:
        log_func(f"  Positions evaluated : {cls_m['n_positions']:>12,}")
        log_func(f"  Positive rate       : {cls_m['positive_rate']:>12.4%}  (splice / all)")
        log_func(f"\n  Classification AUPRC")
        log_func(f"  {'Binary (splice vs background)':<38s} {cls_m['binary_auprc']:.4f}")
        log_func(f"  {'Mean per-class AUPRC':<38s} {cls_m['mean_splice_class_auprc']:.4f}")
        for name in SPLICE_CLASS_NAMES:
            if name in cls_m["per_class_auprc"]:
                auprc = cls_m["per_class_auprc"][name]
                n = cls_m["per_class_n_positives"][name]
                log_func(f"    {name:<36s} {auprc:.4f}  (n={n:,})")

    if usage_m and usage_m.get("usage_n_conditions_evaluated", 0) > 0:
        # Add newline before usage section only if we printed classification metrics
        prefix = "\n  " if cls_m else "  "
        log_func(
            f"{prefix}Usage Pearson r  (evaluated on "
            f"{usage_m['usage_n_conditions_evaluated']} / "
            f"{usage_m['usage_n_conditions_total']} conditions, "
            f"{usage_m['usage_n_observations']:,} observations)"
        )
        log_func(f"  {'Mean r':<38s} {usage_m['usage_mean_pearson_r']:.4f}")
        log_func(f"  {'Median r':<38s} {usage_m['usage_median_pearson_r']:.4f}")
    elif usage_m:
        prefix = "\n  " if cls_m else "  "
        log_func(f"{prefix}Usage: no valid conditions found (too few observations per condition)")








# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Set up logging -------------------------------------------------------
    log_path = out_dir / "eval.log"
    logger = setup_logging(log_path)
    
    logger.info(f"=== evaluate_splice  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    logger.info(f"Log: {log_path}")

    # -- Resolve checkpoint and model config ---------------------------------
    pth_path, default_cfg_path = resolve_checkpoint(args.checkpoint)
    model_cfg_path = Path(args.model_config) if args.model_config is not None else default_cfg_path
    if not model_cfg_path.exists():
        sys.exit(f"Model config not found: {model_cfg_path}")
    model_cfg = load_config(model_cfg_path)
    logger.info(f"Checkpoint   : {pth_path}")
    logger.info(f"Model config : {model_cfg_path}")

    # -- Load data config and resolve species --------------------------------
    data_cfg_path = Path(args.data_config)
    if not data_cfg_path.exists():
        sys.exit(f"Data config not found: {data_cfg_path}")
    
    species_specs = load_data_config(data_cfg_path, args.eval_species)
    logger.info(f"Data config  : {data_cfg_path}")
    logger.info(f"Eval species : {[s['name'] for s in species_specs]}")
    
    # Match each data species to model's organism_index by species name
    # If species not found, allow cross-species evaluation (use organism_index 0 for classification)
    for spec in species_specs:
        match_result = match_species_to_model(model_cfg, spec)
        if match_result is None:
            # Cross-species evaluation: species not in training data
            # Use organism_index 0 for classification head (species-agnostic)
            spec["organism_index"] = 0
            spec["cross_species"] = True
            
            # Collect trained species for info message
            model_species_list = []
            for ms in model_cfg.get('species_specs', []):
                species = extract_species_from_path(ms.get('annotation_parquet', ''))
                if species:
                    common_name = next((k for k, v in SPECIES_NAME_MAP.items() if v == species), species)
                    model_species_list.append(common_name)
            
            logger.info(
                f"  '{spec['name']}' not in model training data → cross-species evaluation"
            )
            logger.info(
                f"    Model trained on: {', '.join(model_species_list) if model_species_list else 'unknown'}"
            )
            logger.info(
                f"    Using organism_index 0 for classification (species-agnostic head)"
            )
            if spec.get("usage_parquet"):
                logger.info(
                    f"    Will match '{spec['name']}' usage conditions against all trained species"
                )
        else:
            # Same-species evaluation: direct match
            org_idx, model_spec = match_result
            spec["organism_index"] = org_idx
            spec["cross_species"] = False
            model_species = extract_species_from_path(model_spec['annotation_parquet'])
            logger.info(
                f"  Matched '{spec['name']}' → organism_index {org_idx} ({model_species})"
            )

    # Use test_bed from data config (default to val_bed if test_bed not present)
    bed_files = []
    for spec in species_specs:
        bed_file = spec.get("test_bed") or spec.get("val_bed")
        if not bed_file:
            sys.exit(f"Species '{spec['name']}' missing both 'test_bed' and 'val_bed' in data config")
        bed_files.append(bed_file)
    logger.info(f"BED files    : {bed_files}")

    if args.gene_overlap_annotation is not None and len(args.gene_overlap_annotation) != len(species_specs):
        sys.exit(
            f"--gene-overlap-annotation: expected {len(species_specs)} file(s), "
            f"got {len(args.gene_overlap_annotation)}"
        )

    def species_needs_inference(spec: dict) -> bool:
        """Return True when this species still needs inference in the current run."""
        if args.skip_predictions:
            return False
        if args.overwrite:
            return True
        org_idx = spec["organism_index"]
        org_name = spec.get("name", ORGANISM_NAMES.get(org_idx, f"organism_{org_idx}"))
        pred_npz = out_dir / f"predictions_{org_name}.npz"
        usage_json = out_dir / f"usage_{org_name}.json"
        usage_npz = out_dir / f"usage_{org_name}.npz"
        return not (pred_npz.exists() and (usage_json.exists() or usage_npz.exists()))

    needs_inference = [species_needs_inference(spec) for spec in species_specs]

    device = torch.device(args.device)

    model_loaded = False
    gpu_released_after_inference = False
    if not args.skip_predictions and any(needs_inference):
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
        model = build_model(model_cfg, ckpt, device, logger=logger)
        
        # Extract usage heads from the built model (usage heads are now built in build_model)
        usage_heads: dict[int, nn.Module] = {}
        if model.splice_sites_usage_head is not None:
            for org_key, head in model.splice_sites_usage_head.items():
                usage_heads[int(org_key)] = head
        
        model_loaded = True

        from torch.utils.data import DataLoader
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteAnnotation,
            SpliceSiteDataset,
            SpliceSiteUsageIndex,
            collate_splice,
        )
    elif not args.skip_predictions:
        logger.info("All predictions already exist. Inference is not needed; skipping model/GPU initialization.")

    all_results: dict[str, dict] = {}

    need_usage_arrays = not args.skip_plots
    if args.skip_predictions and args.skip_plots:
        logger.info("Using stats-first loading mode: raw usage arrays are skipped for faster exact metrics.")


    for i, (spec, bed_file) in enumerate(zip(species_specs, bed_files)):
        org_idx: int = spec["organism_index"]
        org_name: str = spec.get("name", ORGANISM_NAMES.get(org_idx, f"organism_{org_idx}"))

        logger.info(f"{'='*62}")
        logger.info(f"  {org_name.upper()}")
        logger.info(f"{'='*62}")

        # Check if predictions already exist
        pred_npz = out_dir / f"predictions_{org_name}.npz"
        usage_json = out_dir / f"usage_{org_name}.json"
        usage_npz = out_dir / f"usage_{org_name}.npz"
        if (not args.overwrite and pred_npz.exists() and (usage_json.exists() or usage_npz.exists())):
            logger.info(f"[{org_name}] Predictions already exist. Skipping generation (use --overwrite to force).")
            cls_probs, cls_labels, usage_stats, usage_per_cond = load_predictions(
                out_dir,
                org_name,
                require_usage_arrays=need_usage_arrays,
            )
            seq_len = model_cfg.get("sequence_length", 131_072)
            
            # Reconstruct usage_results from saved files
            usage_results: dict[str, tuple[dict, str]] = {}
            # Check for cross-species usage files (usage_{org_name}_from_*.npz)
            for usage_file in out_dir.glob(f"usage_{org_name}_from_*.npz"):
                suffix = usage_file.stem.replace(f"usage_{org_name}_", "")
                source_name = suffix.replace("from_", "")
                
                # Load usage data directly from NPZ file
                u = np.load(usage_file)
                loaded_usage_per_cond: dict = {}
                if "cond_ids" in u:
                    cond_ids = u["cond_ids"].astype(np.int32, copy=False)
                    pred = u["pred"].astype(np.float32, copy=False)
                    true = u["true"].astype(np.float32, copy=False)
                    if cond_ids.size > 0:
                        order = np.argsort(cond_ids, kind="stable")
                        cond_sorted = cond_ids[order]
                        pred_sorted = pred[order]
                        true_sorted = true[order]
                        uniq, starts, counts = np.unique(cond_sorted, return_index=True, return_counts=True)
                        for c, s, k in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
                            loaded_usage_per_cond[int(c)] = {
                                "pred": pred_sorted[s:s + k],
                                "true": true_sorted[s:s + k],
                            }
                
                usage_results[suffix] = (loaded_usage_per_cond, source_name)
            
            # If no cross-species files, use the standard usage file
            if not usage_results and usage_per_cond:
                usage_results[""] = (usage_per_cond, org_name)

        elif args.skip_predictions:
            logger.info(f"[{org_name}] Loading saved predictions …")
            cls_probs, cls_labels, usage_stats, usage_per_cond = load_predictions(
                out_dir,
                org_name,
                require_usage_arrays=need_usage_arrays,
            )
            seq_len = model_cfg.get("sequence_length", 131_072)
            
            # Reconstruct usage_results from saved files
            usage_results: dict[str, tuple[dict, str]] = {}
            # Check for cross-species usage files (usage_{org_name}_from_*.npz)
            for usage_file in out_dir.glob(f"usage_{org_name}_from_*.npz"):
                suffix = usage_file.stem.replace(f"usage_{org_name}_", "")
                source_name = suffix.replace("from_", "")
                
                # Load usage data directly from NPZ file
                u = np.load(usage_file)
                loaded_usage_per_cond: dict = {}
                if "cond_ids" in u:
                    cond_ids = u["cond_ids"].astype(np.int32, copy=False)
                    pred = u["pred"].astype(np.float32, copy=False)
                    true = u["true"].astype(np.float32, copy=False)
                    if cond_ids.size > 0:
                        order = np.argsort(cond_ids, kind="stable")
                        cond_sorted = cond_ids[order]
                        pred_sorted = pred[order]
                        true_sorted = true[order]
                        uniq, starts, counts = np.unique(cond_sorted, return_index=True, return_counts=True)
                        for c, s, k in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
                            loaded_usage_per_cond[int(c)] = {
                                "pred": pred_sorted[s:s + k],
                                "true": true_sorted[s:s + k],
                            }
                
                usage_results[suffix] = (loaded_usage_per_cond, source_name)
            
            # If no cross-species files, use the standard usage file
            if not usage_results and usage_per_cond:
                usage_results[""] = (usage_per_cond, org_name)
        else:
            # -- Load annotation (for gene filtering if provided) ----------------
            seq_len = model_cfg.get("sequence_length", 131_072)
            if args.gene_overlap_annotation is not None:
                gene_intervals = build_gene_intervals(args.gene_overlap_annotation[i])
                n_genes = sum(len(iv) for iv in gene_intervals.values())
                logger.info(
                    f"[{org_name}] Loaded {n_genes:,} merged gene intervals "
                    f"across {len(gene_intervals)} chromosomes"
                )
                filtered_bed = out_dir / f"filtered_bed_{org_name}.bed"
                n_kept, n_total = filter_bed_by_gene_overlap(
                    bed_file, gene_intervals, filtered_bed, seq_len,
                )
                logger.info(
                    f"[{org_name}] Gene-overlap filter: kept {n_kept:,} / "
                    f"{n_total:,} windows ({100*n_kept/max(n_total,1):.1f}%)"
                )
                bed_file = str(filtered_bed)

            # -- Dataset & loader --------------------------------------------
            logger.info(f"[{org_name}] Loading annotation from {spec['annotation_parquet']} …")
            annotation = SpliceSiteAnnotation(spec["annotation_parquet"])

            usage_index: SpliceSiteUsageIndex | None = None
            
            # For cross-species evaluation, we may have multiple usage heads to evaluate
            # Map: organism_index -> (condition_mapping, source_species_name)
            usage_head_configs: dict[int, tuple[dict[int, int], str]] = {}
            
            if spec.get("usage_parquet") and not args.skip_usage:
                logger.info(f"[{org_name}] Loading usage index from {spec['usage_parquet']} …")
                # usage.parquet is already in 0-based coordinates (verified by direct overlap with annotations)
                # Use usage_coord_base=0 to prevent incorrect -1 conversion.
                # Keep all conditions per observed site so unobserved site-condition pairs are treated as 0 usage.
                usage_index = SpliceSiteUsageIndex(
                    spec["usage_parquet"],
                    min_coverage=args.min_coverage,
                    usage_coord_base=0,
                    observed_conditions_only=False,
                )
                
                # Match conditions between model and data
                if spec.get("cross_species", False):
                    # Cross-species: match against all model species' usage heads
                    cross_species_mappings = match_cross_species_usage_conditions(
                        model_cfg, spec, logger=logger
                    )
                    
                    if cross_species_mappings:
                        # Evaluate with ALL usage heads that have overlapping conditions
                        for usage_org_idx, condition_mapping in cross_species_mappings.items():
                            # Get source species name
                            source_spec = next(
                                (s for s in model_cfg.get("species_specs", []) 
                                 if s["organism_index"] == usage_org_idx),
                                None
                            )
                            if source_spec:
                                source_species = extract_species_from_path(
                                    source_spec.get("annotation_parquet", "")
                                )
                                source_name = next(
                                    (k for k, v in SPECIES_NAME_MAP.items() if v == source_species),
                                    f"org{usage_org_idx}"
                                )
                                usage_head_configs[usage_org_idx] = (condition_mapping, source_name)
                                logger.info(
                                    f"[{org_name}] Will evaluate with {source_name} usage head "
                                    f"(organism_index {usage_org_idx}, {len(condition_mapping)} conditions)"
                                )
                    else:
                        logger.info(f"[{org_name}] No overlapping usage conditions found with any trained species")
                else:
                    # Same-species: first add the matching head
                    data_condition_indices, model_to_data_map = match_usage_conditions(
                        model_cfg, spec, logger=logger
                    )
                    
                    if model_to_data_map is not None:
                        # Invert mapping: we need data_idx -> model_idx for accumulation
                        condition_mapping = {v: k for k, v in model_to_data_map.items()}
                        usage_head_configs[org_idx] = (condition_mapping, org_name)
                        logger.info(
                            f"[{org_name}] Using condition mapping: "
                            f"{len(condition_mapping)} matched conditions"
                        )

                    # Also evaluate with all OTHER trained usage heads that have overlapping conditions
                    all_cross_mappings = match_cross_species_usage_conditions(
                        model_cfg, spec, logger=logger
                    )
                    for other_org_idx, other_cond_mapping in all_cross_mappings.items():
                        if other_org_idx == org_idx:
                            continue  # Already handled above
                        source_spec = next(
                            (s for s in model_cfg.get("species_specs", [])
                             if s["organism_index"] == other_org_idx),
                            None,
                        )
                        if source_spec:
                            source_species = extract_species_from_path(
                                source_spec.get("annotation_parquet", "")
                            )
                            source_name = next(
                                (k for k, v in SPECIES_NAME_MAP.items() if v == source_species),
                                f"org{other_org_idx}",
                            )
                            usage_head_configs[other_org_idx] = (other_cond_mapping, source_name)
                            logger.info(
                                f"[{org_name}] Will also evaluate with {source_name} usage head "
                                f"(organism_index {other_org_idx}, {len(other_cond_mapping)} conditions)"
                            )

            logger.info(f"[{org_name}] Building dataset from {bed_file} …")
            dataset = SpliceSiteDataset(
                genome=spec["genome"],
                bed_file=bed_file,
                annotation=annotation,
                usage_index=usage_index,
                sequence_length=seq_len,
                organism_index=org_idx,
                max_sites=model_cfg.get("max_sites", 1024),
            )
            
            total_windows = len(dataset)
            
            # Random sampling if max_windows is specified
            if args.max_windows is not None and args.max_windows < total_windows:
                import random
                from torch.utils.data import Subset
                
                random.seed(args.seed)
                indices = random.sample(range(total_windows), args.max_windows)
                dataset = Subset(dataset, indices)
                logger.info(
                    f"[{org_name}] Randomly sampled {len(dataset):,} / {total_windows:,} windows "
                    f"(seed={args.seed})"
                )
            else:
                logger.info(f"[{org_name}] {total_windows:,} windows to evaluate")

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_splice,
            )

            # -- Inference ---------------------------------------------------
            # Classification predictions (same for all usage heads)
            logger.info(f"[{org_name}] Running classification inference …")
            cls_probs, cls_labels, _ = collect_predictions(
                model=model,
                usage_heads=usage_heads,
                loader=loader,
                device=device,
                organism_index=org_idx,
                seq_len=seq_len,
                skip_usage=True,  # Skip usage for now, compute separately per head
                condition_mapping=None,
            )
            
            # Usage predictions (one per usage head configuration)
            usage_results: dict[str, tuple[dict, str]] = {}  # suffix -> (usage_per_cond, source_name)
            
            if usage_head_configs and not args.skip_usage:
                for usage_org_idx, (condition_mapping, source_name) in usage_head_configs.items():
                    logger.info(
                        f"[{org_name}] Running usage inference with {source_name} head "
                        f"(organism_index {usage_org_idx}) …"
                    )
                    _, _, usage_per_cond = collect_predictions(
                        model=model,
                        usage_heads=usage_heads,
                        loader=loader,
                        device=device,
                        organism_index=usage_org_idx,
                        seq_len=seq_len,
                        skip_usage=False,
                        condition_mapping=condition_mapping,
                    )
                    
                    # Create suffix for filenames:
                    # Use "from_{source_name}" whenever the head's organism_index differs
                    # from the eval species (including additional heads in same-species evals).
                    if usage_org_idx != org_idx or spec.get("cross_species", False):
                        suffix = f"from_{source_name}"
                    else:
                        suffix = ""
                    
                    usage_results[suffix] = (usage_per_cond, source_name)
            
            # Save predictions
            # Classification predictions (once, no usage)
            npz_kwargs: dict = dict(
                cls_probs=cls_probs.astype(np.float32),
                cls_labels=cls_labels.astype(np.int64),
            )
            cls_npz_path = out_dir / f"predictions_{org_name}.npz"
            np.savez_compressed(cls_npz_path, **npz_kwargs)
            logger.info(f"  Saved: {cls_npz_path}")
            
            # Usage predictions (one file per usage head)
            if usage_results:
                import tempfile
                import shutil
                
                for suffix, (usage_per_cond, source_name) in usage_results.items():
                    result_name = f"{org_name}_{suffix}" if suffix else org_name
                    usage_path = out_dir / f"usage_{result_name}.npz"
                    
                    # Use save_predictions to create usage file, extract just the usage part
                    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
                        tmpdir_path = Path(tmpdir)
                        save_predictions(tmpdir_path, "temp", cls_probs, cls_labels, usage_per_cond)
                        # Move only usage file
                        temp_usage = tmpdir_path / "usage_temp.npz"
                        if temp_usage.exists():
                            shutil.move(str(temp_usage), str(usage_path))
                            logger.info(f"  Saved: {usage_path}")
                            
            
            usage_stats = {}

            # If there are no remaining species that still require inference,
            # free GPU model memory immediately before CPU metrics/plotting.
            if model_loaded and not any(needs_inference[i + 1:]) and not gpu_released_after_inference:
                logger.info("=" * 62)
                logger.info("Inference complete. Releasing GPU memory...")
                logger.info("=" * 62)

                model.cpu()
                del model
                for h in usage_heads.values():
                    h.cpu()
                del usage_heads
                model_loaded = False

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                logger.info("GPU memory released. Proceeding with metrics and plotting on CPU...")
                gpu_released_after_inference = True

        logger.info(f"{'='*62}")
        logger.info(f"  {org_name.upper()} - Metrics & Plotting")
        logger.info(f"{'='*62}")

        # -- Filter positions to gene-overlapping sites ----------------------
        if args.gene_overlap_annotation is not None:
            gene_intervals = build_gene_intervals(args.gene_overlap_annotation[i])

            # Determine which BED was used for predictions
            filtered_bed_path = out_dir / f"filtered_bed_{org_name}.bed"
            if filtered_bed_path.exists():
                mask_bed = str(filtered_bed_path)
            else:
                mask_bed = str(bed_file)

            gene_mask = build_gene_overlap_mask(
                mask_bed, gene_intervals, seq_len,
            )
            n_in_gene = int(gene_mask.sum())
            n_total_pos = len(gene_mask)
            logger.info(
                f"[{org_name}] Position-level gene filter: {n_in_gene:,} / "
                f"{n_total_pos:,} positions in genes ({100*n_in_gene/max(n_total_pos,1):.1f}%)"
            )

            cls_probs = cls_probs[gene_mask]
            cls_labels = cls_labels[gene_mask]

        # -- Metrics ---------------------------------------------------------
        logger.info(f"[{org_name}] Computing metrics …")
        cls_m = compute_classification_metrics(cls_probs, cls_labels)
        
        # Compute usage metrics for each usage head variant
        usage_metrics_by_head: dict[str, dict] = {}
        per_tissue_by_head: dict[str, dict] = {}
        
        if usage_results:
            for suffix, (usage_per_cond, source_name) in usage_results.items():
                result_key = suffix if suffix else "same_species"
                logger.info(
                    f"[{org_name}] Computing usage metrics for {source_name} head" +
                    (f" ({suffix})" if suffix else "")
                )
                usage_m = compute_usage_metrics(usage_per_cond) if usage_per_cond else None
                if usage_m:
                    usage_metrics_by_head[result_key] = usage_m
                    print_metrics(
                        f"{org_name} ({source_name} head)" if suffix else org_name,
                        cls_m if not suffix else {},  # Only print classification once
                        usage_m,
                        logger=logger
                    )
                
                # Plots for this usage head
                if not args.skip_plots and usage_per_cond:
                    plot_suffix = f"_{suffix}" if suffix else ""
                    logger.info(
                        f"[{org_name}] Generating plots for {source_name} head" +
                        (f" ({suffix})" if suffix else "")
                    )
                    
                    # Only plot classification PR curve once
                    if not suffix or len(usage_results) == 1:
                        plot_pr_curves(
                            cls_probs, cls_labels, org_name,
                            out_dir / f"pr_curve_{org_name}.png"
                        )
                    
                    # Usage plots for each head
                    plot_usage_density(
                        usage_per_cond, f"{org_name} ({source_name})",
                        out_dir / f"usage_density_{org_name}{plot_suffix}.png"
                    )
                    
                    # Per-tissue metrics and plots
                    if args.per_tissue and spec.get("usage_parquet"):
                        usage_metadata_path = Path(spec["usage_parquet"]).with_suffix(".json")
                        per_tissue_metrics = plot_usage_correlation_by_tissue(
                            usage_per_cond, usage_metadata_path, out_dir,
                            f"{org_name}{plot_suffix}"
                        )
                        if per_tissue_metrics:
                            per_tissue_by_head[result_key] = per_tissue_metrics
        
        # Build results structure
        all_results[org_name] = {**cls_m}
        
        # Add usage metrics
        if len(usage_metrics_by_head) == 1:
            # Single usage head - add directly for backward compatibility
            single_metrics = next(iter(usage_metrics_by_head.values()))
            all_results[org_name].update(single_metrics)
            
            if per_tissue_by_head:
                all_results[org_name]["per_tissue_usage"] = next(iter(per_tissue_by_head.values()))
        elif len(usage_metrics_by_head) > 1:
            # Multiple usage heads - nest under "usage_by_head"
            all_results[org_name]["usage_by_head"] = usage_metrics_by_head
            
            if per_tissue_by_head:
                all_results[org_name]["per_tissue_by_head"] = per_tissue_by_head

        del cls_probs, cls_labels, usage_results, usage_stats

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- Release GPU memory after all species are processed ------------------
    if not args.skip_predictions and model_loaded:
        logger.info("="*62)
        logger.info("Species-by-species evaluation complete. Releasing GPU memory...")
        logger.info("="*62)

        model.cpu()
        del model
        for h in usage_heads.values():
            h.cpu()
        del usage_heads
        model_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("GPU memory released.")

    # -- Multi-species macro-average -----------------------------------------
    if len(species_specs) > 1:
        keys_to_avg = [
            "binary_auprc", "mean_splice_class_auprc",
            "usage_pearson_r", "usage_mean_pearson_r", "usage_median_pearson_r",
        ]
        logger.info(f"{'='*62}")
        logger.info("  MACRO-AVERAGE ACROSS SPECIES")
        logger.info(f"{'='*62}")
        for k in keys_to_avg:
            vals = [v[k] for v in all_results.values()
                    if k in v and not np.isnan(v.get(k, float("nan")))]
            if vals:
                logger.info(f"  {k:<38s} {np.mean(vals):.4f}")

    # -- Save JSON -----------------------------------------------------------
    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None)

    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
