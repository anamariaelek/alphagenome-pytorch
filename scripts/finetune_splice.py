#!/usr/bin/env python
"""AlphaGenome splice-site fine-tuning script.

Trains splice-site classification (5-class: Donor+/Acceptor+/Donor-/Acceptor-/Background)
and optionally splice-site usage prediction on top of a pretrained AlphaGenome trunk.

Preprocessing:
    # 1. Convert splice-site annotations to Parquet
    python scripts/convert_splice_sites_to_parquet.py \\
        --annotation splice_sites.gff \\
        --genome hg38.fa \\
        --output splice_annotation.parquet

    # 2. (Optional) Convert Spliser usage data to Parquet
    python scripts/convert_splice_usage_to_parquet.py \\
        --input-dir /path/to/spliser/Homo_sapiens/ \\
        --output splice_usage.parquet

Usage:
    # Linear probe (frozen backbone, train classification head only)
    python scripts/finetune_splice.py --mode linear-probe \\
        --genome hg38.fa \\
        --annotation-parquet splice_annotation.parquet \\
        --train-bed train.bed --val-bed val.bed \\
        --pretrained-weights model.pth

    # LoRA fine-tuning with classification + usage heads
    python scripts/finetune_splice.py --mode lora \\
        --genome hg38.fa \\
        --annotation-parquet splice_annotation.parquet \\
        --usage-parquet /path/to/splice_usage.parquet \\
        --train-bed train.bed --val-bed val.bed \\
        --pretrained-weights model.pth \\
        --lora-rank 8

    # LoRA fine-tuning, also fully unfreezing the encoder/decoder conv
    # towers and output embedders (in addition to LoRA-adapted attention,
    # splice heads, and species embeddings)
    python scripts/finetune_splice.py --mode lora \\
        --genome hg38.fa \\
        --annotation-parquet splice_annotation.parquet \\
        --train-bed train.bed --val-bed val.bed \\
        --pretrained-weights model.pth \\
        --train-components encoder,decoder,embedder_128bp,embedder_1bp

    # Full fine-tuning (all parameters)
    python scripts/finetune_splice.py --mode full \\
        --genome hg38.fa \\
        --annotation-parquet splice_annotation.parquet \\
        --train-bed train.bed --val-bed val.bed \\
        --pretrained-weights model.pth

    # Multi-GPU with DDP
    torchrun --nproc_per_node=4 scripts/finetune_splice.py --mode lora ...

    # Multi-species training (human + mouse) via YAML config
    # config.yaml:
    #   species:
    #     - name: human
    #       genome: hg38.fa
    #       annotation_parquet: human_splice.parquet
    #       usage_parquet: human_usage.parquet   # optional
    #       train_bed: human_train.bed
    #       val_bed: human_val.bed
    #       organism_index: 0
    #     - name: mouse
    #       genome: mm10.fa
    #       annotation_parquet: mouse_splice.parquet
    #       train_bed: mouse_train.bed
    #       val_bed: mouse_val.bed
    #       organism_index: 1
    python scripts/finetune_splice.py --mode lora \\
        --pretrained-weights model.pth \\
        --config config.yaml

    # Resume from checkpoint
    python scripts/finetune_splice.py ... --resume auto
    python scripts/finetune_splice.py ... --resume path/to/checkpoint.pth

    # Graceful shutdown (saves checkpoint_preempt.pth)
    kill -USR1 <pid>
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Workaround for torch.compile bug in quantization pattern matcher
import torch._inductor.config
torch._inductor.config.post_grad_fusion_options = {}

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# AlphaGenome imports
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.extensions.finetuning import (
    # Model
    TransferConfig,
    # Training
    create_lr_scheduler,
    # Distributed
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    print_rank0,
    barrier,
    broadcast_object,
    # Logging
    TrainingLogger,
    setup_output_logging,
    # Checkpointing
    find_latest_checkpoint,
    setup_preemption_handler,
)
from alphagenome_pytorch.extensions.finetuning.adapters import get_adapter_params
from alphagenome_pytorch.extensions.finetuning.checkpointing import save_checkpoint, load_checkpoint
from alphagenome_pytorch.extensions.finetuning.heads import (
    create_splice_classification_finetuning_head,
    create_splice_usage_finetuning_head,
)
from alphagenome_pytorch.extensions.finetuning.transfer import (
    load_trunk,
    remove_all_heads,
    prepare_for_transfer,
)
from alphagenome_pytorch.extensions.finetuning.splice_training import (
    SpliceTrainMetrics,
    train_epoch_splice,
    validate_splice,
)
from alphagenome_pytorch.utils.paths import expand_path, expand_paths_in_dict


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULTS = {
    # Data
    "sequence_length": 131072,
    "organism_index": 0,
    "max_sites": 1024,
    "usage_coord_base": 0,
    "observed_conditions_only": False,
    "usage_delta_from_mean": False,
    # Model
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_targets": "q_proj,v_proj",
    "train_species_embeddings": True,
    "train_components": "",
    # Training
    "epochs": 10,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "lr": 1e-4,
    "weight_decay": 0.1,
    "warmup_steps": 500,
    "lr_schedule": "cosine",
    "cls_weight": 1.0,
    "usage_weight": 1.0,
    "max_grad_norm": 1.0,
    "num_workers": 4,
    # Logging
    "wandb_project": "alphagenome-splice",
    "log_every": 500,
    "save_every": 1,
    # Output
    "output_dir": "finetuning_output",
}

# Model components (top-level AlphaGenome attributes) that --train-components
# may unfreeze in addition to whatever the selected --mode already trains.
TRAINABLE_COMPONENTS = {
    "encoder",         # SequenceEncoder: DNA embedder + down-sampling conv blocks
    "tower",           # TransformerTower: attention/pair-update/MLP blocks
    "decoder",         # SequenceDecoder: up-sampling conv blocks
    "embedder_128bp",  # OutputEmbedder for the 128bp resolution
    "embedder_1bp",    # OutputEmbedder for the 1bp resolution
    "embedder_pair",   # OutputPair embedder feeding the contact-map head
    "organism_embed",  # Top-level organism/species embedding table
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AlphaGenome splice-site fine-tuning script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file (CLI flags override config values)",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["linear-probe", "lora", "full"],
        default="lora",
        help=(
            "Training mode: "
            "'linear-probe' (frozen backbone, train splice head(s) only), "
            "'lora' (LoRA adapters + heads), "
            "'full' (all parameters)"
        ),
    )

    # Data arguments
    data = parser.add_argument_group("Data")
    data.add_argument("--genome", type=str, required=False, help="Reference genome FASTA")
    data.add_argument(
        "--annotation-parquet",
        type=str,
        required=False,
        help="Splice-site annotation Parquet produced by convert_splice_sites_to_parquet.py",
    )
    data.add_argument(
        "--usage-parquet",
        type=str,
        default=None,
        help=(
            "Path to the splice-site usage Parquet file (_usage.parquet) produced by "
            "convert_splice_usage_to_parquet.py. The sibling JSON metadata file "
            "(_usage.json) must exist at the same path. "
            "When provided, enables the splice-usage head in addition to classification."
        ),
    )
    data.add_argument("--train-bed", type=str, required=False, help="Training regions BED file")
    data.add_argument("--val-bed", type=str, required=False, help="Validation regions BED file")
    data.add_argument("--sequence-length", type=int, default=DEFAULTS["sequence_length"])
    data.add_argument(
        "--organism-index",
        type=int,
        default=DEFAULTS["organism_index"],
        help="Organism index for organism-specific embeddings (0=human, 1=mouse)",
    )
    data.add_argument(
        "--max-sites",
        type=int,
        default=DEFAULTS["max_sites"],
        help="Maximum splice sites per sequence window for usage targets",
    )
    data.add_argument(
        "--cache-genome",
        action="store_true",
        help="Cache genome in memory (~12 GB for hg38)",
    )
    data.add_argument(
        "--usage-coord-base",
        type=int,
        default=0,
        choices=[0, 1],
        help="Coordinate base in usage parquet: 0 for 0-based (default), 1 if using 1-based Spliser format",
    )
    data.add_argument(
        "--observed-conditions-only",
        action="store_true",
        help="Only compute usage loss for observed (position, condition) pairs. "
             "When False (default), unobserved conditions are treated as 0 and included in loss.",
    )
    data.add_argument(
        "--usage-delta-from-mean",
        action="store_true",
        help="Compute usage loss on the deviation of each condition from the per-site mean "
             "across observed conditions (MSE on deltas). When False (default), compute MSE "
             "on absolute usage values.",
    )

    # Model arguments
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument(
        "--pretrained-weights",
        type=str,
        required=False,
        help="Pretrained AlphaGenome weights (.pth)",
    )
    model_grp.add_argument("--lora-rank", type=int, default=DEFAULTS["lora_rank"], help="LoRA rank (0 to disable)")
    model_grp.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"], help="LoRA alpha scaling")
    model_grp.add_argument(
        "--lora-targets",
        type=str,
        default=DEFAULTS["lora_targets"],
        help="Comma-separated module name substrings for LoRA",
    )
    model_grp.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Model dtype policy",
    )
    model_grp.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to trade compute for memory",
    )
    model_grp.add_argument(
        "--train-species-embeddings",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["train_species_embeddings"],
        help="Whether to train organism/species embedding tables",
    )
    model_grp.add_argument(
        "--train-components",
        type=str,
        default=DEFAULTS["train_components"],
        help=(
            "Comma-separated list of additional model components to fully "
            "fine-tune (unfreeze all their parameters), on top of whatever "
            "the selected --mode already trains (heads, LoRA-adapted "
            "attention, species embeddings). Only applies in 'linear-probe' "
            "and 'lora' modes ('full' already trains everything). Valid "
            f"names: {', '.join(sorted(TRAINABLE_COMPONENTS))}"
        ),
    )

    # Training arguments
    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    train_grp.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    train_grp.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULTS["gradient_accumulation_steps"],
        help="Accumulate gradients over N batches before an optimizer step",
    )
    train_grp.add_argument("--lr", type=float, default=DEFAULTS["lr"], help="Learning rate")
    train_grp.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    train_grp.add_argument("--warmup-steps", type=int, default=DEFAULTS["warmup_steps"])
    train_grp.add_argument(
        "--lr-schedule",
        type=str,
        default=DEFAULTS["lr_schedule"],
        choices=["cosine", "constant"],
    )
    train_grp.add_argument(
        "--cls-weight",
        type=float,
        default=DEFAULTS["cls_weight"],
        help="Weight for the splice classification loss term",
    )
    train_grp.add_argument(
        "--usage-weight",
        type=float,
        default=DEFAULTS["usage_weight"],
        help="Weight for the splice usage loss term (only used when --usage-parquet is provided)",
    )
    train_grp.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    train_grp.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULTS["max_grad_norm"],
        help="Max gradient norm for clipping (0 to disable)",
    )
    train_grp.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    train_grp.add_argument("--compile", action="store_true", help="Use torch.compile on the model")
    train_grp.add_argument("--seed", type=int, default=None, help="Random seed")

    # Logging arguments
    log_grp = parser.add_argument_group("Logging")
    log_grp.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    log_grp.add_argument("--wandb-project", type=str, default=DEFAULTS["wandb_project"])
    log_grp.add_argument("--wandb-entity", type=str, default=None)
    log_grp.add_argument("--log-every", type=int, default=DEFAULTS["log_every"], help="Log every N steps")
    log_grp.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to a file where stdout will be tee'd (default: <output-dir>/train.log)",
    )

    # Output arguments
    out_grp = parser.add_argument_group("Output")
    out_grp.add_argument("--output-dir", type=str, default=DEFAULTS["output_dir"])
    out_grp.add_argument("--run-name", type=str, default=None)
    out_grp.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])

    # Resume
    resume_grp = parser.add_argument_group("Resume / Checkpointing")
    resume_grp.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path or 'auto' to find the latest checkpoint in --output-dir",
    )

    args = parser.parse_args()
    cli_flags = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }

    def _load_yaml_config(path: str) -> dict[str, Any]:
        try:
            import yaml
        except ImportError:
            parser.error("YAML config support requires PyYAML (`pip install pyyaml`).")
        config_path = Path(path)
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")
        with config_path.open() as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            parser.error("YAML config root must be a mapping/dictionary")
        
        # Expand paths in config
        path_keys = {
            "genome", "annotation_parquet", "usage_parquet",
            "train_bed", "val_bed", "test_bed", "pretrained_weights",
            "output_dir", "log_file"
        }
        data = expand_paths_in_dict(data, path_keys)
        
        return data

    def _apply_config_scalar(attr: str, config: dict[str, Any], key: str | None = None) -> None:
        flag = f"--{attr.replace('_', '-')}"
        if flag in cli_flags:
            return
        config_key = key or attr
        if config_key in config and config[config_key] is not None:
            setattr(args, attr, config[config_key])

    config_data = _load_yaml_config(args.config) if args.config else {}

    for attr in (
        "mode",
        "genome",
        "annotation_parquet",
        "usage_parquet",
        "train_bed",
        "val_bed",
        "sequence_length",
        "organism_index",
        "max_sites",
        "cache_genome",
        "usage_coord_base",
        "observed_conditions_only",
        "usage_delta_from_mean",
        "pretrained_weights",
        "lora_rank",
        "lora_alpha",
        "lora_targets",
        "train_species_embeddings",
        "train_components",
        "dtype",
        "gradient_checkpointing",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "lr",
        "weight_decay",
        "warmup_steps",
        "lr_schedule",
        "cls_weight",
        "usage_weight",
        "max_grad_norm",
        "num_workers",
        "compile",
        "seed",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "log_every",
        "output_dir",
        "run_name",
        "save_every",
        "resume",
    ):
        _apply_config_scalar(attr, config_data)

    if "--no-amp" not in cli_flags:
        if "use_amp" in config_data:
            args.no_amp = not bool(config_data["use_amp"])
        elif "no_amp" in config_data:
            args.no_amp = bool(config_data["no_amp"])

    # Parse multi-species block from YAML config if present.
    # Format:
    #   species:
    #     - name: human
    #       genome: hg38.fa
    #       annotation_parquet: human.parquet
    #       usage_parquet: human_usage.parquet   # optional
    #       train_bed: human_train.bed
    #       val_bed: human_val.bed
    #       organism_index: 0
    #     - name: mouse
    #       ...
    species_specs: list[dict] | None = None
    if "species" in config_data:
        raw_species = config_data["species"]
        if not isinstance(raw_species, list) or len(raw_species) < 1:
            parser.error("YAML 'species' must be a non-empty list of species configurations")
        species_specs = []
        for i, s in enumerate(raw_species):
            if not isinstance(s, dict):
                parser.error(f"Species entry #{i} must be a mapping with genome/annotation/bed files")
            for field in ("genome", "annotation_parquet", "train_bed", "val_bed"):
                if not s.get(field):
                    parser.error(f"Species entry #{i} is missing required field '{field}'")
            # Paths are already expanded by _load_yaml_config
            species_specs.append({
                "name": s.get("name", f"species_{i}"),
                "genome": s["genome"],
                "annotation_parquet": s["annotation_parquet"],
                "usage_parquet": s.get("usage_parquet"),
                "train_bed": s["train_bed"],
                "val_bed": s["val_bed"],
                "organism_index": int(s.get("organism_index", i)),
            })

    if species_specs is not None:
        # Multi-species mode: per-species data comes from the YAML species block.
        # Only --pretrained-weights is required from the flat CLI args.
        if not args.pretrained_weights:
            parser.error("--pretrained-weights is required (or provide it in --config)")
    else:
        # Single-species mode: validate required flat args
        for flag, value in (
            ("--genome", args.genome),
            ("--annotation-parquet", args.annotation_parquet),
            ("--train-bed", args.train_bed),
            ("--val-bed", args.val_bed),
            ("--pretrained-weights", args.pretrained_weights),
        ):
            if not value:
                parser.error(f"{flag} is required (or provide it in --config)")
        # Build a unified single-species spec for downstream code
        # Expand paths from CLI args
        species_specs = [{
            "name": "species_0",
            "genome": expand_path(args.genome),
            "annotation_parquet": expand_path(args.annotation_parquet),
            "usage_parquet": expand_path(args.usage_parquet),
            "train_bed": expand_path(args.train_bed),
            "val_bed": expand_path(args.val_bed),
            "organism_index": args.organism_index,
        }]
        # Also expand pretrained_weights from CLI
        if args.pretrained_weights:
            args.pretrained_weights = expand_path(args.pretrained_weights)

    # Parse usage_loss_weights from config if present
    usage_loss_weights = None
    if "usage_loss_weights" in config_data and config_data["usage_loss_weights"] is not None:
        usage_loss_weights = config_data["usage_loss_weights"]
    args.usage_loss_weights = usage_loss_weights

    # Epochs over which to linearly ramp the trajectory/shape/smoothness loss weights
    # (0 = apply at full strength from epoch 0).
    args.usage_traj_warmup_epochs = int(config_data.get("usage_traj_warmup_epochs", 0) or 0)

    # Fraction of GPU memory this process may use (1.0 = no cap; <1.0 applies
    # torch.cuda.set_per_process_memory_fraction). Useful to leave headroom / share a GPU.
    args.cuda_memory_fraction = float(config_data.get("cuda_memory_fraction", 1.0) or 1.0)

    args.species_specs = species_specs
    return args


# =============================================================================
# Data Loading
# =============================================================================

def create_datasets(args: argparse.Namespace, rank: int):
    """Create training and validation SpliceSiteDatasets.

    Supports both single-species and multi-species modes.  In the
    multi-species case each species produces its own pair of datasets which
    are then concatenated with :class:`torch.utils.data.ConcatDataset`.

    Returns:
        ``(train_dataset, val_dataset, species_n_conditions)`` where
        *species_n_conditions* is a ``dict[organism_index, n_conditions]``
        containing only species that have a usage parquet.  This dict may be
        empty when no species provides usage data.  Different species are
        allowed to have different numbers of conditions.
    """
    from alphagenome_pytorch.extensions.finetuning.datasets import CachedGenome
    from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
        SpliceSiteAnnotation,
        SpliceSiteUsageIndex,
        SpliceSiteDataset,
    )
    from torch.utils.data import ConcatDataset

    train_datasets: list = []
    val_datasets: list = []
    species_n_conditions: dict[int, int] = {}  # organism_index → n_conditions
    tissue_cond_groups_by_org: dict[int, list] = {}  # organism_index → per-tissue condition groups

    for spec in args.species_specs:
        name = spec["name"]
        print_rank0(f"[{name}] Loading annotation from: {spec['annotation_parquet']}", rank)
        annotation = SpliceSiteAnnotation(spec["annotation_parquet"])

        usage_index = None
        if spec.get("usage_parquet"):
            print_rank0(f"[{name}] Loading usage index from: {spec['usage_parquet']}", rank)
            usage_index = SpliceSiteUsageIndex(
                spec["usage_parquet"],
                usage_coord_base=args.usage_coord_base,
                observed_conditions_only=args.observed_conditions_only,
            )
            print_rank0(f"  [{name}] Using usage_coord_base={args.usage_coord_base} (1=Spliser/1-based, 0=already 0-based)", rank)
            print_rank0(f"  [{name}] observed_conditions_only={args.observed_conditions_only} (True=only observed, False=unobserved as 0)", rank)
            cond = usage_index.n_conditions
            species_n_conditions[spec["organism_index"]] = cond
            tissue_cond_groups_by_org[spec["organism_index"]] = usage_index.tissue_cond_groups
            print_rank0(f"  [{name}] Usage conditions: {cond} "
                        f"({len(usage_index.tissue_cond_groups)} tissue groups for trajectory loss)", rank)

        genome = CachedGenome(spec["genome"]) if args.cache_genome else spec["genome"]

        print_rank0(f"[{name}] Creating train dataset...", rank)
        train_ds = SpliceSiteDataset(
            genome=genome,
            bed_file=spec["train_bed"],
            annotation=annotation,
            usage_index=usage_index,
            sequence_length=args.sequence_length,
            organism_index=spec["organism_index"],
            max_sites=args.max_sites,
        )

        print_rank0(f"[{name}] Creating validation dataset...", rank)
        val_ds = SpliceSiteDataset(
            genome=genome,
            bed_file=spec["val_bed"],
            annotation=annotation,
            usage_index=usage_index,
            sequence_length=args.sequence_length,
            organism_index=spec["organism_index"],
            max_sites=args.max_sites,
        )

        print_rank0(f"  [{name}] Train: {len(train_ds):,}  Val: {len(val_ds):,}", rank)
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
    else:
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        print_rank0(
            f"Combined — Train: {len(train_dataset):,}  Val: {len(val_dataset):,}", rank
        )

    return train_dataset, val_dataset, species_n_conditions, tissue_cond_groups_by_org


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, Any, Any]:
    """Create data loaders backed by species-grouped batch samplers.

    Each batch is guaranteed to contain sequences from a single species,
    which allows per-species usage heads with different ``n_conditions``.
    """
    from alphagenome_pytorch.extensions.finetuning.splice_datasets import SpeciesGroupedSampler
    train_sampler = SpeciesGroupedSampler(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    val_sampler = SpeciesGroupedSampler(val_dataset, batch_size=batch_size, shuffle=False, seed=seed)

    from alphagenome_pytorch.extensions.finetuning.splice_datasets import collate_splice

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_splice,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_splice,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, train_sampler, val_sampler


# =============================================================================
# Model Setup
# =============================================================================

def create_model(
    args: argparse.Namespace,
    species_n_conditions: dict[int, int],
    device: torch.device,
) -> tuple[nn.Module, dict[int, nn.Module], list[torch.nn.Parameter]]:
    """Create and configure the model for splice fine-tuning.

    Args:
        args: Parsed command-line arguments.
        species_n_conditions: Mapping of organism_index → n_conditions for
            species that have a usage Parquet.  Empty dict → no usage head.
            Each species may have a different number of conditions, enabling
            independent per-species usage heads.
        device: Torch device.
        rank: Process rank.
        world_size: Number of DDP processes.
        local_rank: Local rank for GPU assignment.

    Returns:
        Tuple of ``(model, usage_heads, trainable_params)`` where
        *usage_heads* is a ``dict[organism_index, nn.Module]``
        (empty when no species provides usage data).
    """
    print(f"Loading pretrained model from {args.pretrained_weights}")
    dtype_policy = (
        DtypePolicy.full_float32() if args.dtype == "float32" else DtypePolicy.mixed_precision()
    )
    print(f"Dtype policy: {dtype_policy}")

    # --- Calculate num_organisms and initialization mapping BEFORE creating model ---
    num_organisms = max(s["organism_index"] for s in args.species_specs) + 1
    # Parse mapping from config/args (dict: new_org_idx -> pretrained_org_idx)
    organism_embedding_init = getattr(args, "organism_embedding_init", None)

    # Default: identity mapping (organism i → pretrained organism i).
    # Override via organism_embedding_init e.g. {0: 1} to init organism 0
    # from pretrained organism 1 (useful for cross-species transfer).
    if organism_embedding_init is None:
        organism_embedding_init = {i: i for i in range(num_organisms)}

    # Validate that all keys in organism_embedding_init are valid organism indices
    invalid_keys = [k for k in organism_embedding_init.keys() if k >= num_organisms or k < 0]
    if invalid_keys:
        raise ValueError(
            f"Invalid organism indices in organism_embedding_init: {invalid_keys}. "
            f"Model has {num_organisms} organism(s) with valid indices 0-{num_organisms-1}. "
            f"Check your config's organism_embedding_init and species organism_index values."
        )

    print(f"Creating model with {num_organisms} organism(s)")
    print(f"Organism initialization mapping: {organism_embedding_init}")

    model = AlphaGenome(
        num_organisms=num_organisms,
        gradient_checkpointing=args.gradient_checkpointing,
        dtype_policy=dtype_policy,
    )

    # Load trunk weights (exclude heads so we replace them below)
    # Note: This will have missing keys for organism embeddings beyond pretrained num_organisms
    model = load_trunk(model, args.pretrained_weights, exclude_heads=True)

    # Initialize organism embeddings for all mapped organisms from pretrained organisms.
    # This also covers newly added organisms (e.g., rat/rabbit from mouse).
    import torch
    weights_path = args.pretrained_weights
    if weights_path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file as _safetensors_load
        except ImportError:
            raise ImportError(
                "safetensors is required to load .safetensors checkpoints. "
                "Install it with: pip install safetensors"
            )
        state_dict = _safetensors_load(weights_path, device='cpu')
    else:
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
    # Get pretrained organism embeddings count
    pretrained_organism_embed = state_dict.get('organism_embed.weight')
    if pretrained_organism_embed is not None:
        pretrained_num_organisms = pretrained_organism_embed.shape[0]
        print(f"Pretrained model has {pretrained_num_organisms} organism(s)")
        
        # Initialize organism embeddings from specified pretrained organisms
        organism_embed_modules = [
            ('organism_embed', model.organism_embed),
            ('embedder_128bp.organism_embed', model.embedder_128bp.organism_embed),
            ('embedder_1bp.organism_embed', model.embedder_1bp.organism_embed),
            ('embedder_pair.organism_embed', model.embedder_pair.organism_embed),
        ]
        
        for new_org_idx, pretrained_org_idx in organism_embedding_init.items():
            if pretrained_org_idx >= pretrained_num_organisms:
                print(f"[Warning] Cannot initialize organism {new_org_idx} from pretrained organism {pretrained_org_idx} "
                      f"(pretrained model only has {pretrained_num_organisms} organisms)")
                continue

            print(f"Initializing organism {new_org_idx} embeddings from pretrained organism {pretrained_org_idx}")
            for name, module in organism_embed_modules:
                pretrained_key = f'{name}.weight'
                if pretrained_key not in state_dict:
                    print(f"[Warning] Missing '{pretrained_key}' in checkpoint; skipping {name} copy")
                    continue
                if new_org_idx >= module.weight.data.shape[0]:
                    print(f"[Warning] Target organism index {new_org_idx} out of range for {name} "
                          f"(size={module.weight.data.shape[0]}); skipping")
                    continue
                module.weight.data[new_org_idx] = state_dict[pretrained_key][pretrained_org_idx].clone()

    # Freeze backbone first for non-full modes
    if args.mode != "full":
        for param in model.parameters():
            param.requires_grad = False

    # Keep species embedding tables trainable by default (configurable).
    train_species_embeddings = bool(getattr(args, "train_species_embeddings", True))
    for emb in (
        model.organism_embed,
        model.embedder_128bp.organism_embed,
        model.embedder_1bp.organism_embed,
        model.embedder_pair.organism_embed,
    ):
        emb.weight.requires_grad = train_species_embeddings

    species_embedding_params = [
        model.organism_embed.weight,
        model.embedder_128bp.organism_embed.weight,
        model.embedder_1bp.organism_embed.weight,
        model.embedder_pair.organism_embed.weight,
    ]


    # Remove all existing heads (including splice_sites_classification_head)
    model = remove_all_heads(model)

    # Load pretrained classification head weights (state_dict already loaded above)
    head_prefix = "splice_sites_classification_head.conv."
    pretrained_head_weights = {k[len(head_prefix):]: v for k, v in state_dict.items() if k.startswith(head_prefix)}

    # Create a fresh classification head
    from alphagenome_pytorch.extensions.finetuning.heads import create_splice_classification_finetuning_head
    cls_head = create_splice_classification_finetuning_head(num_organisms=num_organisms)

    # Copy pretrained classification head weights for each organism according to the mapping
    if pretrained_head_weights:
        for new_org_idx, pretrained_org_idx in organism_embedding_init.items():
            try:
                cls_head.conv.weight.data[new_org_idx] = pretrained_head_weights["weight"][pretrained_org_idx].clone()
                cls_head.conv.bias.data[new_org_idx] = pretrained_head_weights["bias"][pretrained_org_idx].clone()
            except Exception as e:
                print(f"[Warning] Could not copy head weights for organism {new_org_idx} from pretrained organism {pretrained_org_idx}: {e}")
        print(f"Initialized classification head from pretrained weights (mapping: {organism_embedding_init})")
    else:
        print(f"Created splice classification head (5-class, 1bp, {num_organisms} organism(s)), random init (no pretrained head found)")

    model.splice_sites_classification_head = cls_head

    # Create per-species usage heads (one per organism that has usage data)
    # Store as ModuleDict under model.splice_sites_usage_head
    usage_heads: dict[int, nn.Module] = {}
    for org_idx, n_cond in species_n_conditions.items():
        if n_cond > 0:
            usage_heads[org_idx] = create_splice_usage_finetuning_head(
                n_conditions=n_cond,
                num_organisms=num_organisms,
            )
            print(f"Created splice usage head for organism {org_idx} "
                  f"({n_cond} conditions, 1bp, {num_organisms} organism(s))")
    
    # Store usage heads in model.splice_sites_usage_head (as ModuleDict for multi-organism)
    if usage_heads:
        model.splice_sites_usage_head = nn.ModuleDict({str(k): v for k, v in usage_heads.items()})
    else:
        model.splice_sites_usage_head = None

    # Configure trainable parameters based on training mode
    trainable_params: list[torch.nn.Parameter] = []

    if args.mode == "linear-probe":
        # Only splice heads are trainable (backbone frozen above)
        trainable_params.extend(list(cls_head.parameters()))
        for h in usage_heads.values():
            trainable_params.extend(list(h.parameters()))
        if train_species_embeddings:
            for p in species_embedding_params:
                if p.requires_grad and all(p is not q for q in trainable_params):
                    trainable_params.append(p)
        print("Mode: linear-probe (frozen backbone, heads only)")

    elif args.mode == "lora":
        if args.lora_rank > 0:
            lora_targets = [t.strip() for t in args.lora_targets.split(",")]
            print(f"Applying LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
            print(f"  Target modules: {lora_targets}")
            config = TransferConfig(
                mode="lora",
                lora_targets=lora_targets,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
            )
            model = prepare_for_transfer(model, config)
            trainable_params = get_adapter_params(model)
        # Always add head parameters (created after freeze → requires_grad=True)
        trainable_params.extend(list(cls_head.parameters()))
        for h in usage_heads.values():
            trainable_params.extend(list(h.parameters()))
        if train_species_embeddings:
            for p in species_embedding_params:
                if p.requires_grad and all(p is not q for q in trainable_params):
                    trainable_params.append(p)
        mode_desc = f"lora (rank={args.lora_rank})" if args.lora_rank > 0 else "lora (rank=0, heads only)"
        print(f"Mode: {mode_desc}")

    elif args.mode == "full":
        # All model parameters
        trainable_params = list(model.parameters())
        for h in usage_heads.values():
            trainable_params.extend(list(h.parameters()))
        print("Mode: full (all parameters trainable)")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Optionally unfreeze additional components on top of the selected mode
    # (heads / LoRA-adapted attention / species embeddings). No-op for
    # 'full' mode, where every parameter is already trainable.
    train_components_str = getattr(args, "train_components", "") or ""
    extra_train_components = [c.strip() for c in train_components_str.split(",") if c.strip()]
    if extra_train_components:
        invalid = [c for c in extra_train_components if c not in TRAINABLE_COMPONENTS]
        if invalid:
            raise ValueError(
                f"Invalid --train-components entries: {invalid}. "
                f"Valid names: {sorted(TRAINABLE_COMPONENTS)}"
            )
        if args.mode != "full":
            component_map = {
                "encoder": model.encoder,
                "tower": model.tower,
                "decoder": model.decoder,
                "embedder_128bp": model.embedder_128bp,
                "embedder_1bp": model.embedder_1bp,
                "embedder_pair": model.embedder_pair,
                "organism_embed": model.organism_embed,
            }
            seen_params = {id(p) for p in trainable_params}
            for name in extra_train_components:
                module = component_map[name]
                n_added = 0
                for p in module.parameters():
                    p.requires_grad = True
                    if id(p) not in seen_params:
                        trainable_params.append(p)
                        seen_params.add(id(p))
                        n_added += 1
                print(f"Also fine-tuning component '{name}' ({n_added:,} additional params)")
        else:
            print(f"--train-components ignored in 'full' mode (all parameters already trainable)")

    model = model.to(device)
    model_module = model
    # Extract usage_heads dict from model for training loop compatibility
    if model_module.splice_sites_usage_head is not None:
        usage_heads = {int(k): v for k, v in model_module.splice_sites_usage_head.items()}
    else:
        usage_heads = {}

    # Optionally compile
    if args.compile:
        # Use rank=0 for single-process (no DDP)
        print_rank0("Compiling model with torch.compile...", 0)
        import torch._inductor.config as inductor_config
        inductor_config.group_fusion = False
        model = torch.compile(model)

    # Parameter counts (model_module.parameters() now includes usage heads)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model_module.parameters())
    print_rank0(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)", 0)

    return model, usage_heads, trainable_params


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Main training function."""

    args = parse_args()

    # Support loading organism embedding initialization mapping from YAML config.
    # If present in config, inject as attribute for create_model
    if hasattr(args, 'config') and args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            config_yaml = yaml.safe_load(f)
        if 'organism_embedding_init' in config_yaml:
            # Ensure keys are int (YAML may parse as str)
            mapping = {int(k): int(v) for k, v in config_yaml['organism_embedding_init'].items()}
            setattr(args, 'organism_embedding_init', mapping)

    # Single-process/single-GPU only
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Optionally cap this process to a fraction of the GPU's memory (1.0 = no cap).
    if device.type == 'cuda' and 0.0 < args.cuda_memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device.index or 0)
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"CUDA memory fraction: {args.cuda_memory_fraction:.2f} "
              f"(~{args.cuda_memory_fraction * total_gb:.1f} / {total_gb:.1f} GB)")
    import random
    import numpy as np
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    print(f"Device: {device}")

    # Output directory
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / str(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    setup_output_logging(output_dir, 0, log_file=args.log_file)

    # Resolve resume checkpoint
    resume_path = None
    if args.resume == "auto":
        resume_path = find_latest_checkpoint(output_dir)
        if resume_path:
            print(f"Auto-resume: found {resume_path}")
        else:
            print("Auto-resume: no checkpoint found, starting fresh")
    elif args.resume:
        resume_path = Path(args.resume)

    # Create datasets
    train_dataset, val_dataset, species_n_conditions, tissue_cond_groups_by_org = create_datasets(args, 0)

    # Set rank and world_size for single-process (no DDP)
    rank = 0
    world_size = 1

    # Create dataloaders
    train_loader, val_loader, train_sampler, _ = create_dataloaders(
        train_dataset,
        val_dataset,
        args.batch_size,
        args.num_workers,
        seed=args.seed or 0,
    )
    print(f"Train batches: {len(train_loader):,}, Val batches: {len(val_loader):,}")

    class_weights: torch.Tensor | None = None

    # Create model
    model, usage_heads, trainable_params = create_model(
        args, species_n_conditions, device
    )
    # When torch.compile wraps the model, _orig_mod holds the original PyTorch module.
    # Always checkpoint from the uncompiled module so state-dict keys have no _orig_mod. prefix.
    model_module = getattr(model, '_orig_mod', model)
    usage_modules: dict[int, nn.Module] = dict(usage_heads)

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    total_steps = (args.epochs * len(train_loader)) // args.gradient_accumulation_steps
    scheduler = create_lr_scheduler(optimizer, args.warmup_steps, total_steps, schedule=args.lr_schedule)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total optimizer steps: {total_steps:,}")
    print(f"LR schedule: {args.lr_schedule} (warmup: {args.warmup_steps} steps)")

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    wandb_run_id = None

    if resume_path and resume_path.exists():
        print_rank0(f"Resuming from: {resume_path}", rank)
        ckpt = load_checkpoint(
            resume_path,
            model=model_module,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu",
        )
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        wandb_run_id = ckpt.get("wandb_run_id")
        # Restore per-species usage head weights if present
        if usage_modules and "usage_heads_state_dicts" in ckpt:
            for k_str, sd in ckpt["usage_heads_state_dicts"].items():
                org_idx = int(k_str)
                if org_idx in usage_modules:
                    usage_modules[org_idx].load_state_dict(sd)
            print_rank0("  Restored usage head weights from checkpoint", rank)
        elif usage_modules and "usage_head_state_dict" in ckpt:
            # Backward compat: single-species checkpoint
            if len(usage_modules) == 1:
                org_idx = next(iter(usage_modules))
                usage_modules[org_idx].load_state_dict(ckpt["usage_head_state_dict"])
                print_rank0("  Restored usage head weights from checkpoint (legacy key)", rank)
        print_rank0(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}", rank)

    # Config for logging
    config = {
        "mode": args.mode,
        "species_specs": args.species_specs,
        "sequence_length": args.sequence_length,
        "max_sites": args.max_sites,
        "usage_coord_base": args.usage_coord_base,
        "observed_conditions_only": args.observed_conditions_only,
        "usage_delta_from_mean": args.usage_delta_from_mean,
        "usage_loss_weights": getattr(args, "usage_loss_weights", None),
        "species_n_conditions": species_n_conditions,
        "cls_weight": args.cls_weight,
        "usage_weight": args.usage_weight,
        "pretrained_weights": args.pretrained_weights,
        "lora_rank": args.lora_rank if args.mode == "lora" else None,
        "lora_alpha": args.lora_alpha if args.mode == "lora" else None,
        "lora_targets": args.lora_targets if args.mode == "lora" else None,
        "train_species_embeddings": args.train_species_embeddings,
        "train_components": args.train_components,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lr_schedule": args.lr_schedule,
        "total_steps": total_steps,
        "n_trainable_params": sum(p.numel() for p in trainable_params),
        "use_amp": not args.no_amp,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dtype": args.dtype,
        "world_size": world_size,
        "seed": args.seed,
        "log_every": args.log_every,
        "resumed_from": str(resume_path) if resume_path else None,
    }

    # Logger (rank 0 only)
    logger = TrainingLogger(
        output_dir=output_dir,
        rank=rank,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=run_name,
        config=config,
        resume_id=wandb_run_id if resume_path else None,
    )

    use_amp = not args.no_amp
    current_epoch = start_epoch
    species_name_by_org = {int(s["organism_index"]): s["name"] for s in args.species_specs}

    def _save_preempt():
        """Save a preemption checkpoint (called on SIGUSR1)."""
        if is_main_process(rank):
            ckpt_path = output_dir / "checkpoint_preempt.pth"
            extra: dict[str, Any] = {}
            if usage_modules:
                extra["usage_heads_state_dicts"] = {
                    str(k): v.state_dict() for k, v in usage_modules.items()
                }
            save_checkpoint(
                path=ckpt_path,
                epoch=max(0, current_epoch - 1),
                model=model_module,
                optimizer=optimizer,
                val_loss=best_val_loss,
                track_names=[],
                modality="splice",
                resolutions=(1,),
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                wandb_run_id=logger.wandb_run_id,
                **extra,
            )
            print(f"Preemption checkpoint saved to {ckpt_path}")

    handler = setup_preemption_handler(_save_preempt, 0, 1)

    # Validate usage head setup before training (check for coordinate mismatch)
    if usage_modules:
        print("Validating usage head configuration...")
        total_valid_pairs = 0
        n_batches_checked = 0
        max_check_batches = min(5, len(train_loader))
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_check_batches:
                break
            if "usage_mask" in batch:
                n_valid = batch["usage_mask"].sum().item()
                total_valid_pairs += n_valid
                n_batches_checked += 1
        
        if n_batches_checked > 0:
            if total_valid_pairs == 0:
                print("\n" + "!" * 60)
                print("⚠️  CRITICAL ERROR: NO VALID USAGE PAIRS FOUND!")
                print("!" * 60)
                print("\nThe usage head will NOT train (zero gradient).")
                print("Most likely cause: COORDINATE MISMATCH")
                print(f"\nCurrent setting: --usage-coord-base {args.usage_coord_base}")
                print("\nTroubleshooting:")
                print("  • Spliser data (default): use --usage-coord-base 1")
                print("  • Already 0-based:        use --usage-coord-base 0")
                print("  • Check chromosome naming in annotation vs usage parquet")
                print("  • Verify genome versions match")
                print("\nAborting training to prevent wasted computation.")
                print("Fix the coordinate base and restart.\n")
                sys.exit(1)
            else:
                print(f"Total valid (position, condition) pairs: {total_valid_pairs:,}")
                print(f"Usage head will receive gradients during training.\n")
        else:
            print("Warning: No usage data found in training batches.\n")

    # Training loop
    print("\n" + "=" * 60)
    print(f"Starting training (epoch {start_epoch} to {args.epochs})")
    print("=" * 60)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if handler.preempted:
                print_rank0("Preemption flag set — saving and exiting.", rank)
                handler.save_and_exit()
                break

            current_epoch = epoch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Training epoch
            epoch_start_time = time.monotonic()
            train_metrics: SpliceTrainMetrics = train_epoch_splice(
                model=model,
                usage_head=usage_heads or None,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                cls_weight=args.cls_weight,
                usage_weight=args.usage_weight,
                class_weights=class_weights,
                use_amp=use_amp,
                accumulation_steps=args.gradient_accumulation_steps,
                log_every=args.log_every,
                epoch=epoch,
                logger=logger,
                max_grad_norm=args.max_grad_norm,
                usage_delta_from_mean=args.usage_delta_from_mean,
                usage_loss_weights=args.usage_loss_weights,
                tissue_cond_groups=tissue_cond_groups_by_org,
                usage_traj_warmup_epochs=getattr(args, "usage_traj_warmup_epochs", 0),
            )

            if handler.preempted:
                print_rank0("Preemption flag set — saving and exiting.", rank)
                handler.save_and_exit()
                break

            # Validation epoch
            val_metrics: SpliceTrainMetrics = validate_splice(
                model=model,
                usage_head=usage_heads or None,
                val_loader=val_loader,
                device=device,
                cls_weight=args.cls_weight,
                usage_weight=args.usage_weight,
                class_weights=class_weights,
                use_amp=use_amp,
                usage_delta_from_mean=args.usage_delta_from_mean,
                usage_loss_weights=args.usage_loss_weights,
                tissue_cond_groups=tissue_cond_groups_by_org,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            train_loss = train_metrics.loss
            val_loss = val_metrics.loss
            current_lr = scheduler.get_last_lr()[0]
            is_best = val_loss < best_val_loss

            # Print epoch summary
            epoch_elapsed = time.monotonic() - epoch_start_time
            
            def format_time(seconds: float) -> str:
                """Format seconds as hours:minutes or minutes:seconds."""
                if seconds >= 3600:
                    hours = int(seconds // 3600)
                    mins = int((seconds % 3600) // 60)
                    return f"{hours}h{mins}m"
                else:
                    mins = int(seconds // 60)
                    secs = int(seconds % 60)
                    return f"{mins}m{secs}s"
            

            train_bce = getattr(train_metrics, "usage_bce_loss", None)
            train_delta = getattr(train_metrics, "usage_delta_loss", None)
            train_traj = getattr(train_metrics, "usage_trajectory_loss", None)
            train_tcorr = getattr(train_metrics, "usage_trajectory_corr", None)
            val_bce = getattr(val_metrics, "usage_bce_loss", None)
            val_delta = getattr(val_metrics, "usage_delta_loss", None)
            val_traj = getattr(val_metrics, "usage_trajectory_loss", None)
            val_tcorr = getattr(val_metrics, "usage_trajectory_corr", None)
            summary = (
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}  "
                f"train_cls_loss={train_metrics.cls_loss:.4f}  "
                f"train_usage_loss={train_metrics.usage_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_cls_loss={val_metrics.cls_loss:.4f}  "
                f"val_usage_loss={val_metrics.usage_loss:.4f}  "
                f"lr={current_lr:.2e}\n"
                f"  Timing: {format_time(epoch_elapsed)} ({format_time(train_metrics.elapsed_s)} train + {format_time(val_metrics.elapsed_s)} val)"
            )
            if any(v is not None for v in (train_bce, train_delta, train_traj, train_tcorr,
                                           val_bce, val_delta, val_traj, val_tcorr)):
                summary += "\n  [usage breakdown]"
                train_parts = []
                if train_bce is not None:
                    train_parts.append(f"train_bce_loss={train_bce:.4f}")
                if train_delta is not None:
                    train_parts.append(f"train_delta_loss={train_delta:.4f}")
                if train_traj is not None:
                    train_parts.append(f"train_trajectory_loss={train_traj:.4f}")
                if train_tcorr is not None:
                    train_parts.append(f"train_trajectory_corr={train_tcorr:.3f}")
                if train_parts:
                    summary += "\n    " + "  ".join(train_parts)

                val_parts = []
                if val_bce is not None:
                    val_parts.append(f"val_bce_loss={val_bce:.4f}")
                if val_delta is not None:
                    val_parts.append(f"val_delta_loss={val_delta:.4f}")
                if val_traj is not None:
                    val_parts.append(f"val_trajectory_loss={val_traj:.4f}")
                if val_tcorr is not None:
                    val_parts.append(f"val_trajectory_corr={val_tcorr:.3f}")
                if val_parts:
                    summary += "\n    " + "  ".join(val_parts)
            print(summary)

            # Log epoch metrics
            extra = {
                "train_cls_loss": train_metrics.cls_loss,
                "train_usage_loss": train_metrics.usage_loss,
                "val_cls_loss": val_metrics.cls_loss,
                "val_usage_loss": val_metrics.usage_loss,
            }
            if train_bce is not None:
                extra["train_bce_loss"] = train_bce
            if train_delta is not None:
                extra["train_delta_loss"] = train_delta
            if train_traj is not None:
                extra["train_trajectory_loss"] = train_traj
            if train_tcorr is not None:
                extra["train_trajectory_corr"] = train_tcorr
            if val_bce is not None:
                extra["val_bce_loss"] = val_bce
            if val_delta is not None:
                extra["val_delta_loss"] = val_delta
            if val_traj is not None:
                extra["val_trajectory_loss"] = val_traj
            if val_tcorr is not None:
                extra["val_trajectory_corr"] = val_tcorr
            if val_metrics.species_metrics:
                for org_idx, species_vals in sorted(val_metrics.species_metrics.items()):
                    species_name = species_name_by_org.get(org_idx, f"org_{org_idx}")
                    for metric_name, metric_val in species_vals.items():
                        extra[f"{metric_name}_{species_name}"] = metric_val
            logger.log_epoch(epoch, train_loss, val_loss, current_lr, is_best, extra)

            # Save checkpoints
            usage_extra: dict[str, Any] = {}
            if usage_modules:
                usage_extra["usage_heads_state_dicts"] = {
                    str(k): v.state_dict() for k, v in usage_modules.items()
                }

            if is_best:
                best_val_loss = val_loss
                save_checkpoint(
                    path=output_dir / "best_model.pth",
                    epoch=epoch,
                    model=model_module,
                    optimizer=optimizer,
                    val_loss=val_loss,
                    track_names=[],
                    modality="splice",
                    resolutions=(1,),
                    scheduler=scheduler,
                    best_val_loss=best_val_loss,
                    wandb_run_id=logger.wandb_run_id,
                    **usage_extra,
                )
                # Also save numbered checkpoint for this best epoch
                save_checkpoint(
                    path=output_dir / f"epoch_{epoch:02d}.pth",
                    epoch=epoch,
                    model=model_module,
                    optimizer=optimizer,
                    val_loss=val_loss,
                    track_names=[],
                    modality="splice",
                    resolutions=(1,),
                    scheduler=scheduler,
                    best_val_loss=best_val_loss,
                    wandb_run_id=logger.wandb_run_id,
                    **usage_extra,
                )
                print(f"  Saved best model (val_loss={val_loss:.4f})")
            
            # Aggressive cleanup at the end of each epoch to prevent RAM accumulation
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        logger.finish()
        handler.unregister()

    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
