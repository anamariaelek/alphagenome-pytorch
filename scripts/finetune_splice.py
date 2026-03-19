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
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULTS = {
    # Data
    "sequence_length": 131072,
    "organism_index": 0,
    "max_sites": 1024,
    # Model
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_targets": "q_proj,v_proj",
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
    "num_workers": 4,
    # Logging
    "wandb_project": "alphagenome-splice",
    "log_every": 500,
    "save_every": 1,
    # Output
    "output_dir": "finetuning_output",
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
    train_grp.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    train_grp.add_argument("--compile", action="store_true", help="Use torch.compile on the model")
    train_grp.add_argument("--seed", type=int, default=None, help="Random seed")

    # Logging arguments
    log_grp = parser.add_argument_group("Logging")
    log_grp.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    log_grp.add_argument("--wandb-project", type=str, default=DEFAULTS["wandb_project"])
    log_grp.add_argument("--wandb-entity", type=str, default=None)
    log_grp.add_argument("--log-every", type=int, default=DEFAULTS["log_every"], help="Log every N steps")

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
        "pretrained_weights",
        "lora_rank",
        "lora_alpha",
        "lora_targets",
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
        species_specs = [{
            "name": "species_0",
            "genome": args.genome,
            "annotation_parquet": args.annotation_parquet,
            "usage_parquet": args.usage_parquet,
            "train_bed": args.train_bed,
            "val_bed": args.val_bed,
            "organism_index": args.organism_index,
        }]

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

    for spec in args.species_specs:
        name = spec["name"]
        print_rank0(f"[{name}] Loading annotation from: {spec['annotation_parquet']}", rank)
        annotation = SpliceSiteAnnotation(spec["annotation_parquet"])

        usage_index = None
        if spec.get("usage_parquet"):
            print_rank0(f"[{name}] Loading usage index from: {spec['usage_parquet']}", rank)
            usage_index = SpliceSiteUsageIndex(spec["usage_parquet"])
            cond = usage_index.n_conditions
            species_n_conditions[spec["organism_index"]] = cond
            print_rank0(f"  [{name}] Usage conditions: {cond}", rank)

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

    return train_dataset, val_dataset, species_n_conditions


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, Any, Any]:
    """Create data loaders backed by species-grouped batch samplers.

    Each batch is guaranteed to contain sequences from a single species,
    which allows per-species usage heads with different ``n_conditions``.
    """
    from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
        DistributedSpeciesGroupedSampler,
        SpeciesGroupedSampler,
    )

    if world_size > 1:
        train_sampler = DistributedSpeciesGroupedSampler(
            train_dataset, batch_size=batch_size,
            num_replicas=world_size, rank=rank,
            shuffle=True, seed=seed,
        )
        val_sampler = DistributedSpeciesGroupedSampler(
            val_dataset, batch_size=batch_size,
            num_replicas=world_size, rank=rank,
            shuffle=False, seed=seed,
        )
    else:
        train_sampler = SpeciesGroupedSampler(
            train_dataset, batch_size=batch_size, shuffle=True, seed=seed,
        )
        val_sampler = SpeciesGroupedSampler(
            val_dataset, batch_size=batch_size, shuffle=False, seed=seed,
        )

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
    rank: int,
    world_size: int,
    local_rank: int,
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
    print_rank0(f"Loading pretrained model from {args.pretrained_weights}", rank)

    dtype_policy = (
        DtypePolicy.full_float32() if args.dtype == "float32" else DtypePolicy.mixed_precision()
    )
    print_rank0(f"Dtype policy: {dtype_policy}", rank)

    model = AlphaGenome(
        gradient_checkpointing=args.gradient_checkpointing,
        dtype_policy=dtype_policy,
    )

    # Load trunk weights (exclude heads so we replace them below)
    model = load_trunk(model, args.pretrained_weights, exclude_heads=True)

    # Freeze backbone first for non-full modes
    if args.mode != "full":
        for param in model.parameters():
            param.requires_grad = False

    # Remove all existing heads (including splice_sites_classification_head)
    model = remove_all_heads(model)

    # Create and attach a fresh splice classification head
    num_organisms = max(s["organism_index"] for s in args.species_specs) + 1
    cls_head = create_splice_classification_finetuning_head(num_organisms=num_organisms)
    model.splice_sites_classification_head = cls_head
    print_rank0(
        f"Created splice classification head (5-class, 1bp, {num_organisms} organism(s))", rank
    )

    # Create per-species usage heads (one per organism that has usage data)
    usage_heads: dict[int, nn.Module] = {}
    for org_idx, n_cond in species_n_conditions.items():
        if n_cond > 0:
            usage_heads[org_idx] = create_splice_usage_finetuning_head(
                n_conditions=n_cond,
                num_organisms=num_organisms,
            )
            print_rank0(
                f"Created splice usage head for organism {org_idx} "
                f"({n_cond} conditions, 1bp, {num_organisms} organism(s))",
                rank,
            )

    # Configure trainable parameters based on training mode
    trainable_params: list[torch.nn.Parameter] = []

    if args.mode == "linear-probe":
        # Only splice heads are trainable (backbone frozen above)
        trainable_params.extend(list(cls_head.parameters()))
        for h in usage_heads.values():
            trainable_params.extend(list(h.parameters()))
        print_rank0("Mode: linear-probe (frozen backbone, heads only)", rank)

    elif args.mode == "lora":
        if args.lora_rank > 0:
            lora_targets = [t.strip() for t in args.lora_targets.split(",")]
            print_rank0(f"Applying LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}", rank)
            print_rank0(f"  Target modules: {lora_targets}", rank)
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
        mode_desc = f"lora (rank={args.lora_rank})" if args.lora_rank > 0 else "lora (rank=0, heads only)"
        print_rank0(f"Mode: {mode_desc}", rank)

    elif args.mode == "full":
        # All model parameters
        trainable_params = list(model.parameters())
        for h in usage_heads.values():
            trainable_params.extend(list(h.parameters()))
        print_rank0("Mode: full (all parameters trainable)", rank)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Move to device
    model = model.to(device)
    for h in usage_heads.values():
        h.to(device)

    # Wrap with DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        usage_heads = {
            k: DDP(v, device_ids=[local_rank], output_device=local_rank)
            for k, v in usage_heads.items()
        }
        print_rank0("Model(s) wrapped with DistributedDataParallel", rank)

    # Optionally compile
    if args.compile:
        print_rank0("Compiling model with torch.compile...", rank)
        import torch._inductor.config as inductor_config
        inductor_config.group_fusion = False
        model = torch.compile(model)
        usage_heads = {k: torch.compile(v) for k, v in usage_heads.items()}

    # Parameter counts
    model_module = model.module if isinstance(model, DDP) else model
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model_module.parameters())
    for h in usage_heads.values():
        usage_m = h.module if isinstance(h, DDP) else h
        n_total += sum(p.numel() for p in usage_m.parameters())
    print_rank0(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)", rank)

    return model, usage_heads, trainable_params


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Main training function."""
    args = parse_args()

    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + rank)
        print_rank0(f"Random seed: {args.seed} (+ rank offset)", rank)

    if world_size > 1:
        print_rank0(f"Distributed training with {world_size} GPUs", rank)
    print_rank0(f"Device: {device}", rank)

    # Output directory
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_name
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")
    setup_output_logging(output_dir, rank)
    barrier()

    # Resolve resume checkpoint
    resume_path = None
    if args.resume == "auto":
        resume_path = find_latest_checkpoint(output_dir)
        if is_main_process(rank):
            if resume_path:
                print(f"Auto-resume: found {resume_path}")
            else:
                print("Auto-resume: no checkpoint found, starting fresh")
    elif args.resume:
        resume_path = Path(args.resume)

    # Create datasets
    train_dataset, val_dataset, species_n_conditions = create_datasets(args, rank)

    # Create dataloaders
    train_loader, val_loader, train_sampler, _ = create_dataloaders(
        train_dataset,
        val_dataset,
        args.batch_size,
        args.num_workers,
        world_size,
        rank,
        seed=args.seed or 0,
    )
    print_rank0(f"Train batches: {len(train_loader):,}, Val batches: {len(val_loader):,}", rank)

    # Compute class weights (rank 0 scans training BED(s), then broadcasts).
    # For multi-species, weights are computed per-species and averaged.
    if is_main_process(rank):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import compute_splice_class_weights
        print("Computing splice class weights from training windows...")
        cw_list = []
        for spec in args.species_specs:
            cw = compute_splice_class_weights(
                annotation_parquet=spec["annotation_parquet"],
                bed_file=spec["train_bed"],
                sequence_length=args.sequence_length,
            )
            print(f"  [{spec['name']}] Class weights: {[f'{w:.3f}' for w in cw.tolist()]}")
            cw_list.append(cw)
        if len(cw_list) == 1:
            final_cw = cw_list[0]
        else:
            final_cw = torch.stack(cw_list).mean(dim=0)
            print(f"  Averaged class weights: {[f'{w:.3f}' for w in final_cw.tolist()]}")
        class_weights_list = final_cw.tolist()
    else:
        class_weights_list = None
    class_weights_list = broadcast_object(class_weights_list, src=0)
    class_weights: torch.Tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(device)

    # Create model
    model, usage_heads, trainable_params = create_model(
        args, species_n_conditions, device, rank, world_size, local_rank
    )
    model_module = model.module if isinstance(model, DDP) else model
    # Unwrapped usage modules for state_dict access
    usage_modules: dict[int, nn.Module] = {
        k: (v.module if isinstance(v, DDP) else v) for k, v in usage_heads.items()
    }

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    total_steps = (args.epochs * len(train_loader)) // args.gradient_accumulation_steps
    scheduler = create_lr_scheduler(optimizer, args.warmup_steps, total_steps, schedule=args.lr_schedule)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    print_rank0(f"Gradient accumulation: {args.gradient_accumulation_steps}", rank)
    print_rank0(f"Effective batch size: {effective_batch_size}", rank)
    print_rank0(f"Total optimizer steps: {total_steps:,}", rank)
    print_rank0(f"LR schedule: {args.lr_schedule} (warmup: {args.warmup_steps} steps)", rank)

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
        "species_n_conditions": species_n_conditions,
        "cls_weight": args.cls_weight,
        "usage_weight": args.usage_weight,
        "pretrained_weights": args.pretrained_weights,
        "lora_rank": args.lora_rank if args.mode == "lora" else None,
        "lora_alpha": args.lora_alpha if args.mode == "lora" else None,
        "lora_targets": args.lora_targets if args.mode == "lora" else None,
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

    handler = setup_preemption_handler(_save_preempt, rank, world_size)

    # Training loop
    print_rank0("\n" + "=" * 60, rank)
    print_rank0(f"Starting training (epoch {start_epoch} to {args.epochs})", rank)
    print_rank0("=" * 60, rank)

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
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            train_loss = train_metrics.loss
            val_loss = val_metrics.loss
            current_lr = scheduler.get_last_lr()[0]
            is_best = val_loss < best_val_loss

            # Print epoch summary
            if is_main_process(rank):
                summary = (
                    f"Epoch {epoch}: "
                    f"train_loss={train_loss:.4f}  "
                    f"train_cls_loss={train_metrics.cls_loss:.4f}  "
                    f"train_usage_loss={train_metrics.usage_loss:.4f}  "
                    f"train_accuracy={train_metrics.accuracy:.4f}  "
                    f"val_loss={val_loss:.4f}  "
                    f"val_cls_loss={val_metrics.cls_loss:.4f}  "
                    f"val_usage_loss={val_metrics.usage_loss:.4f}  "
                    f"val_accuracy={val_metrics.accuracy:.4f}  "
                    f"lr={current_lr:.2e}"
                )
                if val_metrics.usage_loss > 0:
                    summary += f"  val_usage_loss={val_metrics.usage_loss:.4f}"
                print(summary)

            # Log epoch metrics
            extra = {
                "train_cls_loss": train_metrics.cls_loss,
                "train_usage_loss": train_metrics.usage_loss,
                "train_accuracy": train_metrics.accuracy,
                "val_cls_loss": val_metrics.cls_loss,
                "val_usage_loss": val_metrics.usage_loss,
                "val_accuracy": val_metrics.accuracy,
            }
            # Per-class validation accuracy
            for i, acc in enumerate(val_metrics.per_class_acc):
                extra[f"val_acc_class{i}"] = acc

            logger.log_epoch(epoch, train_loss, val_loss, current_lr, is_best, extra)

            # Save checkpoints
            if is_main_process(rank):
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
                    print(f"  Saved best model (val_loss={val_loss:.4f})")

                if epoch % args.save_every == 0:
                    save_checkpoint(
                        path=output_dir / f"checkpoint_epoch{epoch}.pth",
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

            barrier()

    except KeyboardInterrupt:
        print_rank0("\nTraining interrupted by user", rank)
    finally:
        logger.finish()
        handler.unregister()
        cleanup_distributed()

    print_rank0(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}", rank)
    print_rank0(f"Output: {output_dir}", rank)


if __name__ == "__main__":
    main()
