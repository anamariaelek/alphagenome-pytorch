#!/usr/bin/env python
"""Reproduce the splice fine-tuning GPU workload, simplified and sped up.

Builds AlphaGenome model with  gradient-checkpointing and LoRA adapters
and runs the forward -> loss -> backward -> optimizer-step loop 
with randomly-initialized weights and synthetic random batches.

Usage:
    python scripts/gpu_reproduce_training.py --steps 600
    python scripts/gpu_reproduce_training.py --hours 6

"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn
from torch.amp import autocast

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.extensions.finetuning import TransferConfig
from alphagenome_pytorch.extensions.finetuning.adapters import get_adapter_params
from alphagenome_pytorch.extensions.finetuning.transfer import prepare_for_transfer
from alphagenome_pytorch.extensions.finetuning.heads import (
    create_splice_classification_finetuning_head,
    create_splice_usage_finetuning_head,
)
from alphagenome_pytorch.extensions.finetuning.splice_losses import (
    splice_classification_loss,
    splice_usage_loss,
)

# Same set as scripts/finetune_splice.py's TRAINABLE_COMPONENTS
TRAINABLE_COMPONENTS = {
    "encoder", "tower", "decoder", "embedder_128bp", "embedder_1bp",
    "embedder_pair", "organism_embed",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce splice fine-tuning GPU load with synthetic data")

    # Duration control: run until whichever limit is hit first
    p.add_argument("--steps", type=int, default=600, help="Number of optimizer steps to run")
    p.add_argument("--hours", type=float, default=None, help="Optional wall-clock limit in hours (overrides --steps if reached first)")

    # Data shape
    p.add_argument("--sequence-length", type=int, default=131072)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--num-organisms", type=int, default=5)
    p.add_argument("--n-conditions", type=int, default=90, help="Usage-head condition count (synthetic)")
    p.add_argument("--max-sites", type=int, default=1024)

    # Model / training mode
    p.add_argument("--mode", type=str, choices=["linear-probe", "lora", "full"], default="lora")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-targets", type=str, default="q_proj,v_proj")
    p.add_argument(
        "--train-components", type=str, default="",
        help=f"Comma-separated, subset of: {sorted(TRAINABLE_COMPONENTS)}",
    )
    p.add_argument("--dtype", type=str, choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")

    # Optimizer
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--log-every", type=int, default=5, help="Print a heartbeat every N optimizer steps")
    return p.parse_args()


def make_synthetic_batch(
    batch_size: int, seq_len: int, num_organisms: int, n_conditions: int,
    max_sites: int, device: torch.device,
) -> dict:
    """Random data matching the real dataset's tensor shapes/dtypes."""
    seq = torch.zeros(batch_size, seq_len, 4, device=device)
    idx = torch.randint(0, 4, (batch_size, seq_len), device=device)
    seq.scatter_(2, idx.unsqueeze(-1), 1.0)

    organism_index = torch.randint(0, num_organisms, (batch_size,), device=device)
    classification_labels = torch.randint(0, 5, (batch_size, seq_len), device=device)

    usage_positions = torch.randint(0, seq_len, (batch_size, max_sites), device=device)
    usage_values = torch.rand(batch_size, max_sites, n_conditions, device=device)
    usage_mask = torch.rand(batch_size, max_sites, n_conditions, device=device) > 0.5

    return dict(
        sequence=seq,
        organism_index=organism_index,
        classification_labels=classification_labels,
        usage_positions=usage_positions,
        usage_values=usage_values,
        usage_mask=usage_mask,
    )


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module, list]:
    dtype_policy = DtypePolicy.full_float32() if args.dtype == "float32" else DtypePolicy.mixed_precision()
    print(f"Dtype policy: {dtype_policy}")

    model = AlphaGenome(
        num_organisms=args.num_organisms,
        gradient_checkpointing=args.gradient_checkpointing,
        dtype_policy=dtype_policy,
    )  # random init — no pretrained weights needed for a hardware reproduction

    if args.mode != "full":
        for param in model.parameters():
            param.requires_grad = False

    cls_head = create_splice_classification_finetuning_head(num_organisms=args.num_organisms)
    usage_head = create_splice_usage_finetuning_head(
        n_conditions=args.n_conditions, num_organisms=args.num_organisms,
    )

    trainable_params: list[torch.nn.Parameter] = []

    if args.mode == "linear-probe":
        trainable_params.extend(cls_head.parameters())
        trainable_params.extend(usage_head.parameters())
    elif args.mode == "lora":
        if args.lora_rank > 0:
            lora_targets = [t.strip() for t in args.lora_targets.split(",")]
            config = TransferConfig(
                mode="lora", lora_targets=lora_targets,
                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            )
            model = prepare_for_transfer(model, config)
            trainable_params = get_adapter_params(model)
        trainable_params.extend(cls_head.parameters())
        trainable_params.extend(usage_head.parameters())
    elif args.mode == "full":
        trainable_params = list(model.parameters())
        trainable_params.extend(usage_head.parameters())
    else:
        raise ValueError(args.mode)

    extra_components = [c.strip() for c in args.train_components.split(",") if c.strip()]
    if extra_components and args.mode != "full":
        component_map = {
            "encoder": model.encoder, "tower": model.tower, "decoder": model.decoder,
            "embedder_128bp": model.embedder_128bp, "embedder_1bp": model.embedder_1bp,
            "embedder_pair": model.embedder_pair, "organism_embed": model.organism_embed,
        }
        seen = {id(p) for p in trainable_params}
        for name in extra_components:
            for p in component_map[name].parameters():
                p.requires_grad = True
                if id(p) not in seen:
                    trainable_params.append(p)
                    seen.add(id(p))
        print(f"Also training components: {extra_components}")

    model = model.to(device)
    cls_head = cls_head.to(device)
    usage_head = usage_head.to(device)

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in usage_head.parameters())
    print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/max(n_total,1):.2f}%)")

    return model, cls_head, usage_head, trainable_params


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available to PyTorch. Aborting.", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")
    print(f"Device: {device}  ({torch.cuda.get_device_name(device)})")
    print(f"Sequence length: {args.sequence_length}  Batch size: {args.batch_size}  "
          f"Grad accum: {args.grad_accum}  Mode: {args.mode}")

    model, cls_head, usage_head, trainable_params = build_model(args, device)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    amp_enabled = args.dtype == "bfloat16"
    deadline = time.monotonic() + args.hours * 3600 if args.hours else None

    print(f"\nStarting reproduction run: {args.steps} optimizer steps"
          + (f", up to {args.hours:.2f}h wall-clock" if args.hours else "") + "\n")

    optimizer.zero_grad()
    start = time.monotonic()
    step = 0
    micro_batch = 0

    while step < args.steps:
        if deadline is not None and time.monotonic() >= deadline:
            print("Wall-clock limit reached.")
            break

        batch = make_synthetic_batch(
            args.batch_size, args.sequence_length, args.num_organisms,
            args.n_conditions, args.max_sites, device,
        )

        with autocast("cuda", enabled=amp_enabled):
            outputs = model.forward(
                batch["sequence"], batch["organism_index"],
                resolutions=(1,), channels_last=False, embeddings_only=True,
            )
            emb_1bp = outputs["embeddings_1bp"]

            cls_out = cls_head(emb_1bp, batch["organism_index"], channels_last=True)
            cls_loss_val, _ = splice_classification_loss(cls_out["logits"], batch["classification_labels"])

            usage_out = usage_head(emb_1bp, batch["organism_index"], channels_last=True)
            usage_loss_val, _ = splice_usage_loss(
                usage_out["logits"], batch["usage_positions"], batch["usage_values"], batch["usage_mask"],
                usage_loss_weights={"bce": 1.0},
            )

            total_loss = cls_loss_val + usage_loss_val

        if not torch.isfinite(total_loss):
            print(f"  WARNING: non-finite loss at micro-batch {micro_batch} — skipping")
            micro_batch += 1
            continue

        (total_loss / args.grad_accum).backward()
        micro_batch += 1

        if micro_batch % args.grad_accum == 0:
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad] + list(usage_head.parameters()),
                    args.max_grad_norm,
                )
                if not torch.isfinite(grad_norm):
                    print(f"  WARNING: non-finite grad norm ({grad_norm}) at step {step} — skipping optimizer step")
                    optimizer.zero_grad()
                    step += 1
                    continue

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % args.log_every == 0 or step == args.steps:
                elapsed = time.monotonic() - start
                allocated = torch.cuda.memory_allocated(device) / 1e9
                reserved = torch.cuda.memory_reserved(device) / 1e9
                print(
                    f"  step {step:5d}/{args.steps} | loss={total_loss.item():.4f} | "
                    f"{elapsed/max(step,1):.2f}s/step | mem_alloc={allocated:.2f}GB mem_reserved={reserved:.2f}GB",
                    flush=True,
                )

    total_elapsed = time.monotonic() - start
    print(f"\nCompleted {step} optimizer steps over {total_elapsed/3600:.2f}h.")


if __name__ == "__main__":
    main()
