"""Training loop for splice-site fine-tuning of AlphaGenome.

Provides :func:`train_epoch_splice` and :func:`validate_splice` which jointly
optimise two heads:

1. **Classification** – the pretrained
   ``model.splice_sites_classification_head`` (5-class softmax at 1 bp).
2. **Usage** – a new :class:`~alphagenome_pytorch.heads.SpliceSitesUsageHead`
   with ``n_conditions`` outputs specific to your dataset (sigmoid, BCE loss).

Both heads receive the same NCL 1 bp trunk embeddings.  The model trunk is
called with ``embeddings_only=True, channels_last=False`` to skip all other
heads and avoid the full decoder overhead when only 1 bp embeddings are needed.

Example
-------
::

    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.extensions.finetuning.heads import (
        create_splice_usage_finetuning_head,
    )
    from alphagenome_pytorch.extensions.finetuning.splice_training import (
        train_epoch_splice, validate_splice,
    )

    model = AlphaGenome.from_pretrained('model.pth')
    usage_head = create_splice_usage_finetuning_head(n_conditions=62)

    params = list(model.splice_sites_classification_head.parameters()) + \\
             list(usage_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    for epoch in range(10):
        train_loss = train_epoch_splice(
            model, usage_head, train_loader, optimizer, scheduler, device,
        )
        val_metrics = validate_splice(model, usage_head, val_loader, device)
"""

from __future__ import annotations

import math
import time

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from alphagenome_pytorch.extensions.finetuning.splice_losses import (
    splice_classification_loss,
    splice_usage_loss,
)

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader

    from alphagenome_pytorch.extensions.finetuning.logging import TrainingLogger


@dataclass
class SpliceTrainMetrics:
    """Aggregated training/validation metrics for one epoch."""

    loss: float = 0.0
    cls_loss: float = 0.0
    usage_loss: float = 0.0
    n_batches: int = 0
    n_usage_valid_pairs: int = 0  # total (position, condition) pairs with observed usage
    elapsed_s: float = 0.0   # wall-clock seconds for this epoch
    latency_ms: float = 0.0  # average batch latency in milliseconds


def train_epoch_splice(
    model: nn.Module,
    usage_head: "nn.Module | dict[int, nn.Module] | None",
    train_loader: "DataLoader",
    optimizer: "Optimizer",
    scheduler: "LambdaLR | None",
    device: torch.device,
    cls_weight: float = 1.0,
    usage_weight: float = 1.0,
    class_weights: Tensor | None = None,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    log_every: int = 50,
    epoch: int = 0,
    logger: "TrainingLogger | None" = None,
    max_grad_norm: float = 1.0,
    usage_delta_from_mean: bool = False,
    usage_loss_weights: dict | None = None,
) -> SpliceTrainMetrics:
    """Train the splice classification and usage heads for one epoch.

    The model trunk is run with ``embeddings_only=True`` to extract 1 bp NCL
    embeddings, which are then passed to both heads.

    Args:
        model: AlphaGenome trunk + splice heads.  The classification head
            is accessed via ``model.splice_sites_classification_head``.
        usage_head: Per-dataset usage head (or ``None`` to skip usage loss).
            May also be a ``dict[organism_index, nn.Module]`` for multi-species
            training where each species has its own usage head with a different
            number of conditions.  In that case
            :class:`~alphagenome_pytorch.extensions.finetuning.splice_datasets\
.SpeciesGroupedSampler` must be used so every batch is single-species.
        train_loader: DataLoader yielding dicts from :class:`~alphagenome_pytorch\
.extensions.finetuning.splice_datasets.SpliceSiteDataset` via
            :func:`~alphagenome_pytorch.extensions.finetuning.splice_datasets\
.collate_splice`.
        optimizer: Optimizer covering classification head params + usage head
            params (and any adapter/LoRA params on the trunk).
        scheduler: Optional LR scheduler; stepped once per optimizer step.
        device: Device to run on.
        cls_weight: Scalar weight for the classification loss (default: 1.0).
        usage_weight: Scalar weight for the usage loss (default: 1.0).
        class_weights: Optional 1-D Tensor of length 5 for
            :func:`~alphagenome_pytorch.extensions.finetuning.splice_losses\
.splice_classification_loss` class reweighting;
            see :func:`~alphagenome_pytorch.extensions.finetuning.splice_losses\
.compute_splice_class_weights`.
        use_amp: Use ``torch.amp.autocast`` for mixed-precision training
            (default: ``True``, only active on CUDA).
        accumulation_steps: Gradient accumulation steps (default: 1).
        log_every: Print training stats every N optimizer steps.
        epoch: Current epoch number (used for logging only).

    Returns:
        :class:`SpliceTrainMetrics` with averaged losses and accuracies.
    """
    model.train()
    if isinstance(usage_head, dict):
        for h in usage_head.values():
            h.train()
            h.to(device)
    elif usage_head is not None:
        usage_head.train()
        usage_head.to(device)

    metrics = SpliceTrainMetrics()
    # For tracking separate loss components if both are used
    bce_loss_sum = 0.0
    delta_loss_sum = 0.0
    n_dual_batches = 0
    # For robust epoch-level correlation
    # No correlation tracking
    step = 0
    amp_device = device.type if hasattr(device, "type") else str(device).split(":")[0]
    amp_enabled = use_amp and amp_device == "cuda"

    optimizer.zero_grad()

    epoch_start = time.perf_counter()
    step_start  = time.perf_counter()

    batch_latencies = []
    recent_batch_times = []  # Track recent batch times for rolling average
    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.perf_counter()
        seq = batch["sequence"].to(device)
        org_idx = batch["organism_index"].to(device)
        cls_labels = batch["classification_labels"].to(device)

        # Resolve per-batch usage head (dict keyed by organism_index)
        if isinstance(usage_head, dict):
            batch_org = int(batch["organism_index"][0].item())
            active_usage_head = usage_head.get(batch_org)
        else:
            active_usage_head = usage_head


        # Single-GPU mode
        with nullcontext():
            with autocast(amp_device, enabled=amp_enabled):
                # Run trunk (encoder + transformer + decoder) to get 1 bp embeddings
                outputs = model.forward(
                    seq, org_idx,
                    resolutions=(1,),
                    channels_last=False,
                    embeddings_only=True,
                )
                emb_1bp = outputs["embeddings_1bp"]  # (B, TRUNK_DIM, S) NCL

                # ── Classification loss ─────────────────────────────────────────
                cls_out = model.splice_sites_classification_head(
                    emb_1bp, org_idx, channels_last=True
                )
                # Apply loss mask if present in batch
                loss_mask = batch.get("loss_mask")
                if loss_mask is not None:
                    loss_mask = loss_mask.to(device)
                cls_loss_val, cls_acc = splice_classification_loss(
                    cls_out["logits"], cls_labels, class_weights=class_weights, loss_mask=loss_mask
                )
                total_loss = cls_weight * cls_loss_val

                # ── Usage loss (optional) ────────────────────────────────────────
                usage_loss_val = torch.tensor(0.0, device=device)
                usage_corr = {}
                usage_bce = None
                usage_delta = None
                if active_usage_head is not None and "usage_positions" in batch:
                    usage_pos = batch["usage_positions"].to(device)
                    usage_vals = batch["usage_values"].to(device)
                    usage_mask = batch["usage_mask"].to(device)

                    usage_out = active_usage_head(emb_1bp, org_idx, channels_last=True)
                    logits = usage_out["logits"]
                    # Weighted combination logic
                    if usage_delta_from_mean and usage_loss_weights is not None:
                        bce_w = usage_loss_weights.get("bce", 1.0)
                        delta_w = usage_loss_weights.get("delta_mse", 1.0)
                        # BCE loss
                        bce_loss, bce_corr = splice_usage_loss(
                            logits, usage_pos, usage_vals, usage_mask, delta_from_mean=False
                        )
                        delta_loss, delta_corr = splice_usage_loss(
                            logits, usage_pos, usage_vals, usage_mask, delta_from_mean=True
                        )
                        usage_loss_val = bce_w * bce_loss + delta_w * delta_loss
                        usage_corr = {"n_valid": bce_corr.get("n_valid", 0),
                                      "bce_loss": bce_loss.item(),
                                      "delta_loss": delta_loss.item()}
                        bce_loss_sum += bce_loss.item()
                        delta_loss_sum += delta_loss.item()
                        n_dual_batches += 1
                        usage_bce = bce_loss.item()
                        usage_delta = delta_loss.item()
                    else:
                        usage_loss_val, usage_corr = splice_usage_loss(
                            logits, usage_pos, usage_vals, usage_mask,
                            delta_from_mean=usage_delta_from_mean,
                        )
                        if usage_delta_from_mean:
                            usage_delta = usage_loss_val.item()
                        else:
                            usage_bce = usage_loss_val.item()
                    total_loss = total_loss + usage_weight * usage_loss_val
            if not torch.isfinite(total_loss):
                continue

            # Scale for accumulation
            (total_loss / accumulation_steps).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping to prevent exploding gradients
            if max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                # Skip optimizer step if gradients contain NaN/Inf
                if not torch.isfinite(grad_norm):
                    print(
                        f"  WARNING: Non-finite grad norm ({grad_norm:.4f}) at "
                        f"epoch {epoch} batch {batch_idx+1} — skipping step"
                    )
                    optimizer.zero_grad()
                    step += 1
                    continue

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            step += 1

            # Accumulate metrics
            metrics.loss += total_loss.item()
            metrics.cls_loss += cls_loss_val.item()
            metrics.usage_loss += usage_loss_val.item()
            # No correlation tracking
            metrics.n_usage_valid_pairs += usage_corr.get("n_valid", 0)
            metrics.n_batches += 1

            if log_every > 0 and step % log_every == 0:
                avg = metrics.loss / metrics.n_batches
                avg_cls = metrics.cls_loss / metrics.n_batches
                avg_usg = metrics.usage_loss / metrics.n_batches
                # Compute rolling average of recent batch times (in seconds)
                avg_batch_time = sum(recent_batch_times) / len(recent_batch_times) if recent_batch_times else 0.0
                elapsed = time.perf_counter() - step_start
                sps = log_every / elapsed  # optimizer steps per second
                # Add usage_bce and usage_delta to log if present
                usage_bce_str = f" usage_bce={usage_bce:.4f}" if usage_bce is not None else ""
                usage_delta_str = f" usage_mse_delta={usage_delta:.4f}" if usage_delta is not None else ""
                print(
                    f"  Epoch {epoch} step {step:5d} | "
                    f"loss={avg:.4f}  cls={avg_cls:.4f}  usage={avg_usg:.4f}" +
                    usage_bce_str + usage_delta_str +
                    f"  {sps:.2f} steps/s  batch_time={avg_batch_time:.2f}s"
                )
                # Add to logger
                log_metrics = {
                    "epoch": epoch,
                    "train_loss": avg,
                    "train_cls_loss": avg_cls,
                    "train_usage_loss": avg_usg,
                    "steps_per_sec": sps,
                    "avg_batch_time_s": avg_batch_time,
                }
                if usage_bce is not None:
                    log_metrics["train_usage_bce_loss"] = usage_bce
                if usage_delta is not None:
                    log_metrics["train_usage_mse_delta_loss"] = usage_delta
                if logger is not None:
                    logger.log_step(log_metrics)
                step_start = time.perf_counter()
        # Record batch latency (for each optimizer step)
        batch_end = time.perf_counter()
        batch_time = batch_end - batch_start
        batch_latencies.append(batch_time * 1000.0)  # ms for metrics
        recent_batch_times.append(batch_time)  # seconds for logging
        # Keep only recent batch times (rolling window of 100 batches)
        if len(recent_batch_times) > 100:
            recent_batch_times.pop(0)
        # Limit batch_latencies to prevent unbounded memory growth
        if len(batch_latencies) > 1000:
            # Keep rolling average by downsampling
            batch_latencies = batch_latencies[-1000:]
        
        # Periodic memory cleanup to prevent accumulation
        if (batch_idx + 1) % 100 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    # Average
    if metrics.n_batches > 0:
        metrics.loss /= metrics.n_batches
        metrics.cls_loss /= metrics.n_batches
        metrics.usage_loss /= metrics.n_batches
        metrics.latency_ms = sum(batch_latencies) / len(batch_latencies)
    metrics.elapsed_s = time.perf_counter() - epoch_start

    # Add extra logging for dual loss
    if n_dual_batches > 0:
        metrics.bce_loss = bce_loss_sum / n_dual_batches
        metrics.delta_loss = delta_loss_sum / n_dual_batches

    # Warn when usage head is present but received zero gradient signal.
    # This typically means there are no matching (position, condition) pairs
    # between the annotation and usage parquet — usually caused by a coordinate
    # mismatch (e.g. usage parquet generated with the wrong position correction).
    has_usage_head = (
        (isinstance(usage_head, dict) and any(v is not None for v in usage_head.values()))
        or (usage_head is not None and not isinstance(usage_head, dict))
    )
    if has_usage_head and metrics.n_usage_valid_pairs == 0 and metrics.n_batches > 0:
        print(
            f"\n{'='*70}\n"
            f" WARNING: Usage head received NO gradient!\n"
            f"{'='*70}\n"
            f"  No valid (position, condition) pairs were found in {metrics.n_batches} batches.\n"
            f"  The usage head will NOT train — predictions will remain at ~0.5.\n\n"
            f"  Most likely cause: COORDINATE MISMATCH\n"
            f"  → If your usage parquet is from Spliser (1-based), use --usage-coord-base 1\n"
            f"  → If already converted to 0-based, use --usage-coord-base 0\n\n"
            f"  Check your --usage-coord-base setting and re-run training.\n"
            f"{'='*70}\n"
        )


    return metrics


@torch.no_grad()
def validate_splice(
    model: nn.Module,
    usage_head: "nn.Module | dict[int, nn.Module] | None",
    val_loader: "DataLoader",
    device: torch.device,
    cls_weight: float = 1.0,
    usage_weight: float = 1.0,
    class_weights: Tensor | None = None,
    use_amp: bool = True,
    usage_delta_from_mean: bool = False,
    usage_loss_weights: dict | None = None,
) -> SpliceTrainMetrics:
    """Evaluate the splice heads on the validation set.

    Args:
        model: AlphaGenome model (same as :func:`train_epoch_splice`).
        usage_head: Per-dataset usage head, or ``None``.  May be a
            ``dict[organism_index, nn.Module]`` for multi-species training;
            see :func:`train_epoch_splice`.
        val_loader: DataLoader from a validation
            :class:`~alphagenome_pytorch.extensions.finetuning.splice_datasets\
.SpliceSiteDataset`.
        device: Device to run on.
        cls_weight: Classification loss weight (default: 1.0).
        usage_weight: Usage loss weight (default: 1.0).
        class_weights: Optional class weights for classification loss.
        use_amp: Use ``torch.amp.autocast`` (default: ``True``).

    Returns:
        :class:`SpliceTrainMetrics` averaged over the validation set.
    """
    model.eval()
    if isinstance(usage_head, dict):
        for h in usage_head.values():
            h.eval()
    elif usage_head is not None:
        usage_head.eval()

    metrics = SpliceTrainMetrics()
    bce_loss_sum = 0.0
    delta_loss_sum = 0.0
    n_dual_batches = 0
    # No correlation tracking
    amp_device = device.type if hasattr(device, "type") else str(device).split(":")[0]
    amp_enabled = use_amp and amp_device == "cuda"

    val_start = time.perf_counter()

    batch_latencies = []
    for batch_idx, batch in enumerate(val_loader):
        batch_start = time.perf_counter()
        seq = batch["sequence"].to(device)
        org_idx = batch["organism_index"].to(device)
        cls_labels = batch["classification_labels"].to(device)

        # Resolve per-batch usage head
        if isinstance(usage_head, dict):
            batch_org = int(batch["organism_index"][0].item())
            active_usage_head = usage_head.get(batch_org)
        else:
            active_usage_head = usage_head

        with autocast(amp_device, enabled=amp_enabled):
            outputs = model.forward(
                seq, org_idx,
                resolutions=(1,),
                channels_last=False,
                embeddings_only=True,
            )
            emb_1bp = outputs["embeddings_1bp"]

            cls_out = model.splice_sites_classification_head(
                emb_1bp, org_idx, channels_last=True
            )
            # Apply loss mask if present in batch
            loss_mask = batch.get("loss_mask")
            if loss_mask is not None:
                loss_mask = loss_mask.to(device)
            cls_loss_val, cls_acc = splice_classification_loss(
                cls_out["logits"], cls_labels, class_weights=class_weights, loss_mask=loss_mask
            )
            total_loss = cls_weight * cls_loss_val

            usage_loss_val = torch.tensor(0.0, device=device)
            usage_corr = {}
            if active_usage_head is not None and "usage_positions" in batch:
                usage_pos = batch["usage_positions"].to(device)
                usage_vals = batch["usage_values"].to(device)
                usage_mask = batch["usage_mask"].to(device)

                usage_out = active_usage_head(emb_1bp, org_idx, channels_last=True)
                logits = usage_out["logits"]
                # Weighted combination logic
                if usage_delta_from_mean and usage_loss_weights is not None:
                    bce_w = usage_loss_weights.get("bce", 1.0)
                    delta_w = usage_loss_weights.get("delta_mse", 1.0)
                    # BCE loss
                    bce_loss, bce_corr = splice_usage_loss(
                        logits, usage_pos, usage_vals, usage_mask, delta_from_mean=False
                    )
                    # Delta-from-mean loss
                    delta_loss, delta_corr = splice_usage_loss(
                        logits, usage_pos, usage_vals, usage_mask, delta_from_mean=True
                    )
                    usage_loss_val = bce_w * bce_loss + delta_w * delta_loss
                    usage_corr = {"n_valid": bce_corr.get("n_valid", 0),
                                  "bce_loss": bce_loss.item(),
                                  "delta_loss": delta_loss.item()}
                    bce_loss_sum += bce_loss.item()
                    delta_loss_sum += delta_loss.item()
                    n_dual_batches += 1
                else:
                    usage_loss_val, usage_corr = splice_usage_loss(
                        logits, usage_pos, usage_vals, usage_mask,
                        delta_from_mean=usage_delta_from_mean,
                    )
                total_loss = total_loss + usage_weight * usage_loss_val

        metrics.loss += total_loss.item()
        metrics.cls_loss += cls_loss_val.item()
        metrics.usage_loss += usage_loss_val.item()
        # No correlation tracking
        metrics.n_batches += 1
        batch_end = time.perf_counter()
        batch_latencies.append((batch_end - batch_start) * 1000.0)  # ms
        # Limit batch_latencies in validation too
        if len(batch_latencies) > 1000:
            batch_latencies = batch_latencies[-1000:]
        
        # Periodic memory cleanup in validation
        if (batch_idx + 1) % 100 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    if metrics.n_batches > 0:
        metrics.loss /= metrics.n_batches
        metrics.cls_loss /= metrics.n_batches
        metrics.usage_loss /= metrics.n_batches
        metrics.latency_ms = sum(batch_latencies) / len(batch_latencies)
    metrics.elapsed_s = time.perf_counter() - val_start

    if n_dual_batches > 0:
        metrics.bce_loss = bce_loss_sum / n_dual_batches
        metrics.delta_loss = delta_loss_sum / n_dual_batches

    # No correlation logging

    return metrics
