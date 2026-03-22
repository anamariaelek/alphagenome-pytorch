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
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module, unwrapping DistributedDataParallel if needed."""
    return model.module if isinstance(model, DDP) else model

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
    accuracy: float = 0.0
    # Per-class accuracy (class 0-3 = splice sites, 4 = background)
    per_class_acc: dict[str, float] = field(default_factory=dict)
    n_batches: int = 0
    elapsed_s: float = 0.0   # wall-clock seconds for this epoch


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
.splice_classification_loss` class reweighting.  Strongly recommended;
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
    step = 0
    amp_device = device.type if hasattr(device, "type") else str(device).split(":")[0]
    amp_enabled = use_amp and amp_device == "cuda"

    optimizer.zero_grad()

    epoch_start = time.perf_counter()
    step_start  = time.perf_counter()

    for batch_idx, batch in enumerate(train_loader):
        seq = batch["sequence"].to(device)
        org_idx = batch["organism_index"].to(device)
        cls_labels = batch["classification_labels"].to(device)

        # Resolve per-batch usage head (dict keyed by organism_index)
        if isinstance(usage_head, dict):
            batch_org = int(batch["organism_index"][0].item())
            active_usage_head = usage_head.get(batch_org)
        else:
            active_usage_head = usage_head

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
            cls_out = _unwrap(model).splice_sites_classification_head(
                emb_1bp, org_idx, channels_last=True
            )
            cls_loss_val, cls_acc = splice_classification_loss(
                cls_out["logits"], cls_labels, class_weights=class_weights
            )
            total_loss = cls_weight * cls_loss_val

            # ── Usage loss (optional) ────────────────────────────────────────
            usage_loss_val = torch.tensor(0.0, device=device)
            if active_usage_head is not None and "usage_positions" in batch:
                usage_pos = batch["usage_positions"].to(device)
                usage_vals = batch["usage_values"].to(device)
                usage_mask = batch["usage_mask"].to(device)

                usage_out = active_usage_head(emb_1bp, org_idx, channels_last=True)
                usage_loss_val = splice_usage_loss(
                    usage_out["logits"], usage_pos, usage_vals, usage_mask
                )
                total_loss = total_loss + usage_weight * usage_loss_val

        # Scale for accumulation
        (total_loss / accumulation_steps).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            step += 1

            # Accumulate metrics
            metrics.loss += total_loss.item()
            metrics.cls_loss += cls_loss_val.item()
            metrics.usage_loss += usage_loss_val.item()
            metrics.accuracy += cls_acc.get("accuracy", 0.0)
            for k, v in cls_acc.items():
                metrics.per_class_acc[k] = metrics.per_class_acc.get(k, 0.0) + v
            metrics.n_batches += 1

            if log_every > 0 and step % log_every == 0:
                avg = metrics.loss / metrics.n_batches
                avg_cls = metrics.cls_loss / metrics.n_batches
                avg_usg = metrics.usage_loss / metrics.n_batches
                elapsed = time.perf_counter() - step_start
                sps = log_every / elapsed  # optimizer steps per second
                print(
                    f"  Epoch {epoch} step {step:5d} | "
                    f"loss={avg:.4f}  cls={avg_cls:.4f}  usage={avg_usg:.4f}  "
                    f"{sps:.2f} steps/s"
                )
                if logger is not None:
                    logger.log_step({
                        "epoch": epoch,
                        "train_loss": avg,
                        "train_cls_loss": avg_cls,
                        "train_usage_loss": avg_usg,
                        "steps_per_sec": sps,
                    })
                step_start = time.perf_counter()

    # Average
    if metrics.n_batches > 0:
        metrics.loss /= metrics.n_batches
        metrics.cls_loss /= metrics.n_batches
        metrics.usage_loss /= metrics.n_batches
        metrics.accuracy /= metrics.n_batches
        metrics.per_class_acc = {
            k: v / metrics.n_batches for k, v in metrics.per_class_acc.items()
        }
    metrics.elapsed_s = time.perf_counter() - epoch_start

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
    amp_device = device.type if hasattr(device, "type") else str(device).split(":")[0]
    amp_enabled = use_amp and amp_device == "cuda"

    val_start = time.perf_counter()

    for batch in val_loader:
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

            cls_out = _unwrap(model).splice_sites_classification_head(
                emb_1bp, org_idx, channels_last=True
            )
            cls_loss_val, cls_acc = splice_classification_loss(
                cls_out["logits"], cls_labels, class_weights=class_weights
            )
            total_loss = cls_weight * cls_loss_val

            usage_loss_val = torch.tensor(0.0, device=device)
            if active_usage_head is not None and "usage_positions" in batch:
                usage_pos = batch["usage_positions"].to(device)
                usage_vals = batch["usage_values"].to(device)
                usage_mask = batch["usage_mask"].to(device)

                usage_out = active_usage_head(emb_1bp, org_idx, channels_last=True)
                usage_loss_val = splice_usage_loss(
                    usage_out["logits"], usage_pos, usage_vals, usage_mask
                )
                total_loss = total_loss + usage_weight * usage_loss_val

        metrics.loss += total_loss.item()
        metrics.cls_loss += cls_loss_val.item()
        metrics.usage_loss += usage_loss_val.item()
        metrics.accuracy += cls_acc.get("accuracy", 0.0)
        for k, v in cls_acc.items():
            metrics.per_class_acc[k] = metrics.per_class_acc.get(k, 0.0) + v
        metrics.n_batches += 1

    if metrics.n_batches > 0:
        metrics.loss /= metrics.n_batches
        metrics.cls_loss /= metrics.n_batches
        metrics.usage_loss /= metrics.n_batches
        metrics.accuracy /= metrics.n_batches
        metrics.per_class_acc = {
            k: v / metrics.n_batches for k, v in metrics.per_class_acc.items()
        }
    metrics.elapsed_s = time.perf_counter() - val_start

    return metrics
