"""Loss functions for splice-site fine-tuning.

Provides two task-specific losses:

:func:`splice_classification_loss`
    Weighted cross-entropy over 5 classes (Donor+, Acceptor+, Donor-,
    Acceptor-, Background).  Background positions typically outnumber
    real splice sites by ~10,000:1, so class reweighting is advised.

:func:`splice_usage_loss`
    Masked binary cross-entropy over *observed* (position, condition) pairs.
    Predictions are gathered at sparse splice-site positions, then compared
    with per-condition SSE values only where the usage index has data.

:func:`compute_splice_class_weights`
    Utility to auto-compute class weights from a splice annotation and a
    BED file, accounting for the heavy class imbalance.
"""

from __future__ import annotations

from alphagenome_pytorch import metrics
import torch
import torch.nn.functional as F
from torch import Tensor


def splice_classification_loss(
    logits: Tensor,
    labels: Tensor,
    class_weights: Tensor | None = None,
    loss_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compute cross-entropy loss for splice-site classification.

    Args:
        logits: Raw (pre-softmax) model output, shape ``(B, S, 5)`` NLC.
        labels: Integer class labels, shape ``(B, S)``; values in ``{0,1,2,3,4}``.
        class_weights: Optional 1-D tensor of length 5 with per-class weights.
            Applied via :func:`torch.nn.functional.cross_entropy`'s ``weight``
            argument.  If ``None``, all classes are equally weighted (which will
            under-train non-background classes; see
            :func:`compute_splice_class_weights`).
        loss_mask: Optional boolean mask, shape ``(B, S)``. When provided,
            only positions where mask is True contribute to the loss.
            Useful for masking loss to specific gene regions.

    Returns:
        Tuple of:
        - ``loss``:  scalar mean cross-entropy.
        - ``metrics``: dict with ``'accuracy'`` (overall) and
          ``'acc_cls0'`` … ``'acc_cls4'`` per-class accuracy strings.
    """
    B, S, C = logits.shape
    logits_flat = logits.reshape(B * S, C)
    labels_flat = labels.reshape(B * S)

    if class_weights is not None:
        class_weights = class_weights.to(logits.device)

    # Apply loss mask if provided
    if loss_mask is not None:
        mask_flat = loss_mask.reshape(B * S)
        # Only compute loss on masked positions
        if mask_flat.any():
            logits_flat = logits_flat[mask_flat]
            labels_flat = labels_flat[mask_flat]
        else:
            # No valid positions in mask: return zero loss
            return torch.tensor(0.0, device=logits.device), {"accuracy": 0.0}

    loss = F.cross_entropy(logits_flat, labels_flat, weight=class_weights, reduction="mean")

    metrics: dict[str, float] = {}
    with torch.no_grad():
        preds = logits_flat.argmax(dim=-1)
        correct = preds == labels_flat
        metrics["accuracy"] = correct.float().mean().item()
        for c in range(C):
            mask = labels_flat == c
            if mask.any():
                metrics[f"acc_cls{c}"] = correct[mask].float().mean().item()

    return loss, metrics


def splice_usage_loss(
    predictions: Tensor,
    usage_positions: Tensor,
    usage_values: Tensor,
    usage_mask: Tensor,
) -> tuple[Tensor, dict]:
    """Masked binary cross-entropy loss for per-condition splice-site usage.

    Gathers model predictions at sparse splice-site positions and computes
    BCE only for ``(position, condition)`` pairs where the usage index has
    an observation (as indicated by *usage_mask*).

    Args:
        predictions: Raw logits from
            :class:`~alphagenome_pytorch.heads.SpliceSitesUsageHead`,
            shape ``(B, S, n_conditions)`` NLC (pre-sigmoid).
        usage_positions: Window-relative position indices for observed splice
            sites, shape ``(B, max_sites)``.  Padding positions are ``-1``
            and are ignored.
        usage_values: Target SSE fractions in ``[0, 1]``, shape
            ``(B, max_sites, n_conditions)``.  Unobserved entries are 0.
        usage_mask: Boolean mask indicating observed
            ``(position, condition)`` pairs, shape
            ``(B, max_sites, n_conditions)``.

    Returns:
        Tuple of:
        - ``loss``: Scalar mean binary cross-entropy over all observed entries.
        - ``metrics``: dict with ``'correlation'`` (overall) and
          ``'corr_cond0'`` … ``'corr_cond{n_conditions-1}'`` per-condition correlation strings.
    """
    _B, max_sites, n_conditions = usage_values.shape

    # Valid sites: position index ≥ 0
    valid_site_mask = usage_positions >= 0   # (B, max_sites)

    # Clamp positions for safe gather indexing (invalid positions will be masked)
    positions_clamped = usage_positions.clamp(min=0)   # (B, max_sites)

    # Gather predictions at splice site positions
    # predictions: (B, S, n_cond), positions_clamped: (B, max_sites)
    pos_exp = positions_clamped.unsqueeze(-1).expand(-1, -1, n_conditions)
    gathered = predictions.gather(1, pos_exp)  # (B, max_sites, n_cond)

    # Combine: need a valid position AND an observed (position, condition) pair
    valid_site_exp = valid_site_mask.unsqueeze(-1)       # (B, max_sites, 1)
    final_mask = usage_mask & valid_site_exp             # (B, max_sites, n_cond)

    n_valid = final_mask.sum()
    if n_valid == 0:
        # No observations in this batch — return zero loss with gradient and empty metrics dict
        loss = (predictions * 0.0).sum()
        metrics_dict = {"correlation": float("nan"), "n_valid": 0}
        return loss, metrics_dict

    loss = F.binary_cross_entropy_with_logits(
        gathered[final_mask],
        usage_values[final_mask],
        reduction="mean",
    )

    # Calculate correlation metrics
    metrics_dict = {}
    with torch.no_grad():
        pred_sigmoid = torch.sigmoid(gathered[final_mask])
        true_vals = usage_values[final_mask]

        if pred_sigmoid.numel() > 1:
            corr = torch.corrcoef(torch.stack([pred_sigmoid, true_vals]))[0, 1].item()
            metrics_dict["correlation"] = corr
        else:
            metrics_dict["correlation"] = float("nan")
        metrics_dict["n_valid"] = int(n_valid)
    return loss, metrics_dict


def compute_splice_class_weights(
    annotation_parquet: str,
    bed_file: str,
    sequence_length: int = 131_072,
    n_classes: int = 5,
) -> Tensor:
    """Compute per-class weights from annotation density over training windows.

    Counts how many positions in each window belong to each class, then
    returns weights inversely proportional to class frequency following the
    median-frequency balancing strategy.

    Args:
        annotation_parquet: Path to annotation Parquet produced by
            ``scripts/convert_splice_sites_to_parquet.py``.
        bed_file: BED file with training genomic windows.
        sequence_length: Model input window size in bp (default: 131,072).
        n_classes: Number of classes (default: 5).

    Returns:
        Float32 Tensor of shape ``(n_classes,)`` suitable for passing to
        :func:`splice_classification_loss` as *class_weights*.
    """
    import numpy as np
    from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
        SpliceSiteAnnotation,
        _load_intervals_from_bed,
        BACKGROUND_CLASS,
    )

    annot = SpliceSiteAnnotation(annotation_parquet)
    intervals, _ = _load_intervals_from_bed(bed_file)

    counts = np.zeros(n_classes, dtype=np.float64)
    half = sequence_length // 2

    for chrom, start, end in intervals:
        center = (start + end) // 2
        win_start = center - half
        win_end = center + half
        if win_start < 0:
            continue
        pos, cls = annot.query(chrom, win_start, win_end)
        n_bg = sequence_length - len(pos)
        counts[BACKGROUND_CLASS] += n_bg
        for c in cls:
            counts[c] += 1

    total = counts.sum()
    if total == 0:
        return torch.ones(n_classes)

    # Median-frequency balancing: w_c = median_freq / freq_c
    freq = counts / total
    median_freq = float(np.median(freq[freq > 0]))
    weights = np.where(freq > 0, median_freq / freq, 0.0)
    return torch.tensor(weights, dtype=torch.float32)
