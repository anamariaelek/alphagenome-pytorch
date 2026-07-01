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
    delta_from_mean: bool = False,
    usage_loss_weights: dict[str, float] | None = None,
    return_vals: bool = False,
) -> tuple:
    """Masked loss for per-condition splice-site usage.

    Gathers model predictions at sparse splice-site positions and computes
    the loss only for ``(position, condition)`` pairs where the usage index has
    an observation (as indicated by *usage_mask*).

    By default (``delta_from_mean=False``) uses binary cross-entropy with
    logits (BCE) to predict absolute usage values in ``[0, 1]``.

        ``delta_from_mean`` is retained for backward compatibility, but the
        effective behavior is now controlled by ``usage_loss_weights``.

        When *usage_loss_weights* is provided, the loss can combine multiple
        components in a single call. Supported keys are:

        - ``bce``: BCE on absolute usage values.
        - ``delta_mse``: MSE on deviation from the per-site mean.
        - ``trajectory_pearson``: ``1 - PearsonR`` over each site's observed
            condition trajectory, averaged across sites.

        Missing keys default to ``0.0`` when a weight dictionary is passed.

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
        delta_from_mean: Deprecated compatibility flag. The loss is controlled
            by ``usage_loss_weights``; when no weights are provided, BCE is
            used.
        usage_loss_weights: Optional dict of component weights. When provided,
            the function combines the enabled usage loss terms instead of
            returning a single loss component.

    Returns:
        Tuple of:
        - ``loss``: Scalar mean loss over all observed entries.
        - ``metrics``: dict with ``'correlation'`` (overall) and
          ``'n_valid'`` count.
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
        if return_vals:
            return loss, metrics_dict, torch.tensor([]), torch.tensor([])
        return loss, metrics_dict

    if usage_loss_weights is not None:
        total_loss = (predictions * 0.0).sum()
        metrics_dict = {"correlation": float("nan"), "n_valid": int(n_valid)}

        bce_w = float(usage_loss_weights.get("bce", 0.0))
        delta_w = float(usage_loss_weights.get("delta_mse", 0.0))
        traj_w = float(usage_loss_weights.get("trajectory_pearson", 0.0))

        gathered_sigmoid = torch.sigmoid(gathered)

        if bce_w != 0.0:
            bce_loss = F.binary_cross_entropy_with_logits(
                gathered[final_mask],
                usage_values[final_mask],
                reduction="mean",
            )
            total_loss = total_loss + bce_w * bce_loss
            metrics_dict["bce_loss"] = bce_loss.item()
            pred_vals = gathered_sigmoid[final_mask].detach()
            true_vals = usage_values[final_mask]
            if pred_vals.numel() > 1:
                metrics_dict["correlation"] = torch.corrcoef(torch.stack([pred_vals, true_vals]))[0, 1].item()
            else:
                metrics_dict["correlation"] = float("nan")

        if delta_w != 0.0:
            n_obs = final_mask.sum(-1, keepdim=True).float().clamp(min=1)
            mean_preds = (gathered_sigmoid * final_mask).sum(-1, keepdim=True) / n_obs
            mean_targets = (usage_values * final_mask).sum(-1, keepdim=True) / n_obs
            delta_preds = gathered_sigmoid - mean_preds
            delta_targets = usage_values - mean_targets
            delta_loss = F.mse_loss(
                delta_preds[final_mask],
                delta_targets[final_mask],
                reduction="mean",
            )
            total_loss = total_loss + delta_w * delta_loss
            metrics_dict["delta_loss"] = delta_loss.item()

        if traj_w != 0.0:
            traj_loss, traj_metrics = _trajectory_pearson_loss(
                gathered_sigmoid, usage_values, final_mask
            )
            total_loss = total_loss + traj_w * traj_loss
            metrics_dict["trajectory_loss"] = traj_loss.item()
            metrics_dict.update(traj_metrics)

        if return_vals:
            pred_vals = gathered_sigmoid[final_mask].detach()
            true_vals = usage_values[final_mask]
            return total_loss, metrics_dict, pred_vals, true_vals
        return total_loss, metrics_dict

    loss = F.binary_cross_entropy_with_logits(
        gathered[final_mask],
        usage_values[final_mask],
        reduction="mean",
    )
    pred_vals = torch.sigmoid(gathered[final_mask]).detach()
    true_vals = usage_values[final_mask]

    # Calculate correlation metrics
    metrics_dict = {}
    with torch.no_grad():
        if pred_vals.numel() > 1:
            corr = torch.corrcoef(torch.stack([pred_vals, true_vals]))[0, 1].item()
            metrics_dict["correlation"] = corr
        else:
            metrics_dict["correlation"] = float("nan")
        metrics_dict["n_valid"] = int(n_valid)
    if return_vals:
        return loss, metrics_dict, pred_vals, true_vals
    return loss, metrics_dict


def _trajectory_pearson_loss(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Compute a per-site Pearson-R trajectory loss over observed conditions."""
    flat_predictions = predictions.reshape(-1, predictions.shape[-1])
    flat_targets = targets.reshape(-1, targets.shape[-1])
    flat_mask = mask.reshape(-1, mask.shape[-1])

    site_losses = []
    site_corrs = []

    for pred_site, true_site, site_mask in zip(flat_predictions, flat_targets, flat_mask):
        if int(site_mask.sum().item()) < 2:
            continue

        pred_vals = pred_site[site_mask]
        true_vals = true_site[site_mask]
        corr = metrics.pearson_r(pred_vals, true_vals, dim=0)
        site_losses.append(1.0 - corr)
        site_corrs.append(corr.detach())

    if not site_losses:
        zero = (predictions * 0.0).sum()
        return zero, {"trajectory_corr": float("nan"), "n_trajectory_sites": 0}

    loss = torch.stack(site_losses).mean()
    corr_mean = torch.stack(site_corrs).mean().item()
    return loss, {"trajectory_corr": corr_mean, "n_trajectory_sites": len(site_losses)}


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
