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
    tissue_cond_groups: list[list[int]] | None = None,
    usage_coverage: Tensor | None = None,
    traj_min_timepoints: int = 3,
    traj_var_floor: float = 1e-3,
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

    # Defensive: drop any "observed" pair whose target is non-finite (e.g. a
    # degenerate SSE = NaN that slipped through the source data's coverage
    # filter). A single such entry would otherwise poison the whole batch's
    # BCE loss (and any downstream per-tissue trajectory terms) with NaN.
    finite_target_mask = torch.isfinite(usage_values)
    if not finite_target_mask.all():
        n_bad_targets = int((~finite_target_mask & final_mask).sum())
        if n_bad_targets > 0:
            print(f"  WARNING: {n_bad_targets} observed usage target(s) are non-finite (NaN/Inf SSE) — excluding from loss")
        final_mask = final_mask & finite_target_mask

    # Separately flag non-finite *predictions* at observed positions — this
    # indicates the model itself produced inf/NaN logits (e.g. attention
    # overflow from weight drift during full/partial unfreezing), which is a
    # training-instability signal distinct from a bad target value and
    # deserves a louder, distinguishable warning.
    finite_pred_mask = torch.isfinite(gathered)
    if not finite_pred_mask.all():
        n_bad_preds = int((~finite_pred_mask & final_mask).sum())
        if n_bad_preds > 0:
            print(f"  WARNING: {n_bad_preds} model prediction(s) are non-finite (inf/NaN logits) at observed usage positions — likely weight/activation instability, not a data issue")
        final_mask = final_mask & finite_pred_mask

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

        bce_w    = float(usage_loss_weights.get("bce", 0.0))
        delta_w  = float(usage_loss_weights.get("delta_mse", 0.0))
        pear_w   = float(usage_loss_weights.get("trajectory_pearson", 0.0))
        shape_w  = float(usage_loss_weights.get("trajectory_shape", 0.0))
        smooth_w = float(usage_loss_weights.get("traj_smooth", 0.0))

        gathered_sigmoid = torch.sigmoid(gathered)

        # Per-tissue condition groups isolate WITHIN-tissue temporal dynamics; without them
        # the "trajectory"/"delta" terms collapse to a single all-conditions group, which is
        # dominated by between-tissue level differences (the legacy behaviour).
        groups = tissue_cond_groups or [list(range(n_conditions))]

        if bce_w != 0.0:
            bce_loss = F.binary_cross_entropy_with_logits(
                gathered[final_mask], usage_values[final_mask], reduction="mean",
            )
            total_loss = total_loss + bce_w * bce_loss
            metrics_dict["bce_loss"] = bce_loss.item()

        # pooled correlation (level+tissue; kept for backwards-compatible logging)
        pred_vals = gathered_sigmoid[final_mask].detach()
        true_vals = usage_values[final_mask]
        if pred_vals.numel() > 1:
            metrics_dict["correlation"] = torch.corrcoef(torch.stack([pred_vals, true_vals]))[0, 1].item()

        if delta_w != 0.0:
            delta_loss = _per_tissue_delta_mse(
                gathered_sigmoid, usage_values, final_mask, groups)
            total_loss = total_loss + delta_w * delta_loss
            metrics_dict["delta_loss"] = delta_loss.item()

        if shape_w != 0.0:
            shape_loss, shape_metrics = _per_tissue_shape_loss(
                gathered_sigmoid, usage_values, final_mask, groups,
                coverage=usage_coverage, min_tp=traj_min_timepoints, var_floor=traj_var_floor)
            total_loss = total_loss + shape_w * shape_loss
            metrics_dict["trajectory_loss"] = shape_loss.item()
            metrics_dict.update(shape_metrics)

        if pear_w != 0.0:
            pear_loss, pear_metrics = _per_tissue_pearson_loss(
                gathered_sigmoid, usage_values, final_mask, groups,
                min_tp=traj_min_timepoints, var_floor=traj_var_floor)
            total_loss = total_loss + pear_w * pear_loss
            metrics_dict.setdefault("trajectory_loss", pear_loss.item())
            metrics_dict.update(pear_metrics)

        if smooth_w != 0.0:
            smooth_loss = _per_tissue_smoothness(gathered_sigmoid, groups)
            total_loss = total_loss + smooth_w * smooth_loss
            metrics_dict["smooth_loss"] = smooth_loss.item()

        # Always report the within-tissue temporal Pearson as a monitoring metric,
        # even when no shape term is weighted (this is the number that actually
        # reflects trajectory learning, unlike the pooled 'correlation').
        if "trajectory_corr" not in metrics_dict and tissue_cond_groups is not None:
            with torch.no_grad():
                _, _pm = _per_tissue_pearson_loss(
                    gathered_sigmoid, usage_values, final_mask, groups,
                    min_tp=traj_min_timepoints, var_floor=traj_var_floor)
            metrics_dict["trajectory_corr"] = _pm.get("trajectory_corr", float("nan"))
            metrics_dict["n_trajectory_sites"] = _pm.get("n_trajectory_sites", 0)

        if return_vals:
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


def _tissue_centered(pred: Tensor, tgt: Tensor, m: Tensor, idx: list[int]):
    """Slice one tissue's conditions and mean-center pred/target over its observed
    timepoints. Returns (dp, dt, mm, n) — masked centered deviations and obs counts,
    all shaped (B, max_sites, T_tissue) / (B, max_sites)."""
    p = pred[..., idx]
    t = tgt[..., idx]
    mm = m[..., idx].to(pred.dtype)
    n = mm.sum(-1)                                   # (B, sites)
    denom = n.clamp(min=1).unsqueeze(-1)
    mu_p = (p * mm).sum(-1, keepdim=True) / denom
    mu_t = (t * mm).sum(-1, keepdim=True) / denom
    dp = (p - mu_p) * mm
    dt = (t - mu_t) * mm
    return dp, dt, mm, n


def _per_tissue_delta_mse(pred, tgt, mask, groups) -> Tensor:
    """MSE on per-(site,tissue) mean-centered trajectories, over observed timepoints.
    Isolates temporal shape from tissue-level offsets (unlike a global-mean delta)."""
    se_sum = (pred * 0.0).sum()
    n_sum = pred.new_zeros(())
    for idx in groups:
        dp, dt, mm, _ = _tissue_centered(pred, tgt, mask, idx)
        se_sum = se_sum + ((dp - dt) ** 2 * mm).sum()
        n_sum = n_sum + mm.sum()
    return se_sum / n_sum.clamp(min=1.0)


def _per_tissue_shape_loss(pred, tgt, mask, groups, coverage=None,
                           min_tp: int = 3, var_floor: float = 1e-3
                           ) -> tuple[Tensor, dict[str, float]]:
    """Per-(site,tissue) centered-MSE shape loss, weighted by the true trajectory's
    amplitude (so dynamic sites dominate) and, optionally, mean read coverage."""
    num = (pred * 0.0).sum()
    wsum = pred.new_zeros(())
    n_traj = 0
    for idx in groups:
        dp, dt, mm, n = _tissue_centered(pred, tgt, mask, idx)
        denom = n.clamp(min=1)
        se = ((dp - dt) ** 2).sum(-1) / denom               # (B, sites) per-traj MSE
        var_t = (dt ** 2).sum(-1) / denom                   # true temporal variance
        ok = (n >= min_tp) & (var_t > var_floor)
        w = ok.to(pred.dtype) * var_t.clamp(min=0).sqrt()   # weight by true amplitude
        if coverage is not None:
            cov = coverage[..., idx].to(pred.dtype)
            w = w * (cov * mm).sum(-1) / denom              # mean coverage over obs tps
        num = num + (se * w).sum()
        wsum = wsum + w.sum()
        n_traj += int(ok.sum().item())
    loss = num / wsum.clamp(min=1e-6)
    return loss, {"n_trajectory_sites": n_traj}


def _per_tissue_pearson_loss(pred, tgt, mask, groups,
                             min_tp: int = 3, var_floor: float = 1e-3, eps: float = 1e-8
                             ) -> tuple[Tensor, dict[str, float]]:
    """1 - Pearson r per (site, tissue) over its timepoints, averaged. Vectorised."""
    losses = []
    corrs = []
    for idx in groups:
        dp, dt, mm, n = _tissue_centered(pred, tgt, mask, idx)
        var_p = (dp ** 2).sum(-1)
        var_t = (dt ** 2).sum(-1)
        # Regularise the norms with var_floor *inside* the sqrt so a flat prediction
        # (var_p -> 0) yields r -> 0 with a bounded gradient (a bare eps blows up).
        denom = (var_p + var_floor).sqrt() * (var_t + var_floor).sqrt()
        r = (dp * dt).sum(-1) / denom                                 # (B, sites)
        # Require only a non-flat TARGET (a flat prediction against a dynamic target
        # should be *penalised*, r -> 0, not skipped).
        ok = (n >= min_tp) & (var_t > var_floor)
        if ok.any():
            losses.append((1.0 - r)[ok])
            corrs.append(r[ok].detach())
    if not losses:
        return (pred * 0.0).sum(), {"trajectory_corr": float("nan"), "n_trajectory_sites": 0}
    loss = torch.cat(losses).mean()
    corr_mean = torch.cat(corrs).mean().item()
    return loss, {"trajectory_corr": corr_mean, "n_trajectory_sites": int(sum(l.numel() for l in losses))}


def _per_tissue_smoothness(pred, groups) -> Tensor:
    """Second-difference smoothness penalty on each predicted per-tissue trajectory.
    Groups must be timepoint-ordered. Applied to all predicted timepoints (a prior,
    independent of which are observed)."""
    terms = []
    for idx in groups:
        if len(idx) < 3:
            continue
        p = pred[..., idx]
        d2 = p[..., 2:] - 2.0 * p[..., 1:-1] + p[..., :-2]
        terms.append((d2 ** 2).mean())
    if not terms:
        return (pred * 0.0).sum()
    return torch.stack(terms).mean()


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
