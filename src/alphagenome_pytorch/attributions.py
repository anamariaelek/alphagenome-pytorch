"""Input attribution methods for splice-site predictions.

Provides gradient-based attribution scores (saliency, input-times-gradient,
integrated gradients, SmoothGrad) that explain a scalar model output in terms
of the one-hot encoded input DNA sequence. Built for the splice-site
classification and usage heads (:mod:`alphagenome_pytorch.heads`), but the
core ``compute_gradients`` / ``integrated_gradients`` functions work with any
scalar target derived from ``AlphaGenome.forward``'s output dict.

Typical usage::

    from alphagenome_pytorch.attributions import (
        classification_target, usage_target, integrated_gradients,
        summarize_attributions,
    )

    target_fn = classification_target(position=5000, class_idx=0)  # Donor+
    attr = integrated_gradients(model, dna_sequence, organism_index, target_fn)
    contribution = summarize_attributions(attr, dna_sequence)  # (B, S)
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.nn as nn

# A target function maps the model's forward() output dict to a scalar
# (per-batch-element) tensor of shape (B,) that attributions are computed for.
TargetFn = Callable[[dict], torch.Tensor]


# ---------------------------------------------------------------------------
# Target functions
# ---------------------------------------------------------------------------

def classification_target(position: int, class_idx: int) -> TargetFn:
    """Target: splice-site classification logit for ``class_idx`` at 1bp ``position``.

    Class indices match :data:`alphagenome_pytorch.heads.SpliceSitesClassificationHead`:
    0=Donor+, 1=Acceptor+, 2=Donor-, 3=Acceptor-, 4=Background.

    Args:
        position: 0-based index along the sequence axis (1bp resolution).
        class_idx: Splice-site class index (0-4).
    """
    def target_fn(outputs: dict) -> torch.Tensor:
        logits = outputs["splice_sites_classification"]["logits"]  # (B, S, 5) NLC
        return logits[:, position, class_idx]
    return target_fn


def usage_target(position: int, condition_idx: "int | list[int]") -> TargetFn:
    """Target: splice-site usage logit for ``condition_idx`` at 1bp ``position``.

    Uses the pre-sigmoid logits (not the squashed prediction) so that
    attributions reflect the model's raw output scale. Requires the model to
    have been run with an organism whose ``splice_sites_usage_head`` is
    attached (see ``build_model`` in ``scripts/evaluate_splice.py``).

    Args:
        position: 0-based index along the sequence axis (1bp resolution).
        condition_idx: Index into the usage head's condition/track axis. Pass a
            list of indices (e.g. every timepoint condition for one tissue) to
            attribute their *mean* logit instead of a single condition — useful
            for a tissue-level attribution aggregated across timepoints.
    """
    def target_fn(outputs: dict) -> torch.Tensor:
        logits = outputs["splice_sites_usage"]["logits"]  # (B, S, T) NLC
        if isinstance(condition_idx, int):
            return logits[:, position, condition_idx]
        return logits[:, position, condition_idx].mean(dim=-1)
    return target_fn


# ---------------------------------------------------------------------------
# Core gradient computation
# ---------------------------------------------------------------------------

def _autocast_context(model: nn.Module, device: torch.device):
    """Match the autocast behavior of ``AlphaGenome.predict()``.

    ``model.forward()`` unconditionally casts its input to ``compute_dtype``
    but relies on an active ``autocast`` context to also cast the (float32)
    weights; calling ``forward()`` directly without it raises a dtype
    mismatch in the first conv layer.
    """
    dtype_policy = getattr(model, "dtype_policy", None)
    if dtype_policy is None or dtype_policy.compute_dtype == torch.float32:
        return nullcontext()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.autocast(device_type=device_type, dtype=dtype_policy.compute_dtype)


def _prepare_organism_index(
    organism_index: "torch.Tensor | int",
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(organism_index, int):
        return torch.full((batch_size,), organism_index, dtype=torch.long, device=device)
    return organism_index.to(device)


def compute_gradients(
    model: nn.Module,
    dna_sequence: torch.Tensor,
    organism_index: "torch.Tensor | int",
    target_fn: TargetFn,
    resolutions: tuple[int, ...] = (1,),
) -> torch.Tensor:
    """Compute raw gradients of a scalar target w.r.t. the one-hot input sequence.

    Args:
        model: An :class:`~alphagenome_pytorch.model.AlphaGenome` instance (eval mode).
        dna_sequence: One-hot encoded input, shape ``(B, S, 4)``. Does not need
            ``requires_grad`` set beforehand; this is handled internally.
        organism_index: Organism index, scalar int or ``(B,)`` tensor.
        target_fn: Function mapping the model's forward-pass output dict to a
            ``(B,)`` scalar tensor (e.g. :func:`classification_target`).
        resolutions: Resolutions to compute in the forward pass. Splice heads
            need 1bp embeddings, so this defaults to ``(1,)``.

    Returns:
        Gradient tensor of shape ``(B, S, 4)``, same dtype/device as input.
    """
    device = dna_sequence.device
    batch_size = dna_sequence.shape[0]
    org_idx = _prepare_organism_index(organism_index, batch_size, device)

    x = dna_sequence.detach().clone().float().requires_grad_(True)

    with torch.enable_grad():
        with _autocast_context(model, device):
            outputs = model.forward(x, org_idx, resolutions=resolutions, channels_last=True)
            target = target_fn(outputs)
        grad, = torch.autograd.grad(target.sum(), x)

    return grad.detach()


def input_x_gradient(
    model: nn.Module,
    dna_sequence: torch.Tensor,
    organism_index: "torch.Tensor | int",
    target_fn: TargetFn,
    resolutions: tuple[int, ...] = (1,),
) -> torch.Tensor:
    """Gradient * input attribution, a.k.a. "saliency times input".

    Cheap (single forward/backward pass) approximation to Integrated Gradients
    that works well for one-hot inputs since ``input`` is 0/1.

    Returns:
        Attribution tensor of shape ``(B, S, 4)``.
    """
    grad = compute_gradients(model, dna_sequence, organism_index, target_fn, resolutions=resolutions)
    return grad * dna_sequence.float()


def integrated_gradients(
    model: nn.Module,
    dna_sequence: torch.Tensor,
    organism_index: "torch.Tensor | int",
    target_fn: TargetFn,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 20,
    resolutions: tuple[int, ...] = (1,),
) -> torch.Tensor:
    """Integrated Gradients attribution (Sundararajan et al. 2017).

    Approximates the path integral of gradients along a straight line from
    ``baseline`` to ``dna_sequence`` via a Riemann sum, one forward/backward
    pass per step (steps are *not* batched together, to keep peak memory at
    the same level as a single-sample forward pass on this full-genome model).

    Args:
        baseline: Reference input, same shape as ``dna_sequence``. Defaults to
            an all-zeros sequence (the common baseline for one-hot DNA).
        steps: Number of interpolation steps. More steps → smoother/more
            accurate attribution at proportionally higher compute cost.

    Returns:
        Attribution tensor of shape ``(B, S, 4)``, already scaled by
        ``(input - baseline)``.
    """
    x = dna_sequence.float()
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.float().to(x.device)

    diff = x - baseline
    device = x.device
    batch_size = x.shape[0]
    org_idx = _prepare_organism_index(organism_index, batch_size, device)

    accumulated = torch.zeros_like(x)
    # Trapezoidal rule over alpha in [0, 1], endpoints included.
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i, alpha in enumerate(alphas):
        interpolated = (baseline + alpha * diff).detach().requires_grad_(True)
        with torch.enable_grad():
            with _autocast_context(model, device):
                outputs = model.forward(interpolated, org_idx, resolutions=resolutions, channels_last=True)
                target = target_fn(outputs)
            grad, = torch.autograd.grad(target.sum(), interpolated)
        weight = 0.5 if i in (0, len(alphas) - 1) else 1.0
        accumulated += weight * grad.detach()
    avg_grad = accumulated / steps

    return avg_grad * diff


def smoothgrad(
    model: nn.Module,
    dna_sequence: torch.Tensor,
    organism_index: "torch.Tensor | int",
    target_fn: TargetFn,
    num_samples: int = 20,
    noise_level: float = 0.15,
    method: str = "input_x_gradient",
    resolutions: tuple[int, ...] = (1,),
) -> torch.Tensor:
    """SmoothGrad attribution: average gradients over noisy copies of the input.

    Gaussian noise (std = ``noise_level``) is added directly to the one-hot
    input in continuous space (the perturbed input is *not* renormalized back
    to one-hot); this is the standard SmoothGrad recipe applied to a
    differentiable relaxation of the sequence.

    Args:
        num_samples: Number of noisy samples to average over.
        noise_level: Standard deviation of the additive Gaussian noise.
        method: Either ``"gradient"`` (plain saliency) or ``"input_x_gradient"``
            — which per-sample attribution to average before returning.

    Returns:
        Attribution tensor of shape ``(B, S, 4)``.
    """
    if method not in ("gradient", "input_x_gradient"):
        raise ValueError(f"method must be 'gradient' or 'input_x_gradient', got {method!r}")

    x = dna_sequence.float()
    device = x.device
    batch_size = x.shape[0]
    org_idx = _prepare_organism_index(organism_index, batch_size, device)

    accumulated = torch.zeros_like(x)
    for _ in range(num_samples):
        noisy = (x + noise_level * torch.randn_like(x)).detach().requires_grad_(True)
        with torch.enable_grad():
            with _autocast_context(model, device):
                outputs = model.forward(noisy, org_idx, resolutions=resolutions, channels_last=True)
                target = target_fn(outputs)
            grad, = torch.autograd.grad(target.sum(), noisy)
        grad = grad.detach()
        if method == "input_x_gradient":
            grad = grad * x
        accumulated += grad

    return accumulated / num_samples


# ---------------------------------------------------------------------------
# Summarization / projection helpers
# ---------------------------------------------------------------------------

def summarize_attributions(
    attributions: torch.Tensor,
    dna_sequence: torch.Tensor,
    mode: str = "contribution",
) -> torch.Tensor:
    """Reduce per-channel (A/C/G/T) attributions to a single score per position.

    Args:
        attributions: Attribution tensor, shape ``(B, S, 4)``.
        dna_sequence: One-hot input the attributions were computed for, shape
            ``(B, S, 4)``. Used by the ``"contribution"`` mode to select the
            attribution of the observed base.
        mode:
            - ``"contribution"``: attribution at the observed (reference) base
              only, i.e. ``sum(attributions * dna_sequence, dim=-1)``. This is
              the standard summary for gradient*input / integrated gradients.
            - ``"l2"``: L2 norm across the 4 bases at each position. Useful for
              plain-gradient (saliency) maps where the sign is less meaningful.

    Returns:
        Tensor of shape ``(B, S)``.
    """
    if mode == "contribution":
        return (attributions * dna_sequence.float()).sum(dim=-1)
    elif mode == "l2":
        return attributions.norm(dim=-1)
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'contribution' or 'l2'")


def hypothetical_contributions(attributions: torch.Tensor) -> torch.Tensor:
    """Mean-center attributions across the base axis for sequence-logo plotting.

    Subtracting the per-position mean turns raw attributions into "hypothetical
    contribution scores" (as in DeepLIFT/TF-MoDISco): each base's height shows
    how much it would have contributed relative to the average of A/C/G/T at
    that position, independent of which base is actually present.

    Returns:
        Tensor of shape ``(B, S, 4)``.
    """
    return attributions - attributions.mean(dim=-1, keepdim=True)


__all__ = [
    "TargetFn",
    "classification_target",
    "usage_target",
    "compute_gradients",
    "input_x_gradient",
    "integrated_gradients",
    "smoothgrad",
    "summarize_attributions",
    "hypothetical_contributions",
]
