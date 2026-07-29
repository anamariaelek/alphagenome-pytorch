"""Unit tests for alphagenome_pytorch.attributions.

Uses a lightweight linear mock model (instead of the full AlphaGenome) so
gradient-based attributions can be checked against closed-form expectations
without paying the cost of a full 131kb forward/backward pass.
"""

import pytest
import torch
import torch.nn as nn

from alphagenome_pytorch.attributions import (
    classification_target,
    usage_target,
    compute_gradients,
    input_x_gradient,
    integrated_gradients,
    smoothgrad,
    summarize_attributions,
    hypothetical_contributions,
)


class LinearMockModel(nn.Module):
    """A model whose 'splice_sites_classification'/'splice_sites_usage' logits
    are an exact linear function of the input, so gradients are constant and
    attribution methods have closed-form expected outputs.
    """

    def __init__(self, seq_len=16, n_classes=5, n_conditions=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # Fixed (non-trainable) per-position, per-base, per-class weights.
        self.cls_weight = torch.randn(seq_len, 4, n_classes, generator=g)
        self.usage_weight = torch.randn(seq_len, 4, n_conditions, generator=g)

    def forward(self, dna_sequence, organism_index, resolutions=(1,), channels_last=True):
        # dna_sequence: (B, S, 4)
        cls_logits = torch.einsum("bsc,scn->bsn", dna_sequence, self.cls_weight)
        usage_logits = torch.einsum("bsc,scn->bsn", dna_sequence, self.usage_weight)
        return {
            "splice_sites_classification": {"logits": cls_logits, "probs": cls_logits.softmax(-1)},
            "splice_sites_usage": {"logits": usage_logits, "predictions": usage_logits.sigmoid()},
        }


@pytest.fixture
def model():
    return LinearMockModel()


@pytest.fixture
def onehot_sequence():
    torch.manual_seed(0)
    seq_len = 16
    idx = torch.randint(0, 4, (1, seq_len))
    return torch.nn.functional.one_hot(idx, num_classes=4).float()


def test_classification_target_selects_correct_scalar(model, onehot_sequence):
    outputs = model(onehot_sequence, organism_index=0)
    target_fn = classification_target(position=3, class_idx=2)
    expected = outputs["splice_sites_classification"]["logits"][:, 3, 2]
    assert torch.allclose(target_fn(outputs), expected)


def test_usage_target_selects_correct_scalar(model, onehot_sequence):
    outputs = model(onehot_sequence, organism_index=0)
    target_fn = usage_target(position=5, condition_idx=1)
    expected = outputs["splice_sites_usage"]["logits"][:, 5, 1]
    assert torch.allclose(target_fn(outputs), expected)


def test_compute_gradients_matches_linear_weight(model, onehot_sequence):
    position, class_idx = 4, 0
    target_fn = classification_target(position=position, class_idx=class_idx)
    grad = compute_gradients(model, onehot_sequence, 0, target_fn)

    assert grad.shape == onehot_sequence.shape
    # For a linear model, d(logit)/d(input) at `position` is exactly the weight
    # column for that position/class; all other positions have zero gradient.
    expected_at_pos = model.cls_weight[position, :, class_idx]
    assert torch.allclose(grad[0, position], expected_at_pos, atol=1e-5)
    other_positions = torch.arange(grad.shape[1]) != position
    assert torch.allclose(grad[0, other_positions], torch.zeros_like(grad[0, other_positions]))


def test_input_x_gradient_shape_and_zero_baseline_contribution(model, onehot_sequence):
    target_fn = classification_target(position=2, class_idx=1)
    attr = input_x_gradient(model, onehot_sequence, 0, target_fn)
    assert attr.shape == onehot_sequence.shape

    # summarize with "contribution" should equal the target logit's dependence
    # on only the observed base, i.e. grad_at_observed_base * 1.
    contribution = summarize_attributions(attr, onehot_sequence, mode="contribution")
    grad = compute_gradients(model, onehot_sequence, 0, target_fn)
    expected = (grad * onehot_sequence).sum(-1)
    assert torch.allclose(contribution, expected, atol=1e-5)


def test_integrated_gradients_matches_input_x_gradient_for_linear_model(model, onehot_sequence):
    # Gradients of a linear function are constant along the straight-line path,
    # so IG (zero baseline) should reduce exactly to grad * input regardless of steps.
    target_fn = classification_target(position=7, class_idx=3)
    ig = integrated_gradients(model, onehot_sequence, 0, target_fn, steps=5)
    ixg = input_x_gradient(model, onehot_sequence, 0, target_fn)
    assert torch.allclose(ig, ixg, atol=1e-4)


def test_integrated_gradients_respects_custom_baseline(model, onehot_sequence):
    target_fn = classification_target(position=1, class_idx=0)
    baseline = torch.full_like(onehot_sequence, 0.25)  # uniform "no information" baseline
    ig = integrated_gradients(model, onehot_sequence, 0, target_fn, baseline=baseline, steps=8)

    grad = compute_gradients(model, onehot_sequence, 0, target_fn)  # constant grad (linear model)
    expected = grad * (onehot_sequence - baseline)
    assert torch.allclose(ig, expected, atol=1e-4)


def test_smoothgrad_shape_and_low_noise_matches_input_x_gradient(model, onehot_sequence):
    target_fn = classification_target(position=6, class_idx=2)
    torch.manual_seed(42)
    sg = smoothgrad(
        model, onehot_sequence, 0, target_fn,
        num_samples=30, noise_level=1e-6, method="input_x_gradient",
    )
    ixg = input_x_gradient(model, onehot_sequence, 0, target_fn)
    assert sg.shape == onehot_sequence.shape
    assert torch.allclose(sg, ixg, atol=1e-3)


def test_smoothgrad_invalid_method_raises(model, onehot_sequence):
    target_fn = classification_target(position=0, class_idx=0)
    with pytest.raises(ValueError):
        smoothgrad(model, onehot_sequence, 0, target_fn, method="bogus")


def test_summarize_attributions_l2_matches_norm(model, onehot_sequence):
    target_fn = classification_target(position=3, class_idx=1)
    attr = input_x_gradient(model, onehot_sequence, 0, target_fn)
    l2 = summarize_attributions(attr, onehot_sequence, mode="l2")
    assert torch.allclose(l2, attr.norm(dim=-1))


def test_summarize_attributions_invalid_mode_raises(model, onehot_sequence):
    target_fn = classification_target(position=0, class_idx=0)
    attr = input_x_gradient(model, onehot_sequence, 0, target_fn)
    with pytest.raises(ValueError):
        summarize_attributions(attr, onehot_sequence, mode="bogus")


def test_hypothetical_contributions_are_mean_centered(model, onehot_sequence):
    target_fn = classification_target(position=3, class_idx=1)
    attr = input_x_gradient(model, onehot_sequence, 0, target_fn)
    hyp = hypothetical_contributions(attr)
    assert torch.allclose(hyp.mean(dim=-1), torch.zeros(hyp.shape[:-1]), atol=1e-6)


def test_organism_index_as_tensor_batch(model):
    torch.manual_seed(1)
    seq = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 16)), num_classes=4).float()
    org_idx = torch.tensor([0, 1])
    target_fn = classification_target(position=0, class_idx=0)
    grad = compute_gradients(model, seq, org_idx, target_fn)
    assert grad.shape == seq.shape
