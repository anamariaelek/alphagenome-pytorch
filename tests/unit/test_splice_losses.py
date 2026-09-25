"""Unit tests for splice_losses module."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import pytest


# ─── splice_classification_loss ──────────────────────────────────────────────


class TestSpliceClassificationLoss:
    def test_uniform_logits_loss_magnitude(self):
        """With uniform logits and all-background labels, loss ≈ log(5)."""
        from alphagenome_pytorch.extensions.finetuning.splice_losses import (
            splice_classification_loss,
        )

        B, S = 2, 100
        logits = torch.zeros(B, S, 5)   # uniform → prob = 0.2 each
        labels = torch.full((B, S), 4, dtype=torch.long)  # all background

        loss, metrics = splice_classification_loss(logits, labels)
        expected = math.log(5)
        assert abs(loss.item() - expected) < 0.01
        assert "accuracy" in metrics
        assert "acc_cls4" in metrics

    def test_class_weights_change_loss(self):
        """Class weights alter the aggregate loss when per-class CEs differ."""
        from alphagenome_pytorch.extensions.finetuning.splice_losses import (
            splice_classification_loss,
        )

        B, S = 1, 4
        # Very confident correct logits for class 0 (CE ≈ 0);
        # uniform logits for class 4 targets (CE = log(5) ≈ 1.609).
        # The two CEs are very different → weighting changes the mean.
        logits_c0 = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])
        logits_c4 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        logits = torch.stack([logits_c0, logits_c4, logits_c0, logits_c4]).unsqueeze(0)
        labels = torch.tensor([[0, 4, 0, 4]])

        loss_unweighted, _ = splice_classification_loss(logits, labels)

        # Heavily upweight class 0 (low CE) vs class 4 (high CE)
        weights = torch.tensor([10.0, 1.0, 1.0, 1.0, 0.1])
        loss_weighted, _ = splice_classification_loss(logits, labels, class_weights=weights)
        assert not torch.isclose(loss_unweighted, loss_weighted)

    def test_metrics_keys_complete(self):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import (
            splice_classification_loss,
        )

        B, S = 1, 20
        logits = torch.randn(B, S, 5)
        labels = torch.randint(0, 5, (B, S))
        _, metrics = splice_classification_loss(logits, labels)
        for key in ["accuracy", "acc_cls0", "acc_cls1", "acc_cls2", "acc_cls3", "acc_cls4"]:
            assert key in metrics, f"Missing key: {key}"

    def test_perfect_predictions_high_accuracy(self):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import (
            splice_classification_loss,
        )

        B, S = 1, 10
        labels = torch.arange(S).remainder(5)
        # Make logits confidently correct
        logits = torch.full((B, S, 5), -100.0)
        for i in range(S):
            logits[0, i, labels[i].item()] = 100.0
        _, metrics = splice_classification_loss(logits.unsqueeze(0).squeeze(0), labels)
        # Accuracy should be 1.0 (or very close)
        # Note: squeeze above is no-op; fix properly:
        logits = torch.full((B, S, 5), -100.0)
        for i in range(S):
            logits[0, i, labels[i].item()] = 100.0
        _, metrics = splice_classification_loss(logits, labels.unsqueeze(0))
        assert metrics["accuracy"] > 0.99

    def test_gradient_flows(self):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import (
            splice_classification_loss,
        )

        B, S = 1, 16
        logits = torch.randn(B, S, 5, requires_grad=True)
        labels = torch.randint(0, 5, (B, S))
        loss, _ = splice_classification_loss(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


# ─── splice_usage_loss ───────────────────────────────────────────────────────


class TestSpliceUsageLoss:
    def test_no_valid_positions_returns_zero(self):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 2, 1000, 3, 10
        predictions = torch.rand(B, S, n_cond)
        usage_positions = torch.full((B, max_sites), -1, dtype=torch.long)
        usage_values = torch.zeros(B, max_sites, n_cond, 1)
        usage_mask = torch.zeros(B, max_sites, n_cond, 1, dtype=torch.bool)

        loss, _ = splice_usage_loss(predictions, usage_positions, usage_values, usage_mask)
        assert loss.item() == 0.0

    def test_single_valid_observation(self):
        """A single valid observation should produce a non-zero loss."""
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 500, 2, 4
        # splice_usage_loss expects sigmoid outputs (values in [0,1])
        predictions = torch.full((B, S, n_cond), 0.5)
        predictions[0, 100, 0] = 0.9   # high probability at the target position

        usage_positions = torch.full((B, max_sites), -1, dtype=torch.long)
        usage_positions[0, 0] = 100  # single valid position
        usage_values = torch.zeros(B, max_sites, n_cond, 1)
        usage_values[0, 0, 0, 0] = 1.0   # target = 1.0
        usage_mask = torch.zeros(B, max_sites, n_cond, 1, dtype=torch.bool)
        usage_mask[0, 0, 0, 0] = True     # only (batch=0, site=0, cond=0, rep=0) observed

        loss, _ = splice_usage_loss(predictions, usage_positions, usage_values, usage_mask)
        # BCE(0.9, 1.0) = -log(0.9) ≈ 0.105
        assert loss.item() > 0.0
        assert torch.isfinite(loss)

    def test_target_zero_at_high_pred_high_loss(self):
        """High prediction with target=0 should give high BCE loss."""
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 100, 1, 2
        # predictions are raw LOGITS (the loss applies sigmoid internally); a large
        # positive logit -> prob ~1, so target=0 gives a high BCE.
        predictions = torch.full((B, S, n_cond), 7.0)

        usage_positions = torch.full((B, max_sites), -1, dtype=torch.long)
        usage_positions[0, 0] = 50
        usage_values = torch.zeros(B, max_sites, n_cond, 1)  # target = 0
        usage_mask = torch.zeros(B, max_sites, n_cond, 1, dtype=torch.bool)
        usage_mask[0, 0, 0, 0] = True

        loss, _ = splice_usage_loss(predictions, usage_positions, usage_values, usage_mask)
        # BCE_with_logits(7.0, 0) = softplus(7.0) ≈ 7.0
        assert loss.item() > 3.0

    def test_gradient_flows_through_usage_loss(self):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 200, 2, 3
        # splice_usage_loss expects sigmoid outputs in [0,1]
        raw = torch.randn(B, S, n_cond)
        predictions = torch.sigmoid(raw).detach().requires_grad_(True)
        usage_positions = torch.tensor([[50, 120, -1]])
        usage_values = torch.rand(B, max_sites, n_cond, 1)
        usage_mask = torch.zeros(B, max_sites, n_cond, 1, dtype=torch.bool)
        usage_mask[0, 0, :, 0] = True
        usage_mask[0, 1, 0, 0] = True

        loss, _ = splice_usage_loss(predictions, usage_positions, usage_values, usage_mask)
        loss.backward()
        assert predictions.grad is not None

    def test_gradient_zero_loss_still_has_grad(self):
        """Even the zero-loss path should produce a valid gradient graph."""
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 100, 2, 4
        predictions = torch.randn(B, S, n_cond, requires_grad=True)
        usage_positions = torch.full((B, max_sites), -1, dtype=torch.long)
        usage_values = torch.zeros(B, max_sites, n_cond, 1)
        usage_mask = torch.zeros(B, max_sites, n_cond, 1, dtype=torch.bool)

        loss, _ = splice_usage_loss(predictions, usage_positions, usage_values, usage_mask)
        assert loss.item() == 0.0
        # should not raise on backward
        loss.backward()

    def test_trajectory_pearson_penalizes_flat_trajectory(self):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 64, 3, 1
        usage_positions = torch.tensor([[12]])
        usage_values = torch.tensor([[[0.0, 0.5, 1.0]]]).unsqueeze(-1)
        usage_mask = torch.ones(B, max_sites, n_cond, 1, dtype=torch.bool)

        flat_predictions = torch.zeros(B, S, n_cond)
        flat_predictions[0, 12, :] = 0.0

        shaped_predictions = torch.zeros(B, S, n_cond)
        shaped_predictions[0, 12, :] = torch.tensor([-3.0, 0.0, 3.0])

        flat_loss, flat_metrics = splice_usage_loss(
            flat_predictions,
            usage_positions,
            usage_values,
            usage_mask,
            usage_loss_weights={"trajectory_pearson": 1.0},
        )
        shaped_loss, shaped_metrics = splice_usage_loss(
            shaped_predictions,
            usage_positions,
            usage_values,
            usage_mask,
            usage_loss_weights={"trajectory_pearson": 1.0},
        )

        assert shaped_loss.item() < flat_loss.item()
        assert flat_metrics["trajectory_corr"] <= shaped_metrics["trajectory_corr"]

    def test_delta_mse_penalizes_magnitude_on_dynamic_trajectory(self):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 64, 3, 1
        usage_positions = torch.tensor([[12]])
        usage_values = torch.tensor([[[0.0, 0.5, 1.0]]]).unsqueeze(-1)  # clearly dynamic trajectory
        usage_mask = torch.ones(B, max_sites, n_cond, 1, dtype=torch.bool)

        exact_logits = torch.zeros(B, S, n_cond)
        exact_logits[0, 12, :] = torch.logit(usage_values[0, 0, :, 0].clamp(1e-4, 1 - 1e-4))

        wrong_magnitude = torch.zeros(B, S, n_cond)
        wrong_magnitude[0, 12, :] = torch.logit(torch.tensor([0.45, 0.5, 0.55]))  # right shape, tiny magnitude

        exact_loss, exact_metrics = splice_usage_loss(
            exact_logits, usage_positions, usage_values, usage_mask,
            usage_loss_weights={"delta_mse": 1.0},
        )
        wrong_loss, _ = splice_usage_loss(
            wrong_magnitude, usage_positions, usage_values, usage_mask,
            usage_loss_weights={"delta_mse": 1.0},
        )

        assert exact_loss.item() < wrong_loss.item()
        assert exact_metrics["n_trajectory_sites"] == 1

    def test_delta_mse_ignores_flat_and_noisy_trajectories(self):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 64, 9, 2
        usage_positions = torch.tensor([[12, 20]])
        # site 0: near-flat/wobbly (should be excluded); site 1: single-outlier spike (should
        # also be excluded even though its raw variance/range is large).
        usage_values = torch.tensor([[
            [0.50, 0.48, 0.51, 0.49, 0.50, 0.52, 0.49, 0.50, 0.51],
            [0.78, 0.80, 0.81, 0.75, 0.05, 0.79, 0.75, 0.67, 0.70],
        ]]).unsqueeze(-1)
        usage_mask = torch.ones(B, max_sites, n_cond, 1, dtype=torch.bool)
        predictions = torch.zeros(B, S, n_cond)  # arbitrary; loss should be exactly 0 (no eligible sites)

        loss, metrics = splice_usage_loss(
            predictions, usage_positions, usage_values, usage_mask,
            usage_loss_weights={"delta_mse": 1.0},
        )
        assert loss.item() == 0.0
        assert metrics["n_trajectory_sites"] == 0

    # ─── Replicate-level BCE ────────────────────────────────────────────────

    def test_bce_sums_every_replicate_independently(self):
        """Every real replicate observation should contribute its own BCE term,
        not be collapsed to a single pooled value -- with a single replicate
        per (site,condition), this reduces to the pre-replicate behavior."""
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 50, 1, 1
        predictions = torch.full((B, S, n_cond), 2.0)  # sigmoid(2.0) ≈ 0.881
        usage_positions = torch.tensor([[10]])

        # Two replicates at the same (site, condition) with different SSE.
        usage_values = torch.tensor([[[[0.2, 0.8]]]])   # (B, max_sites, n_cond, max_rep)
        usage_mask = torch.ones(B, max_sites, n_cond, 2, dtype=torch.bool)

        loss_two_rep, metrics_two_rep = splice_usage_loss(
            predictions, usage_positions, usage_values, usage_mask,
            usage_loss_weights={"bce": 1.0},
        )
        # Manually average the two replicates' unweighted BCE terms (equal coverage
        # implied since usage_coverage=None -> uniform weight => plain mean).
        expected = F.binary_cross_entropy_with_logits(
            torch.tensor([2.0, 2.0]), torch.tensor([0.2, 0.8]), reduction="mean")
        assert loss_two_rep.item() == pytest.approx(expected.item(), rel=1e-5)
        assert metrics_two_rep["n_valid"] == 2   # replicate-level count, not condition-level

    def test_bce_coverage_weighting_favors_higher_coverage_replicate(self):
        """With usage_coverage supplied, a high-coverage replicate should pull
        the weighted-mean BCE closer to its own (unweighted) value than an
        equal-weight average would."""
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 50, 1, 1
        usage_positions = torch.tensor([[10]])
        usage_values = torch.tensor([[[[0.1, 0.9]]]])
        usage_mask = torch.ones(B, max_sites, n_cond, 2, dtype=torch.bool)

        # A non-zero logit so each replicate's individual BCE term actually
        # differs (BCE_with_logits(0, y) is identically log(2) regardless of y,
        # which would make any weighting scheme indistinguishable).
        predictions = torch.full((B, S, n_cond), 2.0)

        # Replicate 0 (target 0.1) has far more reads than replicate 1 (target 0.9).
        usage_coverage_high_first = torch.tensor([[[[100.0, 5.0]]]])
        usage_coverage_high_second = torch.tensor([[[[5.0, 100.0]]]])

        loss_a, _ = splice_usage_loss(
            predictions, usage_positions, usage_values, usage_mask,
            usage_loss_weights={"bce": 1.0}, usage_coverage=usage_coverage_high_first)
        loss_b, _ = splice_usage_loss(
            predictions, usage_positions, usage_values, usage_mask,
            usage_loss_weights={"bce": 1.0}, usage_coverage=usage_coverage_high_second)
        unweighted, _ = splice_usage_loss(
            predictions, usage_positions, usage_values, usage_mask,
            usage_loss_weights={"bce": 1.0})

        bce_low = F.binary_cross_entropy_with_logits(torch.tensor(2.0), torch.tensor(0.1)).item()
        bce_high = F.binary_cross_entropy_with_logits(torch.tensor(2.0), torch.tensor(0.9)).item()

        # loss_a is weighted toward matching target 0.1 -> closer to bce_low than
        # the unweighted 50/50 mean is; loss_b toward 0.9 -> closer to bce_high.
        assert abs(loss_a.item() - bce_low) < abs(unweighted.item() - bce_low)
        assert abs(loss_b.item() - bce_high) < abs(unweighted.item() - bce_high)
        assert loss_a.item() != pytest.approx(loss_b.item(), rel=1e-4)

    def test_bce_reduces_to_pooled_mean_when_replicates_agree(self):
        """When all replicates at a (site,condition) agree exactly, coverage
        weighting shouldn't matter and this should equal the single-value BCE."""
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 50, 1, 1
        predictions = torch.full((B, S, n_cond), 1.5)
        usage_positions = torch.tensor([[10]])
        usage_values = torch.full((B, max_sites, n_cond, 3), 0.7)
        usage_mask = torch.ones(B, max_sites, n_cond, 3, dtype=torch.bool)
        usage_coverage = torch.tensor([[[[3.0, 40.0, 17.0]]]])

        loss, _ = splice_usage_loss(
            predictions, usage_positions, usage_values, usage_mask,
            usage_loss_weights={"bce": 1.0}, usage_coverage=usage_coverage)
        expected = F.binary_cross_entropy_with_logits(torch.tensor(1.5), torch.tensor(0.7))
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    # ─── tissue_contrast ────────────────────────────────────────────────────

    def test_tissue_contrast_rewards_matching_marker_enrichment(self):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, max_sites = 1, 64, 1
        # Two tissues x two timepoints each = 4 conditions. Tissue 0 (conds 0,1) is
        # a true marker (usage ~0.9, well above tissue 1's ~0.1) -- true contrast
        # ~0.8, comfortably clearing the default marker_floor=0.25.
        n_cond = 4
        usage_positions = torch.tensor([[12]])
        usage_values = torch.tensor([[[[0.9], [0.85], [0.1], [0.15]]]])  # (B,1,4,1)
        usage_mask = torch.ones(B, max_sites, n_cond, 1, dtype=torch.bool)
        groups = [[0, 1], [2, 3]]

        matching_logits = torch.zeros(B, S, n_cond)
        matching_logits[0, 12, :] = torch.logit(torch.tensor([0.9, 0.85, 0.1, 0.15]).clamp(1e-4, 1 - 1e-4))

        compressed_logits = torch.zeros(B, S, n_cond)
        compressed_logits[0, 12, :] = torch.logit(torch.tensor([0.55, 0.55, 0.45, 0.45]))  # right sign, flat

        match_loss, match_metrics = splice_usage_loss(
            matching_logits, usage_positions, usage_values, usage_mask,
            usage_loss_weights={"tissue_contrast": 1.0}, tissue_cond_groups=groups,
        )
        compressed_loss, _ = splice_usage_loss(
            compressed_logits, usage_positions, usage_values, usage_mask,
            usage_loss_weights={"tissue_contrast": 1.0}, tissue_cond_groups=groups,
        )

        assert match_loss.item() < compressed_loss.item()
        assert match_metrics["n_marker_sites"] >= 1

    def test_tissue_contrast_ignores_non_marker_sites(self):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, max_sites, n_cond = 1, 64, 1, 4
        usage_positions = torch.tensor([[12]])
        # Both tissues at a similar usage level -- no marker, true contrast near 0.
        usage_values = torch.tensor([[[[0.5], [0.52], [0.48], [0.5]]]])
        usage_mask = torch.ones(B, max_sites, n_cond, 1, dtype=torch.bool)
        groups = [[0, 1], [2, 3]]
        predictions = torch.zeros(B, S, n_cond)  # arbitrary

        loss, metrics = splice_usage_loss(
            predictions, usage_positions, usage_values, usage_mask,
            usage_loss_weights={"tissue_contrast": 1.0}, tissue_cond_groups=groups,
        )
        assert loss.item() == 0.0
        assert metrics["n_marker_sites"] == 0
