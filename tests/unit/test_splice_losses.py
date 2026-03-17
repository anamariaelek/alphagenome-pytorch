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
        usage_values = torch.zeros(B, max_sites, n_cond)
        usage_mask = torch.zeros(B, max_sites, n_cond, dtype=torch.bool)

        loss = splice_usage_loss(predictions, usage_positions, usage_values, usage_mask)
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
        usage_values = torch.zeros(B, max_sites, n_cond)
        usage_values[0, 0, 0] = 1.0   # target = 1.0
        usage_mask = torch.zeros(B, max_sites, n_cond, dtype=torch.bool)
        usage_mask[0, 0, 0] = True     # only (batch=0, site=0, cond=0) observed

        loss = splice_usage_loss(predictions, usage_positions, usage_values, usage_mask)
        # BCE(0.9, 1.0) = -log(0.9) ≈ 0.105
        assert loss.item() > 0.0
        assert torch.isfinite(loss)

    def test_target_zero_at_high_pred_high_loss(self):
        """High prediction with target=0 should give high BCE loss."""
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 100, 1, 2
        # Use a value close to 1 (valid sigmoid range)
        predictions = torch.full((B, S, n_cond), 0.999)

        usage_positions = torch.full((B, max_sites), -1, dtype=torch.long)
        usage_positions[0, 0] = 50
        usage_values = torch.zeros(B, max_sites, n_cond)  # target = 0
        usage_mask = torch.zeros(B, max_sites, n_cond, dtype=torch.bool)
        usage_mask[0, 0, 0] = True

        loss = splice_usage_loss(predictions, usage_positions, usage_values, usage_mask)
        # BCE(0.999, 0) = -log(1 - 0.999) = -log(0.001) ≈ 6.9
        assert loss.item() > 3.0

    def test_gradient_flows_through_usage_loss(self):
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 200, 2, 3
        # splice_usage_loss expects sigmoid outputs in [0,1]
        raw = torch.randn(B, S, n_cond)
        predictions = torch.sigmoid(raw).detach().requires_grad_(True)
        usage_positions = torch.tensor([[50, 120, -1]])
        usage_values = torch.rand(B, max_sites, n_cond)
        usage_mask = torch.zeros(B, max_sites, n_cond, dtype=torch.bool)
        usage_mask[0, 0, :] = True
        usage_mask[0, 1, 0] = True

        loss = splice_usage_loss(predictions, usage_positions, usage_values, usage_mask)
        loss.backward()
        assert predictions.grad is not None

    def test_gradient_zero_loss_still_has_grad(self):
        """Even the zero-loss path should produce a valid gradient graph."""
        from alphagenome_pytorch.extensions.finetuning.splice_losses import splice_usage_loss

        B, S, n_cond, max_sites = 1, 100, 2, 4
        predictions = torch.randn(B, S, n_cond, requires_grad=True)
        usage_positions = torch.full((B, max_sites), -1, dtype=torch.long)
        usage_values = torch.zeros(B, max_sites, n_cond)
        usage_mask = torch.zeros(B, max_sites, n_cond, dtype=torch.bool)

        loss = splice_usage_loss(predictions, usage_positions, usage_values, usage_mask)
        assert loss.item() == 0.0
        # should not raise on backward
        loss.backward()
