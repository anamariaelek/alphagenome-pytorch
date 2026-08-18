"""
Unit tests for TrainingLogger's CSV logging (epoch-level and step-level).

Covers the schema-growth bug where a later call's metrics dict gaining a new key
(e.g. a trajectory-loss term that only turns nonzero once warm-up ends, or a
per-species validation column that only appears once that species has data) used
to either silently corrupt column alignment (`log_epoch`, which built a brand-new
`csv.DictWriter` from each call's own keys with no memory of the original header)
or silently drop the new metric from the file forever (`log_step`, whose fieldname
set was locked in from the very first call and never grew).
"""

import csv

import pytest

from alphagenome_pytorch.extensions.finetuning.logging import TrainingLogger


@pytest.mark.unit
class TestEpochLogCsv:
    def _read_csv(self, path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames), list(reader)

    def test_new_key_mid_run_grows_header_without_corrupting_earlier_rows(self, tmp_path):
        logger = TrainingLogger(output_dir=str(tmp_path), use_wandb=False)

        logger.log_epoch(1, 0.5, 0.4, 1e-4, is_best=True, extra={"train_bce_loss": 0.5})
        logger.log_epoch(2, 0.45, 0.38, 9e-5, is_best=True, extra={"train_bce_loss": 0.45})
        # epoch 3 introduces a key the earlier epochs never had (e.g. warm-up ending)
        logger.log_epoch(
            3, 0.40, 0.35, 8e-5, is_best=True,
            extra={"train_bce_loss": 0.40, "train_usage_mse_delta_loss": 0.02},
        )
        logger.finish()

        fieldnames, rows = self._read_csv(tmp_path / "epoch_log.csv")

        assert "train_usage_mse_delta_loss" in fieldnames
        assert len(rows) == 3
        for row in rows:
            assert set(row.keys()) == set(fieldnames)  # every row aligned to the header

        assert rows[0]["train_usage_mse_delta_loss"] == ""  # backfilled, not misaligned
        assert rows[1]["train_usage_mse_delta_loss"] == ""
        assert rows[2]["train_usage_mse_delta_loss"] == "0.02"
        assert rows[2]["epoch"] == "3"

    def test_resumed_logger_adopts_existing_header_and_keeps_growing_it(self, tmp_path):
        logger1 = TrainingLogger(output_dir=str(tmp_path), use_wandb=False)
        logger1.log_epoch(1, 0.5, 0.4, 1e-4, is_best=True, extra={"train_bce_loss": 0.5})
        logger1.finish()

        # simulate a fresh process resuming into the same output dir
        logger2 = TrainingLogger(output_dir=str(tmp_path), use_wandb=False)
        logger2.log_epoch(
            2, 0.45, 0.38, 9e-5, is_best=True,
            extra={"train_bce_loss": 0.45, "train_usage_mse_delta_loss": 0.02},
        )
        logger2.finish()

        fieldnames, rows = self._read_csv(tmp_path / "epoch_log.csv")

        assert len(rows) == 2
        assert rows[0]["epoch"] == "1"
        assert rows[0]["train_usage_mse_delta_loss"] == ""
        assert rows[1]["epoch"] == "2"
        assert rows[1]["train_usage_mse_delta_loss"] == "0.02"


@pytest.mark.unit
class TestStepLogCsv:
    def _read_csv(self, path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames), list(reader)

    def test_new_key_mid_run_grows_header_instead_of_being_dropped(self, tmp_path):
        logger = TrainingLogger(output_dir=str(tmp_path), use_wandb=False)

        logger.log_step({"train_loss": 0.5, "train_bce_loss": 0.5})
        logger.log_step({"train_loss": 0.45, "train_bce_loss": 0.45})
        # step 3 introduces a key the earlier steps never had
        logger.log_step({
            "train_loss": 0.40, "train_bce_loss": 0.40,
            "train_usage_trajectory_loss": 0.02,
        })
        logger.finish()

        fieldnames, rows = self._read_csv(tmp_path / "training_log.csv")

        assert "train_usage_trajectory_loss" in fieldnames
        assert len(rows) == 3
        for row in rows:
            assert set(row.keys()) == set(fieldnames)  # every row aligned to the header

        assert rows[0]["train_usage_trajectory_loss"] == ""  # backfilled, not dropped
        assert rows[1]["train_usage_trajectory_loss"] == ""
        assert rows[2]["train_usage_trajectory_loss"] == "0.02"

    def test_resumed_logger_with_mismatched_first_call_stays_aligned(self, tmp_path):
        logger1 = TrainingLogger(output_dir=str(tmp_path), use_wandb=False)
        logger1.log_step({"train_loss": 0.5, "train_bce_loss": 0.5})
        logger1.log_step({"train_loss": 0.45, "train_bce_loss": 0.45})
        logger1.finish()

        # simulate a fresh process resuming, whose very first logged step already
        # has a key the on-disk header doesn't
        logger2 = TrainingLogger(output_dir=str(tmp_path), use_wandb=False)
        logger2.log_step({
            "train_loss": 0.40, "train_bce_loss": 0.40,
            "train_usage_trajectory_loss": 0.02,
        })
        logger2.finish()

        fieldnames, rows = self._read_csv(tmp_path / "training_log.csv")

        assert "train_usage_trajectory_loss" in fieldnames
        assert len(rows) == 3
        for row in rows:
            assert set(row.keys()) == set(fieldnames)
        assert rows[0]["train_usage_trajectory_loss"] == ""
        assert rows[1]["train_usage_trajectory_loss"] == ""
        assert rows[2]["train_usage_trajectory_loss"] == "0.02"
