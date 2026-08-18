"""Training logging utilities for AlphaGenome fine-tuning.

Provides TrainingLogger for CSV and optional W&B logging with rank awareness.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from alphagenome_pytorch.extensions.finetuning.distributed import is_main_process


class _Tee:
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def fileno(self) -> int:
        return self._streams[0].fileno()

    def isatty(self) -> bool:
        return False


def setup_output_logging(
    output_dir: Path,
    rank: int,
    log_file: "str | Path | None" = None,
) -> None:
    """Tee stdout to a log file on rank 0.

    All subsequent ``print()`` calls (and anything written to *sys.stdout*)
    will be mirrored to the log file in addition to the terminal.
    Safe to call multiple times — subsequent calls on the same process are
    no-ops once the tee is already installed.

    Args:
        output_dir: Directory that already exists (created by caller).
        rank: Process rank; only rank 0 writes the file.
        log_file: Optional explicit path for the log file.  When given it is
            used as-is (absolute or relative); when omitted the log is written
            to ``output_dir/train.log`` (previous default behaviour).
    """
    if not is_main_process(rank):
        return
    # Already tee'd — don't double-wrap.
    if isinstance(sys.stdout, _Tee):
        return
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = Path(output_dir) / "train.log"
    log_fh = open(log_path, "a", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    print(f"Logging to {log_path}")


class TrainingLogger:
    """Logger for training metrics with optional W&B integration.

    Handles both step-level (batch) and epoch-level logging to CSV files
    and optionally to Weights & Biases. Only logs on rank 0 in distributed mode.

    Attributes:
        output_dir: Directory for log files.
        rank: Process rank (0 for main process).
        use_wandb: Whether W&B logging is enabled.
        step: Current step counter.

    Example:
        >>> logger = TrainingLogger(
        ...     output_dir=Path("output"),
        ...     rank=0,
        ...     use_wandb=True,
        ...     wandb_project="my-project",
        ...     config={"lr": 1e-4, "epochs": 10},
        ... )
        >>> logger.log_step({"loss": 0.5, "lr": 1e-4})
        >>> logger.log_epoch(1, train_loss=0.5, val_loss=0.4, lr=1e-4)
        >>> logger.finish()
    """

    def __init__(
        self,
        output_dir: Path,
        rank: int = 0,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        run_name: str | None = None,
        config: dict | None = None,
        resume_id: str | None = None,
    ) -> None:
        """Initialize the training logger.

        Args:
            output_dir: Directory for log files (created if it doesn't exist).
            rank: Process rank for distributed training. Only rank 0 logs.
            use_wandb: Whether to enable Weights & Biases logging.
            wandb_project: W&B project name.
            wandb_entity: W&B entity (team/user).
            run_name: Name for this run.
            config: Configuration dict to save and log to W&B.
            resume_id: W&B run ID for resuming a previous run.
        """
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.use_wandb = use_wandb and is_main_process(rank)
        self.step = 0
        self.resume_id = resume_id

        # Only main process handles logging
        if not is_main_process(rank):
            self.csv_file = None
            self.csv_writer = None
            self._csv_fieldnames = None
            self.wandb = None
            return

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CSV logging
        self.csv_path = self.output_dir / "training_log.csv"
        self.csv_file = None
        self.csv_writer = None
        self._csv_fieldnames: list[str] | None = None
        self._epoch_csv_fieldnames: list[str] | None = None
        self._epoch_rows: list[dict[str, Any]] = []

        # Save config
        if config:
            config_path = self.output_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)
            print(f"Saved config to {config_path}")

        # Initialize W&B if requested
        if self.use_wandb:
            try:
                import wandb

                self.wandb = wandb
                wandb.init(
                    project=wandb_project or "alphagenome-finetune",
                    entity=wandb_entity,
                    name=run_name,
                    config=config,
                    dir=str(self.output_dir),
                    id=resume_id,
                    resume="allow" if resume_id else None,
                )
                print(f"W&B initialized: {wandb.run.url}" + (" (resumed)" if resume_id else ""))
            except ImportError:
                print("Warning: wandb not installed, disabling W&B logging")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def _ensure_csv(self, fieldnames: list[str]) -> None:
        """Initialize CSV file with headers if not already done.

        On resume (an existing non-empty file), adopts the file's actual on-disk
        header rather than trusting only this call's metrics keys — otherwise a
        resumed process whose first logged step has a different key set than the
        one that originally established the file (e.g. after a code change) would
        silently misalign every row appended from that point on.
        """
        if not is_main_process(self.rank):
            return
        if self.csv_writer is None:
            if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
                with open(self.csv_path, newline="") as f:
                    existing_header = next(csv.reader(f), [])
                self._csv_fieldnames = existing_header or fieldnames
                write_header = not existing_header
            else:
                self._csv_fieldnames = fieldnames
                write_header = True
            self.csv_file = open(self.csv_path, "a", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self._csv_fieldnames)
            if write_header:
                self.csv_writer.writeheader()
            self.csv_file.flush()

    def _grow_csv_header(self, new_fields: list[str]) -> None:
        """Add newly-seen fields to the CSV header.

        Rewrites the file once (reading back every row written so far and
        backfilling '' for the new columns on those rows), then reopens the
        append-mode writer bound to the expanded schema. Only runs the first time
        each new metric key appears — not on every step — so it stays cheap even
        for a long, high-frequency step log; the common case (no new keys) is a
        plain append via the existing writer.
        """
        if not is_main_process(self.rank):
            return
        self.csv_file.close()
        with open(self.csv_path, newline="") as f:
            old_rows = list(csv.DictReader(f))
        self._csv_fieldnames = self._csv_fieldnames + new_fields
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writeheader()
            for row in old_rows:
                writer.writerow({k: row.get(k, "") for k in self._csv_fieldnames})
        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self._csv_fieldnames)

    def log_step(self, metrics: dict[str, Any]) -> None:
        """Log metrics for a training step.

        Args:
            metrics: Dictionary of metric name -> value.
                     'step' and 'timestamp' are added automatically.
        """
        if not is_main_process(self.rank):
            return

        self.step += 1
        metrics["step"] = self.step
        metrics["timestamp"] = datetime.now().isoformat()

        # CSV logging
        fieldnames = ["step", "timestamp"] + [
            k for k in sorted(metrics.keys()) if k not in ["step", "timestamp"]
        ]
        self._ensure_csv(fieldnames)

        new_fields = [k for k in fieldnames if k not in self._csv_fieldnames]
        if new_fields:
            self._grow_csv_header(new_fields)

        # Only write fields that exist in the header
        row = {k: v for k, v in metrics.items() if k in self._csv_fieldnames}
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # W&B logging
        if self.use_wandb:
            self.wandb.log(metrics, step=self.step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        is_best: bool = False,
        extra: dict[str, Any] | None = None,
        histograms: dict[str, list[float]] | None = None,
    ) -> None:
        """Log epoch-level metrics.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.
            lr: Current learning rate.
            is_best: Whether this is the best model so far.
            extra: Additional scalar metrics to log.
            histograms: Dict of metric_name -> list of values for histogram logging
                       (only logged to W&B, not CSV).
        """
        if not is_main_process(self.rank):
            return

        metrics: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": lr,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            metrics.update(extra)

        # Append to epoch log (scalars only). A plain per-call DictWriter (the old
        # approach) derives fieldnames fresh from *this* epoch's metrics dict every
        # time — if a later epoch's dict gains a key an earlier one didn't have (e.g.
        # a trajectory-loss term that only turns nonzero once warm-up ends, or a
        # per-species validation column that only appears once that species has
        # data), the row silently gets written with a different field count/order
        # than the header, corrupting column alignment for the whole file with no
        # error raised. Instead: track a fieldname list that only ever grows, and
        # rewrite the file (all rows, backfilling '' for columns a row predates)
        # whenever a new key appears, so the header always matches every row.
        epoch_log_path = self.output_dir / "epoch_log.csv"
        if self._epoch_csv_fieldnames is None:
            # First call this run: adopt an existing file's header + rows (resume
            # case) so we keep appending consistently instead of starting fresh.
            if epoch_log_path.exists() and epoch_log_path.stat().st_size > 0:
                with open(epoch_log_path, newline="") as f:
                    reader = csv.DictReader(f)
                    self._epoch_csv_fieldnames = list(reader.fieldnames or [])
                    self._epoch_rows = [
                        {k: v for k, v in row.items() if k in self._epoch_csv_fieldnames}
                        for row in reader
                    ]
            else:
                self._epoch_csv_fieldnames = []
                self._epoch_rows = []

        new_fields = [k for k in metrics.keys() if k not in self._epoch_csv_fieldnames]
        if new_fields:
            self._epoch_csv_fieldnames = self._epoch_csv_fieldnames + new_fields

        self._epoch_rows.append({k: metrics.get(k, "") for k in self._epoch_csv_fieldnames})
        with open(epoch_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._epoch_csv_fieldnames)
            writer.writeheader()
            writer.writerows(self._epoch_rows)

        # W&B logging
        if self.use_wandb:
            wandb_metrics: dict[str, Any] = {
                "epoch": epoch,
                "epoch/train_loss": train_loss,
                "epoch/val_loss": val_loss,
                "epoch/learning_rate": lr,
            }
            if extra:
                for k, v in extra.items():
                    wandb_metrics[f"epoch/{k}"] = v
            # Log histograms for distributions
            if histograms:
                for k, values in histograms.items():
                    wandb_metrics[f"epoch/{k}"] = self.wandb.Histogram(values)
            self.wandb.log(wandb_metrics, step=self.step)

    @property
    def wandb_run_id(self) -> str | None:
        """Get the current W&B run ID for checkpoint saving."""
        if self.use_wandb and self.wandb and self.wandb.run:
            return self.wandb.run.id
        return None

    def finish(self) -> None:
        """Close logger and finalize W&B run."""
        if self.csv_file:
            self.csv_file.close()
        if self.use_wandb and self.wandb:
            self.wandb.finish()


__all__ = ["TrainingLogger", "setup_output_logging"]
