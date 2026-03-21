"""Splice-site datasets for AlphaGenome fine-tuning.

Provides Dataset classes for loading genomic data together with:
  1. Splice-site classification labels built from an annotation Parquet
     (produced by ``scripts/convert_splice_sites_to_parquet.py``).
  2. Sparse splice-site usage targets built from a Spliser usage Parquet.

The sequence input mirrors :class:`GenomicDataset`: BED intervals are
expanded from their center to *sequence_length* (default 131,072 bp),
matching the standard AlphaGenome input window.

Example
-------
::

    from alphagenome_pytorch.extensions.finetuning.datasets import CachedGenome
    from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
        SpliceSiteAnnotation, SpliceSiteUsageIndex, SpliceSiteDataset,
        collate_splice,
    )
    from torch.utils.data import DataLoader

    genome  = CachedGenome('hg38.fa')
    annot   = SpliceSiteAnnotation('human_splice_sites.parquet')
    usage   = SpliceSiteUsageIndex('/path/to/spliser/Homo_sapiens/', min_coverage=10)

    ds = SpliceSiteDataset(
        genome=genome,
        bed_file='train_regions.bed',
        annotation=annot,
        usage_index=usage,
        organism_index=0,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_splice)
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset

# Class integer → label mapping (matches SpliceSitesClassificationHead)
SITE_TYPE_TO_CLASS: dict[str, int] = {
    "Donor+": 0,
    "Acceptor+": 1,
    "Donor-": 2,
    "Acceptor-": 3,
}
BACKGROUND_CLASS: int = 4


class SpliceSiteAnnotation:
    """In-memory index of splice-site positions for fast interval queries.

    Loads the annotation Parquet produced by
    ``scripts/convert_splice_sites_to_parquet.py`` and builds per-chromosome
    sorted arrays for O(log n) range lookups via :func:`numpy.searchsorted`.

    Args:
        parquet_path: Path to the annotation Parquet file.
    """

    def __init__(self, parquet_path: str | Path) -> None:
        import pandas as pd

        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Annotation Parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        df["Chromosome"] = df["Chromosome"].astype(str)
        df["Position"] = df["Position"].astype(int)

        # Build per-chromosome sorted index
        self._positions: dict[str, np.ndarray] = {}
        self._classes: dict[str, np.ndarray] = {}

        for chrom, grp in df.groupby("Chromosome"):
            grp = grp.sort_values("Position")
            self._positions[chrom] = grp["Position"].to_numpy(dtype=np.int64)
            # Map SiteType string → int class
            self._classes[chrom] = np.array(
                [SITE_TYPE_TO_CLASS.get(st, BACKGROUND_CLASS) for st in grp["SiteType"]],
                dtype=np.int64,
            )

        total = sum(len(v) for v in self._positions.values())
        print(
            f"SpliceSiteAnnotation: loaded {total:,} sites across "
            f"{len(self._positions)} chromosomes from {parquet_path.name}"
        )

    @property
    def chromosomes(self) -> set[str]:
        return set(self._positions.keys())

    def query(self, chrom: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions, classes) within [start, end).

        Args:
            chrom: Chromosome name.
            start: Window start, 0-based inclusive.
            end: Window end, 0-based exclusive.

        Returns:
            Tuple of ``(positions, classes)`` numpy int64 arrays, both
            sorted by position.  Empty arrays if no sites in range.
        """
        if chrom not in self._positions:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        pos = self._positions[chrom]
        lo = int(np.searchsorted(pos, start, side="left"))
        hi = int(np.searchsorted(pos, end, side="left"))
        return pos[lo:hi], self._classes[chrom][lo:hi]


class SpliceSiteUsageIndex:
    """In-memory index of per-condition splice-site usage values.

    Loads a Spliser ``_usage.parquet`` file, applies coverage filters, then
    builds a lookup dictionary keyed by ``(chromosome, position)``.

    Args:
        usage_parquet: Path to the ``_usage.parquet`` file produced by
            ``scripts/convert_splice_usage_to_parquet.py``.  The companion
            JSON metadata file is expected at the same path with a ``.json``
            extension (e.g. ``splice_usage.json`` next to
            ``splice_usage.parquet``).
        min_coverage: Minimum ``Alpha + Beta`` to include a site
            (default: 10).
        alpha_min: Optional minimum ``Alpha`` count.
        usage_coord_base: Coordinate base in the parquet (1 or 0).
            Use ``1`` (default) for Spliser output, which is 1-based.
    """

    def __init__(
        self,
        usage_parquet: str | Path,
        min_coverage: int = 10,
        alpha_min: int | None = None,
        usage_coord_base: int = 1,
    ) -> None:
        import pandas as pd

        usage_parquet = Path(usage_parquet)
        usage_json = usage_parquet.with_suffix(".json")

        for p in (usage_parquet, usage_json):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")

        with open(usage_json) as f:
            metadata = json.load(f)

        self._class_labels: dict[str, int] = metadata["class_labels"]
        self._condition_labels: dict[str, int] = metadata["condition_labels"]
        self.n_conditions: int = len(self._condition_labels)
        none_class = self._class_labels.get("None", 4)

        df = pd.read_parquet(usage_parquet)

        # Filter: exclude "None" class and apply coverage thresholds
        df = df[df["Label"] != none_class].copy()
        df = df[df["Alpha"] + df["Beta"] >= min_coverage]
        if alpha_min is not None:
            df = df[df["Alpha"] >= alpha_min]

        if usage_coord_base == 1:
            df["Position"] = df["Position"] - 1  # → 0-based

        df["Chromosome"] = df["Chromosome"].astype(str)

        # Build lookup: (chrom, pos) → [(condition_idx, sse), ...]
        self._lookup: dict[tuple[str, int], list[tuple[int, float]]] = {}
        for row in df.itertuples(index=False):
            key = (str(row.Chromosome), int(row.Position))
            entry = (int(row.Condition), float(row.SSE))
            self._lookup.setdefault(key, []).append(entry)

        print(
            f"SpliceSiteUsageIndex: loaded {len(self._lookup):,} sites, "
            f"{self.n_conditions} conditions from {usage_parquet.name}"
        )

    @property
    def condition_labels(self) -> dict[str, int]:
        return self._condition_labels

    def query(
        self, chrom: str, positions: np.ndarray
    ) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
        """Return usage data for positions that have observed values.

        Args:
            chrom: Chromosome name.
            positions: Array of 0-based genomic positions to query.

        Returns:
            Tuple of three parallel lists:
            - ``site_positions``: subset of input positions with at least one
              observation.
            - ``values``: per-site float32 array of shape (n_conditions,) with
              SSE values (0.0 for unobserved conditions).
            - ``masks``: per-site bool array of shape (n_conditions,) indicating
              observed conditions.
        """
        site_positions: list[int] = []
        values_list: list[np.ndarray] = []
        masks_list: list[np.ndarray] = []

        for pos in positions:
            key = (chrom, int(pos))
            entries = self._lookup.get(key)
            if not entries:
                continue
            vals = np.zeros(self.n_conditions, dtype=np.float32)
            mask = np.zeros(self.n_conditions, dtype=bool)
            for cond_idx, sse in entries:
                if 0 <= cond_idx < self.n_conditions:
                    vals[cond_idx] = sse
                    mask[cond_idx] = True
            site_positions.append(int(pos))
            values_list.append(vals)
            masks_list.append(mask)

        return site_positions, values_list, masks_list


def _load_intervals_from_bed(
    bed_path: str,
) -> tuple[list[tuple[str, int, int]], set[str]]:
    """Load genomic intervals from a BED file (0-based half-open)."""
    intervals: list[tuple[str, int, int]] = []
    chromosomes: set[str] = set()

    with open(bed_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            intervals.append((chrom, start, end))
            chromosomes.add(chrom)

    return intervals, chromosomes


class SpliceSiteDataset(Dataset):
    """Dataset for fine-tuning AlphaGenome splice-site heads.

    Returns batches with one-hot sequence + classification labels and
    (optionally) sparse usage targets, suitable for
    :func:`collate_splice`.

    Args:
        genome: Pre-built :class:`~alphagenome_pytorch.extensions.finetuning.\
datasets.CachedGenome` instance **or** a path string (FASTA).
            Passing a ``CachedGenome`` shares memory across train/val splits.
        bed_file: BED file defining training windows (chrom, start, end).
            Intervals are expanded from their center to ``sequence_length``.
        annotation: :class:`SpliceSiteAnnotation` built from the output of
            ``scripts/convert_splice_sites_to_parquet.py``.
        usage_index: Optional :class:`SpliceSiteUsageIndex` for per-condition
            usage targets.  When ``None``, the batch will not contain usage
            keys.
        sequence_length: Model input length in bp (default: 131,072).
            Intervals shorter than this are expanded; longer ones are
            truncated from their center.
        organism_index: Organism index passed to the model (0 = human,
            1 = mouse).  All examples in this dataset share the same index.
        max_sites: Maximum number of usage sites returned per window.
            Positions are sorted by genomic coordinate; surplus sites are
            dropped.  Padded with ``-1`` up to ``max_sites``.
    """

    def __init__(
        self,
        genome: Any,   # CachedGenome | str
        bed_file: str,
        annotation: SpliceSiteAnnotation,
        usage_index: SpliceSiteUsageIndex | None = None,
        sequence_length: int = 131_072,
        organism_index: int = 0,
        max_sites: int = 1024,
    ) -> None:
        # Defer heavy import to avoid hard dep at module import time
        from alphagenome_pytorch.extensions.finetuning.datasets import (
            CachedGenome,
            _ensure_genomic_deps,
        )
        from alphagenome_pytorch.utils.sequence import sequence_to_onehot

        self._sequence_to_onehot = sequence_to_onehot
        _ensure_genomic_deps()

        self.sequence_length = sequence_length
        self.organism_index = organism_index
        self.max_sites = max_sites
        self.annotation = annotation
        self.usage_index = usage_index

        # Load BED intervals first so we know which chromosomes are needed
        all_intervals, chromosomes = _load_intervals_from_bed(bed_file)

        # Genome backend – pass chromosome set to avoid loading the full genome
        if isinstance(genome, str) or isinstance(genome, Path):
            self._cached_genome = CachedGenome(str(genome), chromosomes=chromosomes)
        else:
            self._cached_genome = genome  # CachedGenome passed directly

        half = sequence_length // 2
        chrom_sizes = self._cached_genome.chrom_sizes
        self._positions: list[tuple[str, int, int]] = []
        n_skipped = n_truncated = 0

        for chrom, start, end in all_intervals:
            if chrom not in chrom_sizes:
                n_skipped += 1
                continue
            center = (start + end) // 2
            win_start = center - half
            win_end = center + half
            if win_start < 0 or win_end > chrom_sizes[chrom]:
                n_skipped += 1
                continue
            if end - start > sequence_length:
                n_truncated += 1
            self._positions.append((chrom, win_start, win_end))

        if n_skipped:
            warnings.warn(
                f"{n_skipped} intervals skipped (out of chromosome bounds)."
            )
        if n_truncated:
            warnings.warn(
                f"{n_truncated} intervals truncated from center to sequence_length="
                f"{sequence_length}."
            )

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        chrom, win_start, win_end = self._positions[idx]

        # ── Sequence ─────────────────────────────────────────────────────────
        seq_np = self._cached_genome.fetch(chrom, win_start, win_end)  # (S,4) uint8
        sequence = torch.from_numpy(seq_np.astype(np.float32))  # (S,4)

        # ── Classification labels ─────────────────────────────────────────────
        labels = np.full(self.sequence_length, BACKGROUND_CLASS, dtype=np.int64)
        site_pos, site_cls = self.annotation.query(chrom, win_start, win_end)
        if len(site_pos) > 0:
            rel_pos = site_pos - win_start               # window-relative, 0-based
            valid = (rel_pos >= 0) & (rel_pos < self.sequence_length)
            labels[rel_pos[valid]] = site_cls[valid]
        classification_labels = torch.from_numpy(labels)   # (S,) int64

        item: dict[str, Any] = {
            "sequence": sequence,
            "organism_index": torch.tensor(self.organism_index, dtype=torch.long),
            "classification_labels": classification_labels,
        }

        # ── Usage targets (sparse) ────────────────────────────────────────────
        if self.usage_index is not None:
            # Query annotation positions that are within the window
            all_site_pos, _ = self.annotation.query(chrom, win_start, win_end)
            if len(all_site_pos) > 0:
                used_pos, val_arrays, mask_arrays = self.usage_index.query(
                    chrom, all_site_pos
                )
            else:
                used_pos = []
                val_arrays = []
                mask_arrays = []

            n_cond = self.usage_index.n_conditions
            n_used = len(used_pos)
            n_pad = self.max_sites - min(n_used, self.max_sites)

            # Trim to max_sites (keep first max_sites sites, sorted by position)
            used_pos = used_pos[: self.max_sites]
            val_arrays = val_arrays[: self.max_sites]
            mask_arrays = mask_arrays[: self.max_sites]

            rel_positions = np.array(
                [p - win_start for p in used_pos], dtype=np.int64
            )

            # Pad to max_sites with -1 / zeros / False
            if n_pad > 0:
                rel_positions = np.concatenate(
                    [rel_positions, np.full(n_pad, -1, dtype=np.int64)]
                )

            if val_arrays:
                vals_2d = np.stack(val_arrays)         # (n_used, n_cond)
                masks_2d = np.stack(mask_arrays)       # (n_used, n_cond)
            else:
                vals_2d = np.zeros((0, n_cond), dtype=np.float32)
                masks_2d = np.zeros((0, n_cond), dtype=bool)

            if n_pad > 0:
                vals_2d = np.concatenate(
                    [vals_2d, np.zeros((n_pad, n_cond), dtype=np.float32)], axis=0
                )
                masks_2d = np.concatenate(
                    [masks_2d, np.zeros((n_pad, n_cond), dtype=bool)], axis=0
                )

            item["usage_positions"] = torch.from_numpy(rel_positions)   # (max_sites,) int64
            item["usage_values"] = torch.from_numpy(vals_2d)            # (max_sites, n_cond)
            item["usage_mask"] = torch.from_numpy(masks_2d)             # (max_sites, n_cond) bool

        return item


def collate_splice(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate a list of :class:`SpliceSiteDataset` items into a batch dict.

    All tensors are stacked along a new batch dimension.  Usage keys are
    only included when present in every item in the batch.

    Returns:
        Dict with keys ``sequence``, ``organism_index``,
        ``classification_labels``, and optionally ``usage_positions``,
        ``usage_values``, ``usage_mask``.
    """
    result: dict[str, Any] = {
        "sequence": torch.stack([b["sequence"] for b in batch]),
        "organism_index": torch.stack([b["organism_index"] for b in batch]),
        "classification_labels": torch.stack([b["classification_labels"] for b in batch]),
    }
    if "usage_positions" in batch[0]:
        result["usage_positions"] = torch.stack([b["usage_positions"] for b in batch])
        result["usage_values"] = torch.stack([b["usage_values"] for b in batch])
        result["usage_mask"] = torch.stack([b["usage_mask"] for b in batch])
    return result


class SpeciesGroupedSampler(BatchSampler):
    """Batch sampler that ensures every batch contains only one species.

    Required for multi-species training when different species have different
    numbers of usage conditions — ``collate_splice`` cannot stack usage tensors
    of incompatible shapes.

    Works with both a plain :class:`SpliceSiteDataset` and a
    :class:`~torch.utils.data.ConcatDataset` of multiple
    :class:`SpliceSiteDataset` instances.

    Args:
        dataset: A :class:`SpliceSiteDataset` or
            :class:`~torch.utils.data.ConcatDataset` thereof.
        batch_size: Number of samples per batch.
        shuffle: Shuffle both within-species indices and the final batch order.
        seed: Base random seed (combined with *epoch* via :meth:`set_epoch`).
        drop_last: If ``True``, drop the last incomplete batch per species.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

        # Build organism_index → list of global dataset indices
        self.species_indices: dict[int, list[int]] = {}
        if hasattr(dataset, "datasets"):  # ConcatDataset
            offset = 0
            for ds in dataset.datasets:
                org_idx = ds.organism_index
                self.species_indices.setdefault(org_idx, []).extend(
                    range(offset, offset + len(ds))
                )
                offset += len(ds)
        else:
            org_idx = dataset.organism_index
            self.species_indices[org_idx] = list(range(len(dataset)))

    def _make_batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        batches: list[list[int]] = []
        for org_idx, indices in self.species_indices.items():
            idx_arr = np.array(indices, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(idx_arr)
            for i in range(0, len(idx_arr), self.batch_size):
                batch = idx_arr[i : i + self.batch_size].tolist()
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            perm = rng.permutation(len(batches))
            batches = [batches[i] for i in perm]
        return batches

    def __iter__(self):
        yield from self._make_batches()

    def __len__(self) -> int:
        total = 0
        for indices in self.species_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += math.ceil(n / self.batch_size)
        return total

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling; call before each epoch."""
        self._epoch = epoch


class DistributedSpeciesGroupedSampler(SpeciesGroupedSampler):
    """Distributed version of :class:`SpeciesGroupedSampler` for DDP training.

    Partitions batches across ranks so each rank receives an exclusive,
    non-overlapping subset while every batch remains single-species.

    Args:
        dataset: As in :class:`SpeciesGroupedSampler`.
        batch_size: Number of samples per batch.
        num_replicas: Total number of DDP processes.
        rank: Rank of the current process.
        shuffle: As in :class:`SpeciesGroupedSampler`.
        seed: As in :class:`SpeciesGroupedSampler`.
        drop_last: As in :class:`SpeciesGroupedSampler`.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        batches = self._make_batches()
        # Pad so total is divisible by num_replicas
        pad = (-len(batches)) % self.num_replicas
        batches = batches + batches[:pad]
        # Interleave: each rank gets every num_replicas-th batch
        yield from batches[self.rank :: self.num_replicas]

    def __len__(self) -> int:
        return math.ceil(super().__len__() / self.num_replicas)
