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
    builds a numpy-based columnar index for fork-safe lookups in DataLoader
    workers.  Reading from numpy arrays does not trigger CPython reference-count
    updates, eliminating the copy-on-write page-duplication that the previous
    Python-dict representation caused in persistent worker processes.

    Args:
        usage_parquet: Path to the ``_usage.parquet`` file produced by
            ``scripts/convert_splice_usage_to_parquet.py``.  The companion
            JSON metadata file is expected at the same path with a ``.json``
            extension (e.g. ``splice_usage.json`` next to
            ``splice_usage.parquet``).
        min_coverage: Optional minimum ``Alpha + Beta`` to include a site
            (default: 10).
        alpha_min: Optional minimum ``Alpha`` count.
        usage_coord_base: Coordinate base in the parquet (1 or 0).
            Use ``1`` for Spliser output, which is 1-based.
            Use ``0`` for 0-based coordinates (e.g. if you already
            post-processed the Spliser output to convert to 0-based).
        observed_conditions_only: If ``True``, usage targets will only be
            returned for conditions with observed coverage.
        max_replicates: Maximum number of biological replicate observations
            kept per (site, condition) in :meth:`query`'s output. Real
            ``usage.parquet`` files carry several replicate rows per (site,
            condition) (each its own ``Alpha``/``Beta``/``SSE``); every one is
            treated as an independent training observation rather than being
            collapsed into a single pooled value. Any replicates beyond this
            cap are folded into the last slot as a coverage-weighted combined
            pseudo-replicate, so no observation is silently dropped.
    """

    def __init__(
        self,
        usage_parquet: str | Path,
        min_coverage: int | None = 10,
        alpha_min: int | None = None,
        usage_coord_base: int = 0,
        observed_conditions_only: bool = False,
        max_replicates: int = 10,
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
        self.observed_conditions_only = observed_conditions_only
        self.max_replicates = max_replicates
        none_class = self._class_labels.get("None", 4)

        # Small Python dicts kept for the tissue_cond_groups property and
        # observed_conditions_only masking.  These are tiny (~n_conditions
        # entries) so CoW pressure from reading them in workers is negligible.
        self._tissue_to_cond_indices: dict[str, list[int]] = {}
        self._cond_idx_to_tissue: dict[int, str] = {}

        for cond_name, cond_idx in self._condition_labels.items():
            tissue = cond_name.split("_")[0]
            self._tissue_to_cond_indices.setdefault(tissue, []).append(cond_idx)
            self._cond_idx_to_tissue[cond_idx] = tissue

        # ── Load and filter parquet ───────────────────────────────────────────
        df = pd.read_parquet(usage_parquet)
        df = df[df["Label"] != none_class].copy()
        if min_coverage is not None:
            df = df[df["Alpha"] + df["Beta"] >= min_coverage]
        if alpha_min is not None:
            df = df[df["Alpha"] >= alpha_min]
        if usage_coord_base == 1:
            df["Position"] = df["Position"] - 1
        df["Chromosome"] = df["Chromosome"].astype(str)

        # ── Encode chromosomes as small integers ──────────────────────────────
        all_chroms = sorted(df["Chromosome"].unique())
        # Small dict: ~25 entries (one per chromosome).
        self._chrom_to_id: dict[str, int] = {c: i for i, c in enumerate(all_chroms)}

        df["_chrom_id"] = df["Chromosome"].map(self._chrom_to_id).astype(np.int32)
        df["Condition"] = df["Condition"].astype(np.int32)
        df["SSE"] = df["SSE"].astype(np.float32)
        df["Position"] = df["Position"].astype(np.int64)

        # ── Sort into (chrom_id, position, condition) order ───────────────────
        df = df.sort_values(["_chrom_id", "Position", "Condition"]).reset_index(drop=True)

        n_rows = len(df)

        # Flat entries arrays — the actual data, stored as C buffers.
        # Numpy reads from these buffers without touching any Python ob_refcnt,
        # making them copy-on-write safe in forked worker processes.
        self._entry_cond_indices: np.ndarray = df["Condition"].to_numpy(dtype=np.int32)
        self._entry_sse_values: np.ndarray = df["SSE"].to_numpy(dtype=np.float32)
        # Per-replicate read coverage (Alpha+Beta) -- every row is one biological
        # replicate observation, not a pre-pooled site/condition value. Used both
        # as the BCE weight (Binomial-correct) and to coverage-weight the pooled
        # reduction the trajectory/contrast loss terms consume.
        self._entry_coverage_values: np.ndarray = (df["Alpha"] + df["Beta"]).to_numpy(dtype=np.float32)

        # ── Build site-level index (CSR-style) ────────────────────────────────
        # One row per unique (chrom_id, position) pair; stores the slice [start,
        # start+count) into the flat entries arrays for that site.
        if n_rows > 0:
            chrom_ids_arr = df["_chrom_id"].to_numpy(dtype=np.int32)
            positions_arr = df["Position"].to_numpy(dtype=np.int64)

            # Detect where a new (chrom_id, position) group begins
            new_site = np.empty(n_rows, dtype=bool)
            new_site[0] = True
            new_site[1:] = (chrom_ids_arr[1:] != chrom_ids_arr[:-1]) | (
                positions_arr[1:] != positions_arr[:-1]
            )
            site_starts = np.where(new_site)[0]  # int64 indices

            self._site_chrom_ids: np.ndarray = chrom_ids_arr[site_starts].astype(np.int32)
            self._site_positions: np.ndarray = positions_arr[site_starts].astype(np.int64)
            self._site_entry_starts: np.ndarray = site_starts.astype(np.int64)

            counts = np.empty(len(site_starts), dtype=np.int32)
            counts[:-1] = (site_starts[1:] - site_starts[:-1]).astype(np.int32)
            counts[-1] = np.int32(n_rows - site_starts[-1])
            self._site_entry_counts: np.ndarray = counts
        else:
            self._site_chrom_ids = np.empty(0, dtype=np.int32)
            self._site_positions = np.empty(0, dtype=np.int64)
            self._site_entry_starts = np.empty(0, dtype=np.int64)
            self._site_entry_counts = np.empty(0, dtype=np.int32)

        # ── Numpy structures for observed_conditions_only masking ─────────────
        # Precomputed CSR arrays so the masking path in query() is also numpy-
        # based and avoids heavy Python-object iteration per site.
        if observed_conditions_only:
            all_tissues = sorted(self._tissue_to_cond_indices.keys())
            self._n_tissues: int = len(all_tissues)
            tissue_to_id = {t: i for i, t in enumerate(all_tissues)}

            # cond_tissue_id[c] = tissue integer id for condition c (or -1)
            cond_tissue_id = np.full(self.n_conditions, -1, dtype=np.int32)
            for cond_name, cond_idx in self._condition_labels.items():
                tid = tissue_to_id.get(cond_name.split("_")[0], -1)
                if 0 <= cond_idx < self.n_conditions:
                    cond_tissue_id[cond_idx] = tid
            self._cond_tissue_id: np.ndarray = cond_tissue_id

            # CSR: tissue_id → flat list of condition indices
            tissue_lists = [
                np.array(sorted(self._tissue_to_cond_indices[t]), dtype=np.int32)
                for t in all_tissues
            ]
            self._tissue_cond_flat: np.ndarray = (
                np.concatenate(tissue_lists) if tissue_lists else np.empty(0, dtype=np.int32)
            )
            starts = np.zeros(len(all_tissues) + 1, dtype=np.int32)
            for i, lst in enumerate(tissue_lists):
                starts[i + 1] = starts[i] + len(lst)
            self._tissue_cond_starts: np.ndarray = starts

        n_sites = len(self._site_positions)
        print(
            f"SpliceSiteUsageIndex: loaded {n_sites:,} sites, "
            f"{self.n_conditions} conditions from {usage_parquet.name}"
        )

    @property
    def condition_labels(self) -> dict[str, int]:
        return self._condition_labels

    @property
    def tissue_cond_groups(self) -> list[list[int]]:
        """Condition indices grouped by tissue, each ordered by timepoint.

        Passed to :func:`splice_usage_loss` so the trajectory/delta terms measure
        *within-tissue* temporal dynamics (a developmental trajectory) rather than
        a single correlation over all tissues concatenated.
        """
        idx_to_name = {v: k for k, v in self._condition_labels.items()}
        def _timepoint(cond_idx: int) -> int:
            # condition labels look like "Brain_1"; sort each tissue by the timepoint
            name = idx_to_name.get(cond_idx, "")
            try:
                return int(name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                return cond_idx
        groups = []
        for tissue in sorted(self._tissue_to_cond_indices):
            idxs = sorted(self._tissue_to_cond_indices[tissue], key=_timepoint)
            groups.append(idxs)
        return groups

    def query(
        self, chrom: str, positions: np.ndarray
    ) -> tuple[list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Return usage data for positions that have observed values.

        Every real biological replicate row at a (site, condition) is kept as
        its own entry along a trailing replicate axis (capped at
        ``max_replicates``, with any overflow folded into the last slot as a
        coverage-weighted combined pseudo-replicate) -- callers that want a
        single representative value per (site, condition) should compute a
        coverage-weighted mean over the replicate axis themselves (this is
        what :func:`~alphagenome_pytorch.extensions.finetuning.splice_losses.
        splice_usage_loss` does internally for its trajectory/contrast terms).

        Args:
            chrom: Chromosome name.
            positions: Array of 0-based genomic positions to query.

        Returns:
            Tuple of four parallel lists:
            - ``site_positions``: subset of input positions with at least one
              observation.
            - ``values``: per-site float32 array of shape
              ``(n_conditions, max_replicates)`` with SSE values (0.0 for
              unobserved/padding replicate slots).
            - ``masks``: per-site bool array of shape
              ``(n_conditions, max_replicates)``, True for real replicate
              observations. When ``observed_conditions_only=True``, a
              condition in a tissue with zero observed timepoints instead
              gets a single synthetic "biological zero" entry at replicate
              slot 0 (mirroring the same behavior when False, where every
              condition gets at least one True slot).
            - ``coverages``: per-site float32 array of shape
              ``(n_conditions, max_replicates)`` with each replicate's own
              ``Alpha+Beta`` read count (0.0 for synthetic/padding slots).
        """
        out_positions: list[int] = []
        values_list: list[np.ndarray] = []
        masks_list: list[np.ndarray] = []
        coverages_list: list[np.ndarray] = []

        if len(positions) == 0:
            return out_positions, values_list, masks_list, coverages_list

        chrom_id = self._chrom_to_id.get(chrom)
        if chrom_id is None:
            return out_positions, values_list, masks_list, coverages_list

        # Slice site arrays to this chromosome only
        chrom_lo = int(np.searchsorted(self._site_chrom_ids, chrom_id, side="left"))
        chrom_hi = int(np.searchsorted(self._site_chrom_ids, chrom_id, side="right"))
        if chrom_lo >= chrom_hi:
            return out_positions, values_list, masks_list, coverages_list

        chrom_pos = self._site_positions[chrom_lo:chrom_hi]          # view, no copy
        chrom_starts = self._site_entry_starts[chrom_lo:chrom_hi]    # view, no copy
        chrom_counts = self._site_entry_counts[chrom_lo:chrom_hi]    # view, no copy

        # Locate all queried positions in one vectorised searchsorted call
        positions_arr = np.asarray(positions, dtype=np.int64)
        idxs = np.searchsorted(chrom_pos, positions_arr, side="left")

        max_rep = self.max_replicates

        for i in range(len(positions_arr)):
            pos = positions_arr[i]
            idx = idxs[i]
            if idx >= len(chrom_pos) or chrom_pos[idx] != pos:
                continue  # position not in index

            entry_start = int(chrom_starts[idx])
            entry_count = int(chrom_counts[idx])
            cond_idx = self._entry_cond_indices[entry_start : entry_start + entry_count]
            sse_val = self._entry_sse_values[entry_start : entry_start + entry_count]
            cov_val = self._entry_coverage_values[entry_start : entry_start + entry_count]

            # Filter to valid condition range
            valid = (cond_idx >= 0) & (cond_idx < self.n_conditions)
            valid_conds = cond_idx[valid]
            valid_sse = sse_val[valid]
            valid_cov = cov_val[valid]

            vals = np.zeros((self.n_conditions, max_rep), dtype=np.float32)
            mask = np.zeros((self.n_conditions, max_rep), dtype=bool)
            covs = np.zeros((self.n_conditions, max_rep), dtype=np.float32)

            if len(valid_conds) > 0:
                # valid_conds is a subsequence of the globally
                # [chrom_id, Position, Condition]-sorted parquet, so entries for
                # this site are already grouped and sorted by Condition -- the
                # "rank within group" (replicate index for a given condition)
                # can be computed with a vectorised run-length scan instead of
                # a per-replicate Python loop.
                change = np.empty(len(valid_conds), dtype=bool)
                change[0] = True
                change[1:] = valid_conds[1:] != valid_conds[:-1]
                group_starts = np.where(change)[0]
                group_id = np.cumsum(change) - 1
                rank = np.arange(len(valid_conds)) - group_starts[group_id]
                capped_rank = np.minimum(rank, max_rep - 1)

                # Scatter-accumulate coverage and coverage*SSE per (condition,
                # replicate-slot); a single non-overflowing replicate reduces
                # to its own (cov, cov*sse) pair, and any replicates beyond
                # max_rep land in the last slot together, combining into a
                # coverage-weighted pseudo-replicate rather than being dropped.
                cov_acc = np.zeros((self.n_conditions, max_rep), dtype=np.float64)
                val_acc = np.zeros((self.n_conditions, max_rep), dtype=np.float64)
                np.add.at(cov_acc, (valid_conds, capped_rank), valid_cov.astype(np.float64))
                np.add.at(
                    val_acc, (valid_conds, capped_rank),
                    valid_sse.astype(np.float64) * valid_cov.astype(np.float64),
                )
                nz = cov_acc > 0
                vals[nz] = (val_acc[nz] / cov_acc[nz]).astype(np.float32)
                covs = cov_acc.astype(np.float32)
                mask[valid_conds, capped_rank] = True

            rep_counter = np.bincount(valid_conds, minlength=self.n_conditions) if len(valid_conds) > 0 \
                else np.zeros(self.n_conditions, dtype=np.int64)

            if not self.observed_conditions_only:
                # Every condition contributes to the loss: real replicates via
                # their own slots (already set above), unobserved conditions
                # via a synthetic biological-zero entry at slot 0.
                unobserved = rep_counter == 0
                mask[unobserved, 0] = True
            else:
                # Tissues with at least one observed condition for this site
                obs_tids = set(self._cond_tissue_id[valid_conds].tolist())
                obs_tids.discard(-1)
                # Unobserved tissues: give every one of their conditions a
                # single synthetic biological-zero entry. Individual
                # unobserved timepoints *within* an otherwise-observed tissue
                # stay unmasked, matching the pre-replicate behavior.
                for tid in range(self._n_tissues):
                    if tid not in obs_tids:
                        ts = int(self._tissue_cond_starts[tid])
                        te = int(self._tissue_cond_starts[tid + 1])
                        mask[self._tissue_cond_flat[ts:te], 0] = True

            out_positions.append(int(pos))
            values_list.append(vals)
            masks_list.append(mask)
            coverages_list.append(covs)

        return out_positions, values_list, masks_list, coverages_list


class SpliceJunctionIndex:
    """In-memory index of per-condition splice-junction read counts.

    Loads a ``junctions.parquet`` produced by
    ``scripts/convert_splice_junctions_to_parquet.py`` (one row per junction,
    condition and sample), aggregates the samples of each condition, and keeps
    the result as flat numpy arrays (fork-safe in DataLoader workers, like
    :class:`SpliceSiteUsageIndex`).

    Args:
        junction_parquet: Path to ``junctions.parquet``; ``junctions.json``
            (``condition_labels``) must sit next to it.
        usage_coord_base: Coordinate base of the parquet (same as for the
            species' usage parquet, since junction ends use its coordinates).
        count_norm: How raw read counts are turned into targets:

            - ``"cpm"`` (default): divide each sample's counts by its total
              junction reads (a library-size proxy; Spliser gives no total
              mapped reads) × 1e6, combine the samples of a condition with
              *replicate_agg*, clip at the *clip_quantile* quantile of the
              non-zero values and divide by their mean, so typical non-zero
              targets are about 1 (as for the pretrained head, which predicts
              ≈0.6 reads per junction, versus ≈230 raw pooled reads).
            - ``"none"``: raw counts, combined with *replicate_agg*.
        replicate_agg: How to combine the samples of a condition: ``"mean"``
            (default; sum divided by the number of samples of that condition)
            or ``"sum"`` (pooled reads).
        clip_quantile: Upper quantile of non-zero targets to clip at (``cpm``).
        min_count: Optional minimum aggregated count per (junction, condition),
            applied after normalisation and before *count_scale*.
        count_scale: Extra factor applied at the end (default 1).
    """

    def __init__(
        self,
        junction_parquet: str | Path,
        usage_coord_base: int = 0,
        count_norm: str = "cpm",
        replicate_agg: str = "mean",
        clip_quantile: float = 0.9999,
        min_count: float | None = None,
        count_scale: float = 1.0,
    ) -> None:
        import pandas as pd

        junction_parquet = Path(junction_parquet)
        junction_json = junction_parquet.with_suffix(".json")
        for p in (junction_parquet, junction_json):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")
        if replicate_agg not in ("sum", "mean"):
            raise ValueError(f"replicate_agg must be 'sum' or 'mean', got {replicate_agg!r}")
        if count_norm not in ("cpm", "none"):
            raise ValueError(f"count_norm must be 'cpm' or 'none', got {count_norm!r}")

        with open(junction_json) as f:
            self._condition_labels: dict[str, int] = json.load(f)["condition_labels"]
        self.n_conditions: int = len(self._condition_labels)

        df = pd.read_parquet(junction_parquet)
        df["Chromosome"] = df["Chromosome"].astype(str)
        df["Strand"] = df["Strand"].astype(str)
        df["Count"] = df["Count"].astype(np.float64)
        if count_norm == "cpm":
            library = df.groupby(["Condition", "Sample"], observed=True)["Count"].transform("sum")
            df["Count"] = df["Count"] / library * 1e6
        key = ["Chromosome", "Donor", "Acceptor", "Strand", "Condition"]
        agg = df.groupby(key, as_index=False, observed=True)["Count"].sum()
        if replicate_agg == "mean":
            n_samples = df.groupby("Condition", observed=True)["Sample"].nunique()
            agg["Count"] = agg["Count"] / agg["Condition"].map(n_samples).to_numpy()
        self.clip_value: float | None = None
        self.norm_mean: float | None = None
        if count_norm == "cpm":
            nonzero = agg["Count"].to_numpy()
            nonzero = nonzero[nonzero > 0]
            if len(nonzero):
                self.clip_value = float(np.quantile(nonzero, clip_quantile))
                agg["Count"] = agg["Count"].clip(upper=self.clip_value)
                self.norm_mean = float(agg.loc[agg["Count"] > 0, "Count"].mean())
                agg["Count"] = agg["Count"] / self.norm_mean
        if min_count is not None:
            agg = agg[agg["Count"] >= min_count]
        agg["Count"] = agg["Count"] * count_scale
        if usage_coord_base == 1:
            agg["Donor"] -= 1
            agg["Acceptor"] -= 1

        all_chroms = sorted(agg["Chromosome"].unique())
        self._chrom_to_id: dict[str, int] = {c: i for i, c in enumerate(all_chroms)}
        agg["_chrom_id"] = agg["Chromosome"].map(self._chrom_to_id).astype(np.int32)
        # Sort by the leftmost end so that a window query is one searchsorted.
        agg["_left"] = np.minimum(agg["Donor"], agg["Acceptor"])
        agg = agg.sort_values(["_chrom_id", "_left"]).reset_index(drop=True)

        self._chrom_ids = agg["_chrom_id"].to_numpy(dtype=np.int32)
        self._left = agg["_left"].to_numpy(dtype=np.int64)
        self._right = np.maximum(agg["Donor"], agg["Acceptor"]).to_numpy(dtype=np.int64)
        self._donor = agg["Donor"].to_numpy(dtype=np.int64)
        self._acceptor = agg["Acceptor"].to_numpy(dtype=np.int64)
        self._plus = (agg["Strand"] == "+").to_numpy()
        self._cond = agg["Condition"].to_numpy(dtype=np.int32)
        self._count = agg["Count"].to_numpy(dtype=np.float32)

        n_junctions = len(agg[["_chrom_id", "Donor", "Acceptor", "Strand"]].drop_duplicates())
        print(
            f"SpliceJunctionIndex: loaded {n_junctions:,} junctions "
            f"({len(agg):,} junction×condition entries), {self.n_conditions} conditions "
            f"from {junction_parquet.name}"
            + (f"; targets: CPM, clipped at {self.clip_value:.4g}, divided by mean {self.norm_mean:.4g}"
               if self.norm_mean else "")
        )

    @property
    def condition_labels(self) -> dict[str, int]:
        return self._condition_labels

    def query(
        self, chrom: str, start: int, end: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Junctions with both ends in ``[start, end)``.

        Returns:
            ``(donor, acceptor, plus_strand, condition, count)`` arrays, one
            entry per (junction, condition), genomic 0-based coordinates.
        """
        chrom_id = self._chrom_to_id.get(chrom)
        if chrom_id is None:
            empty_i = np.empty(0, dtype=np.int64)
            return (empty_i, empty_i, np.empty(0, dtype=bool),
                    np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
        c_lo = int(np.searchsorted(self._chrom_ids, chrom_id, side="left"))
        c_hi = int(np.searchsorted(self._chrom_ids, chrom_id, side="right"))
        left = self._left[c_lo:c_hi]
        lo = c_lo + int(np.searchsorted(left, start, side="left"))
        hi = c_lo + int(np.searchsorted(left, end, side="left"))
        keep = self._right[lo:hi] < end
        sl = slice(lo, hi)
        return (
            self._donor[sl][keep],
            self._acceptor[sl][keep],
            self._plus[sl][keep],
            self._cond[sl][keep],
            self._count[sl][keep],
        )


def build_junction_targets(
    donor: np.ndarray,
    acceptor: np.ndarray,
    plus: np.ndarray,
    cond: np.ndarray,
    count: np.ndarray,
    n_conditions: int,
    max_sites: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn window-relative junctions into junction-head inputs and sparse targets.

    Site lists are built per strand and type (donor+, acceptor+, donor−,
    acceptor−) from the junction ends.  When a list holds more than
    *max_sites* sites, the ones with the most reads (summed over conditions)
    are kept; junctions touching a dropped site are dropped too.

    Args:
        donor, acceptor: Window-relative junction ends.
        plus: True for ``+`` strand junctions.
        cond: Condition index per entry.
        count: Read count per entry.
        n_conditions: Number of conditions T; ``−`` strand entries go to
            track ``T + cond`` (the head's output is ``[+ strand | − strand]``).
        max_sites: P, the length of each site list.

    Returns:
        ``splice_site_positions`` (4, P) int64 padded with −1, ``index``
        (J, 3) int64 rows ``(donor slot, acceptor slot, track)``, and
        ``counts`` (J,) float32.
    """
    positions = np.full((4, max_sites), -1, dtype=np.int64)
    index_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []

    for strand_i, is_plus in enumerate((True, False)):
        sel = plus == is_plus
        if not sel.any():
            continue
        d, a, c, n = donor[sel], acceptor[sel], cond[sel], count[sel]
        keep = np.ones(len(d), dtype=bool)
        slot_maps = []
        for ends in (d, a):
            sites, inv = np.unique(ends, return_inverse=True)
            if len(sites) > max_sites:
                totals = np.bincount(inv, weights=n, minlength=len(sites))
                top = np.sort(np.argsort(-totals, kind="stable")[:max_sites])
                kept = np.zeros(len(sites), dtype=bool)
                kept[top] = True
                keep &= kept[inv]
                sites = sites[top]
            slot_maps.append(sites)
        d_sites, a_sites = slot_maps
        positions[2 * strand_i, : len(d_sites)] = d_sites
        positions[2 * strand_i + 1, : len(a_sites)] = a_sites
        d, a, c, n = d[keep], a[keep], c[keep], n[keep]
        track = c.astype(np.int64) + (0 if is_plus else n_conditions)
        index_parts.append(np.stack(
            [np.searchsorted(d_sites, d), np.searchsorted(a_sites, a), track], axis=1
        ).astype(np.int64))
        count_parts.append(n.astype(np.float32))

    if index_parts:
        index = np.concatenate(index_parts)
        counts = np.concatenate(count_parts)
    else:
        index = np.empty((0, 3), dtype=np.int64)
        counts = np.empty(0, dtype=np.float32)
    return positions, index, counts


def densify_junction_targets(
    junction_index: torch.Tensor,
    junction_counts: torch.Tensor,
    num_sites: int,
    num_tracks: int,
) -> torch.Tensor:
    """Scatter sparse junction targets into a dense ``(B, P, P, num_tracks)`` tensor.

    Args:
        junction_index: (B, J, 3) rows ``(donor slot, acceptor slot, track)``,
            padded with −1 (see :func:`collate_splice`).
        junction_counts: (B, J) counts (0 for padding).
        num_sites: P.
        num_tracks: 2 × number of conditions.
    """
    B = junction_index.shape[0]
    dense = torch.zeros(
        (B, num_sites, num_sites, num_tracks),
        dtype=torch.float32, device=junction_counts.device,
    )
    valid = junction_index[..., 0] >= 0
    b = torch.arange(B, device=junction_index.device)[:, None].expand_as(valid)
    idx = junction_index[valid]
    dense.index_put_(
        (b[valid], idx[:, 0], idx[:, 1], idx[:, 2]),
        junction_counts[valid].float(),
        accumulate=True,
    )
    return dense


def _load_intervals_from_bed(
    bed_path: str,
) -> tuple[list[tuple[str, int, int, int, int]], set[str]]:
    """Load genomic intervals from a BED file (0-based half-open).
    
    Expects BED format with optional mask columns:
    chr, start, end, [gene], [mask_start_rel], [mask_end_rel]
    
    Returns intervals as (chrom, start, end, mask_start_rel, mask_end_rel) tuples.
    Columns 4-5 are RELATIVE coordinates (offset from window start, not absolute genomic coords).
    If mask columns are missing, mask_start_rel=0 and mask_end_rel=sequence_length (full window).
    """
    intervals: list[tuple[str, int, int, int, int]] = []
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
            
            # Parse optional mask columns (columns 4 and 5 are RELATIVE coordinates)
            # These are already offsets from window start, NOT absolute genomic positions
            if len(parts) >= 6:
                mask_start_rel = int(parts[4])
                mask_end_rel = int(parts[5])
            else:
                # No mask columns: will use full window (set to None and handle in __init__)
                mask_start_rel = -1  # Sentinel value
                mask_end_rel = -1
            
            intervals.append((chrom, start, end, mask_start_rel, mask_end_rel))
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
        junction_index: Optional :class:`SpliceJunctionIndex` for junction
            count targets.  When given, items carry
            ``splice_site_positions`` (4, ``max_junction_sites``) and sparse
            ``junction_index`` / ``junction_counts`` (see
            :func:`build_junction_targets`).  Only junctions with both ends
            inside the loss-mask region are used.
        max_junction_sites: Length P of each junction site list (donor+,
            acceptor+, donor−, acceptor−).
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
        lazy_genome: bool = True,
        junction_index: SpliceJunctionIndex | None = None,
        max_junction_sites: int = 256,
    ) -> None:
        # Defer heavy import to avoid hard dep at module import time
        from alphagenome_pytorch.extensions.finetuning.datasets import (
            CachedGenome,
            LazyFastaGenome,
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
        self.junction_index = junction_index
        self.max_junction_sites = max_junction_sites

        # Load BED intervals first so we know which chromosomes are needed
        all_intervals, chromosomes = _load_intervals_from_bed(bed_file)

        # Build a stable chromosome → integer index mapping for coordinate tracking
        self.chrom_names: list[str] = sorted(chromosomes)
        self._chrom_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.chrom_names)}

        # Genome backend. From a FASTA path, default to the lazy backend, which reads
        # only chromosome sizes at startup and fetches windows on demand — avoiding the
        # slow whole-chromosome load + one-hot encoding that CachedGenome does (which is
        # wasteful here since training only touches the BED's windows). Pass
        # ``lazy_genome=False`` to force the whole-chromosome cache, or pass a prebuilt
        # CachedGenome instance directly to share it across splits.
        if isinstance(genome, str) or isinstance(genome, Path):
            if lazy_genome:
                self._cached_genome = LazyFastaGenome(str(genome), chromosomes=chromosomes)
            else:
                self._cached_genome = CachedGenome(str(genome), chromosomes=chromosomes)
        else:
            self._cached_genome = genome  # prebuilt CachedGenome / LazyFastaGenome

        half = sequence_length // 2
        chrom_sizes = self._cached_genome.chrom_sizes
        self._positions: list[tuple[str, int, int]] = []
        self._loss_masks: list[tuple[int, int]] = []  # Store (mask_start_rel, mask_end_rel) per position
        n_skipped = n_truncated = 0

        for chrom, start, end, mask_start_rel, mask_end_rel in all_intervals:
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
            
            # Mask coordinates are already window-relative (or -1 if missing)
            # Clamp to valid range [0, sequence_length]
            if mask_start_rel < 0 or mask_end_rel < 0:
                # No mask columns in BED: use full window
                mask_start_rel = 0
                mask_end_rel = sequence_length
            else:
                # Clamp to sequence bounds
                mask_start_rel = max(0, min(mask_start_rel, sequence_length))
                mask_end_rel = max(0, min(mask_end_rel, sequence_length))
            
            self._positions.append((chrom, win_start, win_end))
            self._loss_masks.append((mask_start_rel, mask_end_rel))

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
        mask_start_rel, mask_end_rel = self._loss_masks[idx]

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

        # ── Loss mask ─────────────────────────────────────────────────────────
        # Create boolean mask: True for positions within [mask_start_rel, mask_end_rel)
        loss_mask = np.zeros(self.sequence_length, dtype=bool)
        if mask_end_rel > mask_start_rel:
            loss_mask[mask_start_rel:mask_end_rel] = True
        loss_mask = torch.from_numpy(loss_mask)  # (S,) bool

        item: dict[str, Any] = {
            "sequence": sequence,
            "organism_index": torch.tensor(self.organism_index, dtype=torch.long),
            "classification_labels": classification_labels,
            "loss_mask": loss_mask,
            "window_start": torch.tensor(win_start, dtype=torch.int64),
            "chrom_idx": torch.tensor(self._chrom_to_idx.get(chrom, -1), dtype=torch.int32),
        }

        # ── Usage targets (sparse) ────────────────────────────────────────────
        if self.usage_index is not None:
            # Query annotation positions that are within the window
            all_site_pos, _ = self.annotation.query(chrom, win_start, win_end)
            
            # Filter positions to only include those within the loss_mask region
            if len(all_site_pos) > 0:
                # Convert to window-relative positions for mask filtering
                rel_pos = all_site_pos - win_start
                # Keep only positions within [mask_start_rel, mask_end_rel)
                mask_filter = (rel_pos >= mask_start_rel) & (rel_pos < mask_end_rel)
                masked_site_pos = all_site_pos[mask_filter]
            else:
                masked_site_pos = all_site_pos
            
            if len(masked_site_pos) > 0:
                used_pos, val_arrays, mask_arrays, cov_arrays = self.usage_index.query(
                    chrom, masked_site_pos
                )
            else:
                used_pos = []
                val_arrays = []
                mask_arrays = []
                cov_arrays = []

            n_cond = self.usage_index.n_conditions
            max_rep = self.usage_index.max_replicates
            n_used = len(used_pos)
            n_pad = self.max_sites - min(n_used, self.max_sites)

            # Trim to max_sites (keep first max_sites sites, sorted by position)
            used_pos = used_pos[: self.max_sites]
            val_arrays = val_arrays[: self.max_sites]
            mask_arrays = mask_arrays[: self.max_sites]
            cov_arrays = cov_arrays[: self.max_sites]

            rel_positions = np.array(
                [p - win_start for p in used_pos], dtype=np.int64
            )

            # Pad to max_sites with -1 / zeros / False
            if n_pad > 0:
                rel_positions = np.concatenate(
                    [rel_positions, np.full(n_pad, -1, dtype=np.int64)]
                )

            if val_arrays:
                vals_3d = np.stack(val_arrays)          # (n_used, n_cond, max_rep)
                masks_3d = np.stack(mask_arrays)        # (n_used, n_cond, max_rep)
                covs_3d = np.stack(cov_arrays)           # (n_used, n_cond, max_rep)
            else:
                vals_3d = np.zeros((0, n_cond, max_rep), dtype=np.float32)
                masks_3d = np.zeros((0, n_cond, max_rep), dtype=bool)
                covs_3d = np.zeros((0, n_cond, max_rep), dtype=np.float32)

            if n_pad > 0:
                vals_3d = np.concatenate(
                    [vals_3d, np.zeros((n_pad, n_cond, max_rep), dtype=np.float32)], axis=0
                )
                masks_3d = np.concatenate(
                    [masks_3d, np.zeros((n_pad, n_cond, max_rep), dtype=bool)], axis=0
                )
                covs_3d = np.concatenate(
                    [covs_3d, np.zeros((n_pad, n_cond, max_rep), dtype=np.float32)], axis=0
                )

            item["usage_positions"] = torch.from_numpy(rel_positions)   # (max_sites,) int64
            item["usage_values"] = torch.from_numpy(vals_3d)            # (max_sites, n_cond, max_rep)
            item["usage_mask"] = torch.from_numpy(masks_3d)             # (max_sites, n_cond, max_rep) bool
            item["usage_coverage"] = torch.from_numpy(covs_3d)          # (max_sites, n_cond, max_rep)

        # ── Junction targets (sparse) ─────────────────────────────────────────
        if self.junction_index is not None:
            donor, acceptor, plus, cond, count = self.junction_index.query(
                chrom, win_start + mask_start_rel, win_start + mask_end_rel
            )
            positions, jx_index, jx_counts = build_junction_targets(
                donor - win_start, acceptor - win_start, plus, cond, count,
                n_conditions=self.junction_index.n_conditions,
                max_sites=self.max_junction_sites,
            )
            item["splice_site_positions"] = torch.from_numpy(positions)  # (4, P) int64
            item["junction_index"] = torch.from_numpy(jx_index)          # (J, 3) int64
            item["junction_counts"] = torch.from_numpy(jx_counts)        # (J,) float32

        return item


def collate_splice(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate a list of :class:`SpliceSiteDataset` items into a batch dict.

    All tensors are stacked along a new batch dimension.  Usage keys are
    only included when present in every item in the batch.

    Returns:
        Dict with keys ``sequence``, ``organism_index``,
        ``classification_labels``, ``loss_mask``, and optionally ``usage_positions``,
        ``usage_values``, ``usage_mask``, ``usage_coverage``, and
        ``splice_site_positions`` (B, 4, P), ``junction_index`` (B, J, 3) and
        ``junction_counts`` (B, J), with J padded to the batch maximum
        (index −1, count 0).
    """
    result: dict[str, Any] = {
        "sequence": torch.stack([b["sequence"] for b in batch]),
        "organism_index": torch.stack([b["organism_index"] for b in batch]),
        "classification_labels": torch.stack([b["classification_labels"] for b in batch]),
        "loss_mask": torch.stack([b["loss_mask"] for b in batch]),
        "window_start": torch.stack([b["window_start"] for b in batch]),
        "chrom_idx": torch.stack([b["chrom_idx"] for b in batch]),
    }
    if "usage_positions" in batch[0]:
        result["usage_positions"] = torch.stack([b["usage_positions"] for b in batch])
        result["usage_values"] = torch.stack([b["usage_values"] for b in batch])
        result["usage_mask"] = torch.stack([b["usage_mask"] for b in batch])
        result["usage_coverage"] = torch.stack([b["usage_coverage"] for b in batch])
    if "junction_index" in batch[0]:
        result["splice_site_positions"] = torch.stack([b["splice_site_positions"] for b in batch])
        n_max = max(1, max(len(b["junction_counts"]) for b in batch))
        index = torch.full((len(batch), n_max, 3), -1, dtype=torch.int64)
        counts = torch.zeros((len(batch), n_max), dtype=torch.float32)
        for i, b in enumerate(batch):
            n = len(b["junction_counts"])
            index[i, :n] = b["junction_index"]
            counts[i, :n] = b["junction_counts"]
        result["junction_index"] = index
        result["junction_counts"] = counts
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
        oversample_to_max: If ``True``, every species is (cyclically) resampled
            up to the window count of the largest species before batching, so
            every species contributes the same number of batches/gradient
            updates per epoch instead of being proportional to its dataset
            size. Each real example is still used floor(target/n) or
            floor(target/n)+1 times per epoch (not iid with-replacement), to
            keep per-epoch exposure even. Has no effect on the species with
            the most windows. Recommended for the training sampler only —
            leave validation unbalanced so reported per-species metrics stay
            representative of the true data distribution.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        oversample_to_max: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.oversample_to_max = oversample_to_max
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

    def _target_size(self) -> int:
        """Window count every species is resampled up to when balancing."""
        return max(len(v) for v in self.species_indices.values())

    def _make_batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        batches: list[list[int]] = []
        target_n = self._target_size() if self.oversample_to_max else None
        for org_idx, indices in self.species_indices.items():
            idx_arr = np.array(indices, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(idx_arr)
            if target_n is not None and len(idx_arr) < target_n:
                n_repeats = math.ceil(target_n / len(idx_arr))
                tiled = np.tile(idx_arr, n_repeats)
                if self.shuffle:
                    # Reshuffle the tiled array (not just each repeat block) so
                    # duplicated examples don't land in the same relative
                    # position/batch every cycle.
                    rng.shuffle(tiled)
                idx_arr = tiled[:target_n]
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
        target_n = self._target_size() if self.oversample_to_max else None
        for indices in self.species_indices.values():
            n = target_n if target_n is not None else len(indices)
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
        oversample_to_max: As in :class:`SpeciesGroupedSampler`.
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
        oversample_to_max: bool = False,
    ) -> None:
        super().__init__(
            dataset, batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last,
            oversample_to_max=oversample_to_max,
        )
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
