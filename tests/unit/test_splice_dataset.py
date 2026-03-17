"""Unit tests for splice_datasets module."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_annotation_df(sites: list[tuple[str, int, str]]) -> pd.DataFrame:
    """Build a minimal annotation DataFrame."""
    df = pd.DataFrame(sites, columns=["Chromosome", "Position", "SiteType"])
    df["Chromosome"] = df["Chromosome"].astype("category")
    df["SiteType"] = pd.Categorical(
        df["SiteType"],
        categories=["Donor+", "Acceptor+", "Donor-", "Acceptor-"],
    )
    df["Position"] = df["Position"].astype("Int64")
    return df


def _make_usage_parquet(rows: list[dict], n_conditions: int = 3) -> tuple[pd.DataFrame, dict]:
    """Build a minimal usage DataFrame and matching metadata dict."""
    meta = {
        "class_labels": {"Donor+": 0, "Acceptor+": 1, "Donor-": 2, "Acceptor-": 3, "None": 4},
        "condition_labels": {f"Cond{i}": i for i in range(n_conditions)},
    }
    df = pd.DataFrame(rows)
    return df, meta


def _write_parquet_tmpfile(df: pd.DataFrame, tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    df.to_parquet(p, index=False)
    return p


def _mock_cached_genome(seq_len: int = 200_000) -> MagicMock:
    """Return a CachedGenome-like mock that provides deterministic one-hot bytes."""
    # Fixed seed one-hot: each position cycles through A/C/G/T
    bases = np.eye(4, dtype=np.uint8)
    seq = np.tile(bases, (seq_len // 4 + 1, 1))[:seq_len]

    mock = MagicMock()
    mock.chrom_sizes = {"chr1": seq_len, "chr2": seq_len}
    mock.fasta_path = "mock.fa"

    def fetch(chrom, start, end, copy=True):
        return seq[start:end].copy()

    mock.fetch.side_effect = fetch
    return mock


# ─── SpliceSiteAnnotation ────────────────────────────────────────────────────


class TestSpliceSiteAnnotation:
    def test_load_and_query(self, tmp_path: Path):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteAnnotation,
            SITE_TYPE_TO_CLASS,
        )

        sites = [
            ("chr1", 1000, "Donor+"),
            ("chr1", 2000, "Acceptor+"),
            ("chr1", 3000, "Donor-"),
            ("chr2", 500, "Acceptor-"),
        ]
        df = _make_annotation_df(sites)
        p = _write_parquet_tmpfile(df, tmp_path, "annot.parquet")

        annot = SpliceSiteAnnotation(str(p))

        # Query window covering first two sites on chr1
        pos, cls = annot.query("chr1", 900, 2500)
        assert len(pos) == 2
        assert 1000 in pos
        assert 2000 in pos
        assert cls[list(pos).index(1000)] == SITE_TYPE_TO_CLASS["Donor+"]
        assert cls[list(pos).index(2000)] == SITE_TYPE_TO_CLASS["Acceptor+"]

    def test_empty_query_for_unknown_chrom(self, tmp_path: Path):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteAnnotation,
        )

        df = _make_annotation_df([("chr1", 1000, "Donor+")])
        p = _write_parquet_tmpfile(df, tmp_path, "annot.parquet")
        annot = SpliceSiteAnnotation(str(p))

        pos, cls = annot.query("chrX", 0, 10000)
        assert len(pos) == 0
        assert len(cls) == 0

    def test_boundary_exclusion(self, tmp_path: Path):
        """Query [start, end) should NOT include position == end."""
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteAnnotation,
        )

        df = _make_annotation_df([("chr1", 1000, "Donor+"), ("chr1", 2000, "Donor+")])
        p = _write_parquet_tmpfile(df, tmp_path, "annot.parquet")
        annot = SpliceSiteAnnotation(str(p))

        pos, _ = annot.query("chr1", 1000, 2000)  # end is exclusive
        assert list(pos) == [1000]


# ─── SpliceSiteUsageIndex ────────────────────────────────────────────────────


class TestSpliceSiteUsageIndex:
    def _build_dir(self, tmp_path: Path, rows: list[dict], n_conditions: int = 3) -> Path:
        df, meta = _make_usage_parquet(rows, n_conditions)
        d = tmp_path / "usage"
        d.mkdir()
        df.to_parquet(d / "_usage.parquet", index=False)
        (d / "_usage.json").write_text(json.dumps(meta))
        return d

    def test_basic_query(self, tmp_path: Path):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteUsageIndex,
        )

        rows = [
            {"Chromosome": "1", "Position": 1001, "SSE": 0.8, "Alpha": 8, "Beta": 2,
             "Label": 0, "Condition": 0},
            {"Chromosome": "1", "Position": 1001, "SSE": 0.5, "Alpha": 5, "Beta": 5,
             "Label": 0, "Condition": 1},
        ]
        d = self._build_dir(tmp_path, rows)
        idx = SpliceSiteUsageIndex(str(d), min_coverage=10, usage_coord_base=1)

        # query position 1000 in 0-based (= 1001 in 1-based)
        site_pos, vals, masks = idx.query("1", np.array([1000]))
        assert len(site_pos) == 1
        assert site_pos[0] == 1000
        assert vals[0].shape == (3,)
        assert masks[0][0] and masks[0][1] and not masks[0][2]
        np.testing.assert_allclose(vals[0][0], 0.8, rtol=1e-5)
        np.testing.assert_allclose(vals[0][1], 0.5, rtol=1e-5)

    def test_coverage_filter(self, tmp_path: Path):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteUsageIndex,
        )

        rows = [
            {"Chromosome": "1", "Position": 1001, "SSE": 0.9, "Alpha": 2, "Beta": 3,
             "Label": 0, "Condition": 0},   # Alpha+Beta=5 < 10 → filtered out
        ]
        d = self._build_dir(tmp_path, rows)
        idx = SpliceSiteUsageIndex(str(d), min_coverage=10, usage_coord_base=1)
        site_pos, vals, masks = idx.query("1", np.array([1000]))
        assert len(site_pos) == 0

    def test_none_class_excluded(self, tmp_path: Path):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteUsageIndex,
        )

        rows = [
            {"Chromosome": "1", "Position": 1001, "SSE": 0.0, "Alpha": 0, "Beta": 20,
             "Label": 4, "Condition": 0},   # Label=4 = "None" → excluded
        ]
        d = self._build_dir(tmp_path, rows)
        idx = SpliceSiteUsageIndex(str(d), min_coverage=10, usage_coord_base=1)
        site_pos, _, _ = idx.query("1", np.array([1000]))
        assert len(site_pos) == 0


# ─── SpliceSiteDataset ───────────────────────────────────────────────────────


class TestSpliceSiteDataset:
    def _build_annotation(self, tmp_path: Path) -> "SpliceSiteAnnotation":
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteAnnotation,
        )

        # Place sites within a known window [5000, 136072)
        sites = [
            ("chr1", 10000, "Donor+"),
            ("chr1", 20000, "Acceptor+"),
            ("chr1", 30000, "Donor-"),
            ("chr1", 40000, "Acceptor-"),
        ]
        df = _make_annotation_df(sites)
        p = tmp_path / "annot.parquet"
        df.to_parquet(p, index=False)
        return SpliceSiteAnnotation(str(p))

    def _write_bed(self, tmp_path: Path, intervals: list[tuple[str, int, int]]) -> str:
        p = tmp_path / "regions.bed"
        with open(p, "w") as f:
            for chrom, s, e in intervals:
                f.write(f"{chrom}\t{s}\t{e}\n")
        return str(p)

    def test_len_and_sequence_shape(self, tmp_path: Path):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteDataset,
        )

        annot = self._build_annotation(tmp_path)
        genome = _mock_cached_genome(seq_len=150_000)
        # Center of [0, 131072] = 65536
        bed = self._write_bed(tmp_path, [("chr1", 0, 131072)])
        ds = SpliceSiteDataset(genome=genome, bed_file=bed, annotation=annot,
                               sequence_length=131072)

        assert len(ds) == 1
        item = ds[0]
        assert item["sequence"].shape == (131072, 4)
        assert item["sequence"].dtype == torch.float32

    def test_classification_labels_shape_and_values(self, tmp_path: Path):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteDataset,
            BACKGROUND_CLASS,
            SITE_TYPE_TO_CLASS,
        )

        annot = self._build_annotation(tmp_path)
        genome = _mock_cached_genome(seq_len=150_000)
        # Window centered at 70536: [70536-65536, 70536+65536) = [5000, 136072)
        bed = self._write_bed(tmp_path, [("chr1", 5000, 136072)])
        ds = SpliceSiteDataset(genome=genome, bed_file=bed, annotation=annot,
                               sequence_length=131072)

        item = ds[0]
        labels = item["classification_labels"]
        assert labels.shape == (131072,)
        assert labels.dtype == torch.int64

        # All four annotated sites should be in this window
        win_start = 5000  # (5000+136072)//2 - 131072//2 = 70536 - 65536 = 5000
        for pos, site_type in [(10000, "Donor+"), (20000, "Acceptor+"),
                               (30000, "Donor-"), (40000, "Acceptor-")]:
            rel = pos - win_start
            assert labels[rel].item() == SITE_TYPE_TO_CLASS[site_type], \
                f"Expected class {SITE_TYPE_TO_CLASS[site_type]} at rel_pos={rel}"

        # Background should dominate
        n_bg = (labels == BACKGROUND_CLASS).sum().item()
        assert n_bg == 131072 - 4

    def test_organism_index_in_batch(self, tmp_path: Path):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteDataset,
        )

        annot = self._build_annotation(tmp_path)
        genome = _mock_cached_genome(seq_len=150_000)
        bed = self._write_bed(tmp_path, [("chr1", 5000, 136072)])
        ds = SpliceSiteDataset(genome=genome, bed_file=bed, annotation=annot,
                               organism_index=1)
        assert ds[0]["organism_index"].item() == 1

    def test_usage_targets_present(self, tmp_path: Path):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteDataset, SpliceSiteUsageIndex,
        )

        annot = self._build_annotation(tmp_path)
        genome = _mock_cached_genome(seq_len=150_000)

        # Build usage index with one observation at position 10000 (1-based = 10001)
        rows = [
            {"Chromosome": "chr1", "Position": 10001, "SSE": 0.75,
             "Alpha": 15, "Beta": 5, "Label": 0, "Condition": 0},
        ]
        meta = {
            "class_labels": {"Donor+": 0, "Acceptor+": 1, "Donor-": 2, "Acceptor-": 3, "None": 4},
            "condition_labels": {"Cond0": 0, "Cond1": 1},
        }
        d = tmp_path / "usage"
        d.mkdir()
        pd.DataFrame(rows).to_parquet(d / "_usage.parquet", index=False)
        (d / "_usage.json").write_text(json.dumps(meta))

        usage_idx = SpliceSiteUsageIndex(str(d), min_coverage=10, usage_coord_base=1)

        bed = self._write_bed(tmp_path, [("chr1", 5000, 136072)])
        ds = SpliceSiteDataset(genome=genome, bed_file=bed, annotation=annot,
                               usage_index=usage_idx, max_sites=10)
        item = ds[0]

        assert "usage_positions" in item
        assert "usage_values" in item
        assert "usage_mask" in item
        assert item["usage_positions"].shape == (10,)
        assert item["usage_values"].shape == (10, 2)
        assert item["usage_mask"].shape == (10, 2)

        # The first non-padded position should correspond to genomic 10000 (0-based)
        win_start = 5000
        rel_10000 = 10000 - win_start  # = 5000
        valid = item["usage_positions"] >= 0
        assert valid.any()
        assert rel_10000 in item["usage_positions"].tolist()

    def test_out_of_bounds_intervals_skipped(self, tmp_path: Path):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
            SpliceSiteDataset,
        )

        annot = self._build_annotation(tmp_path)
        genome = _mock_cached_genome(seq_len=150_000)  # chrom size = 150_000
        # This interval, when expanded to 131072, would go negative
        bed = self._write_bed(tmp_path, [("chr1", 0, 100)])
        with pytest.warns(UserWarning, match="skipped"):
            ds = SpliceSiteDataset(genome=genome, bed_file=bed, annotation=annot,
                                   sequence_length=131072)
        assert len(ds) == 0


# ─── collate_splice ──────────────────────────────────────────────────────────


class TestCollateSplice:
    def _make_item(self, has_usage: bool = True, n_cond: int = 3, max_sites: int = 4):
        item = {
            "sequence": torch.zeros(131072, 4),
            "organism_index": torch.tensor(0),
            "classification_labels": torch.zeros(131072, dtype=torch.long),
        }
        if has_usage:
            item["usage_positions"] = torch.full((max_sites,), -1, dtype=torch.long)
            item["usage_values"] = torch.zeros(max_sites, n_cond)
            item["usage_mask"] = torch.zeros(max_sites, n_cond, dtype=torch.bool)
        return item

    def test_batch_shapes_with_usage(self):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import collate_splice

        batch = [self._make_item(has_usage=True) for _ in range(2)]
        out = collate_splice(batch)
        assert out["sequence"].shape == (2, 131072, 4)
        assert out["classification_labels"].shape == (2, 131072)
        assert out["usage_positions"].shape == (2, 4)
        assert out["usage_values"].shape == (2, 4, 3)
        assert out["usage_mask"].shape == (2, 4, 3)

    def test_batch_shapes_without_usage(self):
        from alphagenome_pytorch.extensions.finetuning.splice_datasets import collate_splice

        batch = [self._make_item(has_usage=False) for _ in range(3)]
        out = collate_splice(batch)
        assert "usage_positions" not in out
        assert out["sequence"].shape == (3, 131072, 4)
