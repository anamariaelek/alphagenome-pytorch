"""Tests for scripts/convert_splice_usage_to_parquet.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from convert_splice_usage_to_parquet import convert_spliser_dir_to_usage_parquet  # noqa: E402

COLUMNS = ["Sample", "Region", "Site", "Strand", "Gene", "SSE", "alpha_count",
           "beta1_count", "beta2_count", "MultiGeneFlag", "Others", "Partners",
           "Competitors"]


def _row(sample, site, strand, alpha, beta, partners):
    sse = alpha / (alpha + beta)
    return [sample, "1", site, strand, "G1", sse, alpha, beta, 0, "False", "[]",
            partners, "[]"]


@pytest.fixture
def combined_tsv(tmp_path):
    rows = [
        # Two replicates with identical values at a + strand donor.
        _row("repA", 1000, "+", 8, 2, "{2000: 8}"),
        _row("repB", 1000, "+", 8, 2, "{2000: 8}"),
        # Two replicates with different values at the matching acceptor.
        _row("repA", 2000, "+", 8, 4, "{1000: 8}"),
        _row("repB", 2000, "+", 6, 6, "{1000: 6}"),
        # Exact duplicate row from Spliser (same sample) is still removed.
        _row("repA", 2000, "+", 8, 4, "{1000: 8}"),
    ]
    path = tmp_path / "Human.Brain.1.combined.tsv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, sep="\t", index=False)
    return path


def test_identical_replicates_are_kept(combined_tsv, tmp_path):
    out = tmp_path / "usage.parquet"
    convert_spliser_dir_to_usage_parquet([combined_tsv], out, min_coverage=10)
    df = pd.read_parquet(out)

    assert list(df.columns) == ["Chromosome", "Position", "SSE", "Alpha", "Beta",
                                "Label", "Condition", "Sample"]
    # + strand donor shifted by -1; both identical replicates survive.
    donor = df[df["Position"] == 999]
    assert sorted(donor["Sample"].astype(str)) == ["repA", "repB"]
    assert (donor["Label"] == 0).all()
    # Acceptor: one row per sample, the exact Spliser duplicate dropped.
    acceptor = df[df["Position"] == 2000]
    assert sorted(acceptor["Sample"].astype(str)) == ["repA", "repB"]
    assert (acceptor["Label"] == 1).all()
    assert len(df) == 4
