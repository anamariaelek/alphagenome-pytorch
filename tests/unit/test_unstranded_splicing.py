"""Strand inference for unstranded (Strand '?') Spliser runs, e.g. Brawand/Khrameeva."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pyfaidx")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from convert_splice_junctions_to_parquet import convert_spliser_to_junction_parquet  # noqa: E402
from shift_unstranded_usage import partner_sides, shift_usage  # noqa: E402
from splice_motifs import MotifStrand  # noqa: E402

# Raw Spliser coordinates: intron (s, p) has first bases genome[s:s+2], last genome[p-2:p].
# + gene: introns (100, 200) GT-AG and (100, 300) GT-AG
# − gene: intron (800, 900) CT-AC
# non-canonical: intron (500, 600) AA-AA


def _genome(tmp_path) -> Path:
    seq = np.array(list("A" * 1000))
    for pos, motif in [(100, "GT"), (198, "AG"), (298, "AG"), (800, "CT"), (898, "AC")]:
        seq[pos:pos + 2] = list(motif)
    fa = tmp_path / "genome.fa"
    fa.write_text(">1\n" + "".join(seq) + "\n")
    return fa


def _rows(strand_plus: str, strand_minus: str):
    rows = [
        ("s1", "1", 100, strand_plus, "{200: 5, 300: 2}"),
        ("s1", "1", 200, strand_plus, "{100: 5}"),
        ("s1", "1", 300, strand_plus, "{100: 2}"),
        ("s1", "1", 900, strand_minus, "{800: 4}"),
        ("s1", "1", 800, strand_minus, "{900: 4}"),
        ("s1", "1", 500, strand_plus, "{600: 3}"),
        ("s1", "1", 600, strand_plus, "{500: 3}"),
    ]
    df = pd.DataFrame(rows, columns=["Sample", "Region", "Site", "Strand", "Partners"])
    df["Gene"] = "G"
    return df


def test_motif_strand(tmp_path):
    ms = MotifStrand(_genome(tmp_path))
    assert ms("1", 100, 200) == "+"
    assert ms("chr1", 800, 900) == "-"   # chr prefix tolerated
    assert ms("1", 500, 600) is None
    assert ms("2", 100, 200) is None     # unknown chromosome


def test_unstranded_junctions_match_stranded(tmp_path):
    """Motif-assigned '?' junctions equal what a stranded run would give."""
    fa = _genome(tmp_path)
    out = {}
    for name, (sp, sm) in {"stranded": ("+", "-"), "unstranded": ("?", "?")}.items():
        d = tmp_path / name
        d.mkdir()
        tsv = d / "Human.Brain.1.combined.tsv"
        df = _rows(sp, sm)
        if name == "stranded":
            df = df[~df["Site"].isin([500, 600])]  # drop the non-canonical intron
        df.to_csv(tsv, sep="\t", index=False)
        res = convert_spliser_to_junction_parquet([tsv], d / "junctions.parquet", genome=fa)
        out[name] = sorted(zip(res["Donor"], res["Acceptor"], res["Strand"].astype(str), res["Count"]))
    assert out["unstranded"] == out["stranded"]
    assert out["unstranded"] == [(99, 200, "+", 5), (99, 300, "+", 2), (900, 799, "-", 4)]


def test_unstranded_junctions_need_genome(tmp_path):
    tsv = tmp_path / "Human.Brain.1.combined.tsv"
    _rows("?", "?").to_csv(tsv, sep="\t", index=False)
    with pytest.raises(ValueError, match="--genome"):
        convert_spliser_to_junction_parquet([tsv], tmp_path / "j.parquet")


def test_shift_assigns_labels(tmp_path):
    fa = _genome(tmp_path)
    tsv = tmp_path / "Human.Brain.1.combined.tsv"
    _rows("?", "?").to_csv(tsv, sep="\t", index=False)
    usage = pd.DataFrame({
        "Chromosome": ["1"] * 7,
        "Position": [100, 200, 300, 800, 900, 500, 600],
        "Label": [np.nan] * 7,
        "SSE": [0.5] * 7,
    })
    sides = partner_sides([tsv], MotifStrand(fa))
    shifted, stats = shift_usage(usage, sides)
    got = dict(zip(shifted["Position"], shifted["Label"]))
    # left ends shifted −1; 0 Donor+, 1 Acceptor+, 2 Donor−, 3 Acceptor−, 4 None
    assert got == {99: 0, 200: 1, 300: 1, 799: 3, 900: 2, 499: 4, 600: 4}
    assert stats["sites_None"] == 2
    # without a genome: shift only, Label untouched
    shifted_nolabel, _ = shift_usage(usage, partner_sides([tsv]))
    assert shifted_nolabel["Label"].isna().all()
