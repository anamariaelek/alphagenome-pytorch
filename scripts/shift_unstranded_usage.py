#!/usr/bin/env python
"""Apply the left-intron-end −1 position shift to an unstranded usage parquet.

``convert_splice_usage_to_parquet.py`` shifts Donor+ and Acceptor− sites by −1
(first intron base → last exon base), but only when Spliser reports a strand.
Unstranded Spliser runs (Strand ``"?"``, e.g. Brawand) get no ``Label`` and no
shift, so their left intron ends sit one base to the right of the reference
coordinates.

The left/right end of the intron does not need the strand: it follows from the
``Partners`` of each site in the Spliser ``.combined.tsv`` files.  Partners at
higher coordinates → the site is a left intron end (Donor+ or Acceptor−) and is
shifted by −1.  Partners on both sides → the side with more summed read
support wins (ties: more partner positions; otherwise unshifted).

Every site of the usage parquet must be found in the combined TSVs, otherwise
the script exits with code 1.  After the shift, a left end at ``p + 1`` can land
on a right end at ``p``; both rows are kept (the count is reported).

With ``--genome``, the script also assigns ``Label`` (needed for training and for
``convert_splice_sites_to_parquet.py``, whose GTF intersect matches on site type).
Each junction's strand comes from its intron dinucleotides (GT-AG/GC-AG/AT-AC → +,
CT-AC/CT-GC/GT-AT → −; see ``splice_motifs.py``), and each site takes the strand
with more read support among the junctions on its chosen side: left end + →
Donor+, left end − → Acceptor−, right end + → Acceptor+, right end − → Donor−.
Sites with no canonical junction (or a tie) get ``None`` (4) and are ignored
downstream.

Usage::

    python scripts/shift_unstranded_usage.py \\
        --input-dir ${spliser_brawand_dir}/Homo_sapiens_brawand \\
        --usage ${test_out_dir}/Homo_sapiens/usage.parquet \\
        --backup ${test_out_dir}/Homo_sapiens/usage.unshifted.parquet \\
        [--genome ${fa_dir}/Homo_sapiens.fa]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PARTNER_RE = re.compile(r"(\d+):\s*(\d+)")


STRAND_COLS = ["plus_right", "minus_right", "plus_left", "minus_left"]


def partner_sides(tsv_files: list[Path], motif_strand=None) -> pd.DataFrame:
    """Per ``(Region, Site)``: summed partner support right/left of the site.

    Returns columns ``Chromosome, Position, right, left, nright, nleft``. With
    *motif_strand* (a :class:`splice_motifs.MotifStrand`), also the read support
    of +/− strand junctions on each side (:data:`STRAND_COLS`).
    """
    sum_cols = ["right", "left"] + (STRAND_COLS if motif_strand is not None else [])
    acc = []
    for f in tsv_files:
        t0 = time.perf_counter()
        df = pd.read_csv(f, sep="\t", usecols=["Region", "Site", "Partners"],
                         dtype={"Region": str, "Partners": str}).dropna()
        df["Site"] = df["Site"].astype(np.int64)
        # Many samples share the same Partners string; parse each unique one once.
        g = df.groupby(["Region", "Site", "Partners"], sort=False).size().reset_index(name="n")
        cols = {c: np.zeros(len(g), dtype=np.int64) for c in sum_cols + ["nright", "nleft"]}
        right, left, nright, nleft = cols["right"], cols["left"], cols["nright"], cols["nleft"]
        for i, (chrom, site, partners) in enumerate(
                zip(g["Region"].values, g["Site"].values, g["Partners"].values)):
            for p, c in PARTNER_RE.findall(partners):
                p, c = int(p), int(c)
                if p > site:
                    right[i] += c
                    nright[i] += 1
                    side = "right"
                elif p < site:
                    left[i] += c
                    nleft[i] += 1
                    side = "left"
                else:
                    continue
                if motif_strand is not None and c > 0:
                    strand = motif_strand(chrom, min(site, p), max(site, p))
                    if strand is not None:
                        cols[("plus_" if strand == "+" else "minus_") + side][i] += c
        n = g["n"].values
        g = g[["Region", "Site"]].assign(
            **{c: cols[c] * n for c in sum_cols}, nright=nright, nleft=nleft)
        agg = {c: (c, "sum") for c in sum_cols}
        agg.update(nright=("nright", "max"), nleft=("nleft", "max"))
        acc.append(g.groupby(["Region", "Site"]).agg(**agg))
        print(f"  {f.name}: {len(df):,} rows in {time.perf_counter() - t0:.0f}s", flush=True)
    res = (pd.concat(acc).groupby(level=[0, 1])
           .agg({**{c: "sum" for c in sum_cols}, "nright": "max", "nleft": "max"}).reset_index())
    return res.rename(columns={"Region": "Chromosome", "Site": "Position"})


def site_labels(sides: pd.DataFrame, left_end: pd.Series) -> np.ndarray:
    """Class label per site from the strand support on its chosen side.

    0 Donor+, 1 Acceptor+, 2 Donor−, 3 Acceptor−, 4 None (no canonical junction
    on that side, or a +/− tie).
    """
    plus = np.where(left_end, sides["plus_right"], sides["plus_left"])
    minus = np.where(left_end, sides["minus_right"], sides["minus_left"])
    label = np.full(len(sides), 4, dtype=np.int64)
    label[(plus > minus) & left_end] = 0
    label[(plus > minus) & ~left_end] = 1
    label[(minus > plus) & ~left_end] = 2
    label[(minus > plus) & left_end] = 3
    return label


def shift_usage(usage: pd.DataFrame, sides: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Shift left-intron-end sites of *usage* by −1. Returns (shifted df, stats)."""
    left_end = (sides["right"] > sides["left"]) | (
        (sides["right"] == sides["left"]) & (sides["nright"] > sides["nleft"]))
    keep = ["Chromosome", "Position", "left_end"]
    labelled = all(c in sides.columns for c in STRAND_COLS)
    if labelled:
        sides = sides.assign(NewLabel=site_labels(sides, left_end.to_numpy()))
        keep.append("NewLabel")
    sides = sides.assign(left_end=left_end)[keep]

    chrom_dtype = usage["Chromosome"].dtype
    usage = usage.copy()
    usage["Chromosome"] = usage["Chromosome"].astype(str)
    sites = usage[["Chromosome", "Position"]].drop_duplicates().merge(
        sides, how="left", on=["Chromosome", "Position"])
    missing = int(sites["left_end"].isna().sum())
    sites["left_end"] = sites["left_end"].fillna(False).astype(bool)
    sites["NewPosition"] = sites["Position"] - sites["left_end"].astype(np.int64)

    new_cols = ["NewPosition"] + (["NewLabel"] if labelled else [])
    usage = usage.merge(sites[["Chromosome", "Position"] + new_cols],
                        on=["Chromosome", "Position"], how="left", validate="m:1")
    usage["Position"] = usage.pop("NewPosition").astype(np.int64)
    usage["Chromosome"] = usage["Chromosome"].astype(chrom_dtype)
    if labelled:
        usage["Label"] = usage.pop("NewLabel").fillna(4).astype(np.int64)

    stats = {
        "rows": len(usage),
        "sites": len(sites),
        "sites_not_in_tsv": missing,
        "sites_shifted": int(sites["left_end"].sum()),
        "sites_colliding_after_shift": int(sites.duplicated(["Chromosome", "NewPosition"], keep=False).sum()),
    }
    if labelled:
        names = ["Donor+", "Acceptor+", "Donor-", "Acceptor-", "None"]
        counts = sites["NewLabel"].fillna(4).astype(int).value_counts()
        stats.update({f"sites_{n}": int(counts.get(i, 0)) for i, n in enumerate(names)})
    return usage, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory with the Spliser *.combined.tsv files used to build --usage.")
    parser.add_argument("--usage", type=Path, required=True, help="Unstranded usage.parquet to shift.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output parquet (default: overwrite --usage).")
    parser.add_argument("--backup", type=Path, default=None,
                        help="Copy the unshifted --usage here before writing.")
    parser.add_argument("--genome", type=Path, default=None,
                        help="Genome FASTA (with .fai): also assign Label from intron motifs.")
    args = parser.parse_args()

    tsv_files = sorted(p for p in args.input_dir.glob("*.combined.tsv") if not p.name.startswith("._"))
    if not tsv_files:
        sys.exit(f"No *.combined.tsv files in {args.input_dir}")
    usage = pd.read_parquet(args.usage)
    if usage["Label"].notna().any():
        sys.exit(f"{args.usage} has Labels: stranded data, or already shifted/labelled by this script.")

    motif_strand = None
    if args.genome is not None:
        from splice_motifs import MotifStrand
        motif_strand = MotifStrand(args.genome)

    print(f"Reading partners from {len(tsv_files)} TSV file(s) in {args.input_dir}")
    sides = partner_sides(tsv_files, motif_strand)
    shifted, stats = shift_usage(usage, sides)
    for k, v in stats.items():
        print(f"{k}: {v:,}")
    if stats["sites_not_in_tsv"]:
        sys.exit("Some usage sites are missing from the TSVs; wrong --input-dir? Nothing written.")

    output = args.output or args.usage
    if args.backup is not None:
        shutil.copy2(args.usage, args.backup)
        print(f"Backup: {args.backup}")
    shifted.to_parquet(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
