#!/usr/bin/env python
"""Quick sanity check: verify splice-site position encoding against canonical GT/AG motifs.

Loads annotation Parquet + genome FASTA and checks whether canonical dinucleotide
motifs are found at the expected offsets. Runs in seconds with no GPU required.

Canonical rules (0-based annotation, pyranges convention):
  Donor+   at position p  → GT at genome[p+1 : p+3]   (1st two intron bases)
  Acceptor+ at position p → AG at genome[p-2 : p]      (last two intron bases)
  Donor-   at position p  → genome[p-2 : p] == 'AC'    (RC of GT, - strand intron starts before p)
  Acceptor- at position p → genome[p+1 : p+3] == 'CT'  (RC of AG)

Usage:
    python scripts/check_splice_positions.py \\
        --annotation splice_sites.parquet \\
        --genome hg38.fa \\
        [--chrom chr1] \\
        [--n-sites 5000] \\
        [--offset 0]     # try -1 / +1 to diagnose off-by-one errors

The --offset flag shifts all positions uniformly, letting you quickly test
whether an off-by-one adjustment would improve the match rate.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def _get_fasta_reader(genome_path: str):
    """Return a function fetch(chrom, start, end) -> str (uppercase)."""
    try:
        import pyfaidx
        fa = pyfaidx.Fasta(genome_path, as_raw=True, sequence_always_upper=True)

        def fetch(chrom, start, end):
            try:
                return str(fa[chrom][start:end])
            except (KeyError, ValueError):
                return None
        return fetch
    except ImportError:
        pass

    # Fallback to pysam
    try:
        import pysam
        fa = pysam.FastaFile(genome_path)

        def fetch(chrom, start, end):
            try:
                return fa.fetch(chrom, start, end).upper()
            except (KeyError, ValueError):
                return None
        return fetch
    except ImportError:
        pass

    sys.exit(
        "ERROR: pyfaidx or pysam is required to read FASTA files.\n"
        "  pip install pyfaidx"
    )


# Expected canonical dinucleotides at the given offset from the annotated position.
# (motif, genome_start_offset, genome_end_offset)
EXPECTED: dict[str, tuple[str, int, int]] = {
    "Donor+":    ("GT", +1, +3),
    "Acceptor+": ("AG", -2,  0),
    "Donor-":    ("AC", -2,  0),   # RC of GT; intron is at lower genomic coords
    "Acceptor-": ("CT", +1, +3),   # RC of AG
}


def check(annotation_path: str, genome_path: str, chrom_filter: str | None,
          n_sites: int, offset: int) -> None:
    import pandas as pd

    print(f"Loading annotation: {annotation_path}")
    df = pd.read_parquet(annotation_path)
    df["Position"] = df["Position"].astype(int)
    df["Chromosome"] = df["Chromosome"].astype(str)

    if chrom_filter:
        df = df[df["Chromosome"] == chrom_filter]
        if df.empty:
            sys.exit(f"No sites found on chromosome '{chrom_filter}'")

    if len(df) > n_sites:
        df = df.sample(n=n_sites, random_state=42)

    print(f"Checking {len(df):,} sites (offset={offset:+d}) …")

    fetch = _get_fasta_reader(genome_path)

    total_by_type: dict[str, int] = defaultdict(int)
    match_by_type: dict[str, int] = defaultdict(int)
    mismatch_examples: dict[str, list[str]] = defaultdict(list)

    for _, row in df.iterrows():
        site_type = row["SiteType"]
        chrom = row["Chromosome"]
        pos = int(row["Position"]) + offset

        if site_type not in EXPECTED:
            continue

        motif, rel_start, rel_end = EXPECTED[site_type]
        seq = fetch(chrom, pos + rel_start, pos + rel_end)

        total_by_type[site_type] += 1
        if seq == motif:
            match_by_type[site_type] += 1
        elif len(mismatch_examples[site_type]) < 3:
            mismatch_examples[site_type].append(
                f"  {chrom}:{pos} → got '{seq}' (expected '{motif}')"
            )

    print()
    print(f"{'SiteType':<14} {'Matched':>8} {'Total':>8} {'Rate':>8}   {'Expected motif @ offset'}")
    print("-" * 65)
    all_ok = True
    for site_type in ["Donor+", "Acceptor+", "Donor-", "Acceptor-"]:
        total = total_by_type.get(site_type, 0)
        matched = match_by_type.get(site_type, 0)
        if total == 0:
            print(f"{site_type:<14} {'N/A':>8} {'0':>8}")
            continue
        rate = matched / total
        motif, rel_start, _ = EXPECTED[site_type]
        offset_str = f"'{motif}' at p{rel_start:+d}"
        flag = "✓" if rate > 0.85 else ("?" if rate > 0.50 else "✗ PROBLEM")
        print(f"{site_type:<14} {matched:>8,} {total:>8,} {rate:>7.1%}   {offset_str}  {flag}")
        if rate < 0.85:
            all_ok = False
            for ex in mismatch_examples.get(site_type, []):
                print(ex)

    print()
    if all_ok:
        print("PASS — position encoding looks correct (>85% motif match for all site types).")
    else:
        print("FAIL — some site types have low motif match rates.")
        print()
        print("Suggestions:")
        print("  1. Re-run with --offset -1 and --offset +1 to detect off-by-one errors.")
        print("  2. Check whether the annotation parquet uses 0-based or 1-based coords")
        print("     (see the --usage-coord-base flag in convert_splice_sites_to_parquet.py).")
        print("  3. Verify chromosome name format (e.g. 'chr1' vs '1').")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify splice-site position encoding against canonical GT/AG motifs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotation", required=True, help="Splice-site annotation Parquet")
    parser.add_argument("--genome", required=True, help="Reference genome FASTA")
    parser.add_argument("--chrom", default=None, help="Restrict check to one chromosome")
    parser.add_argument("--n-sites", type=int, default=5000,
                        help="Number of sites to sample (for speed)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Shift all positions by this value (0, -1, or +1 to probe off-by-one)")
    args = parser.parse_args()
    check(args.annotation, args.genome, args.chrom, args.n_sites, args.offset)


if __name__ == "__main__":
    main()
