#!/usr/bin/env python
"""Build a splice site annotation Parquet from a GTF and/or usage files.

The output Parquet contains one row per unique genomic splice site with columns:
    Chromosome  (category)
    Position    (Int64, 0-based)
    SiteType    (category: Donor+ | Acceptor+ | Donor- | Acceptor-)

Splice-site extraction rules (pyranges uses 0-based half-open [Start, End) coords):
  + strand exons sorted by Start ascending (= transcript order) [e0 … eN]:
      e0 only:       Donor+   at End-1        (no upstream intron)
      e1 … eN-1:    Acceptor+ at Start,  Donor+ at End-1
      eN only:       Acceptor+ at Start          (no downstream intron)

  − strand exons sorted by Start ascending [e0 … eN]
    (e0 is *last* in transcript, eN is *first*):
      e0 only:       Acceptor- at End-1         (no downstream intron in tx direction)
      e1 … eN-1:    Acceptor- at End-1, Donor- at Start
      eN only:       Donor-   at Start           (no upstream intron in tx direction)

Optional --usage-parquet adds the union of GTF-derived sites and sites found
in a Spliser _usage.parquet file (after applying Alpha+Beta and Alpha filters).
The companion _usage.json must reside in the same directory.

Usage
-----
  python scripts/convert_splice_sites_to_parquet.py \\
      --gtf /path/to/gencode.annotation.gtf \\
      --output splice_sites.parquet

  python scripts/convert_splice_sites_to_parquet.py \\
      --gtf /path/to/gencode.annotation.gtf \\
      --usage-parquet /path/to/spliser_output/_usage.parquet \\
      --min-coverage 10 \\
      --output splice_sites.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Site-type label → integer class (matches SpliceSitesClassificationHead)
SITE_TYPE_TO_CLASS = {
    "Donor+": 0,
    "Acceptor+": 1,
    "Donor-": 2,
    "Acceptor-": 3,
}
SITE_TYPES = list(SITE_TYPE_TO_CLASS.keys())


def _extract_from_gtf(gtf_path: Path) -> "pd.DataFrame":
    """Return DataFrame with columns [Chromosome, Position, SiteType]."""
    import pandas as pd

    print(f"Reading GTF: {gtf_path} …")
    t0 = time.perf_counter()

    if gtf_path.suffix == ".parquet":
        df = pd.read_parquet(gtf_path)
    else:
        try:
            import pyranges
            df = pyranges.read_gtf(str(gtf_path)).df
        except ImportError:
            raise ImportError(
                "pyranges is required to read GTF files. "
                "Install it with: pip install pyranges"
            )

    print(f"  Read in {time.perf_counter() - t0:.1f}s — {len(df):,} features")

    # Keep only exon rows and required columns
    exons = df[df["Feature"] == "exon"].copy()
    print(f"  Exons: {len(exons):,}")

    if len(exons) == 0:
        raise ValueError("No exon features found in GTF.")

    for col in ("Chromosome", "Start", "End", "Strand", "transcript_id"):
        if col not in exons.columns:
            raise ValueError(f"Expected column '{col}' not found in GTF.")

    # Ensure correct dtypes
    exons["Start"] = exons["Start"].astype(int)
    exons["End"] = exons["End"].astype(int)
    exons["Chromosome"] = exons["Chromosome"].astype(str)

    sites: list[tuple[str, int, str]] = []

    print("Extracting splice sites per transcript …")
    t0 = time.perf_counter()

    for tx_id, tx_exons in exons.groupby("transcript_id"):
        if len(tx_exons) < 2:
            continue  # Single-exon transcripts have no splice sites

        strand = tx_exons.iloc[0]["Strand"]
        chrom = tx_exons.iloc[0]["Chromosome"]
        tx_sorted = tx_exons.sort_values("Start")
        n = len(tx_sorted)

        for i, (_, exon) in enumerate(tx_sorted.iterrows()):
            start = int(exon["Start"])
            end = int(exon["End"])

            if strand == "+":
                is_first = i == 0
                is_last = i == n - 1
                if not is_first:                  # upstream intron → acceptor
                    sites.append((chrom, start, "Acceptor+"))
                if not is_last:                   # downstream intron → donor
                    sites.append((chrom, end - 1, "Donor+"))
            else:  # "-"
                # Sorted by Start ascending: e0 is LAST in transcript, eN is FIRST
                is_last_in_tx = i == 0
                is_first_in_tx = i == n - 1
                if not is_first_in_tx:            # has upstream intron in tx → acceptor
                    sites.append((chrom, end - 1, "Acceptor-"))
                if not is_last_in_tx:             # has downstream intron in tx → donor
                    sites.append((chrom, start, "Donor-"))

    print(f"  Extracted {len(sites):,} site records in {time.perf_counter() - t0:.1f}s")

    result = pd.DataFrame(sites, columns=["Chromosome", "Position", "SiteType"])
    return result.drop_duplicates(subset=["Chromosome", "Position", "SiteType"])


def _extract_from_usage(
    usage_parquet: Path,
    min_coverage: int,
    alpha_min: int | None,
    usage_coord_base: int,
) -> "pd.DataFrame":
    """Return DataFrame [Chromosome, Position, SiteType] from Spliser usage parquet."""
    import pandas as pd

    usage_json = usage_parquet.with_suffix(".json")

    if not usage_parquet.exists():
        raise FileNotFoundError(f"Usage parquet not found: {usage_parquet}")
    if not usage_json.exists():
        raise FileNotFoundError(f"Usage metadata not found: {usage_json}")

    with open(usage_json) as f:
        metadata = json.load(f)

    class_labels: dict[str, int] = metadata["class_labels"]
    # Reverse mapping: int → label string
    int_to_label = {v: k for k, v in class_labels.items()}
    none_class = class_labels.get("None", 4)

    print(f"Reading usage parquet: {usage_parquet} …")
    t0 = time.perf_counter()
    df = pd.read_parquet(usage_parquet)
    print(f"  Read {len(df):,} rows in {time.perf_counter() - t0:.1f}s")

    # Filter: must be a real splice site class
    df = df[df["Label"] != none_class].copy()

    # Filter by total coverage
    df = df[df["Alpha"] + df["Beta"] >= min_coverage]

    # Filter by alpha if requested
    if alpha_min is not None:
        df = df[df["Alpha"] >= alpha_min]

    print(f"  After filters: {len(df):,} sites")

    if len(df) == 0:
        return pd.DataFrame(columns=["Chromosome", "Position", "SiteType"])

    # Map Label integer → SiteType string
    df["SiteType"] = df["Label"].map(int_to_label)

    # Chromosome: convert to string; numeric chromosomes become e.g. "1" → handle "chr" prefix
    df["Chromosome"] = df["Chromosome"].astype(str)

    # Convert coordinates from 1-based to 0-based if needed
    if usage_coord_base == 1:
        df = df.copy()
        df["Position"] = df["Position"] - 1

    result = df[["Chromosome", "Position", "SiteType"]].drop_duplicates(
        subset=["Chromosome", "Position", "SiteType"]
    )
    return result.reset_index(drop=True)


def _normalize_chromosome(s: str, add_chr: bool, strip_chr: bool) -> str:
    if add_chr and not s.startswith("chr"):
        return "chr" + s
    if strip_chr and s.startswith("chr"):
        return s[3:]
    return s


def convert_splice_sites_to_parquet(
    gtf_path: str | Path | None,
    output_path: str | Path,
    usage_parquet: str | Path | None = None,
    min_coverage: int = 10,
    alpha_min: int | None = None,
    compression: str = "snappy",
    usage_coord_base: int = 0,
    add_chr_prefix: bool = False,
    strip_chr_prefix: bool = False,
) -> None:
    """Build splice site annotation Parquet from GTF and/or Spliser usage data.

    Args:
        gtf_path: Path to GTF/GFF file or a GTF Parquet produced by
            ``convert_gtf_to_parquet.py``. At least one of *gtf_path* or
            *usage_parquet* must be supplied.
        output_path: Destination Parquet file path.
        usage_parquet: Path to a ``_usage.parquet`` file produced by
            ``convert_splice_usage_to_parquet.py``. A sibling ``_usage.json``
            must exist in the same directory. Sites are unioned with
            GTF-derived sites.
        min_coverage: Minimum Alpha+Beta value for usage sites (default 10).
        alpha_min: Optional minimum Alpha value for usage sites.
        compression: Parquet compression codec (``'snappy'``, ``'gzip'``,
            ``'zstd'``, or ``'none'``).
        usage_coord_base: Coordinate base in usage parquet (1 or 0).
            Default is 0: ``convert_splice_usage_to_parquet.py`` already
            outputs 0-based positions (the position adjustments in that
            script convert Spliser coords to match GTF 0-based convention).
            Set to 1 only if the usage parquet was produced by a different
            tool that stores 1-based positions.
        add_chr_prefix: Add ``chr`` prefix to all Chromosome values.
        strip_chr_prefix: Remove ``chr`` prefix from all Chromosome values.
    """
    import pandas as pd

    if gtf_path is None and usage_parquet is None:
        raise ValueError("At least one of --gtf or --usage-parquet must be provided.")

    frames: list[pd.DataFrame] = []

    if gtf_path is not None:
        frames.append(_extract_from_gtf(Path(gtf_path)))

    if usage_parquet is not None:
        frames.append(
            _extract_from_usage(
                Path(usage_parquet),
                min_coverage=min_coverage,
                alpha_min=alpha_min,
                usage_coord_base=usage_coord_base,
            )
        )

    # ── Overlap report ──────────────────────────────────────────────────────
    if len(frames) == 2:
        gtf_df, usage_df = frames
        key = ["Chromosome", "Position", "SiteType"]
        gtf_set = set(map(tuple, gtf_df[key].drop_duplicates().values.tolist()))
        usage_set = set(map(tuple, usage_df[key].drop_duplicates().values.tolist()))
        n_gtf = len(gtf_set)
        n_usage = len(usage_set)
        n_overlap = len(gtf_set & usage_set)
        n_gtf_only = len(gtf_set - usage_set)
        n_usage_only = len(usage_set - gtf_set)
        print(f"\nOverlap between GTF-derived and usage sites:")
        print(f"  GTF sites          : {n_gtf:>10,}")
        print(f"  Usage sites        : {n_usage:>10,}")
        print(f"  Overlap (both)     : {n_overlap:>10,}  "
              f"({100*n_overlap/n_gtf:.1f}% of GTF, "
              f"{100*n_overlap/n_usage:.1f}% of usage)")
        print(f"  GTF only           : {n_gtf_only:>10,}  "
              f"({100*n_gtf_only/n_gtf:.1f}% of GTF)")
        print(f"  Usage only         : {n_usage_only:>10,}  "
              f"({100*n_usage_only/n_usage:.1f}% of usage)")
        print(f"  Union (output)     : {n_gtf + n_usage_only:>10,}")

    combined = pd.concat(frames, ignore_index=True)

    # Chromosome normalisation
    if add_chr_prefix or strip_chr_prefix:
        combined["Chromosome"] = combined["Chromosome"].apply(
            lambda c: _normalize_chromosome(c, add_chr=add_chr_prefix, strip_chr=strip_chr_prefix)
        )

    # Final deduplication and type optimisation
    combined = combined.drop_duplicates(subset=["Chromosome", "Position", "SiteType"])
    combined["Chromosome"] = combined["Chromosome"].astype("category")
    combined["SiteType"] = pd.Categorical(combined["SiteType"], categories=SITE_TYPES)
    combined["Position"] = combined["Position"].astype("Int64")
    combined = combined.sort_values(["Chromosome", "Position"]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compression_arg = None if compression == "none" else compression
    combined.to_parquet(output_path, compression=compression_arg, index=False)

    print(f"\nWrote {len(combined):,} unique splice sites → {output_path}")
    print(f"  Site-type breakdown:")
    for st, n in combined["SiteType"].value_counts().items():
        print(f"    {st:12s}: {n:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build splice site annotation Parquet from GTF/usage data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GTF only
  python scripts/convert_splice_sites_to_parquet.py \\
      --gtf gencode.v47.annotation.gtf \\
      --output human_splice_sites.parquet

  # GTF + usage data union
  python scripts/convert_splice_sites_to_parquet.py \\
      --gtf gencode.v47.annotation.gtf \\
      --usage-parquet /path/to/spliser/Homo_sapiens/_usage.parquet \\
      --min-coverage 10 \\
      --output human_splice_sites.parquet

  # Usage data only, with GTF Parquet as input
  python scripts/convert_splice_sites_to_parquet.py \\
      --gtf gencode.v47.parquet \\
      --usage-parquet /path/to/spliser/Homo_sapiens/_usage.parquet \\
      --alpha-min 5 \\
      --output human_splice_sites.parquet
""",
    )
    parser.add_argument("--gtf", type=Path, default=None,
                        help="GTF/GFF file or GTF Parquet from convert_gtf_to_parquet.py.")
    parser.add_argument("--usage-parquet", type=Path, default=None,
                        help="Path to _usage.parquet produced by convert_splice_usage_to_parquet.py. "
                             "A sibling _usage.json must exist in the same directory.")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output Parquet path.")
    parser.add_argument("--min-coverage", type=int, default=10,
                        help="Minimum Alpha+Beta for usage sites (default: 10).")
    parser.add_argument("--alpha-min", type=int, default=None,
                        help="Optional minimum Alpha count for usage sites.")
    parser.add_argument("--compression", default="snappy",
                        choices=["snappy", "gzip", "zstd", "none"],
                        help="Parquet compression codec (default: snappy).")
    parser.add_argument("--usage-coord-base", type=int, default=0, choices=[0, 1],
                        help="Coordinate base in usage parquet (default: 0 = already 0-based).")
    parser.add_argument("--add-chr-prefix", action="store_true",
                        help="Prepend 'chr' to all chromosome names (e.g. '1' → 'chr1').")
    parser.add_argument("--strip-chr-prefix", action="store_true",
                        help="Remove 'chr' prefix from chromosome names (e.g. 'chr1' → '1').")
    args = parser.parse_args()

    if args.add_chr_prefix and args.strip_chr_prefix:
        parser.error("--add-chr-prefix and --strip-chr-prefix are mutually exclusive.")

    if args.gtf is None and args.usage_parquet is None:
        parser.error("At least one of --gtf or --usage-parquet is required.")

    convert_splice_sites_to_parquet(
        gtf_path=args.gtf,
        output_path=args.output,
        usage_parquet=args.usage_parquet,
        min_coverage=args.min_coverage,
        alpha_min=args.alpha_min,
        compression=args.compression,
        usage_coord_base=args.usage_coord_base,
        add_chr_prefix=args.add_chr_prefix,
        strip_chr_prefix=args.strip_chr_prefix,
    )


if __name__ == "__main__":
    main()
