#!/usr/bin/env python
"""Diagnose coordinate mismatch between GTF-derived and usage splice sites.

Computes overlap between the two sources across:
  1. Chromosome name formats (detects chr prefix mismatch)
  2. Per-site-type position offsets (-3 … +3)

Crucially tested per site type, because Donor+/Acceptor- get a position
adjustment in convert_splice_usage_to_parquet.py while Acceptor+/Donor- do
not.  A different optimal offset per type reveals a bug in that script.

Usage:
    python scripts/diagnose_splice_overlap.py \\
        --gtf-parquet /path/to/splice_sites.parquet \\   # GTF-only run output
        --usage-parquet /path/to/usage.parquet \\
        [--usage-coord-base 1] \\
        [--min-coverage 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SITE_TYPES = ["Donor+", "Acceptor+", "Donor-", "Acceptor-"]


def _load_gtf_sites(path: Path) -> "pd.DataFrame":
    """Load a GTF-only splice_sites.parquet (Position must already be 0-based)."""
    import pandas as pd
    df = pd.read_parquet(path)
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Position"] = df["Position"].astype(int)
    return df[["Chromosome", "Position", "SiteType"]].drop_duplicates()


def _load_usage_sites(path: Path, min_coverage: int, coord_base: int) -> "pd.DataFrame":
    """Load usage.parquet, filter, deduplicate to unique (chrom, pos, sitetype)."""
    import pandas as pd

    usage_json = path.with_suffix(".json")
    with open(usage_json) as f:
        meta = json.load(f)
    int_to_label = {v: k for k, v in meta["class_labels"].items()}
    none_class = meta["class_labels"].get("None", 4)

    df = pd.read_parquet(path)
    df = df[df["Label"] != none_class].copy()
    df = df[(df["Alpha"] + df["Beta"]) >= min_coverage]
    df["SiteType"] = df["Label"].map(int_to_label)
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Position"] = df["Position"].astype(int)
    if coord_base == 1:
        df["Position"] = df["Position"] - 1
    return df[["Chromosome", "Position", "SiteType"]].drop_duplicates()


def _overlap_at_offset(gtf_set: set, usage_df: "pd.DataFrame", offset: int) -> int:
    shifted = set(
        zip(
            usage_df["Chromosome"],
            usage_df["Position"] + offset,
            usage_df["SiteType"],
        )
    )
    return len(gtf_set & shifted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose GTF vs usage splice-site coordinate mismatch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gtf-parquet", required=True, type=Path,
                        help="splice_sites.parquet produced by --gtf only (no --usage-parquet)")
    parser.add_argument("--usage-parquet", required=True, type=Path,
                        help="usage.parquet from convert_splice_usage_to_parquet.py")
    parser.add_argument("--min-coverage", type=int, default=10)
    parser.add_argument("--usage-coord-base", type=int, default=1, choices=[0, 1],
                        help="Coordinate base in usage parquet (default: 1)")
    args = parser.parse_args()

    import pandas as pd

    print("Loading GTF sites …")
    gtf = _load_gtf_sites(args.gtf_parquet)
    print(f"  {len(gtf):,} unique (chrom, pos, type) from GTF")

    print("Loading usage sites …")
    usage = _load_usage_sites(args.usage_parquet, args.min_coverage, args.usage_coord_base)
    print(f"  {len(usage):,} unique (chrom, pos, type) from usage (coord_base={args.usage_coord_base})")

    # ── 1. Chromosome name format check ──────────────────────────────────────
    gtf_chroms = set(gtf["Chromosome"].unique())
    usage_chroms = set(usage["Chromosome"].unique())
    common_chroms = gtf_chroms & usage_chroms
    gtf_only_chroms = gtf_chroms - usage_chroms
    usage_only_chroms = usage_chroms - gtf_chroms

    print(f"\n── Chromosome names ──")
    print(f"  GTF  : {sorted(list(gtf_chroms))[:8]} … ({len(gtf_chroms)} total)")
    print(f"  Usage: {sorted(list(usage_chroms))[:8]} … ({len(usage_chroms)} total)")
    print(f"  Common: {len(common_chroms)}, GTF-only: {len(gtf_only_chroms)}, "
          f"Usage-only: {len(usage_only_chroms)}")
    if gtf_only_chroms:
        print(f"  GTF-only chroms (sample): {sorted(list(gtf_only_chroms))[:6]}")
    if usage_only_chroms:
        print(f"  Usage-only chroms (sample): {sorted(list(usage_only_chroms))[:6]}")

    # Suggest chr fix
    gtf_has_chr = any(c.startswith("chr") for c in gtf_chroms)
    usage_has_chr = any(c.startswith("chr") for c in usage_chroms)
    if gtf_has_chr != usage_has_chr:
        print(f"\n  *** CHROMOSOME NAME MISMATCH: GTF has_chr={gtf_has_chr}, "
              f"usage has_chr={usage_has_chr} ***")
        print("  Fix: use --add-chr-prefix or --strip-chr-prefix in convert_splice_sites_to_parquet.py")

    # Restrict to common chromosomes for position analysis
    gtf_common = gtf[gtf["Chromosome"].isin(common_chroms)]
    usage_common = usage[usage["Chromosome"].isin(common_chroms)]
    print(f"\n  Restricting to {len(common_chroms)} common chromosomes: "
          f"{len(gtf_common):,} GTF sites, {len(usage_common):,} usage sites")

    # ── 2. Per-site-type offset sweep ─────────────────────────────────────────
    print(f"\n── Position offset sweep (per site type) ──")
    print(f"{'Offset':>8}", end="")
    for st in SITE_TYPES:
        print(f"  {st:>12}", end="")
    print(f"  {'All':>10}")
    print("─" * 72)

    gtf_by_type: dict[str, set] = {}
    usage_by_type: dict[str, "pd.DataFrame"] = {}
    for st in SITE_TYPES:
        g = gtf_common[gtf_common["SiteType"] == st]
        gtf_by_type[st] = set(zip(g["Chromosome"], g["Position"]))
        usage_by_type[st] = usage_common[usage_common["SiteType"] == st].copy()

    best: dict[str, tuple[int, float]] = {}  # type → (best_offset, best_rate)
    for offset in range(-3, 4):
        print(f"{offset:>+8}", end="")
        all_match = 0
        all_total = 0
        for st in SITE_TYPES:
            g_set = gtf_by_type[st]
            u_df = usage_by_type[st]
            if len(g_set) == 0 or len(u_df) == 0:
                print(f"  {'N/A':>12}", end="")
                continue
            shifted = set(zip(u_df["Chromosome"], u_df["Position"] + offset))
            n_match = len(g_set & shifted)
            rate = n_match / len(g_set)
            print(f"  {n_match:>7,} {rate:>4.1%}", end="")
            all_match += n_match
            all_total += len(g_set)
            if st not in best or rate > best[st][1]:
                best[st] = (offset, rate)
        total_rate = all_match / all_total if all_total else 0
        print(f"  {all_match:>7,} {total_rate:>3.1%}")

    print()
    print("── Best offset per site type ──")
    for st in SITE_TYPES:
        if st in best:
            off, rate = best[st]
            note = ""
            if off != 0:
                note = f"  ← current code is off by {off:+d}"
            print(f"  {st:12s}: offset {off:+d}  ({rate:.1%} of GTF matched){note}")

    print()
    print("── Interpretation ──")
    offsets = [best[st][0] for st in SITE_TYPES if st in best]
    if all(o == 0 for o in offsets):
        print("  Positions look correct (offset=0 is best for all types).")
        print("  The low overlap in the full run likely has another cause")
        print("  (check --usage-coord-base, or chromosome normalization).")
    elif len(set(offsets)) == 1:
        o = offsets[0]
        print(f"  All site types best at offset {o:+d}: systematic shift.")
        print(f"  Fix: change usage_coord_base or adjust position arithmetic by {o:+d}.")
    else:
        print(f"  Site types differ in best offset: {dict(zip(SITE_TYPES, offsets))}")
        print("  Fix: per-type adjustment in adjust_splice_site_position() is incorrect.")
        adj_types = ["Donor+ (adjusted)", "Acceptor- (adjusted)"]
        no_adj_types = ["Acceptor+ (not adjusted)", "Donor- (not adjusted)"]
        print(f"  Adjusted types  {[offsets[0], offsets[3]]} → expected offset 0")
        print(f"  Unadjusted types {[offsets[1], offsets[2]]} → expected offset 0")


if __name__ == "__main__":
    main()
