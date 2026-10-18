#!/usr/bin/env python
"""Check a regenerated usage parquet (with ``Sample``) against the old one.

The old ``usage.parquet`` files were written without the ``Sample`` column, so
replicate rows with identical values in the same condition were collapsed by
``drop_duplicates``.  The regenerated file keeps them.  Apart from those
restored rows the two files should agree, and this script verifies that.

Hard checks (any failure → exit code 1):

1. ``class_labels`` and ``condition_labels`` in the JSON sidecars are identical.
2. Every old site ``(Chromosome, Position, Label)`` is present in the new file.
3. Per ``(site, Label, Condition)``, the set of unique ``(SSE, Alpha, Beta)``
   values is identical in both files (restricted to old sites).
4. Sites only present in the new file are explained by the cross-condition
   coverage filter: their summed coverage over *deduplicated* rows (what the
   old pipeline saw) is below ``--min-coverage`` / ``--min-alpha``.

Reported only: row counts, number of restored replicate rows, replicates per
``(site, condition)`` before/after, number of new-only sites.

Usage::

    python scripts/compare_usage_parquets.py \\
        --old data/Homo_sapiens/usage.parquet \\
        --new data/Homo_sapiens/usage.new.parquet \\
        --min-coverage 10 --report data/Homo_sapiens/usage.compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SITE = ["Chromosome", "Position", "Label"]
GROUP = SITE + ["Condition"]
VALUES = ["SSE", "Alpha", "Beta"]


def _load(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(path)
    df["Chromosome"] = df["Chromosome"].astype(str)
    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)
    return df, meta


def _replicate_hist(df: pd.DataFrame) -> dict[str, int]:
    counts = df.groupby(GROUP, observed=True).size().value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def compare(
    old_path: Path,
    new_path: Path,
    min_coverage: int | None,
    min_alpha: int | None,
) -> dict:
    old, old_meta = _load(old_path)
    new, new_meta = _load(new_path)
    failures: list[str] = []
    report: dict = {"old": str(old_path), "new": str(new_path)}

    # ── 1. Metadata ─────────────────────────────────────────────────────────
    for key in ("class_labels", "condition_labels"):
        if old_meta.get(key) != new_meta.get(key):
            failures.append(f"{key} differ between JSON sidecars")

    if "Sample" not in new.columns:
        failures.append("new file has no Sample column")
    new_dedup = new.drop(columns=["Sample"], errors="ignore").drop_duplicates()
    old_dedup = old.drop(columns=["Sample"], errors="ignore").drop_duplicates()

    report["rows_old"] = int(len(old))
    report["rows_old_dedup"] = int(len(old_dedup))
    report["rows_new"] = int(len(new))
    report["rows_new_dedup"] = int(len(new_dedup))
    report["rows_restored"] = int(len(new) - len(new_dedup))

    # ── 2. Sites ────────────────────────────────────────────────────────────
    old_sites = old_dedup[SITE].drop_duplicates()
    new_sites = new_dedup[SITE].drop_duplicates()
    sites = old_sites.merge(new_sites, on=SITE, how="outer", indicator=True)
    lost = sites[sites["_merge"] == "left_only"]
    gained = sites[sites["_merge"] == "right_only"][SITE]
    report["sites_old"] = int(len(old_sites))
    report["sites_new"] = int(len(new_sites))
    report["sites_lost"] = int(len(lost))
    report["sites_gained"] = int(len(gained))
    if len(lost):
        failures.append(
            f"{len(lost):,} old sites missing from new file, e.g. "
            f"{lost[SITE].head(5).to_dict('records')}"
        )

    # ── 3. Values per (site, condition), restricted to old sites ──────────
    new_on_old = new_dedup.merge(old_sites, on=SITE, how="inner")
    a = old_dedup[GROUP + VALUES].sort_values(GROUP + VALUES).reset_index(drop=True)
    b = new_on_old[GROUP + VALUES].sort_values(GROUP + VALUES).reset_index(drop=True)
    if len(a) != len(b):
        diff = a.merge(b, how="outer", on=GROUP + VALUES, indicator=True)
        diff = diff[diff["_merge"] != "both"]
        failures.append(
            f"unique (site, condition, values) rows differ: old={len(a):,} "
            f"new={len(b):,}; {len(diff):,} mismatching rows, e.g. "
            f"{diff.head(5).to_dict('records')}"
        )
    else:
        # Float columns: compare with a tolerance; integer columns exactly.
        same = (
            (a[GROUP + ["Alpha", "Beta"]].to_numpy() == b[GROUP + ["Alpha", "Beta"]].to_numpy()).all()
            and np.allclose(a["SSE"].to_numpy(), b["SSE"].to_numpy(), equal_nan=True)
        )
        if not same:
            failures.append("values of (site, condition) rows differ between old and new")

    # ── 4. Gained sites must be explained by the cross-condition filter ───
    if len(gained):
        # The old pipeline summed over deduplicated rows per (Chromosome, Position),
        # across all labels and conditions.
        gained_pos = gained[["Chromosome", "Position"]].drop_duplicates()
        g = new_dedup.merge(gained_pos, on=["Chromosome", "Position"], how="inner")
        agg = g.groupby(["Chromosome", "Position"], as_index=False)[["Alpha", "Beta"]].sum()
        below = pd.Series(False, index=agg.index)
        if min_coverage is not None:
            below |= (agg["Alpha"] + agg["Beta"]) < min_coverage
        if min_alpha is not None:
            below |= agg["Alpha"] < min_alpha
        unexplained = agg[~below]
        report["sites_gained_unexplained"] = int(len(unexplained))
        if len(unexplained):
            failures.append(
                f"{len(unexplained):,} gained sites are not explained by the "
                f"coverage filter, e.g. {unexplained.head(5).to_dict('records')}"
            )

    report["replicates_per_site_condition_old"] = _replicate_hist(old)
    report["replicates_per_site_condition_new"] = _replicate_hist(new)
    report["max_replicates_new"] = int(new.groupby(GROUP, observed=True).size().max())
    report["failures"] = failures
    report["passed"] = not failures
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--old", type=Path, required=True, help="Old usage.parquet (no Sample column).")
    parser.add_argument("--new", type=Path, required=True, help="Regenerated usage parquet (with Sample).")
    parser.add_argument("--min-coverage", type=int, default=None,
                        help="--min-coverage used to build both files (explains gained sites).")
    parser.add_argument("--min-alpha", type=int, default=None,
                        help="--min-alpha used to build both files (explains gained sites).")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report = compare(args.old, args.new, args.min_coverage, args.min_alpha)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report is not None:
        args.report.write_text(text + "\n")
    if report["passed"]:
        print("\nPASSED")
        if report["sites_gained"]:
            print(f"NOTE: {report['sites_gained']:,} sites are new; regenerate the "
                  "splice_sites_*usage*.parquet files derived from this usage file.")
    else:
        print("\nFAILED:")
        for f in report["failures"]:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
