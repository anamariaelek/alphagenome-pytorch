#!/usr/bin/env python
"""Convert Spliser .combined.tsv files to _usage.parquet + _usage.json.

Reads one or more Spliser ``.combined.tsv`` files, classifies each site as
Donor/Acceptor/Both/None, applies a position correction to align coordinates
with GTF-derived exon boundaries, and writes the result as a compact Parquet
file that can be consumed by :class:`SpliceSiteUsageIndex` and by
``convert_splice_sites_to_parquet.py --usage-dir``.

Output files
------------
``_usage.parquet``
    One row per (site, condition) pair with columns:
    ``Chromosome, Position, SSE, Alpha, Beta, Label, Condition``

``_usage.json``
    Companion metadata::

        {
          "class_labels":     {"Donor+": 0, "Acceptor+": 1, "Donor-": 2,
                               "Acceptor-": 3, "None": 4},
          "condition_labels": {"Brain_1": 0, "Brain_2": 1, ...}
        }

Input file format
-----------------
Spliser ``.combined.tsv`` tab-separated files with at least these columns:
``Region, Site, Strand, Partners, SSE, alpha_count, beta1_count, beta2_count``

Filename convention (used to extract tissue/timepoint):
``{anything}.{Tissue}.{Timepoint}.combined.tsv``
e.g. ``Human.Brain.1.combined.tsv`` → tissue=Brain, timepoint=1

Position adjustment (aligns with GTF-derived exon boundary coordinates):
  + strand Donor   : ``Site - 2``
  − strand Acceptor: ``Site - 2``
  All other types  : Site unchanged

After adjustment, positions are stored 1-based to match the expected
``SpliceSiteUsageIndex(usage_coord_base=1)`` default.

Usage
-----
::

  # All .combined.tsv files in a directory → sibling _usage.parquet
  python scripts/convert_splice_usage_to_parquet.py \\
      --input-dir /path/to/spliser/Homo_sapiens/ \\
      --output    /path/to/spliser/Homo_sapiens/_usage.parquet

  # Explicit file list
  python scripts/convert_splice_usage_to_parquet.py \\
      --files Human.Brain.*.combined.tsv \\
      --output human_usage.parquet

  # Choose tissue/timepoint index in the dot-split filename
  python scripts/convert_splice_usage_to_parquet.py \\
      --input-dir /path/to/Mus_musculus/ \\
      --tissue-index 1 --timepoint-index 2 \\
      --output /path/to/Mus_musculus/_usage.parquet
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Core transformation functions
# ──────────────────────────────────────────────────────────────────────────────


def annotate_splice_site_type(df: "pd.DataFrame") -> "pd.DataFrame":
    """Classify each Spliser row as Donor, Acceptor, Both, or None.

    Uses the ``Partners`` column (a stringified dict mapping partner positions
    to read counts) together with ``Strand`` and ``Site`` to determine which
    side of the junction each row represents:

    - ``+`` strand: partners > Site → Donor,  partners < Site → Acceptor
    - ``-`` strand: partners < Site → Donor,  partners > Site → Acceptor

    Args:
        df: Raw combined DataFrame with columns ``Site``, ``Strand``,
            ``Partners``.

    Returns:
        Copy of *df* with an added ``Splice_Site_Type`` column containing
        ``"Donor"``, ``"Acceptor"``, ``"Both"``, or ``"None"``.
    """
    import pandas as pd

    splice_site_types: list[str] = []
    for row in df.itertuples(index=False):
        site = row.Site
        strand = row.Strand
        partners = ast.literal_eval(row.Partners)  # safe alternative to eval()

        if strand == "+":
            donor_partners = [p for p in partners if p > site]
            acceptor_partners = [p for p in partners if p < site]
        else:
            donor_partners = [p for p in partners if p < site]
            acceptor_partners = [p for p in partners if p > site]

        if donor_partners and acceptor_partners:
            splice_site_types.append("Both")
        elif donor_partners:
            splice_site_types.append("Donor")
        elif acceptor_partners:
            splice_site_types.append("Acceptor")
        else:
            splice_site_types.append("None")

    df = df.copy()
    df["Splice_Site_Type"] = splice_site_types
    return df


def adjust_splice_site_position(df: "pd.DataFrame") -> "pd.DataFrame":
    """Shift Donor+/Acceptor- site positions by −2 to match exon-boundary coords.

    Spliser records the first intronic base; the GTF convention used in this
    codebase records the last exonic base.  The correction is:

    - ``+`` strand Donor  : ``Site − 2``  (first intron base → last exon base)
    - ``−`` strand Acceptor: ``Site − 2``
    - All other combinations: Site unchanged

    Args:
        df: DataFrame with ``Site``, ``Strand``, and ``Splice_Site_Type``
            columns.

    Returns:
        Copy of *df* with adjusted ``Site`` values.
    """
    df = df.copy()
    mask = (
        ((df["Strand"] == "+") & (df["Splice_Site_Type"] == "Donor"))
        | ((df["Strand"] == "-") & (df["Splice_Site_Type"] == "Acceptor"))
    )
    df.loc[mask, "Site"] = df.loc[mask, "Site"] - 2
    return df


def convert_usage_to_parquet(
    df: "pd.DataFrame",
    class_labels: dict[str, int] | None = None,
    condition_labels: dict[str, int] | None = None,
    output_path: str | Path | None = None,
    compression: str = "snappy",
) -> "pd.DataFrame | None":
    """Transform the annotated/adjusted DataFrame into the usage Parquet format.

    Steps:
    1. Merge ``beta1_count + beta2_count`` → ``Beta``.
    2. Expand "Both" rows into separate Donor and Acceptor rows.
    3. Concatenate ``Splice_Site_Type + Strand`` as the full class label
       (e.g. ``"Donor+"``, ``"Acceptor-"``); ``"None+/None-"`` → ``"None"``.
    4. Map class strings → integer ``Label`` using *class_labels*.
    5. Build ``Condition`` integer index from ``Tissue_Timepoint`` strings.
    6. Write ``_usage.parquet`` and (sibling) ``_usage.json``.

    Args:
        df: Annotated, position-adjusted DataFrame (output of
            :func:`adjust_splice_site_position`).
        class_labels: Optional mapping ``{label_str: int}``.  Default:
            ``{"Donor+": 0, "Acceptor+": 1, "Donor-": 2, "Acceptor-": 3, "None": 4}``.
        condition_labels: Optional pre-built condition name → index mapping.
            When ``None`` the mapping is derived from the data in
            ``Tissue_Timepoint`` alphabetical order.
        output_path: If given, write Parquet to this path and JSON to the
            sibling ``.json`` path.  If ``None``, return the transformed
            DataFrame.
        compression: Parquet compression codec (``"snappy"``, ``"gzip"``,
            ``"zstd"``, or ``"none"``).

    Returns:
        The transformed DataFrame when *output_path* is ``None``, otherwise
        ``None``.
    """
    import pandas as pd

    required_cols = {"Tissue", "Timepoint", "Region", "Site", "Strand",
                     "Splice_Site_Type", "SSE", "alpha_count",
                     "beta1_count", "beta2_count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    df = df[list(required_cols)].drop_duplicates().copy()

    # ── Beta = beta1 + beta2 ─────────────────────────────────────────────────
    df["beta_count"] = df["beta1_count"] + df["beta2_count"]
    df = df.drop(columns=["beta1_count", "beta2_count"])

    # ── Type casts ───────────────────────────────────────────────────────────
    df["Tissue"] = df["Tissue"].astype(str)
    df["Timepoint"] = df["Timepoint"].astype(int)
    df["Region"] = df["Region"].astype(str)
    df["Site"] = df["Site"].astype(int)
    df["Strand"] = df["Strand"].astype(str)
    df["SSE"] = df["SSE"].astype(float)
    df["alpha_count"] = df["alpha_count"].astype(int)
    df["beta_count"] = df["beta_count"].astype(int)

    # ── Expand "Both" rows ──────────────────────────────────────────────────
    both_mask = df["Splice_Site_Type"] == "Both"
    df_both = df[both_mask]
    df_donor = df_both.copy()
    df_donor["Splice_Site_Type"] = "Donor"
    df_acceptor = df_both.copy()
    df_acceptor["Splice_Site_Type"] = "Acceptor"
    df = pd.concat([df[~both_mask], df_donor, df_acceptor], ignore_index=True)

    # ── Full class label = type + strand (e.g. "Donor+", "Acceptor-") ───────
    df["Splice_Site_Type"] = df["Splice_Site_Type"] + df["Strand"]
    # "None+" / "None-" → "None"
    none_mask = df["Splice_Site_Type"].str.startswith("None")
    df.loc[none_mask, "Splice_Site_Type"] = "None"

    if class_labels is None:
        class_labels = {
            "Donor+": 0,
            "Acceptor+": 1,
            "Donor-": 2,
            "Acceptor-": 3,
            "None": 4,
        }
    df["Label"] = df["Splice_Site_Type"].map(class_labels)
    df = df.drop(columns=["Splice_Site_Type", "Strand"])

    # ── Condition index ──────────────────────────────────────────────────────
    df["Condition_Key"] = df["Tissue"] + "_" + df["Timepoint"].astype(str)
    if condition_labels is None:
        unique_conditions = sorted(df["Condition_Key"].unique())
        condition_labels = {cond: idx for idx, cond in enumerate(unique_conditions)}
    df["Condition_Idx"] = df["Condition_Key"].map(condition_labels)
    df = df.drop(columns=["Tissue", "Timepoint", "Condition_Key"])

    # ── Rename to canonical column names ────────────────────────────────────
    df = df.rename(
        columns={
            "Region": "Chromosome",
            "Site": "Position",
            "alpha_count": "Alpha",
            "beta_count": "Beta",
            "Condition_Idx": "Condition",
        }
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        "class_labels": class_labels,
        "condition_labels": condition_labels,
    }

    # ── Write or return ───────────────────────────────────────────────────────
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compression_arg = None if compression == "none" else compression
        df.to_parquet(output_path, index=False, compression=compression_arg)
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Wrote {len(df):,} rows → {output_path}")
        print(f"Wrote metadata → {json_path}")
        print(f"Conditions ({len(condition_labels)}): {list(condition_labels)[:5]}"
              + (" …" if len(condition_labels) > 5 else ""))
        label_counts = df["Label"].value_counts().sort_index()
        inv_labels = {v: k for k, v in class_labels.items()}
        print("Label breakdown:")
        for lbl, n in label_counts.items():
            print(f"  {inv_labels.get(int(lbl), str(lbl)):12s} ({int(lbl)}): {n:,}")
        return None
    else:
        return df


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────


def _parse_tissue_timepoint(
    filename: str,
    tissue_index: int,
    timepoint_index: int,
) -> tuple[str, int]:
    """Extract tissue and timepoint from a dot-split filename."""
    parts = Path(filename).name.split(".")
    try:
        tissue = parts[tissue_index]
        timepoint = int(parts[timepoint_index])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Cannot extract tissue (index {tissue_index}) / timepoint "
            f"(index {timepoint_index}) from filename '{filename}': {exc}"
        ) from exc
    return tissue, timepoint


def build_combined_dataframe(
    tsv_files: list[Path],
    tissue_index: int,
    timepoint_index: int,
) -> "pd.DataFrame":
    """Load and concatenate all .combined.tsv files into one DataFrame."""
    import pandas as pd

    dfs: list[pd.DataFrame] = []
    for p in tsv_files:
        tissue, timepoint = _parse_tissue_timepoint(p.name, tissue_index, timepoint_index)
        print(f"  Loading {p.name} (tissue={tissue}, timepoint={timepoint}) …", end=" ")
        t0 = time.perf_counter()
        df_temp = pd.read_csv(p, sep="\t")

        # Drop duplicate rows reported by Spliser
        n_before = len(df_temp)
        df_temp = df_temp.drop_duplicates()
        n_dup = n_before - len(df_temp)
        if n_dup:
            print(f"removed {n_dup:,} duplicates; ", end="")

        df_temp["Tissue"] = tissue
        df_temp["Timepoint"] = timepoint
        dfs.append(df_temp)
        print(f"{len(df_temp):,} rows ({time.perf_counter()-t0:.1f}s)")

    return pd.concat(dfs, ignore_index=True)


def convert_spliser_dir_to_usage_parquet(
    tsv_files: list[Path],
    output_path: str | Path,
    tissue_index: int = 1,
    timepoint_index: int = 2,
    class_labels: dict[str, int] | None = None,
    condition_labels: dict[str, int] | None = None,
    compression: str = "snappy",
) -> None:
    """Full pipeline: load → annotate → adjust → export.

    Args:
        tsv_files: List of Spliser ``.combined.tsv`` Path objects.
        output_path: Destination ``_usage.parquet`` file path.
        tissue_index: Zero-based index of the tissue token in the
            dot-split filename (default: 1, e.g. ``Human.Brain.1.combined.tsv``).
        timepoint_index: Zero-based index of the timepoint token (default: 2).
        class_labels: Optional custom label map (see :func:`convert_usage_to_parquet`).
        condition_labels: Optional pre-built condition index map.
        compression: Parquet compression codec.
    """
    print(f"\n{'─'*60}")
    print(f"Step 1/4  Loading {len(tsv_files)} TSV file(s)")
    print(f"{'─'*60}")
    df = build_combined_dataframe(tsv_files, tissue_index, timepoint_index)
    print(f"  Total rows loaded: {len(df):,}")

    print(f"\n{'─'*60}")
    print("Step 2/4  Annotating splice site types")
    print(f"{'─'*60}")
    t0 = time.perf_counter()
    df = annotate_splice_site_type(df)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    type_counts = df["Splice_Site_Type"].value_counts()
    for t, n in type_counts.items():
        print(f"  {t:10s}: {n:,}")

    print(f"\n{'─'*60}")
    print("Step 3/4  Adjusting splice site positions")
    print(f"{'─'*60}")
    df = adjust_splice_site_position(df)
    print("  Done.")

    print(f"\n{'─'*60}")
    print("Step 4/4  Converting to parquet")
    print(f"{'─'*60}")
    convert_usage_to_parquet(
        df,
        class_labels=class_labels,
        condition_labels=condition_labels,
        output_path=output_path,
        compression=compression,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Spliser .combined.tsv files to _usage.parquet + _usage.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All .combined.tsv in directory
  python scripts/convert_splice_usage_to_parquet.py \\
      --input-dir /path/to/spliser/Homo_sapiens/ \\
      --output /path/to/spliser/Homo_sapiens/_usage.parquet

  # Explicit file list (shell glob)
  python scripts/convert_splice_usage_to_parquet.py \\
      --files Human.Brain.*.combined.tsv Human.Heart.*.combined.tsv \\
      --output human_subset_usage.parquet

  # Custom tissue/timepoint position in filename
  # e.g. 'Species.Organism.Tissue.Timepoint.combined.tsv' → indices 2, 3
  python scripts/convert_splice_usage_to_parquet.py \\
      --input-dir /path/to/Mus_musculus/ \\
      --tissue-index 2 --timepoint-index 3 \\
      --output /path/to/Mus_musculus/_usage.parquet
""",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input-dir", "-d",
        type=Path,
        metavar="DIR",
        help="Directory containing *.combined.tsv files.",
    )
    src.add_argument(
        "--files", "-f",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="Explicit list of .combined.tsv files.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output path for _usage.parquet (sibling .json written automatically).",
    )
    parser.add_argument(
        "--tissue-index",
        type=int,
        default=1,
        metavar="N",
        help="Zero-based index of the tissue token in the dot-split filename (default: 1).",
    )
    parser.add_argument(
        "--timepoint-index",
        type=int,
        default=2,
        metavar="N",
        help="Zero-based index of the timepoint token in the dot-split filename (default: 2).",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "gzip", "zstd", "none"],
        help="Parquet compression codec (default: snappy).",
    )
    args = parser.parse_args()

    # ── Resolve input files ───────────────────────────────────────────────────
    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            parser.error(f"--input-dir is not a directory: {args.input_dir}")
        tsv_files = sorted(args.input_dir.glob("*.combined.tsv"))
        if not tsv_files:
            parser.error(f"No *.combined.tsv files found in {args.input_dir}")
        print(f"Found {len(tsv_files)} .combined.tsv file(s) in {args.input_dir}")
    else:
        tsv_files = []
        for p in args.files:
            if not p.exists():
                parser.error(f"File not found: {p}")
            tsv_files.append(p)

    convert_spliser_dir_to_usage_parquet(
        tsv_files=tsv_files,
        output_path=args.output,
        tissue_index=args.tissue_index,
        timepoint_index=args.timepoint_index,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
