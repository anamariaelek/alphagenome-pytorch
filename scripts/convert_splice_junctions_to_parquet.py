#!/usr/bin/env python
"""Convert Spliser .combined.tsv files to junctions.parquet + junctions.json.

Builds splice-junction read counts (donor → acceptor, per condition and sample)
from the ``Partners`` column of Spliser ``.combined.tsv`` files: for each site,
``Partners`` maps every partner site to the number of split reads joining the
two.  These are the targets for the splice-junction head
(:class:`~alphagenome_pytorch.heads.SpliceSitesJunctionHead`).

Each junction is reported by Spliser at both of its ends.  Only the donor end
is used, i.e. the partners downstream of the site on the transcript
(``partner > Site`` on ``+``, ``partner < Site`` on ``−``), so every junction
is emitted exactly once regardless of how the site itself was labelled.

Coordinates are shifted exactly like the site positions in ``usage.parquet``
(see :func:`convert_splice_usage_to_parquet.adjust_splice_site_position`):
``+`` strand donors and ``−`` strand acceptors are moved by −1, so junction
ends line up with the usage sites.

Output files
------------
``junctions.parquet``
    One row per (junction, condition, sample) with columns:
    ``Chromosome, Donor, Acceptor, Strand, Condition, Sample, Count``

``junctions.json``
    ``{"condition_labels": {...}}``.  Taken from ``--usage-json`` when given,
    so that condition indices match the species' ``usage.parquet``.

Usage
-----
::

  python scripts/convert_splice_junctions_to_parquet.py \\
      --input-dir /path/to/spliser/Homo_sapiens/ \\
      --usage-json /path/to/data/Homo_sapiens/usage.json \\
      --output /path/to/data/Homo_sapiens/junctions.parquet \\
      --min-reproducibility 0.5 --strip-chr-names
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from convert_splice_usage_to_parquet import _parse_tissue_timepoint  # noqa: E402

JUNCTION_COLUMNS = ["Chromosome", "Donor", "Acceptor", "Strand", "Condition", "Sample", "Count"]


def extract_junctions(df: "pd.DataFrame") -> "pd.DataFrame":
    """Turn Spliser rows into one row per (sample, junction) seen from its donor.

    Args:
        df: Spliser rows with ``Sample, Region, Site, Strand, Partners``.

    Returns:
        DataFrame with ``Sample, Region, Donor, Acceptor, Strand, Count`` in
        Spliser coordinates (no shift applied), only for counts > 0.
    """
    import pandas as pd

    samples, regions, donors, acceptors, strands, counts = [], [], [], [], [], []
    for row in df.itertuples(index=False):
        site = int(row.Site)
        plus = row.Strand == "+"
        for partner, count in ast.literal_eval(row.Partners).items():
            if count <= 0 or (partner > site) != plus or partner == site:
                continue
            samples.append(row.Sample)
            regions.append(row.Region)
            donors.append(site)
            acceptors.append(int(partner))
            strands.append(row.Strand)
            counts.append(int(count))
    return pd.DataFrame({
        "Sample": samples,
        "Region": regions,
        "Donor": np.asarray(donors, dtype=np.int64),
        "Acceptor": np.asarray(acceptors, dtype=np.int64),
        "Strand": strands,
        "Count": np.asarray(counts, dtype=np.int64),
    })


def adjust_junction_positions(df: "pd.DataFrame") -> "pd.DataFrame":
    """Shift ``+`` donors and ``−`` acceptors by −1 (same rule as usage sites)."""
    df = df.copy()
    plus = df["Strand"] == "+"
    df.loc[plus, "Donor"] -= 1
    df.loc[~plus, "Acceptor"] -= 1
    return df


def filter_reproducible(df: "pd.DataFrame", min_reproducibility: float | None) -> "pd.DataFrame":
    """Keep junctions observed in at least this fraction of the condition's samples."""
    if min_reproducibility is None or len(df) == 0:
        return df
    n_samples = df["Sample"].nunique()
    min_samples = int(np.round(max(0.0, min(1.0, min_reproducibility)) * n_samples))
    if n_samples <= 1 or min_samples <= 0:
        return df
    key = ["Region", "Donor", "Acceptor", "Strand"]
    n_obs = df.groupby(key)["Sample"].transform("nunique")
    return df[n_obs >= min_samples]


def convert_spliser_to_junction_parquet(
    tsv_files: list[Path],
    output_path: str | Path,
    tissue_index: int = 1,
    timepoint_index: int = 2,
    condition_labels: dict[str, int] | None = None,
    compression: str = "snappy",
    strip_chr: bool = False,
    min_count: int = 1,
    min_reproducibility: float | None = None,
) -> "pd.DataFrame":
    """Full pipeline: load → extract junctions → filter → shift → export.

    Args:
        tsv_files: Spliser ``.combined.tsv`` files, one per condition.
        output_path: Destination ``junctions.parquet``.
        tissue_index, timepoint_index: Position of tissue / timepoint in the
            dot-split filename (as in ``convert_splice_usage_to_parquet.py``).
        condition_labels: ``Tissue_Timepoint`` → index.  When given, files
            for conditions not in it are skipped (e.g. empty files that the
            usage converter skipped as well).  When ``None`` it is built from
            the non-empty files in alphabetical order (as for usage files).
        compression: Parquet compression codec.
        strip_chr: Remove ``chr`` prefix from chromosome names.
        min_count: Minimum split-read count per (junction, sample).
        min_reproducibility: Minimum fraction of a condition's samples in
            which a junction must be observed.

    Returns:
        The written DataFrame.
    """
    import pandas as pd

    chunks = []
    keys = set()
    for p in tsv_files:
        tissue, timepoint = _parse_tissue_timepoint(p.name, tissue_index, timepoint_index)
        key = f"{tissue}_{timepoint}"
        if condition_labels is not None and key not in condition_labels:
            # e.g. an empty .combined.tsv, which the usage converter skips too
            print(f"  {p.name}: skipped, {key} is not a condition of the usage file")
            continue
        t0 = time.perf_counter()
        df = pd.read_csv(p, sep="\t", usecols=["Sample", "Region", "Site", "Strand", "Partners"],
                         dtype={"Sample": str, "Region": str, "Site": "Int64", "Strand": str,
                                "Partners": str})
        df = df.dropna(subset=["Region", "Site", "Strand", "Partners"])
        # Same site can be listed once per overlapping gene; keep one copy.
        df = df.drop_duplicates()
        jx = extract_junctions(df)
        n_raw = len(jx)
        jx = jx[jx["Count"] >= min_count]
        jx = filter_reproducible(jx, min_reproducibility)
        if len(jx) == 0:
            print(f"  {p.name}: no junctions, skipped")
            continue
        keys.add(key)
        jx["Condition_Key"] = key
        chunks.append(jx)
        print(f"  {p.name}: {n_raw:,} → {len(jx):,} (junction, sample) rows "
              f"({time.perf_counter() - t0:.1f}s)")

    df = pd.concat(chunks, ignore_index=True)
    df = adjust_junction_positions(df)

    if condition_labels is None:
        condition_labels = {c: i for i, c in enumerate(sorted(keys))}
    missing = sorted(set(condition_labels) - keys)
    if missing:
        print(f"  Note: no junctions for conditions {missing} (no or empty .combined.tsv)")
    df["Condition"] = df["Condition_Key"].map(condition_labels).astype(np.int32)

    df = df.rename(columns={"Region": "Chromosome"})
    if strip_chr:
        df["Chromosome"] = df["Chromosome"].str.replace(r"^chr", "", regex=True)
    df = df[JUNCTION_COLUMNS].copy()
    df["Strand"] = df["Strand"].astype("category")
    df["Sample"] = df["Sample"].astype("category")
    df = df.sort_values(["Chromosome", "Donor", "Acceptor", "Condition"]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression=None if compression == "none" else compression)
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump({"condition_labels": condition_labels}, f, indent=2)
    n_junctions = len(df[["Chromosome", "Donor", "Acceptor", "Strand"]].drop_duplicates())
    print(f"Wrote {len(df):,} rows ({n_junctions:,} unique junctions) → {output_path}")
    return df


def check_against_usage(junctions: "pd.DataFrame", usage_parquet: Path) -> dict:
    """Self-check: junction ends should coincide with ``usage.parquet`` sites.

    Usage files are filtered (coverage, reproducibility), so not every end has
    a usage row; the fraction should still be high if coordinates line up.
    """
    import pandas as pd

    usage = pd.read_parquet(usage_parquet, columns=["Chromosome", "Position"])
    usage["Chromosome"] = usage["Chromosome"].astype(str)
    sites = pd.MultiIndex.from_frame(usage.drop_duplicates())
    jx = junctions[["Chromosome", "Donor", "Acceptor"]].drop_duplicates()
    jx["Chromosome"] = jx["Chromosome"].astype(str)
    donors = pd.MultiIndex.from_arrays([jx["Chromosome"], jx["Donor"]])
    acceptors = pd.MultiIndex.from_arrays([jx["Chromosome"], jx["Acceptor"]])
    # Ends shifted by one in the wrong direction would match here instead.
    donors_off = pd.MultiIndex.from_arrays([jx["Chromosome"], jx["Donor"] + 1])
    report = {
        "donor_in_usage": float(donors.isin(sites).mean()),
        "acceptor_in_usage": float(acceptors.isin(sites).mean()),
        "donor_plus1_in_usage": float(donors_off.isin(sites).mean()),
    }
    print("Junction ends found among usage sites: "
          + ", ".join(f"{k}={v:.3f}" for k, v in report.items()))
    return report


def main() -> None:
    print(f"Command: {' '.join(sys.argv)}\n")
    parser = argparse.ArgumentParser(
        description="Convert Spliser .combined.tsv files to junctions.parquet + junctions.json.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", "-d", type=Path, help="Directory with *.combined.tsv files.")
    src.add_argument("--files", "-f", nargs="+", type=Path, help="Explicit .combined.tsv files.")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output junctions.parquet (sibling .json written automatically).")
    parser.add_argument("--usage-json", type=Path, default=None,
                        help="usage.json whose condition_labels to reuse (recommended). The "
                             "sibling usage.parquet is used for the coordinate self-check.")
    parser.add_argument("--tissue-index", type=int, default=1)
    parser.add_argument("--timepoint-index", type=int, default=2)
    parser.add_argument("--compression", default="snappy", choices=["snappy", "gzip", "zstd", "none"])
    parser.add_argument("--strip-chr-names", action="store_true")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Minimum split reads per (junction, sample) (default: 1).")
    parser.add_argument("--min-reproducibility", type=float, default=None,
                        help="Minimum fraction of a condition's samples with the junction.")
    args = parser.parse_args()

    if args.input_dir is not None:
        tsv_files = sorted(args.input_dir.glob("*.combined.tsv"))
        if not tsv_files:
            parser.error(f"No *.combined.tsv files found in {args.input_dir}")
    else:
        tsv_files = args.files
    print(f"{len(tsv_files)} .combined.tsv file(s)")

    condition_labels = None
    if args.usage_json is not None:
        with open(args.usage_json) as f:
            condition_labels = json.load(f)["condition_labels"]

    df = convert_spliser_to_junction_parquet(
        tsv_files, args.output,
        tissue_index=args.tissue_index, timepoint_index=args.timepoint_index,
        condition_labels=condition_labels, compression=args.compression,
        strip_chr=args.strip_chr_names, min_count=args.min_count,
        min_reproducibility=args.min_reproducibility,
    )
    if args.usage_json is not None:
        usage_parquet = args.usage_json.with_suffix(".parquet")
        if usage_parquet.exists():
            report = check_against_usage(df, usage_parquet)
            with open(args.output.with_suffix(".check.json"), "w") as f:
                json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
