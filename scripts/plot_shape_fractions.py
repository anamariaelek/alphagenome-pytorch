#!/usr/bin/env python3
"""Bar plot of trajectory-shape fractions across every species/tissue in a
trajectory-clustering directory (passed as the first positional argument).

Shape fractions are a **per-site** property of the GP-smoothed trajectories and do NOT depend on
the Ward clustering. For each ``<species>/<tissue>/<prefix>_gp_features.npy`` the per-site shapes
are taken from an existing ``<prefix>_site_shapes.parquet`` or ``<prefix>_clustering_metadata.parquet``
(``ShapeSite`` column) if present; otherwise they are computed with ``classify_site_shapes`` and
written to ``<prefix>_site_shapes.parquet`` (Chromosome/Position/Strand/ShapeSite/GP_mean_SSE) —
so a re-run neither recomputes nor rewrites. Then it plots the fraction, of *all* sites, of the
seven categories:

    non-dynamic high / mid / low, up, down, up-down, down-up

as a stacked bar (the seven sum to 1). One bar per tissue, grouped by species.

Colours follow the cluster-profiles palette (``DYNAMIC_SHAPE_COLORS`` in
``alphagenome_pytorch.clustering``): up = green, down = red, up-down = blue,
down-up = orange, and the non-dynamic bands are three shades of gray.

Also produces:
  - a companion "dynamic only" stacked-count bar plot
  - a per-species grouped bar plot of dynamic-shape counts by tissue
  - a boxplot summarizing, for each dynamic shape (up/down/up-down/down-up),
    the distribution across all species+tissue combinations of that shape's
    fraction *among dynamic sites only* (one value per species+tissue).

Reusable across any clustering output that follows the ``cluster_trajectories.py`` layout
(``gp_splice_usage/`` and any other reference-clustering root). Outputs are written into the
given directory unless ``--out`` is set.

Usage:
    python scripts/plot_shape_fractions.py <clustering-dir> [--out FILE.png]
                                           [--split test] [--nd-high 0.8] [--nd-low 0.2]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Palette ──────────────────────────────────────────────────────────────────
# Dynamic colours copied from alphagenome_pytorch.clustering.DYNAMIC_SHAPE_COLORS
# (the cluster-profiles scheme); non-dynamic is split into three grays (dark=high).
COLORS = {
    "nd_high":  "#525252",   # dark gray
    "nd_mid":   "#969696",   # medium gray
    "nd_low":   "#d9d9d9",   # light gray
    "up":       "#1a7a1a",   # dark green
    "down":     "#a01010",   # dark red
    "up-down":  "#1f77b4",   # blue
    "down-up":  "#ff7f0e",   # orange
}
# Legend / label order (as requested)
LEGEND_ORDER = ["nd_high", "nd_mid", "nd_low", "up", "down", "up-down", "down-up"]
LABELS = {
    "nd_high": "non-dynamic high", "nd_mid": "non-dynamic mid",
    "nd_low": "non-dynamic low", "up": "up", "down": "down",
    "up-down": "up-down", "down-up": "down-up",
}
# Bottom→top stacking: dynamic anchored at 0 (small, keeps them readable), then
# the non-dynamic bands fill up to 1.
STACK_ORDER = ["up", "down", "up-down", "down-up", "nd_low", "nd_mid", "nd_high"]

SPECIES_ORDER = ["human", "macaque", "mouse", "rat", "rabbit", "opossum"]
TISSUE_ORDER = ["Brain", "Midbrain", "Cerebellum", "Heart", "Kidney", "Liver", "Ovary", "Testis"]

# Tissue palette copied from alphagenome_pytorch.plotting.splicing.TISSUE_COLORS
TISSUE_COLORS = {
    "Brain": "#3399cc", "Midbrain": "#34b3e6", "Cerebellum": "#34ccff",
    "Heart": "#cc0100", "Kidney": "#cc9900", "Liver": "#339900",
    "Ovary": "#cc329a", "Testis": "#ff6600",
}
DYN_SHAPES = ["up", "down", "up-down", "down-up"]


def _parquet_has(path, *cols):
    """True if the parquet at `path` exists and has all of `cols` (schema-only check)."""
    if not os.path.exists(path):
        return False
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(path).schema.names)
    return all(c in names for c in cols)


def site_shapes_for(feat_path):
    """Per-site (ShapeSite, GP_mean_SSE) for one combo, reusing an existing annotation when
    present and only computing/writing when needed. Returns ``(shapes, gpmean, source)`` where
    `source` describes where the labels came from. Precedence:

      1. a sibling ``_site_shapes.parquet`` (written by a previous run or by
         ``cluster_trajectories.py --no-cluster``) — read, don't recompute or rewrite;
      2. a sibling ``_clustering_metadata.parquet`` that already carries ``ShapeSite`` — read it;
      3. otherwise compute per-site from ``_gp_features.npy`` (`classify_site_shapes`, no Ward)
         and **write** ``_site_shapes.parquet`` (only in this case).
    """
    base = feat_path[:-len("_gp_features.npy")]
    site_path = base + "_site_shapes.parquet"
    meta_path = base + "_clustering_metadata.parquet"

    if _parquet_has(site_path, "ShapeSite", "GP_mean_SSE"):
        s = pd.read_parquet(site_path, columns=["ShapeSite", "GP_mean_SSE"])
        return s["ShapeSite"].to_numpy(), s["GP_mean_SSE"].to_numpy(), f"existing {os.path.basename(site_path)}"

    if _parquet_has(meta_path, "ShapeSite", "GP_mean_SSE"):
        m = pd.read_parquet(meta_path, columns=["ShapeSite", "GP_mean_SSE"])
        return m["ShapeSite"].to_numpy(), m["GP_mean_SSE"].to_numpy(), f"existing {os.path.basename(meta_path)}"

    # compute per-site, and write _site_shapes.parquet (with coords) if we can
    from alphagenome_pytorch.clustering import classify_site_shapes  # lazy: keeps --help torch-free
    feats = np.load(feat_path)
    shapes = np.asarray(classify_site_shapes(feats))     # per-site up/down/.../non_dynamic
    gpmean = feats.mean(axis=1)
    sites_path = base + "_sites.parquet"
    if os.path.exists(sites_path):
        s = pd.read_parquet(sites_path)
        if len(s) == len(shapes):
            keep = [c for c in ("Chromosome", "Position", "Strand") if c in s.columns]
            meta = s[keep].copy()
            meta["ShapeSite"] = shapes
            meta["GP_mean_SSE"] = gpmean
            meta.to_parquet(site_path, index=False)
            return shapes, gpmean, f"computed → wrote {os.path.basename(site_path)}"
    return shapes, gpmean, "computed (no _sites.parquet — not written)"


def fractions_from_shapes(shapes, gpmean, nd_high, nd_low):
    """The seven category fractions (+ dynamic counts) from per-site shape labels and mean SSE
    (same definitions as ``clustering.shape_fraction_summary``)."""
    n = len(shapes)
    lvl = gpmean[shapes == "non_dynamic"]
    frac = lambda c: c / n if n else 0.0
    cnt = {c: int((shapes == c).sum()) for c in ("up", "down", "up-down", "down-up")}
    return {
        "nd_high": frac(int((lvl >= nd_high).sum())),
        "nd_mid":  frac(int(((lvl > nd_low) & (lvl < nd_high)).sum())),
        "nd_low":  frac(int((lvl <= nd_low).sum())),
        "up":      frac(cnt["up"]),
        "down":    frac(cnt["down"]),
        "up-down": frac(cnt["up-down"]),
        "down-up": frac(cnt["down-up"]),
        "up_n": cnt["up"], "down_n": cnt["down"],
        "up-down_n": cnt["up-down"], "down-up_n": cnt["down-up"],
    }


def collect(root, split, nd_high, nd_low):
    """Per-site shape fractions for every species/tissue under `root`. Per-site labels are read
    from an existing ``_site_shapes.parquet`` / ``_clustering_metadata.parquet`` when present, and
    only computed from ``_gp_features.npy`` (and written to ``_site_shapes.parquet``) otherwise —
    so re-runs neither recompute nor rewrite. Independent of whether Ward clustering ran."""
    rows = []
    feat_files = sorted(p for p in glob.glob(os.path.join(root, "*", "*", f"*_{split}_gp_features.npy"))
                        if not os.path.basename(p).startswith("._"))
    for p in feat_files:
        sp, tis = p.split(os.sep)[-3:-1]
        shapes, gpmean, source = site_shapes_for(p)
        print(f"  [{sp}/{tis}] {len(shapes):,} sites — {source}")
        rows.append(dict(species=sp, tissue=tis, **fractions_from_shapes(shapes, gpmean, nd_high, nd_low)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            f"No *_{split}_gp_features.npy under {root}. (Wrong --split? files carry the split "
            f"token, e.g. '{split}'. Run cluster_trajectories.py first if the directory has no "
            f"GP features.)")
    # Order rows: known species/tissues first (in canonical order), then any extras
    df["_sp"] = df["species"].apply(lambda s: SPECIES_ORDER.index(s) if s in SPECIES_ORDER else len(SPECIES_ORDER))
    df["_ti"] = df["tissue"].apply(lambda t: TISSUE_ORDER.index(t) if t in TISSUE_ORDER else len(TISSUE_ORDER))
    df = df.sort_values(["_sp", "species", "_ti", "tissue"]).drop(columns=["_sp", "_ti"]).reset_index(drop=True)
    return df


def plot(df, out, gap=0.7):
    """Stacked-bar plot: one bar per (species, tissue), grouped by species."""
    species = list(dict.fromkeys(df["species"]))
    # x position for each bar, with a gap between species groups
    x, xt, xl, group_spans, cursor = [], [], [], [], 0.0
    for sp in species:
        sub = df[df["species"] == sp]
        start = cursor
        for _, r in sub.iterrows():
            x.append(cursor); xt.append(cursor); xl.append(r["tissue"]); cursor += 1.0
        group_spans.append((sp, start, cursor - 1.0))
        cursor += gap
    x = np.array(x)

    # compact footprint, large fonts — legible once shrunk into a slide
    fig, ax = plt.subplots(figsize=(max(5.5, 0.34 * len(df) + 1.5), 3.8))
    bottom = np.zeros(len(df))
    for cat in STACK_ORDER:
        vals = df[cat].to_numpy()
        ax.bar(x, vals, bottom=bottom, width=0.82, color=COLORS[cat],
               edgecolor="white", linewidth=0.6)
        bottom += vals

    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of splice sites", fontsize=13)
    ax.set_title("Trajectory-shape composition\nacross species and tissues",
                 fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)
    ax.margins(x=0.01)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.2)

    # species group labels under the tissue labels
    ymin = -0.30
    for sp, lo, hi in group_spans:
        ax.text((lo + hi) / 2, ymin, sp, ha="center", va="top",
                fontsize=12, fontweight="bold", transform=ax.get_xaxis_transform())
        ax.plot([lo - 0.45, hi + 0.45], [ymin + 0.04, ymin + 0.04], color="0.3",
                lw=1.2, clip_on=False, transform=ax.get_xaxis_transform())

    handles = [Patch(facecolor=COLORS[c], edgecolor="white", label=LABELS[c])
               for c in LEGEND_ORDER]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=11, frameon=False, title="Shape", title_fontsize=12)

    fig.subplots_adjust(bottom=0.32, right=0.78)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}")

    # companion: dynamic categories only, as raw counts (non-dynamic excluded)
    dyn = ["up", "down", "up-down", "down-up"]
    fig2, ax2 = plt.subplots(figsize=(max(5.5, 0.34 * len(df) + 1.5), 3.8))
    bottom = np.zeros(len(df))
    for cat in dyn:
        vals = df[f"{cat}_n"].to_numpy()
        ax2.bar(x, vals, bottom=bottom, width=0.82, color=COLORS[cat],
                edgecolor="white", linewidth=0.6)
        bottom += vals
    ax2.set_xticks(xt); ax2.set_xticklabels(xl, rotation=45, ha="right", fontsize=10)
    ax2.set_ylabel("Number of splice sites", fontsize=13)
    ax2.set_title("Dynamic-shape counts\n(non-dynamic excluded)",
                  fontsize=15, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=11)
    ax2.margins(x=0.01)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax2.spines[side].set_linewidth(1.2)
    for sp, lo, hi in group_spans:
        ax2.text((lo + hi) / 2, ymin, sp, ha="center", va="top", fontsize=12,
                 fontweight="bold", transform=ax2.get_xaxis_transform())
        ax2.plot([lo - 0.45, hi + 0.45], [ymin + 0.04, ymin + 0.04], color="0.3",
                 lw=1.2, clip_on=False, transform=ax2.get_xaxis_transform())
    handles2 = [Patch(facecolor=COLORS[c], edgecolor="white", label=LABELS[c]) for c in dyn]
    ax2.legend(handles=handles2, loc="center left", bbox_to_anchor=(1.01, 0.5),
               fontsize=11, frameon=False, title="Shape", title_fontsize=12)
    fig2.subplots_adjust(bottom=0.32, right=0.78)
    out2 = os.path.splitext(out)[0] + "_dynamic_only" + os.path.splitext(out)[1]
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out2}")


def plot_by_species(df, out, ncols=1):
    """One subplot per species (stacked in a single column by default): dynamic-shape
    counts grouped by tissue, each tissue group coloured by its tissue colour (bars
    within a group = up/down/up-down/down-up). Mirrors the evo-devo "# of exons"
    figure style. X-axis shape labels are only drawn under the first (top) panel,
    since they're identical across all species."""
    from matplotlib.ticker import MaxNLocator

    species = [s for s in SPECIES_ORDER if s in set(df["species"])]
    species += [s for s in dict.fromkeys(df["species"]) if s not in species]
    n = len(species)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig_w = 3.6 * ncols
    fig_h = 1.65 * nrows + 0.75  # + extra for the first-to-second gap spacer
    fig = plt.figure(figsize=(fig_w, fig_h))

    if ncols == 1 and nrows > 1:
        # extra invisible spacer row wedged between panel 0 and panel 1 only —
        # every other row keeps its normal, tight spacing
        height_ratios = [1.45, 0.75] + [1.0] * (nrows - 1)
        gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=height_ratios,
                              hspace=0.35)
        axes = []
        for i in range(n):
            row = i if i == 0 else i + 1   # shift rows 1+ down past the spacer
            axes.append(fig.add_subplot(gs[row, 0]))
    else:
        height_ratios = [1.45] + [1.0] * (nrows - 1) if ncols == 1 else None
        gs = fig.add_gridspec(nrows, ncols,
                              height_ratios=height_ratios, hspace=0.15)
        axes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(n)]

    gap = 1.3  # blank space between tissue groups
    for ax_i, sp in enumerate(species):
        ax = axes[ax_i]
        sub = df[df["species"] == sp]
        tissues = [t for t in TISSUE_ORDER if t in set(sub["tissue"])]
        tissues += [t for t in sub["tissue"] if t not in tissues]

        cursor = 0.0
        group_centers, first_xs = [], None
        for t in tissues:
            row = sub[sub["tissue"] == t].iloc[0]
            xs = np.arange(len(DYN_SHAPES)) + cursor
            if first_xs is None:
                first_xs = xs
            vals = [row[f"{s}_n"] for s in DYN_SHAPES]
            ax.bar(xs, vals, width=0.9, color=TISSUE_COLORS.get(t, "#808080"),
                   edgecolor="white", linewidth=0.5)
            group_centers.append((t, xs.mean()))
            cursor += len(DYN_SHAPES) + gap

        # shape names labelled once, under the first (top) panel only
        ax.set_xticks(list(first_xs))
        if ax_i == 0:
            ax.set_xticklabels(DYN_SHAPES, rotation=90, fontsize=12)
            ax.tick_params(axis="x", length=2)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_title(sp, fontsize=17, loc="left", fontweight="bold")
        ax.set_ylabel("# of sites", fontsize=13)
        ax.margins(x=0.01, y=0.08)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(1.1)

    # single shared tissue-colour legend (tissues in canonical order, union over species)
    all_t = [t for t in TISSUE_ORDER if (df["tissue"] == t).any()]
    handles = [Patch(facecolor=TISSUE_COLORS.get(t, "#808080"), label=t) for t in all_t]
    fig.legend(handles=handles, loc="upper center", ncol=len(all_t), fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, 1.03), handlelength=1.2,
               columnspacing=1.4)

    fig.subplots_adjust(top=0.93, left=0.16, right=0.97, bottom=0.05)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}")


def plot_dynamic_fraction_boxplot(df, out):
    """Summarize ``shape_fractions_barplot_dynamic_only.png`` as a boxplot.

    For each species+tissue combination, compute what fraction of that
    combination's *dynamic* sites (up + down + up-down + down-up, non-dynamic
    excluded) fall into each of the four dynamic shapes. This gives one value
    per shape per species+tissue. The boxplot then shows, for each shape, the
    distribution of that value across all species+tissue combinations, with
    the individual species+tissue points overlaid as jittered dots.
    """
    dyn = DYN_SHAPES
    counts = df[[f"{c}_n" for c in dyn]].to_numpy(dtype=float)
    totals = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fracs = np.where(totals[:, None] > 0, counts / totals[:, None], np.nan)
    fracs = pd.DataFrame(fracs, columns=dyn)

    data = [fracs[c].dropna().to_numpy() for c in dyn]

    # small physical footprint (fits neatly into a slide) but large fonts so
    # it stays legible once shrunk down / projected
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    tick_labels = [LABELS[c] for c in dyn]
    box_style = dict(
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        boxprops=dict(linewidth=1.6, edgecolor="0.2"),
        medianprops=dict(color="black", linewidth=2.2),
        whiskerprops=dict(color="0.3", linewidth=1.6),
        capprops=dict(color="0.3", linewidth=1.6),
    )
    import matplotlib
    if tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) >= (3, 9):
        bp = ax.boxplot(data, tick_labels=tick_labels, **box_style)
    else:
        bp = ax.boxplot(data, labels=tick_labels, **box_style)
    for patch, c in zip(bp["boxes"], dyn):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.65)

    # overlay one jittered point per species+tissue combination
    rng = np.random.default_rng(0)
    for i, c in enumerate(dyn):
        y = fracs[c].dropna().to_numpy()
        x = rng.normal(loc=i + 1, scale=0.06, size=len(y))
        ax.scatter(x, y, color="black", s=22, alpha=0.55, zorder=3,
                   edgecolor="white", linewidth=0.4)

    ax.set_ylabel("Fraction of dynamic sites", fontsize=14)
    #ax.set_ylim(0, 1.0)
    ax.set_title("Dynamic-shape composition\nacross species/tissues",
                 fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", labelsize=13, length=0)
    ax.tick_params(axis="y", labelsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root",
                    help="Path to a trajectory-clustering directory containing "
                         "<species>/<tissue>/<prefix>_shape_fractions.csv (or "
                         "_clustering_metadata.parquet), e.g. .../gp_splice_usage.")
    ap.add_argument("--split", default="test", help="Split token in filenames (default: test)")
    ap.add_argument("--out", default=None,
                    help="Output PNG (base name for all figures). "
                         "Default: <root>/shape_fractions_barplot.png")
    ap.add_argument("--nd-high", type=float, default=0.8,
                    help="Non-dynamic high threshold (metadata fallback only)")
    ap.add_argument("--nd-low", type=float, default=0.2,
                    help="Non-dynamic low threshold (metadata fallback only)")
    args = ap.parse_args()

    out = args.out or os.path.join(args.root, "shape_fractions_barplot.png")

    df = collect(args.root, args.split, args.nd_high, args.nd_low)
    # save the tidy table alongside the figure
    table_out = os.path.splitext(out)[0] + "_table.csv"
    df.to_csv(table_out, index=False)
    print(f"{len(df)} species/tissue combinations")
    print(f"Saved -> {table_out}")
    plot(df, out)
    by_sp_out = os.path.splitext(out)[0] + "_by_species" + os.path.splitext(out)[1]
    plot_by_species(df, by_sp_out)
    box_out = os.path.splitext(out)[0] + "_dynamic_only_boxplot" + os.path.splitext(out)[1]
    plot_dynamic_fraction_boxplot(df, box_out)


if __name__ == "__main__":
    main()