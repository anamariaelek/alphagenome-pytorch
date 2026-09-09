#!/usr/bin/env python
"""Shape agreement against the replicate-split noise ceiling, by observed class.

Shape agreement is the per-site Pearson r between the observed and predicted
centred trajectory (spec v4); the ceiling is the replicate-split reliability of
the observed trajectory itself, i.e. the largest r any model could reach at this
data quality. Each bar is a grey ceiling with the achieved median r drawn inside
it, so the empty part of the grey bar is the headroom left.

Inputs are the trajectory-evaluation tables (not the devAS call table, so this is
separate from devas_report.py):
    traj_eval_by_pattern_class_pooled_full.csv   pooled over sites, per species
    traj_eval_by_pattern_class_full.csv          per species x tissue

Both are read with stratifier == 'pattern_true' (bars are keyed by the OBSERVED
class, so a class's sites are fixed regardless of what the model predicted) and
reference == 'counts'.

    python traj_shape_agreement.py --traj-dir DIR --fig-dir DIR --tab-dir DIR

Outputs:
    traj_shape_agreement.png                    overall + per-species panels
    traj_shape_agreement_by_species_tissue.png  species (rows) x tissue (columns) grid
    traj_shape_agreement.csv                    every bar as a row
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PORD = ["up", "down", "up-down", "down-up", "none"]
PABB = {"up": "U", "down": "D", "up-down": "UD", "down-up": "DU", "none": "N"}
# same order as devas_report.py, so the two by-species-tissue grids align row-for-row
SP_ORDER = ["human", "macaque", "mouse", "rat", "rabbit", "opossum", "chicken"]
TORD = ["Brain", "Cerebellum", "Midbrain", "Heart", "Kidney", "Liver", "Ovary", "Testis"]

CEIL = "#c9c9c9"      # replicate-split noise ceiling
MODEL = "#1f6fb4"     # achieved median r
CEIL_EDGE = "#9e9e9e"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def load(traj_dir):
    keep = ["pattern", "n_sites", "r_median", "ceiling_r_median",
            "r_over_ceiling_median", "r_frac_pos"]
    pooled = pd.read_csv(os.path.join(traj_dir, "traj_eval_by_pattern_class_pooled_full.csv"))
    perst = pd.read_csv(os.path.join(traj_dir, "traj_eval_by_pattern_class_full.csv"))
    sel = lambda d: d[(d.stratifier == "pattern_true") & (d.reference == "counts")]
    pooled = sel(pooled)[["scope"] + keep].copy()
    perst = sel(perst)[["Species", "Tissue"] + keep].copy()
    return pooled, perst


def bars(ax, d, xlabels, annotate=True, fs=6.0):
    """Grey ceiling bars with the achieved median r drawn inside each one."""
    x = np.arange(len(d))
    ax.bar(x, d["ceiling_r_median"], width=0.74, color=CEIL,
           edgecolor=CEIL_EDGE, linewidth=0.5, zorder=2)
    ax.bar(x, d["r_median"], width=0.40, color=MODEL, zorder=3)
    ax.axhline(0, color="0.4", lw=0.6, zorder=4)
    if annotate:
        for i, (r, cl) in enumerate(zip(d["r_median"], d["ceiling_r_median"])):
            ax.text(i, cl + 0.025, "%.0f%%" % (100 * r / cl) if cl > 0 else "—",
                    ha="center", va="bottom", fontsize=fs, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(-0.02, 1.10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


def order_classes(d):
    d = d.set_index("pattern").reindex(PORD).reset_index()
    return d


def fig_overall(pooled, fig_dir):
    """Overall bars plus one small panel per species."""
    spp = [s for s in SP_ORDER if s in set(pooled["scope"])]
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.52, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    d = order_classes(pooled[pooled["scope"] == "all species"])
    bars(ax, d, ["%s\nn=%s" % (p, format(int(n), ",")) for p, n in zip(d["pattern"], d["n_sites"])],
         fs=7.0)
    ax.set_ylabel("Shape agreement (median Pearson r)")
    ax.set_title("Shape agreement by observed trajectory class, against the noise ceiling")
    ax.legend(handles=[Patch(facecolor=CEIL, edgecolor=CEIL_EDGE, linewidth=0.5,
                            label="replicate-split noise ceiling"),
                       Patch(facecolor=MODEL, label="model")],
              loc="upper right", ncol=2)

    inner = gs[1, :].subgridspec(1, len(spp), wspace=0.30)
    for k, s in enumerate(spp):
        axk = fig.add_subplot(inner[0, k])
        ds = order_classes(pooled[pooled["scope"] == s])
        # percentages omitted here: five per panel collide. They are in the table.
        bars(axk, ds, [PABB[p] for p in ds["pattern"]], annotate=False)
        axk.set_title("%s (n=%s)" % (s, format(int(ds["n_sites"].sum()), ",")), fontsize=6.5)
        if k:
            axk.set_yticklabels([])
        else:
            axk.set_ylabel("median r")
    fig.text(0.5, 0.005, "U up · D down · UD up-down · DU down-up · N none",
             ha="center", fontsize=6, color="#666666")
    fig.savefig(os.path.join(fig_dir, "traj_shape_agreement.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_by_species_tissue(perst, fig_dir):
    """Species x tissue grid: one row per species, one column per tissue.

    Laid out on the same grid as devas_confusion_by_species_tissue.png (same
    species row order, same tissue column order), so panels for a given
    species x tissue sit at the same position in both figures. Cells with no
    data are left blank rather than closing the gap, which keeps the columns
    comparable down a species.
    """
    sps = [s for s in SP_ORDER if s in set(perst["Species"])]
    tis = [t for t in TORD if t in set(perst["Tissue"])]
    fig, axes = plt.subplots(len(sps), len(tis),
                             figsize=(1.30 * len(tis) + 0.8, 1.15 * len(sps) + 0.8),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.22, "wspace": 0.14})
    axes = np.atleast_2d(axes)
    present = set()
    for i, s in enumerate(sps):
        for j, t in enumerate(tis):
            ax = axes[i, j]
            d = perst[(perst.Species == s) & (perst.Tissue == t)]
            if d.empty:
                ax.set_axis_off()
                continue
            present.add((i, j))
            d = order_classes(d)
            bars(ax, d, [PABB[p] for p in d["pattern"]], annotate=False)
            # inside the axes, not as a title: the rows sit close enough that a
            # title reads as belonging to the panel above it
            ax.text(0.98, 0.99, "n=%s" % format(int(d["n_sites"].sum()), ","),
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=5.4, color="#777777")
            ax.tick_params(labelbottom=False, labelleft=False)
    # Tick labels on the outermost drawn panel of each row and column: sharex/sharey
    # would otherwise leave a row or column whose edge cell is blank unlabelled.
    for i in range(len(sps)):
        js = [j for (ii, j) in present if ii == i]
        if js:
            axes[i, min(js)].tick_params(labelleft=True)
    for j in range(len(tis)):
        iis = [i for (i, jj) in present if jj == j]
        if iis:
            ax = axes[max(iis), j]
            ax.set_xticks(range(len(PORD)))
            ax.set_xticklabels([PABB[p] for p in PORD])
            ax.tick_params(labelbottom=True)
    for j, t in enumerate(tis):
        axes[0, j].text(0.5, 1.22, t, transform=axes[0, j].transAxes,
                        ha="center", va="bottom", fontsize=7.5)
    for i, s in enumerate(sps):
        axes[i, 0].text(-0.42, 0.5, s.capitalize(), transform=axes[i, 0].transAxes,
                        ha="right", va="center", fontsize=7.5, rotation=90)
    fig.supylabel("Shape agreement (median Pearson r)", fontsize=8, x=0.02)
    fig.supxlabel("Observed trajectory class", fontsize=8, y=0.01)
    fig.legend(handles=[Patch(facecolor=CEIL, edgecolor=CEIL_EDGE, linewidth=0.5,
                             label="replicate-split noise ceiling"),
                        Patch(facecolor=MODEL, label="model")],
               loc="center left", bbox_to_anchor=(0.995, 0.5), ncol=1, fontsize=7.5)
    fig.suptitle("Shape agreement vs noise ceiling, per species x tissue x observed class",
                 x=0.5, y=1.035, fontsize=9)
    fig.text(0.5, -0.022, "U up · D down · UD up-down · DU down-up · N none",
             ha="center", fontsize=6.5, color="#666666")
    fig.savefig(os.path.join(fig_dir, "traj_shape_agreement_by_species_tissue.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj-dir", default="devas/traj_eval",
                   help="directory holding the two traj_eval_by_pattern_class tables")
    p.add_argument("--fig-dir", default="devas/figures")
    p.add_argument("--tab-dir", default="devas/tables")
    a = p.parse_args()

    style()
    pooled, perst = load(a.traj_dir)
    for d in (a.fig_dir, a.tab_dir):
        os.makedirs(d, exist_ok=True)

    fig_overall(pooled, a.fig_dir)
    fig_by_species_tissue(perst, a.fig_dir)

    out = pd.concat([
        pooled.assign(level=np.where(pooled["scope"] == "all species", "overall", "species"),
                      Species=pooled["scope"], Tissue="all"),
        perst.assign(level="species x tissue", scope=perst["Species"]),
    ], ignore_index=True)
    out["frac_of_ceiling_medians"] = (out["r_median"] / out["ceiling_r_median"]).round(4)
    out = out[["level", "Species", "Tissue", "pattern", "n_sites", "r_median",
               "ceiling_r_median", "frac_of_ceiling_medians",
               "r_over_ceiling_median", "r_frac_pos"]]
    out.to_csv(os.path.join(a.tab_dir, "traj_shape_agreement.csv"), index=False)

    ov = order_classes(pooled[pooled["scope"] == "all species"])
    print(ov.assign(frac=(ov.r_median / ov.ceiling_r_median).round(3))
            [["pattern", "n_sites", "r_median", "ceiling_r_median", "frac"]]
            .round(3).to_string(index=False), flush=True)
    print("%d rows written; %d species x tissue panels"
          % (len(out), perst.groupby(["Species", "Tissue"]).ngroups), flush=True)


if __name__ == "__main__":
    main()
