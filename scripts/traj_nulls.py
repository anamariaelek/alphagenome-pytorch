#!/usr/bin/env python
"""Predicted trajectories against all three null baselines.

The three nulls of the trajectory spec (v4):

    N1  flat at the site's OBSERVED mean usage   - an oracle null, not realisable
    N2  flat at the model's own PREDICTED mean   - the realisable flat null
    N3  the organ-mean centred trajectory        - shared developmental signal

N1 and N2 coincide once trajectories are centred (any flat series has zero
centred variance, so both reduce to RMSE_flat = sigma_obs). They separate only in
absolute SSE units, where N1 gets the site's level for free and N2 must predict
it. N3 exists only in centred space. The figures therefore report absolute space
(model vs N1 vs N2) and centred space (model vs flat vs N3, with the
replicate-split noise ceiling) side by side, which is the only way all three
nulls are visible at once.

Inputs (trajectory evaluation, not the devAS call table):
    traj_eval_summary.csv                        per organ x reference x band
    traj_eval_by_pattern_class_pooled_full.csv   pooled, per observed class
    traj_eval_by_pattern_class_full.csv          per species x tissue x class

    python traj_nulls.py --traj-dir DIR --fig-dir DIR --tab-dir DIR

Outputs:
    traj_nulls.png            per organ: absolute RMSE, centred RMSE, frac beating
    traj_nulls_by_class.png   per observed trajectory class (centred space only)
    traj_nulls.csv            every plotted value

Reference series is 'counts' (pooled Alpha/(Alpha+Beta)) and band 'all'; the
stored-SSE reference agrees closely (median r differs by -0.014).
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SP_ORDER = ["human", "macaque", "rabbit", "rat", "mouse", "opossum", "chicken"]
TORD = ["Brain", "Cerebellum", "Midbrain", "Heart", "Kidney", "Liver", "Ovary", "Testis"]
PORD = ["up", "down", "up-down", "down-up", "none"]

MODEL = "#1f6fb4"    # the model, everywhere
N1 = "#cfcfcf"       # oracle flat null (observed mean)
N2 = "#7a7a7a"       # realisable flat null (predicted mean)
FLAT = "#9e9e9e"     # the two flat nulls, collapsed in centred space
N3 = "#8c6bb1"       # organ-mean trajectory null
CEILING = "#2e7d32"  # replicate-split noise ceiling


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6, "legend.fontsize": 6.8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def organ_order(d):
    d = d.copy()
    d["_s"] = d.Species.map({s: i for i, s in enumerate(SP_ORDER)})
    d["_t"] = d.Tissue.map({t: i for i, t in enumerate(TORD)})
    return d.sort_values(["_s", "_t"]).drop(columns=["_s", "_t"]).reset_index(drop=True)


def load(traj_dir):
    S = pd.read_csv(os.path.join(traj_dir, "traj_eval_summary.csv"))
    S = organ_order(S[(S.reference == "counts") & (S.band == "all")])
    cls = pd.read_csv(os.path.join(traj_dir, "traj_eval_by_pattern_class_pooled_full.csv"))
    cls = cls[(cls.stratifier == "pattern_true") & (cls.reference == "counts")]
    per = pd.read_csv(os.path.join(traj_dir, "traj_eval_by_pattern_class_full.csv"))
    per = per[(per.stratifier == "pattern_true") & (per.reference == "counts")]
    return S, cls, organ_order(per)


def dots(ax, y, series, xmax=None):
    """One row per organ: a connector spanning the values, then a dot per series."""
    vals = np.vstack([v for _, v, _, _ in series])
    for i in range(len(y)):
        col = vals[:, i]
        if np.isfinite(col).any():
            ax.plot([np.nanmin(col), np.nanmax(col)], [y[i], y[i]],
                    color="#d8d8d8", lw=0.9, zorder=1)
    for label, v, colour, marker in series:
        # N1 is nearly white so it needs an outline to read on the panel background.
        ax.scatter(v, y, s=15 if marker == "o" else 20, c=colour, marker=marker,
                   linewidths=0.5, edgecolors="#8a8a8a" if colour == N1 else "none",
                   zorder=3, label=label)
    ax.set_ylim(len(y) - 0.5, -0.5)
    if xmax:
        ax.set_xlim(0, xmax)
    ax.grid(axis="x", color="#eeeeee", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def fig_organs(S, fig_dir):
    lab = ["%s %s" % (s, t.lower()) for s, t in zip(S.Species, S.Tissue)]
    y = np.arange(len(S))
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 8.6), sharey=True,
                             gridspec_kw={"wspace": 0.22, "width_ratios": [1, 1, 0.95]})
    leg = dict(loc="lower left", bbox_to_anchor=(0, 1.035), markerscale=1.1,
               handletextpad=0.3, borderpad=0, labelspacing=0.25)

    ax = axes[0]
    dots(ax, y, [("N1 flat at observed mean (oracle)", S.rmse_abs_n1_median, N1, "o"),
                 ("N2 flat at predicted mean", S.rmse_abs_n2_median, N2, "o"),
                 ("model", S.rmse_abs_median, MODEL, "D")])
    ax.set_yticks(y, lab)
    ax.set_xlabel("median RMSE, absolute (SSE units)")
    ax.legend(**leg)

    ax = axes[1]
    dots(ax, y, [("flat null (N1 = N2 centred)", S.rmse_flat_median, FLAT, "o"),
                 ("N3 organ-mean trajectory", S.rmse_n3_median, N3, "o"),
                 ("replicate-split noise ceiling", S.ceiling_rmse_median, CEILING, "o"),
                 ("model", S.rmse_median, MODEL, "D")])
    ax.set_xlabel("median RMSE, centred (SSE units)")
    ax.legend(ncol=2, **leg)

    ax = axes[2]
    dots(ax, y, [("vs N2, absolute", S.frac_beating_n2_abs, N2, "o"),
                 ("vs flat, centred", S.frac_beating_flat, FLAT, "o"),
                 ("vs N3, centred", S.frac_beating_n3, N3, "o")], xmax=1.0)
    ax.axvline(0.5, color="#c62828", lw=0.8, ls="--", zorder=2)
    ax.set_xlabel("fraction of sites beating the null\n(a null is beaten right of the dashed line)")
    # inside the panel: every organ sits far left, and an outside legend collides
    # with the centred-space one.
    ax.legend(loc="upper right", markerscale=1.1, handletextpad=0.3,
              labelspacing=0.25)

    fig.suptitle("Predicted developmental trajectories against all three nulls "
                 "(reference: pooled counts, all usage bands)", x=0.5, y=1.045, fontsize=9.5)
    fig.text(0.995, -0.035,
             "'vs N2, absolute' and 'vs flat, centred' coincide exactly in all 46 organs: "
             "model and N2 share the same predicted level, so the level error cancels and\n"
             "both comparisons reduce to the centred residual against sigma_obs. N1 keeps "
             "the observed level, which is why it is unreachable in absolute space.",
             ha="right", va="top", fontsize=6.5, color="#666666")
    fig.savefig(os.path.join(fig_dir, "traj_nulls.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def cls_order(d, key="pattern"):
    return d.set_index(key).reindex(PORD).reset_index()


def fig_classes(cls, per, fig_dir):
    spp = [s for s in SP_ORDER if s in set(cls.scope)]
    fig = plt.figure(figsize=(8.6, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.42, wspace=0.24)
    x = np.arange(len(PORD))
    w = 0.26

    d = cls_order(cls[cls.scope == "all species"])
    ax = fig.add_subplot(gs[0, 0])
    # The class tables store MEDIAN PER-SITE RATIOS, not median null RMSEs, so the
    # null RMSE cannot be recovered by dividing medians. Plot the ratios as stored:
    # 1.0 is a tie with the null, below 1.0 beats it.
    ax.bar(x - w / 2, d.rmse_over_flat_median, w, color=FLAT, label="vs flat (N1 = N2)")
    ax.bar(x + w / 2, d.rmse_over_n3_median, w, color=N3, label="vs N3 organ-mean")
    ax.axhline(1.0, color="#c62828", lw=0.8, ls="--")
    ax.set_xticks(x, PORD)
    ax.set_ylabel("median per-site RMSE ratio\nmodel / null")
    ax.set_title("Error relative to each null (1.0 = tie)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=2,
              handletextpad=0.3, borderpad=0)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x - w / 2, d.frac_beating_flat, w, color=FLAT, label="vs flat")
    ax.bar(x + w / 2, d.beat_n3, w, color=N3, label="vs N3")
    ax.axhline(0.5, color="#c62828", lw=0.8, ls="--")
    ax.set_xticks(x, PORD)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("fraction of sites beating the null")
    ax.set_title("Fraction of sites beating each null")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=2,
              handletextpad=0.3, borderpad=0)

    inner = gs[1, :].subgridspec(1, len(spp), wspace=0.16)
    for k, s in enumerate(spp):
        axk = fig.add_subplot(inner[0, k])
        ds = cls_order(cls[cls.scope == s])
        axk.bar(x - w / 2, ds.frac_beating_flat, w, color=FLAT)
        axk.bar(x + w / 2, ds.beat_n3, w, color=N3)
        axk.axhline(0.5, color="#c62828", lw=0.8, ls="--")
        axk.set_xticks(x, ["U", "D", "UD", "DU", "N"])
        axk.set_ylim(0, 0.75)
        axk.set_title(s, fontsize=7)
        if k:
            axk.set_yticklabels([])
        else:
            axk.set_ylabel("frac beating null")
    fig.text(0.5, 0.012, "U up · D down · UD up-down · DU down-up · N none. "
             "Class tables carry the centred nulls only, so N1 and N2 stay collapsed here.",
             ha="center", fontsize=6.5, color="#666666")
    fig.suptitle("Nulls by observed trajectory class", x=0.5, y=0.985, fontsize=9.5)
    fig.savefig(os.path.join(fig_dir, "traj_nulls_by_class.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj-dir", default="devas/traj_eval")
    p.add_argument("--fig-dir", default="devas/figures")
    p.add_argument("--tab-dir", default="devas/tables")
    a = p.parse_args()

    style()
    S, cls, per = load(a.traj_dir)
    for d in (a.fig_dir, a.tab_dir):
        os.makedirs(d, exist_ok=True)
    fig_organs(S, a.fig_dir)
    fig_classes(cls, per, a.fig_dir)

    org = S[["Species", "Tissue", "n_sites", "rmse_abs_median", "rmse_abs_n1_median",
             "rmse_abs_n2_median", "frac_beating_n2_abs", "rmse_median",
             "rmse_flat_median", "rmse_n3_median", "ceiling_rmse_median",
             "frac_beating_flat", "frac_beating_n3", "rmse_over_flat_median"]].copy()
    org.insert(0, "level", "organ")
    org.insert(1, "pattern", "all")
    # Class rows carry median per-site ratios, not median null RMSEs: rmse_n3_median
    # and the absolute-space columns are left empty there rather than reconstructed.
    cl = cls.rename(columns={"scope": "Species"}).assign(
        level=np.where(cls.scope.values == "all species", "class pooled", "class by species"),
        Tissue="all", frac_beating_n3=lambda d: d.beat_n3)
    pc = per.assign(level="class by species x tissue",
                    frac_beating_n3=lambda d: d.beat_n3)
    cols = ["level", "Species", "Tissue", "pattern", "n_sites", "rmse_median",
            "rmse_flat_median", "rmse_n3_median", "ceiling_rmse_median",
            "rmse_abs_median", "rmse_abs_n1_median", "rmse_abs_n2_median",
            "frac_beating_flat", "frac_beating_n3", "frac_beating_n2_abs",
            "rmse_over_flat_median", "rmse_over_n3_median"]
    out = pd.concat([org, cl, pc], ignore_index=True)
    out = out.reindex(columns=cols)
    out.to_csv(os.path.join(a.tab_dir, "traj_nulls.csv"), index=False)

    print(S[["Species", "Tissue", "rmse_abs_median", "rmse_abs_n1_median",
             "rmse_abs_n2_median", "frac_beating_n2_abs", "frac_beating_flat",
             "frac_beating_n3"]].round(3).head(8).to_string(index=False), flush=True)
    print("organs %d | class rows %d | table rows %d"
          % (len(S), len(cl) + len(pc), len(out)), flush=True)


if __name__ == "__main__":
    main()
