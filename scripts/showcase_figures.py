#!/usr/bin/env python
"""Single-example, slide-sized versions of two evaluation figures.

Companions to the species x tissue grids: the grids show the whole picture,
these two show one cell of it at a size that reads on a slide.

  devas_confusion_human_brain_liver.png    human Brain and Liver - two cells of
                                           devas_confusion_by_species_tissue.png
  traj_shape_agreement_human_brain_liver.png  the same two cells of
                                           traj_shape_agreement_by_species_tissue.png,
                                           drawn like the top panel of
                                           traj_shape_agreement.png

Both read the same tables as the grid figures and write the numbers they plot
next to the figure, so the two versions cannot drift apart.

Usage:
    python showcase_figures.py --calls-dir DIR --traj-dir DIR \
        --fig-dir DIR --tab-dir DIR
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from devas_report2 import PORD, PLAB, metrics                  # noqa: E402
from traj_shape_agreement import (CEIL, CEIL_EDGE, MODEL, bars,  # noqa: E402
                                  load, order_classes, style)


def slide_style():
    """House style, tightened for slides."""
    style()
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5,
                         "axes.titlesize": 7.5, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})


def fig_confusion_human(calls_dir, fig_dir, tab_dir, tissues=("Brain", "Liver")):
    """One row-normalised 5-class confusion matrix per tissue, shared colour scale."""
    C = pd.read_parquet(os.path.join(calls_dir, "devas_calls_human.parquet"),
                        columns=["Tissue", "pattern_true", "pattern_pred",
                                 "devAS_true", "devAS_pred", "dpsi_true", "dpsi_pred"])
    mats, stats, tabs = [], [], []
    for ti in tissues:
        g = C[C.Tissue == ti]
        if not len(g):
            raise SystemExit("no human rows for tissue %s" % ti)
        M = (g.groupby(["pattern_true", "pattern_pred"]).size().rename("n").reset_index()
              .pivot(index="pattern_true", columns="pattern_pred", values="n")
              .reindex(index=PORD, columns=PORD).fillna(0).to_numpy())
        R = M / M.sum(1, keepdims=True)
        # same statistics as the grid figure: binary devAS MCC and, among sites
        # called devAS by both, the fraction assigned the same pattern class
        m_ = metrics(g)
        mats.append((M, R))
        stats.append((float(m_["MCC"]), float(m_["pattern_agree"])))
        T = pd.DataFrame(M, index=PORD, columns=PORD).stack().rename("n").reset_index()
        T.columns = ["pattern_true", "pattern_pred", "n"]
        T["frac_of_observed"] = pd.DataFrame(R, index=PORD, columns=PORD).stack().to_numpy()
        T.insert(0, "Species", "human")
        T.insert(1, "Tissue", ti)
        tabs.append(T)

    stem = "devas_confusion_human_" + "_".join(t.lower() for t in tissues)
    pd.concat(tabs, ignore_index=True).to_csv(os.path.join(tab_dir, stem + ".csv"), index=False)

    slide_style()
    n = len(tissues)
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n + 0.75, 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for k, (ti, (M, R), (m, agree)) in enumerate(zip(tissues, mats, stats)):
        ax = axes[k]
        im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5), PLAB, rotation=35, ha="right")
        ax.set_xlabel("Predicted class")
        if k == 0:
            ax.set_yticks(range(5), PLAB)
            ax.set_ylabel("Observed class")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, "%.2f" % R[i, j], ha="center", va="center", fontsize=5.8,
                        color="white" if R[i, j] > 0.55 else "#212121")
        ax.set_title("Human %s\nn=%s · MCC %.2f · agree %.0f%%"
                     % (ti, format(int(M.sum()), ","), m, 100 * agree),
                     fontsize=6.8, pad=3, loc="center")
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.030, pad=0.02)
    cb.set_label("Fraction of observed class", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    p = os.path.join(fig_dir, stem + ".png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    for ti, (M, R), (m, agree) in zip(tissues, mats, stats):
        print("human %s confusion: n=%d  devAS MCC=%.3f  class agreement=%.3f  diagonal %s"
              % (ti, M.sum(), m, agree, np.round(np.diag(R), 3).tolist()), flush=True)
    return p


def fig_shape_human(traj_dir, fig_dir, tab_dir, species="human", tissues=("Brain", "Liver")):
    """Shape agreement against the noise ceiling, one panel per tissue, shared y axis."""
    _, perst = load(traj_dir)
    ds, keep = [], []
    for ti in tissues:
        d = perst[(perst.Species == species) & (perst.Tissue == ti)]
        if not len(d):
            raise SystemExit("no rows for %s %s" % (species, ti))
        ds.append(order_classes(d))
        keep.append(ti)
    stem = "traj_shape_agreement_%s_%s" % (species, "_".join(t.lower() for t in keep))
    pd.concat([d.assign(Species=species, Tissue=ti) for ti, d in zip(keep, ds)],
              ignore_index=True).to_csv(os.path.join(tab_dir, stem + ".csv"), index=False)

    lo = min(float(np.nanmin(d["r_median"])) for d in ds)
    lo = min(lo - 0.08, -0.02)  # the shared grid y-limit would clip a negative median r
    slide_style()
    n = len(keep)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    for k, (ti, d) in enumerate(zip(keep, ds)):
        ax = axes[k]
        bars(ax, d, ["%s\nn=%s" % (p, format(int(nn), ",")) for p, nn in
                     zip(d["pattern"], d["n_sites"])], fs=6.0)
        ax.set_ylim(lo, 1.34)       # headroom so the legend clears the bars and the % labels
        ax.set_yticks([round(t, 2) for t in np.arange(-0.25, 1.01, 0.25) if t >= lo])
        ax.set_title("Human %s" % ti, fontsize=7.5, pad=3)
        if k == 0:
            ax.set_ylabel("Shape agreement (median r)")
    axes[0].legend(handles=[Patch(facecolor=CEIL, edgecolor=CEIL_EDGE, linewidth=0.5,
                                  label="replicate-split ceiling"),
                            Patch(facecolor=MODEL, label="model")],
                   loc="upper center", ncol=2, fontsize=6.2, borderaxespad=0.1)
    fig.subplots_adjust(wspace=0.12)
    p = os.path.join(fig_dir, stem + ".png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    for ti, d in zip(keep, ds):
        print("%s %s: %s" % (species, ti,
                             "  ".join("%s r=%.2f ceil=%.2f (%.0f%%)"
                                       % (r.pattern, r.r_median, r.ceiling_r_median,
                                          100 * r.r_median / r.ceiling_r_median)
                                       for r in d.itertuples() if np.isfinite(r.r_median))),
              flush=True)
    return p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calls-dir", required=True)
    p.add_argument("--traj-dir", required=True)
    p.add_argument("--fig-dir", required=True)
    p.add_argument("--tab-dir", required=True)
    p.add_argument("--tissues", nargs="+", default=["Brain", "Liver"],
                   help="tissues drawn as parallel panels in both figures")
    a = p.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    os.makedirs(a.tab_dir, exist_ok=True)
    f1 = fig_confusion_human(a.calls_dir, a.fig_dir, a.tab_dir, tuple(a.tissues))
    f2 = fig_shape_human(a.traj_dir, a.fig_dir, a.tab_dir, tissues=tuple(a.tissues))
    print("wrote", f1, "and", f2, flush=True)


if __name__ == "__main__":
    main()
