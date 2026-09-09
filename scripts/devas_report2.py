#!/usr/bin/env python
"""Regenerate every devAS table and figure from a given call table.

Ports the interactive cells that produced devas/tables/ and devas/figures/
into one runnable pass so the whole report can be rebuilt against a
different prediction store. Usage values are SSE (Spliser, Dent et al.
2021), so amplitude is labelled dSSE.

The GP side of the gp_vs_cubic comparison is a fixed reference from the
earlier Gaussian-process analysis, which ran on its own site set; those
constants do not depend on the prediction store and are carried through
unchanged. stage_age_alignment.png has no prediction-store dependency and
is not regenerated here.

  python devas_report.py --calls <parquet> --fig-dir <d> --tab-dir <d>
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

META_GREY = "#888888"
PORD = ["up", "down", "up-down", "down-up", "none"]
PLAB = ["up", "down", "up-down", "down-up", "not devAS"]
PCOL = dict(zip(PORD, ["#27ac40", "#b30303", "#7d3c98", "#d68910", "#d5d8dc"]))
SP_ORDER = ["human", "macaque", "mouse", "rat", "rabbit", "opossum", "chicken"]
TORD = ["Brain", "Cerebellum", "Midbrain", "Heart", "Kidney", "Liver", "Ovary", "Testis"]
# Diverging at MCC = 0, with the *positive* (better-than-chance) side green and the
# negative side purple, so red never marks a good result. Colour-blind safe.
MCC_CMAP = "PRGn"

# Gaussian-process reference, human brain and liver, from the earlier analysis.
GP = {
    ("Brain", "recall"):   {"down": 0.018, "down-up": 0.000, "none": 0.306, "up": 0.872, "up-down": 0.045},
    ("Brain", "marginal"): {"down": 0.009, "down-up": 0.001, "none": 0.261, "up": 0.722, "up-down": 0.006},
    ("Liver", "recall"):   {"down": 0.456, "down-up": 0.048, "none": 0.516, "up": 0.222, "up-down": 0.010},
    ("Liver", "marginal"): {"down": 0.342, "down-up": 0.021, "none": 0.506, "up": 0.116, "up-down": 0.015},
}
GP_OVERALL = {"Brain": 0.400, "Liver": 0.478}
GP_MCC = {"Brain": 0.162, "Liver": 0.044}
GP_N = {"Brain": 15544, "Liver": 10177}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titleweight": "normal", "axes.titlelocation": "left",
        "lines.linewidth": 1.2, "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def mcc(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    tp = (a & b).sum(); tn = (~a & ~b).sum()
    fp = (~a & b).sum(); fn = (a & ~b).sum()
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0


def metrics(g, ref="true"):
    t = g["devAS_" + ref].to_numpy(bool)
    p = g["devAS_pred"].to_numpy(bool)
    tp = int((t & p).sum()); fp = int((~t & p).sum())
    fn = int((t & ~p).sum()); tn = int((~t & ~p).sum())
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    both = t & p
    pat = (g.loc[both, "pattern_" + ref].to_numpy() == g.loc[both, "pattern_pred"].to_numpy())
    return pd.Series({
        "n": len(g), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1), "accuracy": (tp + tn) / len(g),
        "MCC": (tp * tn - fp * fn) / den if den > 0 else np.nan,
        "n_both": int(both.sum()), "pattern_agree": pat.mean() if both.any() else np.nan,
        "dsse_r": g["dpsi_" + ref].corr(g["dpsi_pred"]),
        "dsse_rho": g["dpsi_" + ref].corr(g["dpsi_pred"], method="spearman")})


def accuracy(C, tab):
    ACC = C.groupby(["Species", "Tissue"]).apply(metrics, include_groups=False).reset_index()
    ACC.to_csv(os.path.join(tab, "devas_accuracy_by_species_tissue.csv"), index=False)
    print(ACC.groupby("Species")[["sensitivity", "specificity", "precision", "MCC",
                                  "pattern_agree", "dsse_r"]].mean().round(3).to_string(), flush=True)
    return ACC


def counts(C, tab):
    CNT = (C.melt(id_vars=["Species", "Tissue"],
                  value_vars=["pattern_obs", "pattern_true", "pattern_pred"],
                  var_name="pass", value_name="pattern")
            .groupby(["Species", "Tissue", "pass", "pattern"]).size().rename("n").reset_index())
    CNT["pass"] = CNT["pass"].str.replace("pattern_", "")
    CNT.to_csv(os.path.join(tab, "devas_class_counts.csv"), index=False)
    return CNT


def confusion(C, tab):
    CONF = C.groupby(["pattern_true", "pattern_pred"]).size().rename("n").reset_index()
    per_sp = C.groupby(["Species", "pattern_true", "pattern_pred"]).size().rename("n").reset_index()
    pooled = CONF.assign(Species="all")[["Species", "pattern_true", "pattern_pred", "n"]]
    pd.concat([pooled, per_sp], ignore_index=True).to_csv(
        os.path.join(tab, "devas_true_vs_pred_confusion.csv"), index=False)
    return CONF


def fig_mcc(ACC, sp_order, fig_dir):
    H = ACC.pivot(index="Species", columns="Tissue", values="MCC").reindex(
        index=sp_order, columns=TORD)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    im = ax.imshow(H.to_numpy(), cmap=MCC_CMAP, vmin=-0.25, vmax=0.25)
    ax.set_xticks(range(len(TORD)), TORD, rotation=35, ha="right")
    ax.set_yticks(range(len(sp_order)), [s.capitalize() for s in sp_order])
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = H.to_numpy()[i, j]
            ax.text(j, i, "—" if not np.isfinite(v) else "%.2f" % v,
                    ha="center", va="center", fontsize=7.5,
                    color="white" if np.isfinite(v) and abs(v) > 0.72 * 0.25
                    else "#212121")
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Matthews correlation", fontsize=8)
    fig.savefig(os.path.join(fig_dir, "devas_mcc.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_class_composition(CNT, sp_order, fig_dir):
    comp = (CNT[CNT["pass"] != "obs"].groupby(["Species", "pass", "pattern"])["n"].sum()
            .unstack("pattern").reindex(columns=PORD).fillna(0))
    comp = comp.div(comp.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    xs, labels = [], []
    for i, sp in enumerate(sp_order):
        for j, ps in enumerate(["true", "pred"]):
            x = i * 1.0 + (j - 0.5) * 0.36
            bot = 0.0
            for p in PORD:
                v = comp.loc[(sp, ps), p]
                ax.bar(x, v, width=0.32, bottom=bot, color=PCOL[p],
                       edgecolor="white", linewidth=0.5)
                bot += v
            xs.append(x)
            labels.append("obs" if ps == "true" else "pred")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    for i, sp in enumerate(sp_order):
        ax.text(i, -0.115, sp.capitalize(), ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9)
    ax.set_ylabel("Fraction of testable splice sites")
    ax.set_ylim(0, 1)
    h = [plt.Rectangle((0, 0), 1, 1, color=PCOL[p]) for p in PORD]
    ax.legend(h, PLAB, frameon=False, fontsize=7.5, ncol=5,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.savefig(os.path.join(fig_dir, "devas_class_composition.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_observed_by_usage(C, sp_order, fig_dir):
    """Stacked pattern fractions for observed sites, split by max SSE change between organs.
    Bands: <0.3, 0.3-0.7, >0.7 (based on max - min mean_sse_true across tissues/organs)."""
    bands = [("<0.3", 0.0, 0.3), ("0.3-0.7", 0.3, 0.7), (">0.7", 0.7, 1.0)]
    band_labels = [b[0] for b in bands]

    # Calculate max change per site (max - min mean_sse across tissues)
    if "Chromosome" in C.columns and "Position" in C.columns:
        if "Strand" in C.columns:
            site_cols = ["Chromosome", "Position", "Strand"]
        else:
            site_cols = ["Chromosome", "Position"]
        site_sse_range = C.groupby(site_cols)["mean_sse_true"].agg(lambda x: x.max() - x.min()).reset_index()
        site_sse_range.columns = site_cols + ["sse_range"]
        C = C.merge(site_sse_range, on=site_cols, how="left")
    else:
        C["sse_range"] = 0.0

    rows = []
    for sp in sp_order:
        g = C[C.Species == sp]
        for band_label, lo, hi in bands:
            band_mask = (g["sse_range"] > lo) & (g["sse_range"] <= hi)
            band_data = g[band_mask]
            if len(band_data) == 0:
                for p in PORD:
                    rows.append(dict(Species=sp, band=band_label, pattern=p, n=0))
            else:
                counts = band_data["pattern_obs"].value_counts()
                for p in PORD:
                    rows.append(dict(Species=sp, band=band_label, pattern=p, n=counts.get(p, 0)))

    df_rows = pd.DataFrame(rows)
    counts_raw = (df_rows.groupby(["Species", "band", "pattern"])["n"].sum()
                  .unstack("pattern").reindex(columns=PORD).fillna(0))
    comp = counts_raw.div(counts_raw.sum(1), axis=0)

    fig, ax = plt.subplots(figsize=(10.0, 2.8))
    xs, labels = [], []
    for i, sp in enumerate(sp_order):
        for j, band in enumerate(band_labels):
            x = i * 1.5 + j * 0.35
            bot = 0.0
            for p in PORD:
                if (sp, band) in comp.index:
                    v = comp.loc[(sp, band), p]
                else:
                    v = 0.0
                ax.bar(x, v, width=0.30, bottom=bot, color=PCOL[p],
                       edgecolor="white", linewidth=0.5)
                bot += v
            n_sites = int(counts_raw.loc[(sp, band)].sum()) if (sp, band) in counts_raw.index else 0
            ax.text(x, 1.02, "n=%s" % format(n_sites, ","), ha="left", va="bottom",
                    fontsize=6, color=META_GREY, rotation=90)
            xs.append(x)
            labels.append(band)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_xlabel("Max SSE change (max - min across organs)")
    for i, sp in enumerate(sp_order):
        ax.text(i * 1.5 + 0.35, -0.25, sp.capitalize(), ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9)
    ax.set_ylabel("Fraction of observed splice sites")
    ax.set_ylim(0, 1)
    h = [plt.Rectangle((0, 0), 1, 1, color=PCOL[p]) for p in PORD]
    ax.legend(h, PLAB, frameon=False, fontsize=7.5, ncol=1,
              loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(os.path.join(fig_dir, "devas_observed_by_usage.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_observed_class_composition(C, sp_order, fig_dir):
    """Stacked pattern fractions for observed sites, one bar per species (no splitting).
    Uses pattern_true aggregated by tissue first, like CNT."""
    OBS = (C.groupby(["Species", "Tissue", "pattern_true"]).size().rename("n").reset_index()
           .rename(columns={"pattern_true": "pattern"}))
    counts_raw = OBS.groupby(["Species", "pattern"])["n"].sum().unstack("pattern").reindex(columns=PORD).fillna(0)
    comp = counts_raw.div(counts_raw.sum(1), axis=0)

    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    xs, labels = [], []

    for i, sp in enumerate(sp_order):
        x = i
        bot = 0.0
        for p in PORD:
            v = comp.loc[sp, p]
            ax.bar(x, v, width=0.6, bottom=bot, color=PCOL[p],
                   edgecolor="white", linewidth=0.5)
            bot += v
        n_sites = int(counts_raw.loc[sp].sum())
        ax.text(x, 1.02, "n=%s" % format(n_sites, ","), ha="center", va="bottom",
                fontsize=6, color=META_GREY)
        xs.append(x)
        labels.append(sp.capitalize())

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Fraction of observed splice sites (per species)")
    ax.set_ylim(0, 1)
    h = [plt.Rectangle((0, 0), 1, 1, color=PCOL[p]) for p in PORD]
    ax.legend(h, PLAB, frameon=False, fontsize=7.5, ncol=1,
              loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(bottom=0.15, right=0.80)
    fig.savefig(os.path.join(fig_dir, "devas_observed_class_composition.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_observed_by_usage_binary(C, sp_order, fig_dir):
    """Stacked binary fractions (devAS vs non-devAS) for observed sites counted once per species.
    Each site is counted once, and marked as devAS if it's devAS in any organ, non-devAS if it's
    non-devAS in all organs."""
    rows = []
    for sp in sp_order:
        g = C[C.Species == sp]
        # Get unique sites per species (deduplicate by Chromosome/Position/Strand)
        if "Strand" in g.columns:
            sites = g[["Chromosome", "Position", "Strand", "pattern_obs"]].drop_duplicates(
                subset=["Chromosome", "Position", "Strand"])
        else:
            sites = g[["Chromosome", "Position", "pattern_obs"]].drop_duplicates(
                subset=["Chromosome", "Position"])

        is_devas = (sites["pattern_obs"] != "none").sum()
        is_non_devas = (sites["pattern_obs"] == "none").sum()
        rows.append(dict(Species=sp, devAS=int(is_devas), non_devAS=int(is_non_devas)))

    df_rows = pd.DataFrame(rows).set_index("Species")
    counts_raw = df_rows[["devAS", "non_devAS"]]
    comp = counts_raw.div(counts_raw.sum(1), axis=0)

    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    xs, labels = [], []
    colors = {"devAS": "#c0392b", "non_devAS": "#d5d8dc"}

    for i, sp in enumerate(sp_order):
        x = i
        bot = 0.0
        for cat in ["devAS", "non_devAS"]:
            v = comp.loc[sp, cat]
            ax.bar(x, v, width=0.6, bottom=bot, color=colors[cat],
                   edgecolor="white", linewidth=0.5)
            bot += v
        n_sites = int(counts_raw.loc[sp].sum())
        ax.text(x, 1.02, "n=%s" % format(n_sites, ","), ha="center", va="bottom",
                fontsize=6, color=META_GREY)
        xs.append(x)
        labels.append(sp.capitalize())

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Fraction of observed splice sites (per species)")
    ax.set_ylim(0, 1)
    h = [plt.Rectangle((0, 0), 1, 1, color=colors[cat]) for cat in ["devAS", "non_devAS"]]
    ax.legend(h, ["devAS (any organ)", "non-devAS (all organs)"], frameon=False, fontsize=7.5, ncol=1,
              loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(bottom=0.15, right=0.80)
    fig.savefig(os.path.join(fig_dir, "devas_observed_by_usage_binary.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(CONF, fig_dir):
    M = (CONF.pivot(index="pattern_true", columns="pattern_pred", values="n")
         .reindex(index=PORD, columns=PORD).fillna(0).to_numpy())
    R = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5), PLAB, rotation=35, ha="right")
    ax.set_yticks(range(5), PLAB)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, "%.2f" % R[i, j], ha="center", va="center", fontsize=7.5,
                    color="white" if R[i, j] > 0.55 else "#212121")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Fraction of observed class", fontsize=8)
    fig.savefig(os.path.join(fig_dir, "devas_confusion.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_confusion_grid(C, sp_order, fig_dir, tab):
    """One row-normalised 5-class confusion matrix per species x tissue."""
    rows = []
    for (sp, ti), g in C.groupby(["Species", "Tissue"]):
        M = (g.groupby(["pattern_true", "pattern_pred"]).size().rename("n").reset_index()
             .pivot(index="pattern_true", columns="pattern_pred", values="n")
             .reindex(index=PORD, columns=PORD).fillna(0))
        for pt in PORD:
            for pp in PORD:
                rows.append(dict(Species=sp, Tissue=ti, pattern_true=pt,
                                 pattern_pred=pp, n=int(M.loc[pt, pp]),
                                 frac_of_true=float(M.loc[pt, pp] / max(M.loc[pt].sum(), 1))))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(tab, "devas_confusion_by_species_tissue.csv"), index=False)

    sps = [s for s in sp_order if s in set(C.Species)]
    tis = [t for t in TORD if t in set(C.Tissue)]
    ab = ["u", "d", "ud", "du", "n"]
    fig, axes = plt.subplots(len(sps), len(tis), figsize=(1.02 * len(tis) + 0.9,
                                                          1.02 * len(sps) + 0.9))
    axes = np.atleast_2d(axes)
    im = None
    for i, sp in enumerate(sps):
        for j, ti in enumerate(tis):
            ax = axes[i, j]
            sub = T[(T.Species == sp) & (T.Tissue == ti)]
            if sub.empty:
                ax.set_axis_off()
                continue
            R = (sub.pivot(index="pattern_true", columns="pattern_pred", values="frac_of_true")
                 .reindex(index=PORD, columns=PORD).to_numpy())
            im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
            n = int(sub.n.sum())
            ax.set_title("n=%s" % format(n, ","), fontsize=5.2, color=META_GREY,
                         loc="right", pad=1.5)
            ax.tick_params(length=1.5, width=0.4, pad=1)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.4)
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_xticklabels(ab if i == len(sps) - 1 else [], fontsize=5)
            ax.set_yticklabels(ab if j == 0 else [], fontsize=5)
    for j, ti in enumerate(tis):
        axes[0, j].text(0.5, 1.30, ti, transform=axes[0, j].transAxes,
                        ha="center", va="bottom", fontsize=7.5)
    for i, sp in enumerate(sps):
        axes[i, 0].text(-0.55, 0.5, sp.capitalize(), transform=axes[i, 0].transAxes,
                        ha="right", va="center", fontsize=7.5, rotation=90)
    fig.supxlabel("Predicted class", fontsize=8, y=-0.005)
    fig.supylabel("Observed class", fontsize=8, x=0.005)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.015)
        cb.set_label("Fraction of observed class", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.text(0.005, 1.005, "u up   d down   ud up-down   du down-up   n not devAS",
             fontsize=6, color=META_GREY, transform=fig.transFigure)
    fig.savefig(os.path.join(fig_dir, "devas_confusion_by_species_tissue.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    return T


def fig_mcc_by_shape(C, sp_order, fig_dir, tab):
    """One-vs-rest Matthews correlation for each trajectory shape."""
    shapes = ["up", "down", "up-down", "down-up"]
    rows = []
    for (sp, ti), g in C.groupby(["Species", "Tissue"]):
        for sh in shapes:
            t = (g.pattern_true.to_numpy() == sh)
            p = (g.pattern_pred.to_numpy() == sh)
            rows.append(dict(Species=sp, Tissue=ti, shape=sh, n=len(g),
                             n_true=int(t.sum()), n_pred=int(p.sum()),
                             TP=int((t & p).sum()),
                             recall=float((t & p).sum() / t.sum()) if t.sum() else np.nan,
                             MCC=float(mcc(t, p)) if t.sum() and p.sum() else np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(tab, "devas_mcc_by_shape.csv"), index=False)

    sps = [s for s in sp_order if s in set(C.Species)]
    tis = [t for t in TORD if t in set(C.Tissue)]
    vlim = 0.05 * np.ceil(np.nanmax(np.abs(T.MCC.to_numpy())) / 0.05)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6),
                             gridspec_kw=dict(hspace=0.62, wspace=0.22))
    im = None
    for k, sh in enumerate(shapes):
        ax = axes.ravel()[k]
        H = (T[T["shape"] == sh]
             .pivot(index="Species", columns="Tissue", values="MCC")
             .reindex(index=sps, columns=tis))
        A = H.to_numpy(dtype=float)
        im = ax.imshow(A, cmap=MCC_CMAP, vmin=-vlim, vmax=vlim)
        ax.set_xticks(range(len(tis)), tis, rotation=35, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(sps)), [s.capitalize() for s in sps], fontsize=6.5)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                ax.text(j, i, "—" if not np.isfinite(v) else "%.2f" % v,
                        ha="center", va="center", fontsize=6.2,
                        color="white" if np.isfinite(v) and abs(v) > 0.72 * vlim
                        else "#212121")
        ax.set_title(sh, fontsize=8.5, loc="left", color=PCOL[sh])
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cb.set_label("Matthews correlation, one class vs rest", fontsize=7.5)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(fig_dir, "devas_mcc_by_shape.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return T


def fig_dsse(C, sp_order, fig_dir):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0), sharex=True, sharey=True)
    hb = None
    for k, sp in enumerate(sp_order):
        ax = axes.ravel()[k]
        g = C[C.Species == sp]
        hb = ax.hexbin(g["dpsi_true"], g["dpsi_pred"], gridsize=34, extent=(0, 1, 0, 1),
                       bins="log", cmap="magma_r", mincnt=1, linewidths=0)
        ax.plot([0, 1], [0, 1], color=META_GREY, lw=0.8, ls="--")
        r = g["dpsi_true"].corr(g["dpsi_pred"])
        rho = g["dpsi_true"].corr(g["dpsi_pred"], method="spearman")
        ax.set_title("%s   r=%.2f  $\\rho$=%.2f" % (sp.capitalize(), r, rho), fontsize=8.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("Observed amplitude $\\Delta$SSE")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted $\\Delta$SSE")
    fig.colorbar(hb, ax=axes, fraction=0.02, pad=0.02, label="sites")
    fig.savefig(os.path.join(fig_dir, "devas_dpsi_true_vs_pred.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("pooled dSSE r=%.3f rho=%.3f n=%d" % (
        C["dpsi_true"].corr(C["dpsi_pred"]),
        C["dpsi_true"].corr(C["dpsi_pred"], method="spearman"), len(C)), flush=True)


def fig_mean_usage(C, fig_dir, tab):
    H = C[C.Species == "human"]
    S = H.groupby(["Chromosome", "Position"])[["mean_sse_true", "mean_sse_pred"]].mean()
    x, y = S.mean_sse_true, S.mean_sse_pred
    r_all = x.corr(y)
    rho_all = x.corr(y, method="spearman")
    bands = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    BR = pd.DataFrame([dict(band="%.1f-%.1f" % b, lo=b[0], hi=b[1], n=int(m.sum()),
                            r=float(x[m].corr(y[m])))
                       for b in bands for m in [(x > b[0]) & (x <= b[1])]])
    BR.round(4).to_csv(os.path.join(tab, "human_mean_usage_r_bands.csv"), index=False)
    hot = BR.r.to_numpy() >= 0.5 * r_all
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                 gridspec_kw={"width_ratios": [1.05, 1]})
    hb = a1.hexbin(x, y, gridsize=45, bins="log", cmap="magma_r", mincnt=1, linewidths=0)
    a1.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--")
    a1.set_xlabel("observed mean usage")
    a1.set_ylabel("predicted mean usage")
    a1.set_title("Human, one point per site (n=%s)" % format(len(S), ","))
    a1.text(0.04, 0.93, "r = %.2f\n$\\rho$ = %.2f" % (r_all, rho_all),
            transform=a1.transAxes, fontsize=7, va="top")
    cb = fig.colorbar(hb, ax=a1, pad=0.02)
    cb.set_label("sites", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    a2.bar(range(len(BR)), BR.r, width=0.65,
           color=["#c0392b" if h else "#d5d8dc" for h in hot])
    for i, (rv, nv) in enumerate(zip(BR.r, BR.n)):
        a2.text(i, rv + 0.015, "n=%s" % format(int(nv), ","), ha="center",
                fontsize=5.5, color=META_GREY)
    a2.axhline(r_all, color="k", lw=0.8, ls=":")
    a2.text(len(BR) - 0.4, r_all + 0.012, "pooled %.2f" % r_all, ha="right",
            fontsize=6, color=META_GREY)
    a2.set_xticks(range(len(BR)))
    a2.set_xticklabels(BR.band, fontsize=6.5)
    a2.set_xlabel("observed mean usage band")
    a2.set_ylabel("correlation within band")
    a2.set_ylim(0, max(0.85, float(BR.r.max()) + 0.12))
    a2.set_title("Signal is between bands, not within")
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(os.path.join(fig_dir, "human_mean_usage_r.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("human mean usage: n=%d r=%.3f rho=%.3f" % (len(S), r_all, rho_all), flush=True)
    print(BR.round(3).to_string(index=False), flush=True)
    return BR


def gp_vs_cubic(C, fig_dir, tab):
    g = C[(C.Species == "human") & (C.Tissue.isin(["Brain", "Liver"]))]
    rows, summ = [], []
    for tis, gg in g.groupby("Tissue"):
        ct = pd.crosstab(gg.pattern_true, gg.pattern_pred, normalize="index")
        mp = gg.pattern_pred.value_counts(normalize=True)
        m = mcc(gg.devAS_true.values, gg.devAS_pred.values)
        for cls in PORD:
            rows.append(dict(Tissue=tis, cls=cls,
                             cubic_recall=float(ct.loc[cls, cls])
                             if cls in ct.index and cls in ct.columns else np.nan,
                             cubic_marginal=float(mp.get(cls, 0.0)),
                             gp_recall=GP[(tis, "recall")][cls],
                             gp_marginal=GP[(tis, "marginal")][cls]))
        summ += [dict(Tissue=tis, pipeline="GP shape concordance (site-level)",
                      n_sites=GP_N[tis], overall_agreement=GP_OVERALL[tis],
                      binary_MCC=GP_MCC[tis], store="unchanged (earlier GP run)"),
                 dict(Tissue=tis, pipeline="cubic devAS call", n_sites=len(gg),
                      overall_agreement=float((gg.pattern_true == gg.pattern_pred).mean()),
                      binary_MCC=m, store="full test set")]
        print("%s cubic n=%d overall %.3f devAS-MCC %.3f | GP n=%d overall %.3f MCC %.3f"
              % (tis, len(gg), (gg.pattern_true == gg.pattern_pred).mean(), m,
                 GP_N[tis], GP_OVERALL[tis], GP_MCC[tis]), flush=True)
    CMP = pd.DataFrame(rows)
    CMP.round(4).to_csv(os.path.join(tab, "gp_vs_cubic_perclass.csv"), index=False)
    pd.DataFrame(summ).round(3).to_csv(os.path.join(tab, "gp_vs_cubic_summary.csv"), index=False)

    LAB = dict(zip(PORD, PLAB))
    LAB["none"] = "not dynamic"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    xx = np.arange(len(PORD)); w = 0.36
    for ax, tis in zip(axes, ["Brain", "Liver"]):
        s = CMP[CMP.Tissue == tis].set_index("cls")
        ax.bar(xx - w / 2, s.loc[PORD, "gp_recall"], w, color="#7f8c8d",
               label="GP shape concordance")
        ax.bar(xx + w / 2, s.loc[PORD, "cubic_recall"], w, color="#c0392b",
               label="cubic devAS call")
        for i, cls in enumerate(PORD):
            ax.plot([i - w, i - 0.02], [s.loc[cls, "gp_marginal"]] * 2, color="k", lw=1.1, zorder=5)
            ax.plot([i + 0.02, i + w], [s.loc[cls, "cubic_marginal"]] * 2, color="k", lw=1.1, zorder=5)
        ax.set_xticks(xx)
        ax.set_xticklabels([LAB[c] for c in PORD], rotation=30, ha="right")
        ax.set_title("Human %s" % tis.lower())
        ax.set_ylim(0, 1.12)
        ax.margins(x=0.04)
    axes[0].set_ylabel("Fraction of observed sites\ngiven the same label")
    axes[0].plot([], [], color="k", lw=1.1, label="rate that label is predicted overall")
    axes[0].legend(frameon=False, fontsize=6, loc="upper center", ncol=3,
                   bbox_to_anchor=(1.02, 1.16), columnspacing=1.2, handletextpad=.5)
    fig.savefig(os.path.join(fig_dir, "recall_vs_marginal.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return CMP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--tab-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    os.makedirs(a.tab_dir, exist_ok=True)
    style()
    C = pd.read_parquet(a.calls)
    sp_order = [s for s in SP_ORDER if s in set(C.Species)]
    print("calls %s | %d species %s" % (C.shape, len(sp_order), sp_order), flush=True)
    ACC = accuracy(C, a.tab_dir)
    CNT = counts(C, a.tab_dir)
    CONF = confusion(C, a.tab_dir)
    fig_mcc(ACC, sp_order, a.fig_dir)
    fig_class_composition(CNT, sp_order, a.fig_dir)
    fig_observed_class_composition(C, sp_order, a.fig_dir)
    fig_observed_by_usage(C, sp_order, a.fig_dir)
    fig_observed_by_usage_binary(C, sp_order, a.fig_dir)
    fig_confusion(CONF, a.fig_dir)
    fig_confusion_grid(C, sp_order, a.fig_dir, a.tab_dir)
    SHP = fig_mcc_by_shape(C, sp_order, a.fig_dir, a.tab_dir)
    print(SHP.groupby("shape")[["n_true", "n_pred", "TP", "MCC"]].mean().round(3).to_string(),
          flush=True)
    fig_dsse(C, sp_order, a.fig_dir)
    fig_mean_usage(C, a.fig_dir, a.tab_dir)
    gp_vs_cubic(C, a.fig_dir, a.tab_dir)


if __name__ == "__main__":
    main()
