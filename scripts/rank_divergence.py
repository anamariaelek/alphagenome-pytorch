#!/usr/bin/env python
"""Rank model-verified divergent developmental trajectories for attribution.

Input is xspecies_pairs_allfocal.parquet (score_pairs_gw.py).  Candidates are the
rows that are both `divergent` (observed shape classes contrast between the two
species) and `verified` (both sides well predicted AND both predicted shape
classes equal the observed ones).  Those two conditions already imply div_recap,
because matching both shapes forces the predicted contrast to match the observed
one, so no separate recapitulation filter is applied.

Three ranked components, all emitted as columns so the ranking is auditable:

  quality   shape-recovery quality on the DYNAMIC side(s) of the contrast:
            (per-site Pearson r between observed and predicted trajectory) /
            (replicate-split ceiling for that species x tissue), minimised over
            whichever sides are dynamic.  Pearson r is undefined in spirit for a
            static trajectory, so for gain/loss contrasts — where exactly one
            side is static by construction — the static side is scored instead by
            static_rmse, its centred-trajectory RMSE, which measures whether the
            model also predicts flatness.  Ceilings come from
            traj_eval_summary.csv with band == 'all'; reference == 'sse_true' is
            used because the per-site r here is computed against SSE_true, and the
            reference == 'counts' ceiling is carried alongside for comparison.
            Clipped at 1.0 for ranking.  Note this component saturates within
            the candidate pool — verification already requires r >= 0.70 while the
            median ceiling is 0.68-0.78, so most candidates exceed 1.0; the cut is
            therefore set at 1.0, i.e. the model reaches the organ's median
            replicate-split reliability on the dynamic side.
  div_mag   observed divergence magnitude: the largest absolute difference
            between the two species' observed trajectories at any shared
            timepoint (max of gap_focal_high, gap_partner_high).
  margin    divergence-recapitulation margin r_pred - r_true, i.e. how much less
            divergent the model thinks the pair is than it observably is.  Ranked
            on -|margin| so that faithful recapitulation ranks high.

score = mean of the three within-candidate percentile ranks (equal weights).

    python rank_divergence.py [--pairs F] [--traj-dir D] [--out-dir D] [--fig-dir D]

Outputs:
    divergence_ranked.csv    every candidate with all components and the score
    divergence_ranking.png   div_mag vs quality, coloured by divergence class,
                             with the step-5 selection cut drawn
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "/home/elek/sds/sd17d003/Anamaria"
AG = f"{B}/alphagenome_genomicsxai"
XS = f"{AG}/best_model/cross_species_alignment"
TRAJ = f"{AG}/devas/traj_eval"
TORD = ["Brain", "Cerebellum", "Heart", "Kidney", "Liver", "Ovary", "Testis"]
DCOL = {"gain_in_focal": "#1f6fb4", "loss_in_focal": "#d1495b",
        "direction_reversal": "#6a4c93", "profile_shift": "#f0a202"}
# step-5 selection cut, stated once here and reused by select_candidates.py
CUT = dict(quality=1.00, div_mag=0.15, abs_margin=0.60, static_rmse=0.10)


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


def ceilings(traj_dir):
    s = pd.read_csv(f"{traj_dir}/traj_eval_summary.csv")
    s = s[s["band"] == "all"]
    out = {}
    for ref in ["sse_true", "counts"]:
        d = s[s["reference"] == ref]
        out[ref] = {(r.Species, r.Tissue): r.ceiling_r_median for r in d.itertuples()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=f"{XS}/xspecies_pairs_allfocal.parquet")
    p.add_argument("--traj-dir", default=TRAJ)
    p.add_argument("--out-dir", default=XS)
    p.add_argument("--fig-dir", default=f"{AG}/devas/figures")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.fig_dir, exist_ok=True)

    P = pd.read_parquet(a.pairs)
    C = P[P["divergent"] & P["verified"]].copy()
    print("candidates %d of %d pair-tissue rows | human anchors %d"
          % (len(C), len(P), C["human_site"].nunique()), flush=True)

    ceil = ceilings(a.traj_dir)
    for side, sp in [("f", "focal"), ("p", "partner")]:
        for ref in ["sse_true", "counts"]:
            C[f"{side}_ceiling_{ref}"] = [ceil[ref].get((s, t), np.nan)
                                          for s, t in zip(C[sp], C["Tissue"])]
        C[f"{side}_r_frac"] = C[f"{side}_pred_r"] / C[f"{side}_ceiling_sse_true"]
    # quality on the dynamic side(s) only; the static side is scored by RMSE
    fd, pd_ = C["f_dyn"].to_numpy(bool), C["p_dyn"].to_numpy(bool)
    fq, pq = C["f_r_frac"].to_numpy(float), C["p_r_frac"].to_numpy(float)
    C["n_dyn_sides"] = fd.astype(int) + pd_.astype(int)
    C["quality"] = np.where(fd & pd_, np.minimum(fq, pq),
                            np.where(fd, fq, np.where(pd_, pq, np.nan)))
    C["quality_clipped"] = C["quality"].clip(upper=1.0)
    fr, pr = C["f_rmse"].to_numpy(float), C["p_rmse"].to_numpy(float)
    C["static_rmse"] = np.where(fd & pd_, np.nan,
                                np.where(fd, pr, np.where(pd_, fr, np.nan)))
    C["dyn_side"] = np.where(fd & pd_, "both",
                             np.where(fd, "focal", np.where(pd_, "partner", "neither")))
    C["div_mag"] = np.nanmax(
        np.c_[C["gap_focal_high"].to_numpy(), C["gap_partner_high"].to_numpy()], axis=1)
    C["margin"] = C["r_pred"] - C["r_true"]
    C["abs_margin"] = C["margin"].abs()

    n = len(C)
    C["pct_quality"] = C["quality_clipped"].rank(pct=True, na_option="bottom")
    C["pct_div_mag"] = C["div_mag"].rank(pct=True, na_option="bottom")
    C["pct_margin"] = (-C["abs_margin"]).rank(pct=True, na_option="bottom")
    C["score"] = C[["pct_quality", "pct_div_mag", "pct_margin"]].mean(axis=1)
    C["passes_cut"] = ((C["quality"] >= CUT["quality"])
                       & (C["div_mag"] >= CUT["div_mag"])
                       & (C["abs_margin"] <= CUT["abs_margin"])
                       & (C["static_rmse"].fillna(0.0) <= CUT["static_rmse"]))
    C = C.sort_values("score", ascending=False).reset_index(drop=True)
    C["rank"] = np.arange(1, len(C) + 1)
    out = f"{a.out_dir}/divergence_ranked.csv"
    C.to_csv(out, index=False)

    print("quality  median %.3f  q10 %.3f  q90 %.3f  (>1.0: %d)"
          % (C["quality"].median(), C["quality"].quantile(.1),
             C["quality"].quantile(.9), int((C["quality"] > 1).sum())))
    print("div_mag  median %.3f  q10 %.3f  q90 %.3f"
          % (C["div_mag"].median(), C["div_mag"].quantile(.1), C["div_mag"].quantile(.9)))
    print("margin   median %.3f  |margin| median %.3f  q90 %.3f"
          % (C["margin"].median(), C["abs_margin"].median(), C["abs_margin"].quantile(.9)))
    print("dynamic sides:", C["dyn_side"].value_counts().to_dict())
    print("static_rmse quantiles",
          [round(v, 3) for v in C["static_rmse"].quantile([.1, .25, .5, .75, .9])])
    print("quality quantiles by dynamic sides:")
    print(C.groupby("dyn_side")["quality"]
           .quantile([.1, .5, .9]).unstack().round(3).to_string())
    print("\nsensitivity (rows passing div_mag>=%.2f, |margin|<=%.2f):"
          % (CUT["div_mag"], CUT["abs_margin"]))
    base = (C["div_mag"] >= CUT["div_mag"]) & (C["abs_margin"] <= CUT["abs_margin"])
    print("  %-10s %s" % ("quality\\rmse", "  ".join("%6.2f" % s for s in
                                                     [0.05, 0.10, 0.15, 0.20, 9.9])))
    for q in [0.7, 0.9, 1.0, 1.1]:
        cells = [int((base & (C["quality"] >= q)
                      & (C["static_rmse"].fillna(0.0) <= s)).sum())
                 for s in [0.05, 0.10, 0.15, 0.20, 9.9]]
        print("  %-10.2f %s" % (q, "  ".join("%6d" % v for v in cells)))
    print("\ncut quality>=%.2f div_mag>=%.2f |margin|<=%.2f static_rmse<=%.2f"
          " -> %d of %d pass (%.1f%%)"
          % (CUT["quality"], CUT["div_mag"], CUT["abs_margin"], CUT["static_rmse"],
             int(C["passes_cut"].sum()), n, 100 * C["passes_cut"].mean()))
    print("\npassing by species pair:")
    print(C[C["passes_cut"]].groupby(["focal", "partner"]).size().to_string())
    print("\npassing by divergence class:")
    print(C[C["passes_cut"]]["div_class"].value_counts().to_string())
    print("\npassing by tissue:")
    print(C[C["passes_cut"]].groupby("Tissue").size().to_string())
    print("\nunique human anchors passing:", C.loc[C["passes_cut"], "human_site"].nunique())

    # ── figure ──
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1),
                             gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.32))
    ax = axes[0]
    for k, col in DCOL.items():
        d = C[C["div_class"] == k]
        if d.empty:
            continue
        ax.scatter(d["quality"], d["div_mag"], s=7, lw=0, alpha=0.65,
                   color=col, label="%s (n=%d)" % (k.replace("_", " "), len(d)))
    ax.axvline(CUT["quality"], color="#333333", lw=0.7, ls="--")
    ax.axhline(CUT["div_mag"], color="#333333", lw=0.7, ls="--")
    ax.set_xlabel("prediction quality  (dynamic side r / replicate-split ceiling)\n"
                  "unclipped; clipped at 1.0 only for the rank component")
    ax.set_ylabel("observed divergence magnitude\n(max |ΔSSE| across shared timepoints)")
    ax.set_title("Model-verified divergent trajectories (n=%d)" % n)
    ax.legend(loc="upper left", handletextpad=0.3, borderpad=0.2)
    sel = C[C["passes_cut"]]
    ax.annotate("selection cut: %d pass" % len(sel),
                xy=(CUT["quality"], CUT["div_mag"]), xytext=(0.63, 0.06),
                textcoords="axes fraction", fontsize=6.5, color="#333333")

    ax = axes[1]
    ax.scatter(C["r_true"], C["r_pred"], s=7, lw=0, alpha=0.5, color="#7f7f7f",
               label="all candidates")
    ax.scatter(sel["r_true"], sel["r_pred"], s=9, lw=0, alpha=0.85, color="#1f6fb4",
               label="passing the cut")
    lim = [-1.05, 1.05]
    ax.plot(lim, lim, color="#333333", lw=0.7, ls=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("observed cross-species r")
    ax.set_ylabel("predicted cross-species r")
    ax.set_title("Divergence recapitulation")
    ax.legend(loc="upper left", handletextpad=0.3, borderpad=0.2)
    fig.text(0.005, -0.10,
             "candidates: divergent (observed shape classes differ) and verified "
             "(both sides well predicted, both predicted classes correct); "
             "ceilings from traj_eval_summary.csv (reference sse_true, band all); quality is "
             "scored on the dynamic side of the contrast, the static side by RMSE <= %.2f"
             % CUT["static_rmse"],
             fontsize=5.6, color="#666666")
    figp = f"{a.fig_dir}/divergence_ranking.png"
    fig.savefig(figp, bbox_inches="tight")
    fig.savefig(figp.replace(".png", ".pdf"), bbox_inches="tight")
    print("\nwrote", out, "and", figp)


if __name__ == "__main__":
    main()
