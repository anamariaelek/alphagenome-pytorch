#!/usr/bin/env python
"""Score developmental-trajectory divergence for every focal species pair.

Generalises score_pairs.py from a human-anchored orientation to all pairs among
human/mouse/rat/rabbit/opossum, scored transitively through the human anchor: a
human site carrying orthologs in two partners makes those two partners a
comparable pair as well.  Macaque and chicken are excluded — macaque's clustered
test-site universe is 5,291 sites (Brain and Liver tables empty) giving only 189
orthologous pairs, and chicken has no clustered test sites at all.

Definitions are reused unchanged from score_pairs.py / xspecies_score_lineage.py:
per-site shape (obs_shape_site / pred_shape_site), is_well_predicted from
xspecies.py, the divergence class contrast, and div_recap / innovation.
Orientation: within a pair, `focal` is whichever species comes first in SP_ORDER,
so gain_in_focal / loss_in_focal read against that species.

Outputs (--out-dir):
    xspecies_pairs_allfocal.parquet   one row per (focal_site, partner_site, tissue)
    xspecies_pool_coverage.csv        per species pair x tissue funnel counts

    python score_pairs_gw.py [--map F] [--pred-dir D] [--out-dir D]
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

SRC = "/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from alphagenome_pytorch import xspecies as sx  # noqa: E402

B = "/home/elek/sds/sd17d003/Anamaria"
AG = f"{B}/alphagenome_genomicsxai"
PRED = f"{AG}/best_model/preds_intersect_protein_coding"
MAP = f"{AG}/best_model/cross_species_alignment/xspecies_site_map_genomewide.parquet"
TIS = ["Brain", "Cerebellum", "Heart", "Kidney", "Liver", "Testis", "Ovary"]
SP_ORDER = ["human", "mouse", "rat", "rabbit", "opossum"]
PARTNERS = SP_ORDER[1:]
TPS = sx.TPS
DYNSET = {"up", "down", "up-down", "down-up"}
MONO = {"up", "down"}


def dclass(a, b):
    """Divergence class of an observed shape contrast, oriented on the focal side."""
    if not isinstance(a, str) or not isinstance(b, str):
        return None
    if a == b:
        return "conserved"
    da, db = a in DYNSET, b in DYNSET
    if da and not db:
        return "gain_in_focal"
    if db and not da:
        return "loss_in_focal"
    if a in MONO and b in MONO:
        return "direction_reversal"
    return "profile_shift"


def load_species(sp, pred_dir, need):
    """Per-(site, tissue) metrics plus wide true/pred trajectory matrices."""
    d = pd.read_parquet(f"{pred_dir}/{sp}/usage_{sp}.parquet")
    d["Chromosome"] = d["Chromosome"].astype(str)
    d = d.rename(columns={"SSE_true": "true", "SSE_pred": "pred"})
    fr = []
    for t in TIS:
        cp = (f"{pred_dir}/{sp}/pred_gp_splice_usage/{t}/"
              f"{sp}_{t}_test_prediction_clusters.parquet")
        if os.path.exists(cp):
            cd = pd.read_parquet(cp)
            if len(cd) == 0:
                continue
            cd["Chromosome"] = cd["Chromosome"].astype(str)
            cd["Tissue"] = t
            fr.append(cd)
    cl = pd.concat(fr, ignore_index=True)
    cl["site"] = sx._site_series(cl["Chromosome"], cl["Position"], cl["Strand"])
    st = (cl.drop_duplicates(["Chromosome", "Position"])
            .set_index(["Chromosome", "Position"])["Strand"])
    d["Strand"] = [sx.norm_strand(st.get((c, int(p)), "?"))
                   for c, p in zip(d["Chromosome"], d["Position"])]
    d["site"] = sx._site_series(d["Chromosome"], d["Position"], d["Strand"])
    traj = d.loc[d["site"].isin(need), ["site", "Tissue", "Timepoint", "true", "pred"]]
    shp = cl.loc[cl["site"].isin(need),
                 ["site", "Tissue", "obs_shape_site", "pred_shape_site",
                  "same_shape_site", "obs_shape", "pred_shape", "same_shape", "n_obs"]]

    tw = traj.pivot_table(index=["site", "Tissue"], columns="Timepoint",
                          values="true", aggfunc="mean").reindex(columns=TPS)
    pw = traj.pivot_table(index=["site", "Tissue"], columns="Timepoint",
                          values="pred", aggfunc="mean").reindex(columns=TPS)
    T, P = tw.to_numpy(), pw.to_numpy()
    with np.errstate(invalid="ignore", all="ignore"):
        amp = np.nanmax(T, axis=1) - np.nanmin(T, axis=1)
        amp_p = np.nanmax(P, axis=1) - np.nanmin(P, axis=1)
        mt, mp = np.nanmean(T, axis=1), np.nanmean(P, axis=1)
    r = np.array([sx.xcorr(T[i], P[i]) for i in range(len(T))])
    msk = np.isfinite(T) & np.isfinite(P)
    rmse = np.sqrt(np.nanmean(np.where(msk, (T - P) ** 2, np.nan), axis=1))
    S = pd.DataFrame({"site": tw.index.get_level_values(0),
                      "Tissue": tw.index.get_level_values(1),
                      "amp": amp, "amp_pred": amp_p, "mean_true": mt, "mean_pred": mp,
                      "pred_r": r, "pred_rmse": rmse,
                      "n_tp": np.isfinite(T).sum(axis=1)})
    S = S.merge(shp, on=["site", "Tissue"], how="left")
    S["dyn"] = S["obs_shape_site"].isin(DYNSET)
    S["well"] = [sx.is_well_predicted(a, b, c)
                 for a, b, c in zip(S["pred_r"], S["pred_rmse"], S["dyn"])]
    print("%-9s (site,tissue) rows %7d | sites %6d | well %4.1f%% | dynamic %4.1f%%"
          % (sp, len(S), S["site"].nunique(), 100 * S["well"].mean(),
             100 * S["dyn"].mean()), flush=True)
    return S, tw, pw


def members(m):
    """Long table of ortholog-group membership: one row per (human_site, species)."""
    hum = (m[["human_site"]].drop_duplicates()
           .assign(species="human", site=lambda d: d["human_site"], aln_dist=0))
    oth = m.loc[m["species"].isin(PARTNERS),
                ["human_site", "species", "aln_site", "aln_dist"]].rename(
        columns={"aln_site": "site"})
    return pd.concat([hum, oth], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default=MAP)
    p.add_argument("--pred-dir", default=PRED)
    p.add_argument("--out-dir", default=".")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    m = pd.read_parquet(a.map)
    mem = members(m)
    print("map rows %d | groups %d | membership rows %d"
          % (len(m), m["human_site"].nunique(), len(mem)), flush=True)
    print(mem.groupby("species")["site"].nunique().to_dict(), flush=True)

    need = {sp: set(g["site"]) for sp, g in mem.groupby("species")}
    S, TW, PW = {}, {}, {}
    for sp in SP_ORDER:
        S[sp], TW[sp], PW[sp] = load_species(sp, a.pred_dir, need[sp])

    rows = []
    for f_sp, p_sp in itertools.combinations(SP_ORDER, 2):
        fm = mem[mem["species"] == f_sp][["human_site", "site", "aln_dist"]]
        pm = mem[mem["species"] == p_sp][["human_site", "site", "aln_dist"]]
        link = fm.merge(pm, on="human_site", suffixes=("_f", "_p"))
        if link.empty:
            continue
        n_pair_sites = len(link)
        for t in TIS:
            fs = S[f_sp][S[f_sp]["Tissue"] == t].set_index("site")
            ps = S[p_sp][S[p_sp]["Tissue"] == t].set_index("site")
            sub = link[link["site_f"].isin(fs.index) & link["site_p"].isin(ps.index)]
            if sub.empty:
                continue
            F, P_ = fs.loc[sub["site_f"]], ps.loc[sub["site_p"]]
            kf = list(zip(sub["site_f"], [t] * len(sub)))
            kp = list(zip(sub["site_p"], [t] * len(sub)))
            tf, tp = TW[f_sp].loc[kf].to_numpy(), TW[p_sp].loc[kp].to_numpy()
            pf, pp = PW[f_sp].loc[kf].to_numpy(), PW[p_sp].loc[kp].to_numpy()
            rt = np.array([sx.xcorr(tf[i], tp[i]) for i in range(len(tf))])
            rp = np.array([sx.xcorr(pf[i], pp[i]) for i in range(len(pf))])
            both = np.isfinite(tf) & np.isfinite(tp)
            with np.errstate(invalid="ignore", all="ignore"):
                n_sh = both.sum(axis=1)
                gf = np.nanmax(np.where(both, tf - tp, np.nan), axis=1)
                gp = np.nanmax(np.where(both, tp - tf, np.nan), axis=1)
            rows.append(pd.DataFrame({
                "focal": f_sp, "partner": p_sp, "Tissue": t,
                "human_site": sub["human_site"].to_numpy(),
                "focal_site": sub["site_f"].to_numpy(),
                "partner_site": sub["site_p"].to_numpy(),
                "aln_dist_focal": sub["aln_dist_f"].to_numpy(),
                "aln_dist_partner": sub["aln_dist_p"].to_numpy(),
                "f_shape": F["obs_shape_site"].to_numpy(),
                "p_shape": P_["obs_shape_site"].to_numpy(),
                "f_pshape": F["pred_shape_site"].to_numpy(),
                "p_pshape": P_["pred_shape_site"].to_numpy(),
                "f_shape_ok": F["same_shape_site"].to_numpy(),
                "p_shape_ok": P_["same_shape_site"].to_numpy(),
                "f_dyn": F["dyn"].to_numpy(), "p_dyn": P_["dyn"].to_numpy(),
                "f_amp": F["amp"].to_numpy(), "p_amp": P_["amp"].to_numpy(),
                "f_amp_pred": F["amp_pred"].to_numpy(),
                "p_amp_pred": P_["amp_pred"].to_numpy(),
                "f_mean": F["mean_true"].to_numpy(), "p_mean": P_["mean_true"].to_numpy(),
                "f_pred_r": F["pred_r"].to_numpy(), "p_pred_r": P_["pred_r"].to_numpy(),
                "f_rmse": F["pred_rmse"].to_numpy(), "p_rmse": P_["pred_rmse"].to_numpy(),
                "f_well": F["well"].to_numpy(), "p_well": P_["well"].to_numpy(),
                "f_ntp": F["n_tp"].to_numpy(), "p_ntp": P_["n_tp"].to_numpy(),
                "n_shared_tp": n_sh, "r_true": rt, "r_pred": rp,
                "gap_focal_high": gf, "gap_partner_high": gp}))
        print("%-9s vs %-9s ortholog site pairs %6d" % (f_sp, p_sp, n_pair_sites),
              flush=True)

    P = pd.concat(rows, ignore_index=True)
    P["div_class"] = [dclass(x, y) for x, y in zip(P["f_shape"], P["p_shape"])]
    P["pred_div_class"] = [dclass(x, y) for x, y in zip(P["f_pshape"], P["p_pshape"])]
    P["divergent"] = P["div_class"].isin(
        ["gain_in_focal", "loss_in_focal", "direction_reversal", "profile_shift"])
    P["both_well"] = P["f_well"] & P["p_well"]
    P["both_shape_ok"] = (P["f_shape_ok"].astype("boolean").fillna(False)
                          & P["p_shape_ok"].astype("boolean").fillna(False))
    P["verified"] = P["both_well"] & P["both_shape_ok"]
    P["div_recap"] = P["divergent"] & (P["pred_div_class"] == P["div_class"])
    P["innovation"] = P["divergent"] & P["verified"] & P["div_recap"]
    out = f"{a.out_dir}/xspecies_pairs_allfocal.parquet"
    P.to_parquet(out, index=False)

    P["div_verified"] = P["divergent"] & P["verified"]
    P["has_shape"] = P["div_class"].notna()
    cov = (P.assign(n=1)
            .groupby(["focal", "partner", "Tissue"])
            .agg(n_pairs=("n", "sum"),
                 n_both_shape=("has_shape", "sum"),
                 n_divergent=("divergent", "sum"),
                 n_verified=("verified", "sum"),
                 n_div_verified=("div_verified", "sum"),
                 n_innovation=("innovation", "sum"))
            .reset_index())
    cov.to_csv(f"{a.out_dir}/xspecies_pool_coverage.csv", index=False)

    print("\n=== PAIRS", P.shape, "->", out)
    for lab, msk in [
            ("all pairs", np.ones(len(P), bool)),
            ("both sides have shape", P["div_class"].notna()),
            ("divergent", P["divergent"]),
            ("both well-predicted", P["both_well"]),
            ("both shapes correct", P["both_shape_ok"]),
            ("verified", P["verified"]),
            ("divergent + verified", P["divergent"] & P["verified"]),
            ("innovation (div+ver+recap)", P["innovation"])]:
        print("  %-28s %8d" % (lab, int(msk.sum())))
    print("\ninnovation by species pair:")
    print(P[P["innovation"]].groupby(["focal", "partner"]).size().to_string())
    print("\ninnovation by tissue:")
    print(P[P["innovation"]].groupby("Tissue").size().to_string())
    print("\ndivergence class of innovations:")
    print(P[P["innovation"]]["div_class"].value_counts().to_string())
    print("\nunique human anchors with >=1 innovation:",
          P.loc[P["innovation"], "human_site"].nunique())


if __name__ == "__main__":
    main()
