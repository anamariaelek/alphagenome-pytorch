#!/usr/bin/env python
"""Re-run the cubic devAS caller against the full test-set prediction store,
then stratify the trajectory-evaluation metrics by pattern class at full N.

Stage A  run classify_devas.main() with PRED_DIR overridden to
         best_model/preds_intersect_protein_coding_all
Stage B  concatenate the per-species call tables
Stage C  join to the per-site trajectory metrics in devas/traj_eval/ and
         summarise r / RMSE by observed and predicted pattern class

Only the small summary CSVs need to leave the host; the two large inputs
(call tables, per-site metrics) stay in the project tree.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

CLS = ["up", "down", "up-down", "down-up", "none"]


def stage_a(code_dir, base, out_dir, age_table, extra_argv):
    sys.path.insert(0, os.path.join(code_dir, "scripts"))
    import classify_devas as C

    C.PRED_DIR = os.path.join(base, "best_model", "preds_intersect_protein_coding_all")
    assert os.path.isdir(C.PRED_DIR), C.PRED_DIR
    print("[A] PRED_DIR -> %s" % C.PRED_DIR, flush=True)
    print("[A] species  -> %s" % sorted(
        s for s in os.listdir(C.PRED_DIR)
        if os.path.isdir(os.path.join(C.PRED_DIR, s))), flush=True)
    argv = ["classify_devas.py", "--age-table", age_table, "--out-dir", out_dir] + extra_argv
    sys.argv = argv
    C.main()
    return C


def stage_b(out_dir, master):
    import glob
    fs = sorted(f for f in glob.glob(os.path.join(out_dir, "devas_calls_*.parquet"))
                if "all_species" not in os.path.basename(f))
    assert fs, "no call tables written"
    D = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    assert not D.duplicated(["Species", "Tissue", "Chromosome", "Position"]).any(), \
        "duplicate site rows in the concatenated call table"
    D.to_parquet(master, index=False)
    print("[B] %d rows from %d species -> %s" % (len(D), D.Species.nunique(), master), flush=True)
    return D


def cls_summary(df, by):
    """Median r/RMSE statistics plus the optimal single amplitude scale alpha.

    alpha minimises the n_cond-weighted squared error of the centred
    prediction: alpha* = sum(w s_p s_o r) / sum(w s_p^2).
    """
    def med(g, name):
        return np.nanmedian(g[name]) if name in g.columns else np.nan

    out = []
    keys = by if isinstance(by, list) else [by]
    for kv, g in df.groupby(by):
        so = g.sd_o.to_numpy(float)
        sp = g.sd_p.to_numpy(float)
        r = g.r.to_numpy(float)
        w = g.n_cond.to_numpy(float)
        ok = np.isfinite(r) & np.isfinite(so) & np.isfinite(sp)
        a = (np.nansum(w[ok] * sp[ok] * so[ok] * r[ok]) / np.nansum(w[ok] * sp[ok] ** 2)
             if ok.sum() else np.nan)
        rmse_a = np.sqrt(np.clip(a * a * sp ** 2 - 2 * a * sp * so * r + so ** 2, 0, None))
        d = dict(zip(keys, kv if isinstance(kv, tuple) else (kv,)))
        d.update(
            n_sites=len(g),
            r_median=np.nanmedian(r),
            r_q25=np.nanquantile(r, .25),
            r_q75=np.nanquantile(r, .75),
            r_frac_pos=np.nanmean(r > 0),
            ceiling_r_median=med(g, "ceiling_r"),
            r_over_ceiling_median=med(g, "r_over_ceiling"),
            rmse_median=med(g, "rmse"),
            rmse_flat_median=np.nanmedian(so),
            rmse_over_flat_median=med(g, "rmse_over_flat"),
            frac_beating_flat=np.nanmean(g.rmse.to_numpy(float) < so),
            beat_n3=(np.nanmean(g.rmse.to_numpy(float) < g.rmse_n3.to_numpy(float))
                     if "rmse_n3" in g.columns else np.nan),
            rmse_over_n3_median=med(g, "rmse_over_n3"),
            sigma_ratio_median=med(g, "sigma_ratio"),
            pct_amp_median=med(g, "pct_amp"),
            alpha=a,
            frac_beating_flat_rescaled=np.nanmean(rmse_a < so),
            dsse_spearman_median=med(g, "dsse_spearman"),
        )
        out.append(d)
    return pd.DataFrame(out)


def stage_c(D, traj_dir, out_prefix):
    K = ["Species", "Tissue", "Chromosome", "Position"]
    keep = K + ["pattern_true", "pattern_pred", "pattern_obs",
                "devAS_true", "devAS_pred", "dpsi_true"]
    per_org, pooled, cov = [], [], []
    for sp, dg in D.groupby("Species"):
        f = os.path.join(traj_dir, "traj_sites_%s.parquet" % sp)
        if not os.path.exists(f):
            print("[C] %-9s no trajectory metrics, skipped" % sp, flush=True)
            continue
        T = pd.read_parquet(f)
        M = T.merge(dg[keep], on=K, how="inner")
        n_call = len(dg)
        n_traj = (T.reference == T.reference.iloc[0]).sum()
        n_join = (M.reference == M.reference.iloc[0]).sum() if len(M) else 0
        cov.append(dict(Species=sp, n_called=n_call, n_traj_sites=n_traj, n_joined=n_join,
                        frac_of_called=n_join / n_call if n_call else np.nan,
                        frac_of_traj=n_join / n_traj if n_traj else np.nan))
        print("[C] %-9s called %7d | traj %7d | joined %7d" % (sp, n_call, n_traj, n_join), flush=True)
        for ref, mg in M.groupby("reference"):
            for strat in ["pattern_true", "pattern_pred"]:
                a = cls_summary(mg, ["Species", "Tissue", strat]).rename(
                    columns={strat: "pattern"})
                a["stratifier"] = strat
                a["reference"] = ref
                per_org.append(a)
        del T, M
    PO = pd.concat(per_org, ignore_index=True)
    PO.to_csv(out_prefix + "_by_pattern_class_full.csv", index=False)

    # pooled across organs and species, from the joined rows again
    for sp, dg in D.groupby("Species"):
        f = os.path.join(traj_dir, "traj_sites_%s.parquet" % sp)
        if not os.path.exists(f):
            continue
        M = pd.read_parquet(f).merge(dg[keep], on=K, how="inner")
        M["Species"] = sp
        pooled.append(M[["Species", "reference", "pattern_true", "pattern_pred",
                         "sd_o", "sd_p", "r", "n_cond", "rmse", "rmse_n3", "ceiling_r",
                         "r_over_ceiling", "rmse_over_flat", "rmse_over_n3",
                         "sigma_ratio", "pct_amp"]])
    PL = pd.concat(pooled, ignore_index=True)
    rows = []
    for ref, g in PL.groupby("reference"):
        for strat in ["pattern_true", "pattern_pred"]:
            a = cls_summary(g, strat).rename(columns={strat: "pattern"})
            a["stratifier"] = strat
            a["reference"] = ref
            a["scope"] = "all species"
            rows.append(a)
        for sp, gs in g.groupby("Species"):
            a = cls_summary(gs, "pattern_true").rename(columns={"pattern_true": "pattern"})
            a["stratifier"] = "pattern_true"
            a["reference"] = ref
            a["scope"] = sp
            rows.append(a)
    PP = pd.concat(rows, ignore_index=True)
    PP.to_csv(out_prefix + "_by_pattern_class_pooled_full.csv", index=False)
    pd.DataFrame(cov).to_csv(out_prefix + "_class_join_coverage.csv", index=False)
    print(PP[(PP.scope == "all species") & (PP.reference == "sse_true") &
             (PP.stratifier == "pattern_true")]
          .set_index("pattern").reindex(CLS)[
              ["n_sites", "r_median", "r_over_ceiling_median", "frac_beating_flat",
               "beat_n3", "sigma_ratio_median", "alpha"]].round(3).to_string(), flush=True)
    return PO, PP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--age-table", required=True)
    ap.add_argument("--master", default=None)
    ap.add_argument("--out-prefix", default="traj_eval")
    ap.add_argument("--skip-a", action="store_true",
                    help="reuse call tables already in --out-dir")
    a, extra = ap.parse_known_args()
    os.makedirs(a.out_dir, exist_ok=True)
    master = a.master or os.path.join(a.out_dir, "devas_calls_all_species_full.parquet")
    if not a.skip_a:
        stage_a(a.code_dir, a.base, a.out_dir, a.age_table, extra)
    D = stage_b(a.out_dir, master)
    stage_c(D, a.traj_dir, a.out_prefix)


if __name__ == "__main__":
    main()
