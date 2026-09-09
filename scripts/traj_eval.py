#!/usr/bin/env python
"""Evaluate predicted developmental splice-site usage trajectories.

Implements Stages 0-4 of trajectory_evaluation_workflow.md: one shared,
band-stratified site set; a replicate-split noise ceiling; the nulls; and the
per site x organ shape/magnitude pair

    r     = pearson(o_centred, p_centred)          shape
    RMSE  = sqrt(mean_t (p_c[t] - o_c[t])^2)       magnitude, SSE units

with the exact split  RMSE^2 = (sd_p - sd_o)^2 + 2 sd_p sd_o (1 - r)
into an amplitude term and a pattern term.

Usage values are SSE (splice site strength estimate, SpliSER; Dent et al. 2021,
doi:10.1093/nargab/lqab041), not PSI.

Two references for the observed trajectory, reported side by side:
  sse_true -- the stored observed usage the model was trained against;
  counts   -- pooled Alpha/(Alpha+Beta) over the libraries of each condition.

Two spaces, because they answer different questions:
  centred  -- trajectory only. Every flat prediction scores the same here
              (RMSE = sd_o), so the N1 and N2 nulls of the spec coincide; the
              single flat null is reported as rmse_flat.
  absolute -- raw SSE. N1 (flat at the observed site mean, an oracle) and N2
              (flat at the model's own predicted site mean, realisable) differ
              here, and N2 is the baseline the model must beat.

Writes one long-format parquet of per-site rows and a summary CSV.
Requires classify_devas.py and devas_glm.py on the path.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_devas as cd

log = cd.log


# ---------------------------------------------------------------- statistics

def _mmean(x, m):
    """Row-wise mean over the masked entries."""
    n = m.sum(1)
    s = np.where(m, x, 0.0).sum(1)
    return np.divide(s, n, out=np.full(len(n), np.nan), where=n > 0)


def pair_stats(o, p, m):
    """Per-row shape/magnitude statistics of prediction p against observed o.

    o, p : (S, C) float, m : (S, C) bool coverage mask.
    Returns a dict of length-S arrays. Centred quantities use only masked
    entries; sd is the population sd over those entries.
    """
    n = m.sum(1)
    om, pm = _mmean(o, m), _mmean(p, m)
    oc = np.where(m, o - om[:, None], 0.0)
    pc = np.where(m, p - pm[:, None], 0.0)
    sd_o = np.sqrt(_mmean(oc ** 2, m))
    sd_p = np.sqrt(_mmean(pc ** 2, m))
    cov = _mmean(oc * pc, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / (sd_o * sd_p)
    r = np.where((sd_o > 0) & (sd_p > 0), r, np.nan)
    rmse = np.sqrt(_mmean((pc - oc) ** 2, m))
    amp = (sd_p - sd_o) ** 2
    shape = 2.0 * sd_p * sd_o * (1.0 - r)
    rmse_abs = np.sqrt(_mmean((p - o) ** 2, m))
    # absolute-space nulls: flat at the observed mean (oracle) and flat at the
    # model's own predicted mean (realisable)
    rmse_abs_n1 = sd_o
    rmse_abs_n2 = np.sqrt(_mmean((pm[:, None] - o) ** 2, m))
    return dict(n_cond=n, mean_obs=om, mean_pred=pm, sd_o=sd_o, sd_p=sd_p,
                r=r, rmse=rmse, rmse_flat=sd_o, amp_term=amp, shape_term=shape,
                rmse_abs=rmse_abs, rmse_abs_n1=rmse_abs_n1,
                rmse_abs_n2=rmse_abs_n2,
                dsse_obs=np.nanmax(np.where(m, o, np.nan), 1)
                          - np.nanmin(np.where(m, o, np.nan), 1),
                dsse_pred=np.nanmax(np.where(m, p, np.nan), 1)
                           - np.nanmin(np.where(m, p, np.nan), 1))


def organ_null(o, m):
    """N3: the organ's mean centred trajectory, predicted for every site."""
    om = _mmean(o, m)
    oc = np.where(m, o - om[:, None], 0.0)
    cnt = m.sum(0)
    g = np.divide(oc.sum(0), cnt, out=np.zeros(oc.shape[1]), where=cnt > 0)
    gc = np.tile(g, (len(o), 1))
    gm = _mmean(gc, m)
    gc = np.where(m, gc - gm[:, None], 0.0)
    return np.sqrt(_mmean((gc - oc) ** 2, m))


# ------------------------------------------------------------------ per organ

def eval_tissue(species, tis, pt, obs_all, cond_order, args):
    keys = np.unique(pt["key"].to_numpy())
    site_index = pd.Series(np.arange(len(keys)), index=keys)
    obs = obs_all[obs_all["Condition"].isin(cond_order)
                  & obs_all["key"].isin(set(keys))]
    if not len(obs):
        log("  %s/%s: no observed rows" % (species, tis)); return None, None
    I, N, slot_cond, reps = cd.build_slots(obs, site_index, cond_order)
    ncond = len(cond_order)

    # ---- condition-level pooled counts, and an A/B split of the libraries
    Ic = np.zeros((len(keys), ncond)); Nc = np.zeros_like(Ic)
    IA = np.zeros_like(Ic); NA = np.zeros_like(Ic)
    IB = np.zeros_like(Ic); NB = np.zeros_like(Ic)
    for j, cnd in enumerate(cond_order):
        sl = np.where(slot_cond == cnd)[0]
        if not len(sl):
            continue
        Ic[:, j] = I[:, sl].sum(1); Nc[:, j] = N[:, sl].sum(1)
        a, b = sl[0::2], sl[1::2]
        if len(a): IA[:, j] = I[:, a].sum(1); NA[:, j] = N[:, a].sum(1)
        if len(b): IB[:, j] = I[:, b].sum(1); NB[:, j] = N[:, b].sum(1)

    cov = Nc >= args.min_total
    O_counts = np.divide(Ic, Nc, out=np.full_like(Ic, np.nan), where=Nc > 0)

    # ---- predicted and stored-observed usage on the same grid
    def pivot(col):
        v = (pt.pivot_table(index="key", columns="cond_code", values=col,
                            aggfunc="mean")
               .reindex(index=keys, columns=cond_order))
        return v.to_numpy(float)
    P = pivot("SSE_pred"); T = pivot("SSE_true")

    m = cov & np.isfinite(P) & np.isfinite(T) & np.isfinite(O_counts)
    keep = (m.sum(1) >= max(args.min_cond, int(np.ceil(args.frac_covered * ncond))))
    if keep.sum() == 0:
        log("  %s/%s: no site passes the gate (%d conditions)" % (species, tis, ncond))
        return None, None
    m = m[keep]
    P, T, O = P[keep], T[keep], O_counts[keep]
    kk = keys[keep]
    log("  %s/%s: %d conditions, %d/%d sites pass the gate"
        % (species, tis, ncond, keep.sum(), len(keys)))

    rows = []
    for ref, Oref in (("sse_true", T), ("counts", O)):
        st = pair_stats(Oref, P, m)
        st["rmse_n3"] = organ_null(Oref, m)
        d = pd.DataFrame(st)
        d.insert(0, "reference", ref)
        rows.append(d)
    D = pd.concat(rows, ignore_index=True)

    # ---- replicate-split noise ceiling (counts only)
    OA = np.divide(IA, NA, out=np.full_like(IA, np.nan), where=NA > 0)[keep]
    OB = np.divide(IB, NB, out=np.full_like(IB, np.nan), where=NB > 0)[keep]
    mh = m & np.isfinite(OA) & np.isfinite(OB) & (NA[keep] >= args.min_total // 2) \
           & (NB[keep] >= args.min_total // 2)
    ok = mh.sum(1) >= args.min_cond
    ceil = pair_stats(OA, OB, mh) if ok.any() else None

    if args.dump_matrices:
        np.savez_compressed(
            os.path.join(args.out_dir, "matrices_%s_%s.npz" % (species, tis)),
            key=kk, cond=np.asarray(cond_order), O=O, T=T, P=P,
            OA=OA, OB=OB, m=m, mh=mh,
            Nc=Nc[keep], NA=NA[keep], NB=NB[keep])
        log("  %s/%s: matrices dumped" % (species, tis))

    S = len(kk)
    D["key"] = np.tile(kk, 2)
    D["Species"] = species; D["Tissue"] = tis
    if ceil is not None:
        # r between two independent halves is the reliability of ONE half;
        # Spearman-Brown lifts it to the reliability R of the pooled series,
        # and the attenuation bound on any correlation against a series of
        # reliability R is sqrt(R) -- not R itself.
        # RMSE between halves is 2x the full-data noise sd, so the achievable
        # RMSE floor is half of it.
        rh = np.where(ok & (ceil["r"] > 0), ceil["r"], np.nan)
        D["ceiling_r"] = np.tile(np.sqrt(np.clip(2 * rh / (1 + rh), 0, 1)), 2)
        D["reliability"] = np.tile(np.clip(2 * rh / (1 + rh), 0, 1), 2)
        D["ceiling_rmse"] = np.tile(np.where(ok, ceil["rmse"] / 2.0, np.nan), 2)
        D["n_half"] = np.tile(mh.sum(1), 2)
    else:
        D["ceiling_r"] = np.nan; D["ceiling_rmse"] = np.nan
        D["reliability"] = np.nan; D["n_half"] = 0
    return D, dict(species=species, tissue=tis, n_cond=ncond,
                   n_sites_total=len(keys), n_sites_kept=int(keep.sum()),
                   n_libraries=int(I.shape[1]))


# ------------------------------------------------------------------- species

def run_species(species, args):
    pred = cd.read_predictions(species)
    if pred is None or not len(pred):
        log("%s: no prediction parquet, skipped" % species); return None, None
    tissues = sorted(pred["Tissue"].unique())
    if args.tissues:
        tissues = [t for t in tissues if t in args.tissues]
    site_pos = pred[["Chromosome", "Position"]].drop_duplicates()
    chrom_codes = {c: i for i, c in enumerate(sorted(site_pos["Chromosome"].unique()))}
    site_keys_all = cd.encode_key(site_pos["Chromosome"].to_numpy(),
                                  site_pos["Position"].to_numpy(), chrom_codes)

    import pyarrow.parquet as pq
    cands = [c for c in cd.SPECIES_DIRS[species]
             if os.path.exists(os.path.join(cd.DATA_DIR, c, "usage.parquet"))]
    best, best_dir = -1.0, None
    for cand in cands:
        sl = os.path.join(cd.DATA_DIR, cand,
                          "splice_sites_intersect_usage_protein_coding.parquet")
        if os.path.exists(sl):
            t = pq.read_table(sl, columns=["Chromosome", "Position"]).to_pandas()
        else:
            f = pq.ParquetFile(os.path.join(cd.DATA_DIR, cand, "usage.parquet"))
            rgs = range(f.num_row_groups) if len(cands) > 1 else [0]
            t = pd.concat([f.read_row_group(i, columns=["Chromosome", "Position"])
                            .to_pandas() for i in rgs], ignore_index=True)
        k = np.unique(cd.encode_key(t["Chromosome"].astype(str).to_numpy(),
                                    t["Position"].to_numpy(), chrom_codes))
        frac = float(np.isin(site_keys_all, k).mean())
        log("  %s: %.3f of prediction sites present" % (cand, frac))
        if frac > best:
            best, best_dir = frac, cand
    if best_dir is None or best < 0.5:
        log("%s: no matching assembly (best %.3f), skipped" % (species, best))
        return None, None
    log("  using data/%s (%.3f)" % (best_dir, best))

    with open(os.path.join(cd.DATA_DIR, best_dir, "usage.json")) as fh:
        cond_labels = json.load(fh)["condition_labels"]
    name_to_code = {k: int(v) for k, v in cond_labels.items()}
    pred["cond_code"] = pred["Condition_Name"].map(name_to_code)
    miss = pred["cond_code"].isna()
    if miss.any():
        log("  WARNING %d rows with unmapped condition names" % int(miss.sum()))
        pred = pred[~miss]
    pred["cond_code"] = pred["cond_code"].astype(int)
    pred = pred[pred["Tissue"].isin(tissues)]
    pred["key"] = cd.encode_key(pred["Chromosome"].to_numpy(),
                                pred["Position"].to_numpy(), chrom_codes)

    obs_all = cd.scan_usage(os.path.join(cd.DATA_DIR, best_dir, "usage.parquet"),
                            np.unique(pred["key"].to_numpy()),
                            set(pred["cond_code"].unique()), chrom_codes)
    if not len(obs_all):
        log("%s: no observed rows matched" % species); return None, None

    out, meta = [], []
    for tis in tissues:
        pt = pred[pred["Tissue"] == tis]
        cond_tp = (pt[["cond_code", "Timepoint"]].drop_duplicates()
                     .sort_values("Timepoint"))
        cond_order = cond_tp["cond_code"].tolist()
        if len(cond_order) < args.min_cond:
            log("  %s/%s: only %d conditions, skipped" % (species, tis, len(cond_order)))
            continue
        D, mt = eval_tissue(species, tis, pt, obs_all, cond_order, args)
        if D is not None:
            out.append(D); meta.append(mt)
    if not out:
        return None, None
    D = pd.concat(out, ignore_index=True)
    D["Chromosome"] = pd.Series(D["key"] // cd.KEY_SHIFT).map(
        {v: k for k, v in chrom_codes.items()})
    D["Position"] = (D["key"] % cd.KEY_SHIFT).astype(np.int64)
    return D, pd.DataFrame(meta)


# ---------------------------------------------------------------------- main

BANDS = [(0.0, 0.3, "<=0.3"), (0.3, 0.7, "0.3-0.7"), (0.7, 1.01, ">=0.7")]


def band_of(x):
    b = pd.Series(np.full(len(x), "", object), index=x.index)
    for lo, hi, lab in BANDS:
        b[(x >= lo) & (x < hi)] = lab
    return b


def summarise(D):
    D = D.copy()
    D["band"] = band_of(D["mean_obs"])
    D["pct_amp"] = 100.0 * D["amp_term"] / np.where(D["rmse"] ** 2 > 0,
                                                    D["rmse"] ** 2, np.nan)
    D["rmse_over_flat"] = D["rmse"] / D["rmse_flat"]
    D["rmse_over_n3"] = D["rmse"] / D["rmse_n3"]
    D["rmse_over_ceiling"] = D["rmse"] / D["ceiling_rmse"]
    D["r_over_ceiling"] = D["r"] / D["ceiling_r"]
    D["sigma_ratio"] = D["sd_p"] / D["sd_o"]
    D["abs_over_n2"] = D["rmse_abs"] / D["rmse_abs_n2"]

    def agg(g):
        q = lambda c, p: float(np.nanpercentile(g[c], p)) if g[c].notna().any() else np.nan
        return pd.Series(dict(
            n_sites=len(g),
            r_median=g["r"].median(), r_q25=q("r", 25), r_q75=q("r", 75),
            r_frac_pos=float((g["r"] > 0).mean()),
            ceiling_r_median=g["ceiling_r"].median(),
            reliability_median=g["reliability"].median(),
            r_over_ceiling_median=g["r_over_ceiling"].median(),
            rmse_median=g["rmse"].median(), rmse_q25=q("rmse", 25),
            rmse_q75=q("rmse", 75),
            rmse_flat_median=g["rmse_flat"].median(),
            rmse_n3_median=g["rmse_n3"].median(),
            ceiling_rmse_median=g["ceiling_rmse"].median(),
            rmse_over_flat_median=g["rmse_over_flat"].median(),
            frac_beating_flat=float((g["rmse_over_flat"] < 1).mean()),
            frac_beating_n3=float((g["rmse_over_n3"] < 1).mean()),
            rmse_over_ceiling_median=g["rmse_over_ceiling"].median(),
            sigma_ratio_median=g["sigma_ratio"].median(),
            sigma_ratio_q25=q("sigma_ratio", 25), sigma_ratio_q75=q("sigma_ratio", 75),
            pct_amp_median=g["pct_amp"].median(),
            rmse_abs_median=g["rmse_abs"].median(),
            rmse_abs_n1_median=g["rmse_abs_n1"].median(),
            rmse_abs_n2_median=g["rmse_abs_n2"].median(),
            frac_beating_n2_abs=float((g["abs_over_n2"] < 1).mean()),
            dsse_spearman=float(g["dsse_obs"].corr(g["dsse_pred"], method="spearman")),
            dsse_obs_median=g["dsse_obs"].median(),
            dsse_pred_median=g["dsse_pred"].median(),
        ))

    by_all = D.groupby(["Species", "Tissue", "reference"]).apply(agg).reset_index()
    by_all["band"] = "all"
    by_band = (D.groupby(["Species", "Tissue", "reference", "band"])
                 .apply(agg).reset_index())
    return D, pd.concat([by_all, by_band], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=None,
                    help="prediction store (default: the one classify_devas uses)")
    ap.add_argument("--out-dir", default="traj_eval")
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--tissues", nargs="*", default=None)
    ap.add_argument("--min-total", type=int, default=10,
                    help="minimum pooled coverage for a condition to count")
    ap.add_argument("--min-cond", type=int, default=5,
                    help="minimum covered conditions per site")
    ap.add_argument("--frac-covered", type=float, default=0.6)
    ap.add_argument("--dump-matrices", action="store_true",
                    help="save the per-organ observed/predicted matrices for diagnostics")
    args = ap.parse_args()

    if args.pred_dir:
        cd.PRED_DIR = args.pred_dir
    log("prediction store: %s" % cd.PRED_DIR)
    os.makedirs(args.out_dir, exist_ok=True)
    species = args.species or list(cd.SPECIES_DIRS)

    allsum, allmeta = [], []
    for sp in species:
        log("=== %s" % sp)
        t0 = time.time()
        D, meta = run_species(sp, args)
        if D is None:
            continue
        D, S = summarise(D)
        cols = ["Species", "Tissue", "reference", "Chromosome", "Position",
                "band", "n_cond", "n_half", "mean_obs", "mean_pred",
                "sd_o", "sd_p", "sigma_ratio", "r", "rmse", "rmse_flat",
                "rmse_n3", "reliability", "ceiling_r", "ceiling_rmse",
                "r_over_ceiling", "rmse_over_flat",
                "rmse_over_n3", "rmse_over_ceiling", "amp_term", "shape_term",
                "pct_amp", "rmse_abs", "rmse_abs_n1", "rmse_abs_n2",
                "abs_over_n2", "dsse_obs", "dsse_pred"]
        D[cols].to_parquet(os.path.join(args.out_dir, "traj_sites_%s.parquet" % sp),
                           index=False)
        allsum.append(S); allmeta.append(meta)
        log("  %s done in %.0fs, %d site-rows" % (sp, time.time() - t0, len(D)))

    if allsum:
        S = pd.concat(allsum, ignore_index=True)
        S.to_csv(os.path.join(args.out_dir, "traj_eval_summary.csv"), index=False)
        pd.concat(allmeta, ignore_index=True).to_csv(
            os.path.join(args.out_dir, "traj_eval_meta.csv"), index=False)
        k = S[(S.band == "all") & (S.reference == "sse_true")]
        log("--- median r / RMSE / RMSE_flat, reference sse_true ---")
        for _, row in k.iterrows():
            log("  %-8s %-11s n=%6d  r %.3f  rmse %.4f  flat %.4f  beat_flat %.2f"
                % (row.Species, row.Tissue, row.n_sites, row.r_median,
                   row.rmse_median, row.rmse_flat_median, row.frac_beating_flat))


if __name__ == "__main__":
    main()
