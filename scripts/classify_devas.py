#!/usr/bin/env python
"""Call developmental AS (devAS) on observed and on predicted splice-site
usage trajectories, following Mazin et al. 2021 (doi:10.1038/s41588-021-00851-w).

For every species x organ present in the prediction store this script

  1. builds a replicate-level design: one slot per sequencing library, its
     age taken from the aligned-timepoint age table (log days from conception);
  2. fits the paper's quasi-binomial GLM  (Alpha, Beta) ~ a + a^2 + a^3  to the
     observed counts, tests each term by quasi-likelihood ratio F test,
     BH-adjusts within species x organ, and calls devAS when any term passes
     and the spline amplitude dPSI > 0.2;
  3. repeats the identical test on the stored observed usage (SSE_true) and on
     the *predicted* usage (SSE_pred), each converted into pseudo-counts on the
     library's own observed coverage, so all three passes share one design, one
     coverage mask and one set of filters and differ only in the usage values.
     Pseudo-counts carry no within-condition variance, so both of these passes
     are tested against the dispersion of the count-based fit rather than their
     own -- "is this trajectory shape significant at the data's noise level";
  4. classifies all three into up / down / up-down / down-up.

Outputs one parquet per species carrying the three calls side by side
(_obs = counts, _true = stored observed usage, _pred = predicted usage), plus
a run log. The observed-vs-predicted comparison should use the _true / _pred
pair, which is matched in aggregation; _obs is the paper-faithful reference.
Requires devas_glm.py on the path.

Usage
-----
    python classify_devas.py --age-table stage_age_table.csv --out-dir devas
    python classify_devas.py --species human mouse --tissues Brain Cerebellum
    python classify_devas.py --mode drop1        # the stricter per-term test
    python classify_devas.py --verify 200        # refit sites with statsmodels
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devas_glm as dg

BASE = os.environ.get("AGX_BASE", "/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai")
PRED_DIR = os.path.join(BASE, "best_model", "preds_intersect_protein_coding")
DATA_DIR = os.path.join(BASE, "data")

# prediction-store species key -> candidate data/ directory names, in the order
# they are tried; the one whose site coordinates overlap the predictions best
# is used (mouse exists in two assemblies).
SPECIES_DIRS = {
    "human":   ["Homo_sapiens"],
    "macaque": ["Macaca_mulatta"],
    "mouse":   ["Mus_musculus", "Mus_musculus_mm10"],
    "rat":     ["Rattus_norvegicus"],
    "rabbit":  ["Oryctolagus_cuniculus"],
    "opossum": ["Monodelphis_domestica"],
    "chicken": ["Gallus_gallus"],
}
SPECIES_AGE_KEY = {"human": "human", "macaque": "macaque", "mouse": "mouse",
                   "rat": "rat", "rabbit": "rabbit", "opossum": "opossum",
                   "chicken": "chicken"}
KEY_SHIFT = np.int64(2) ** 32


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def encode_key(chrom, pos, chrom_codes):
    c = pd.Series(chrom, dtype="object").map(chrom_codes).fillna(-1).to_numpy(np.int64)
    return c * KEY_SHIFT + np.asarray(pos, np.int64)


def read_predictions(species):
    p = os.path.join(PRED_DIR, species, "usage_%s.parquet" % species)
    if not os.path.exists(p):
        return None
    df = pq.read_table(p).to_pandas()
    df["Chromosome"] = df["Chromosome"].astype(str)
    return df


def scan_usage(usage_path, site_keys, cond_keep, chrom_codes):
    """Stream usage.parquet row group by row group, keeping the wanted
    (site, condition) rows. Returns a DataFrame of key/Condition/Alpha/Beta."""
    f = pq.ParquetFile(usage_path)
    order = np.argsort(site_keys)
    sk = site_keys[order]
    keep_cond = np.zeros(int(max(cond_keep)) + 1, bool)
    keep_cond[list(cond_keep)] = True
    out = []
    n_seen = 0
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=["Chromosome", "Position", "Alpha",
                                          "Beta", "Condition"])
        d = t.to_pandas()
        n_seen += len(d)
        c = d["Condition"].to_numpy()
        m = (c >= 0) & (c < len(keep_cond))
        m &= keep_cond[np.clip(c, 0, len(keep_cond) - 1)]
        if not m.any():
            continue
        d = d[m]
        k = encode_key(d["Chromosome"].astype(str).to_numpy(), d["Position"].to_numpy(),
                       chrom_codes)
        pos = np.searchsorted(sk, k)
        pos = np.clip(pos, 0, len(sk) - 1)
        hit = sk[pos] == k
        if not hit.any():
            continue
        d = d.loc[hit].copy()
        d["key"] = k[hit]
        out.append(d[["key", "Condition", "Alpha", "Beta"]])
    log("  scanned %d rows, kept %d" % (n_seen, sum(len(o) for o in out)))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=["key", "Condition", "Alpha", "Beta"]))


def build_slots(obs, site_index, cond_order):
    """Replicate-level slot matrices for one organ.

    obs        : rows of key/Condition/Alpha/Beta for this organ
    site_index : dict key -> row of the output matrices
    cond_order : conditions in ascending timepoint order

    Returns (I, N, slot_cond) where I and N are (n_sites, n_slots) and
    slot_cond gives the condition of each slot column.
    """
    obs = obs.sort_values(["key", "Condition"], kind="mergesort")
    cc = obs.groupby(["key", "Condition"], sort=False).cumcount().to_numpy()
    reps = {c: int(cc[obs["Condition"].to_numpy() == c].max()) + 1 if (obs["Condition"] == c).any() else 0
            for c in cond_order}
    offs, tot = {}, 0
    for c in cond_order:
        offs[c] = tot
        tot += reps[c]
    slot_cond = np.concatenate([np.full(reps[c], c) for c in cond_order]) if tot else np.array([])
    rows = obs["key"].map(site_index).to_numpy()
    cols = obs["Condition"].map(offs).to_numpy() + cc
    ok = np.isfinite(rows.astype(float)) & (cols < tot)
    I = np.zeros((len(site_index), tot))
    N = np.zeros((len(site_index), tot))
    r = rows[ok].astype(int); cl = cols[ok].astype(int)
    a = obs["Alpha"].to_numpy()[ok].astype(float)
    b = obs["Beta"].to_numpy()[ok].astype(float)
    I[r, cl] = a
    N[r, cl] = a + b
    return I, N, slot_cond, reps


def run_species(species, ages, args):
    pred = read_predictions(species)
    if pred is None or not len(pred):
        log("%s: no prediction parquet, skipped" % species); return None
    tissues = sorted(pred["Tissue"].unique())
    if args.tissues:
        tissues = [t for t in tissues if t in args.tissues]
    site_pos = pred[["Chromosome", "Position"]].drop_duplicates()
    chrom_codes = {c: i for i, c in enumerate(sorted(site_pos["Chromosome"].unique()))}
    site_keys_all = encode_key(site_pos["Chromosome"].to_numpy(),
                               site_pos["Position"].to_numpy(), chrom_codes)

    # pick the data directory whose coordinates match the predictions
    cands = [c for c in SPECIES_DIRS[species]
             if os.path.exists(os.path.join(DATA_DIR, c, "usage.parquet"))]
    best, best_dir = -1.0, None
    for cand in cands:
        # fraction of PREDICTION sites present in this build's site list.
        # Scanned over the whole file (not one row group) whenever the species
        # has more than one candidate assembly, so the choice is decisive.
        sl = os.path.join(DATA_DIR, cand,
                          "splice_sites_intersect_usage_protein_coding.parquet")
        if os.path.exists(sl):
            t = pq.read_table(sl, columns=["Chromosome", "Position"]).to_pandas()
        else:
            f = pq.ParquetFile(os.path.join(DATA_DIR, cand, "usage.parquet"))
            rgs = range(f.num_row_groups) if len(cands) > 1 else [0]
            t = pd.concat([f.read_row_group(i, columns=["Chromosome", "Position"])
                            .to_pandas() for i in rgs], ignore_index=True)
        k = np.unique(encode_key(t["Chromosome"].astype(str).to_numpy(),
                                 t["Position"].to_numpy(), chrom_codes))
        frac = float(np.isin(site_keys_all, k).mean())
        log("  %s: %.3f of prediction sites present (%d site coords)"
            % (cand, frac, len(k)))
        if frac > best:
            best, best_dir = frac, cand
    if len(cands) > 1 and best_dir is not None:
        log("  assembly chosen: %s" % best_dir)
    if best_dir is None:
        log("%s: no usage.parquet found" % species); return None
    if best < 0.5:
        raise RuntimeError("%s: only %.3f of prediction sites are present in "
                           "data/%s -- wrong assembly or wrong site set"
                           % (species, best, best_dir))
    log("  using data/%s (%.3f of prediction sites present)" % (best_dir, best))

    with open(os.path.join(DATA_DIR, best_dir, "usage.json")) as fh:
        cond_labels = json.load(fh)["condition_labels"]
    name_to_code = {k: int(v) for k, v in cond_labels.items()}
    pred["cond_code"] = pred["Condition_Name"].map(name_to_code)
    miss = pred["cond_code"].isna()
    if miss.any():
        log("  WARNING %d prediction rows have unmapped condition names" % int(miss.sum()))
        pred = pred[~miss]
    pred["cond_code"] = pred["cond_code"].astype(int)
    pred = pred[pred["Tissue"].isin(tissues)]
    pred["key"] = encode_key(pred["Chromosome"].to_numpy(), pred["Position"].to_numpy(),
                             chrom_codes)

    usage_path = os.path.join(DATA_DIR, best_dir, "usage.parquet")
    obs_all = scan_usage(usage_path, np.unique(pred["key"].to_numpy()),
                         set(pred["cond_code"].unique()), chrom_codes)
    if not len(obs_all):
        log("%s: no observed rows matched" % species); return None

    age_s = ages[ages["species"] == SPECIES_AGE_KEY[species]].set_index("Timepoint")["dpc"]
    out = []
    for tis in tissues:
        pt = pred[pred["Tissue"] == tis]
        cond_tp = (pt[["cond_code", "Timepoint"]].drop_duplicates()
                     .sort_values("Timepoint"))
        cond_order = cond_tp["cond_code"].tolist()
        tps = cond_tp["Timepoint"].to_numpy()
        if not set(tps).issubset(set(age_s.index)):
            log("  %s/%s: timepoints %s missing from age table, skipped"
                % (species, tis, sorted(set(tps) - set(age_s.index)))); continue
        keys = np.unique(pt["key"].to_numpy())
        site_index = pd.Series(np.arange(len(keys)), index=keys)
        obs = obs_all[obs_all["Condition"].isin(cond_order)
                      & obs_all["key"].isin(set(keys))]
        if not len(obs):
            log("  %s/%s: no observed rows" % (species, tis)); continue
        I, N, slot_cond, reps = build_slots(obs, site_index, cond_order)
        if I.shape[1] <= 5:
            log("  %s/%s: only %d library slots, cannot fit" % (species, tis, I.shape[1]))
            continue
        cond_tp_map = dict(zip(cond_tp["cond_code"], cond_tp["Timepoint"]))
        slot_tp = np.array([cond_tp_map[c] for c in slot_cond])
        la = np.log(age_s.reindex(slot_tp).to_numpy(float))

        SSE = np.divide(I, N, out=np.full_like(I, np.nan), where=N > 0)
        # predicted pseudo-counts on the same coverage
        pv = (pt.pivot_table(index="key", columns="cond_code", values="SSE_pred",
                             aggfunc="mean").reindex(index=keys, columns=cond_order))
        Ppred = pv.to_numpy(float)[:, [cond_order.index(c) for c in slot_cond]]
        tv = (pt.pivot_table(index="key", columns="cond_code", values="SSE_true",
                             aggfunc="mean").reindex(index=keys, columns=cond_order))
        Ptrue = tv.to_numpy(float)[:, [cond_order.index(c) for c in slot_cond]]
        Ipred = np.where(np.isfinite(Ppred), np.rint(np.nan_to_num(Ppred) * N), 0.0)
        Ipred = np.clip(Ipred, 0, N)
        # Slot-level mask: a library slot enters the fit when its own coverage
        # defines a usage value and the model scored that condition.
        mask = (N >= args.min_total) & np.isfinite(Ppred)
        # Filters are applied per CONDITION, not per slot: usage.parquet holds
        # a variable (and sometimes duplicated) number of rows per library, so
        # the slot count is not a trustworthy denominator, whereas the set of
        # conditions is exactly the paper's developmental series.
        ncond = len(cond_order)
        cidx = np.array([cond_order.index(c) for c in slot_cond])
        cov_c = np.zeros((I.shape[0], ncond), bool)
        Npool = np.zeros((I.shape[0], ncond))
        Ipool = np.zeros((I.shape[0], ncond))
        for j in range(ncond):
            s = cidx == j
            Npool[:, j] = N[:, s].sum(1)
            Ipool[:, j] = I[:, s].sum(1)
            cov_c[:, j] = mask[:, s].any(1)
        SSEpool = np.divide(Ipool, Npool, out=np.full_like(Npool, np.nan),
                            where=Npool > 0)
        frac = cov_c.mean(1)
        inter = (cov_c & (SSEpool >= 0.1) & (SSEpool <= 0.9)).sum(1)
        testable = (frac >= args.frac_covered) & (inter >= args.n_intermediate)
        Im, Nm = np.where(mask, I, 0.0), np.where(mask, N, 0.0)
        Ipm = np.where(mask, Ipred, 0.0)
        # need enough libraries for 4 parameters plus residual df, and the
        # developmental series must have >= 6 covered conditions
        testable = testable & ((Nm > 0).sum(1) >= 8) & (cov_c.sum(1) >= 6)

        # Does pooling counts over a condition's rows reproduce the SSE_true
        # stored in the prediction store? Settles whether the extra rows per
        # library are replicate libraries (yes) or something else (no).
        Tc = np.full((I.shape[0], ncond), np.nan)
        for j in range(ncond):
            s = cidx == j
            Tc[:, j] = np.nanmean(np.where(mask[:, s], Ptrue[:, s], np.nan), axis=1)
        SSEmean = np.full((I.shape[0], ncond), np.nan)
        for j in range(ncond):
            sj = cidx == j
            SSEmean[:, j] = np.nanmean(np.where(mask[:, sj], SSE[:, sj], np.nan), axis=1)
        gd = cov_c & np.isfinite(Tc) & np.isfinite(SSEpool)
        if gd.any():
            dd = np.abs(SSEpool[gd] - Tc[gd])
            log("  %s/%s: pooled-vs-stored SSE_true  median|d|=%.4g  p99=%.4g  "
                "frac<0.01=%.3f (n=%d)" % (species, tis, float(np.median(dd)),
                float(np.percentile(dd, 99)), float((dd < 0.01).mean()), int(gd.sum())))
            dm = np.abs(SSEmean[gd] - Tc[gd])
            log("  %s/%s: unweighted-mean-vs-stored SSE_true  median|d|=%.4g  "
                "frac<0.01=%.3f" % (species, tis, float(np.nanmedian(dm)),
                float((dm < 0.01).mean())))
        n_t = int(testable.sum())
        log("  %s/%s: %d sites, %d slots (%d conds), %d testable"
            % (species, tis, len(keys), I.shape[1], len(cond_order), n_t))
        if n_t == 0:
            continue

        sub = np.where(testable)[0]
        Itm = np.where(mask, np.clip(np.rint(np.nan_to_num(Ptrue) * N), 0, N), 0.0)
        # Pass 1: the paper's test on the observed read counts.
        ro = dg.devas_test(Im[sub], Nm[sub], la, mode=args.mode)
        # Passes 2 and 3: the stored observed and predicted usage values turned
        # into pseudo-counts on the same coverage. Both carry no within-
        # condition variance, so their own dispersion would be far too small;
        # they are tested against the dispersion of the observed fit, i.e.
        # "is this trajectory shape significant given the data's noise level".
        rt = dg.devas_test(Itm[sub], Nm[sub], la, mode=args.mode,
                           phi=ro["dispersion"])
        rp = dg.devas_test(Ipm[sub], Nm[sub], la, mode=args.mode,
                           phi=ro["dispersion"])
        so, _, _ = dg.amplitude_and_pattern(la, SSE[sub], mask[sub], dpsi_min=args.dpsi_min)
        st, _, _ = dg.amplitude_and_pattern(la, np.where(mask, Ptrue, np.nan)[sub],
                                            mask[sub], dpsi_min=args.dpsi_min)
        sp, _, _ = dg.amplitude_and_pattern(la, np.where(mask, Ppred, np.nan)[sub],
                                            mask[sub], dpsi_min=args.dpsi_min)

        kk = keys[sub]
        rec = pd.DataFrame({
            "Species": species, "Tissue": tis,
            "Chromosome": [c for c in (kk // KEY_SHIFT)],
            "Position": kk % KEY_SHIFT,
            "n_slots": I.shape[1], "n_used": ro["n_used"],
            "mean_cov": np.nanmean(np.where(mask[sub], N[sub], np.nan), axis=1),
            "mean_sse_true": np.nanmean(np.where(mask[sub], Ptrue[sub], np.nan), axis=1),
            "mean_sse_pred": np.nanmean(np.where(mask[sub], Ppred[sub], np.nan), axis=1),
            "disp_obs": ro["dispersion"], "disp_pred_own": rp["dispersion_own"],
            "dpsi_obs": so["dpsi"], "dpsi_true": st["dpsi"], "dpsi_pred": sp["dpsi"],
            "ratio_obs": so["ratio"], "ratio_true": st["ratio"], "ratio_pred": sp["ratio"],
            "up_timing_obs": so["up_timing"], "down_timing_obs": so["down_timing"],
            "up_timing_true": st["up_timing"], "down_timing_true": st["down_timing"],
            "up_timing_pred": sp["up_timing"], "down_timing_pred": sp["down_timing"],
            "n_turns_obs": so["n_turns"], "n_turns_true": st["n_turns"],
            "n_turns_pred": sp["n_turns"],
            "pattern_obs": so["pattern"], "pattern_true": st["pattern"],
            "pattern_pred": sp["pattern"],
            "devAS_p_obs": ro["devAS_p"], "devAS_p_true": rt["devAS_p"],
            "devAS_p_pred": rp["devAS_p"],
        })
        for t in dg.TERMS:
            rec["padj_%s_obs" % t] = ro["padj_" + t]
            rec["padj_%s_true" % t] = rt["padj_" + t]
            rec["padj_%s_pred" % t] = rp["padj_" + t]
        for lab in ("obs", "true", "pred"):
            rec["devAS_" + lab] = (rec["devAS_p_" + lab]
                                   & (rec["dpsi_" + lab] > args.dpsi_min))
            rec["pattern_" + lab] = np.where(rec["devAS_" + lab],
                                             rec["pattern_" + lab], "none")
        out.append(rec)

        if args.verify and tis == tissues[0]:
            v = dg.verify_against_statsmodels(Im[sub], Nm[sub], la,
                                              n_check=args.verify, mode=args.mode)
            log("  verify vs statsmodels (%s/%s): %s" % (species, tis, v))
            out_v = os.path.join(args.out_dir, "devas_verify_%s.json" % species)
            with open(out_v, "w") as fh:
                json.dump({k: float(x) for k, x in v.items()}, fh, indent=1)

    if not out:
        return None
    res = pd.concat(out, ignore_index=True)
    inv = {v: k for k, v in chrom_codes.items()}
    res["Chromosome"] = res["Chromosome"].map(inv)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--age-table", default="stage_age_table.csv")
    ap.add_argument("--out-dir", default="devas")
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--tissues", nargs="*", default=None)
    ap.add_argument("--mode", choices=["sequential", "drop1"], default="sequential",
                    help="per-term quasi-likelihood F test: R's anova() order "
                         "(sequential, default) or drop1()")
    ap.add_argument("--dpsi-min", type=float, default=0.2)
    ap.add_argument("--min-total", type=int, default=10)
    ap.add_argument("--frac-covered", type=float, default=0.6)
    ap.add_argument("--n-intermediate", type=int, default=4)
    ap.add_argument("--verify", type=int, default=0,
                    help="refit N random sites with statsmodels and log the "
                         "largest discrepancy (first organ of each species)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ages = pd.read_csv(args.age_table)
    species = args.species or [s for s in SPECIES_DIRS
                               if os.path.isdir(os.path.join(PRED_DIR, s))]
    log("species:", species)
    summary = []
    for sp in species:
        t0 = time.time()
        res = run_species(sp, ages, args)
        if res is None:
            continue
        p = os.path.join(args.out_dir, "devas_calls_%s.parquet" % sp)
        res.to_parquet(p, index=False)
        log("%s -> %s  (%d rows, %.1f s)" % (sp, p, len(res), time.time() - t0))
        summary.append(res.groupby("Tissue").agg(
            n_sites=("Position", "size"),
            devAS_obs=("devAS_obs", "sum"),
            devAS_true=("devAS_true", "sum"),
            devAS_pred=("devAS_pred", "sum"),
            agree=("devAS_true", lambda x: 0)).drop(columns=["agree"]).assign(
                Species=sp).reset_index())
    if summary:
        s = pd.concat(summary, ignore_index=True)
        s.to_csv(os.path.join(args.out_dir, "devas_run_summary.csv"), index=False)
        log("\n" + s.to_string(index=False))


if __name__ == "__main__":
    main()
