#!/usr/bin/env python3
"""
analyze_prediction_clusters.py
------------------------------
Batch version of ``examples/notebooks/splice_prediction_clustering.ipynb``: evaluate the
predicted splice-usage trajectories that ``cluster_predictions.py`` annotated, for every
(species, organ) found under ``--preds-dir``.

For each (species, organ) with a ``..._prediction_clusters.parquet`` it:
  * loads the annotated table and adds the RMSE-adjusted shape agreement columns
    (``same_shape_adj`` / ``same_shape_ref_adj``: a categorical shape mismatch whose per-site
    RMSE is below ``--exc-floor`` is a gate-boundary artifact, not a real directional miss);
  * computes per-site RMSE and per-site Pearson r (true vs pred across observed timepoints)
    and pooled trajectory magnitude (``amplitude_r2`` / ``rmse`` / ``trajectory_r2``, the
    numpy ports of ``evaluate_splice.py``'s ``compute_trajectory_magnitude``);
  * saves per-combo diagnostics (shape concordance matrix; per-shape agreement + coverage +
    centroid-distance breakdown) into each combo's own output directory.

Across all combos it writes a summary CSV and three ``species × tissue`` heatmaps
(shape agreement, RMSE, and Pearson r — each broken down by observed shape).

All inputs live under ``--preds-dir`` (the prediction-cluster parquets carry the
obs/pred/ref shape+cluster columns; the raw ``usage_<species>.parquet`` supplies
SSE_true/SSE_pred), so no ``--ref-dir`` is needed.

Usage
-----
  # everything under a model's prediction root
  python scripts/analyze_prediction_clusters.py \
      --preds-dir ~/sds/sd17d003/Anamaria/alphagenome_genomicsxai/<MODEL>/preds_intersect_protein_coding

  # restrict species / organs
  python scripts/analyze_prediction_clusters.py --preds-dir .../preds_intersect_protein_coding \
      --species human mouse --organ Brain Testis
"""

import os
import sys
import glob
import argparse
import logging

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from alphagenome_pytorch.clustering import (
    DYNAMIC_SHAPE_ORDER as SHAPE_ORDER,
    DYNAMIC_SHAPE_COLORS as SHAPE_COLORS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


# ── numpy metric helpers ────────────────────────────────────────────────────────
# Ports of evaluate_splice.py's compute_trajectory_magnitude / _trajectory_excursion_np /
# _amplitude_r2 / _nanmedian_torch_style, kept standalone (as in the notebook) so this script
# has no heavy imports. torch's nanmedian convention (lower-of-two-middle for an even count) is
# reproduced deliberately: it can flip which side of --exc-floor a borderline trajectory falls
# on, so eligibility matches training/evaluation.

def _nanmedian_torch_style(windows):
    sorted_w = np.sort(windows, axis=-1)                 # NaNs sort to the end
    n_valid = np.sum(~np.isnan(windows), axis=-1)
    idx = np.clip((n_valid - 1) // 2, 0, windows.shape[-1] - 1)
    return np.take_along_axis(sorted_w, idx[..., None], axis=-1).squeeze(-1)


def _masked_median_filter(x, mask, win=5):
    N, T = x.shape
    win = max(1, min(win, T if T % 2 == 1 else T - 1))
    half = win // 2
    x_nan = np.where(mask, x, np.nan)
    x_pad = np.pad(x_nan, ((0, 0), (half, half)), mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, win, axis=1)
    n_valid = np.sum(~np.isnan(windows), axis=-1)
    filt = _nanmedian_torch_style(windows)
    return np.where(n_valid > 0, filt, 0.0)


def _trajectory_excursion(dt, mask, win=5):
    dt_filt = _masked_median_filter(dt, mask, win=win)
    denom = np.clip(mask.sum(axis=1), 1, None).astype(np.float64)
    mu_filt = (dt_filt * mask).sum(axis=1) / denom
    dt_filt_c = dt_filt - mu_filt[:, None]
    big = np.finfo(np.float64).max / 4
    exc_max = np.where(mask, dt_filt_c, -big).max(axis=1)
    exc_min = np.where(mask, dt_filt_c, big).min(axis=1)
    return np.maximum(exc_max, -exc_min)


def _amplitude_r2(true_amp, pred_amp):
    if true_amp.size < 2:
        return float("nan")
    ss_tot = np.sum((true_amp - true_amp.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    ss_res = np.sum((pred_amp - true_amp) ** 2)
    return float(1.0 - ss_res / ss_tot)


# ── path helpers (all under preds_dir) ──────────────────────────────────────────

def prefix(sp, tis, split):
    return f"{sp}_{tis.replace(' ', '_')}_{split}"


def preds_parquet_path(preds_dir, sp):
    return os.path.join(preds_dir, sp, f"usage_{sp}.parquet")


def combo_out_dir(preds_dir, sp, tis):
    return os.path.join(preds_dir, sp, "pred_gp_splice_usage", tis.replace(" ", "_"))


def pred_clusters_path(preds_dir, sp, tis, split):
    return os.path.join(combo_out_dir(preds_dir, sp, tis),
                        f"{prefix(sp, tis, split)}_prediction_clusters.parquet")


# ── per-site data + metrics ─────────────────────────────────────────────────────

def _traj_wide(preds_dir, sp, tis, sites=None):
    """Wide (site x timepoint) true/pred/mask matrices for one species/tissue from the raw
    predicted-usage parquet. `sites` restricts to a set of (chrom, pos) pairs."""
    p = pd.read_parquet(preds_parquet_path(preds_dir, sp),
                        columns=["Chromosome", "Position", "Tissue", "Timepoint",
                                 "SSE_true", "SSE_pred"])
    p = p[p["Tissue"] == tis].copy()
    p["Chromosome"] = p["Chromosome"].astype(str)
    if sites is not None:
        site_set = {(str(c), int(pos)) for c, pos in sites}
        p = p[[(c, pos) in site_set for c, pos in zip(p["Chromosome"], p["Position"])]]
    if p.empty:
        return np.empty((0,), dtype=object), np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0), bool)
    p["site"] = list(zip(p["Chromosome"], p["Position"]))
    true_wide = p.pivot_table(index="site", columns="Timepoint", values="SSE_true")
    pred_wide = p.pivot_table(index="site", columns="Timepoint", values="SSE_pred")
    mask = ~true_wide.isna()
    return (true_wide.index.to_numpy(),
            true_wide.fillna(0.0).to_numpy(), pred_wide.fillna(0.0).to_numpy(), mask.to_numpy())


def _site_metrics_all(preds_dir, sp, tis, sites=None):
    """Per-site RMSE and Pearson r in one pass (one `_traj_wide` read). RMSE is the literal
    residual (true - pred) at each observed timepoint; Pearson r is the correlation of the raw
    true/pred values across observed timepoints. Both are NaN for sites with < 2 observed
    timepoints (r additionally NaN when either series has zero variance)."""
    site_keys, true_mat, pred_mat, mask = _traj_wide(preds_dir, sp, tis, sites=sites)
    if len(site_keys) == 0:
        return pd.DataFrame({"Chromosome": pd.Series([], dtype=str),
                             "Position": pd.Series([], dtype=np.int64),
                             "site_rmse": pd.Series([], dtype=float),
                             "site_pearson_r": pd.Series([], dtype=float)})
    n_obs = mask.sum(axis=1)
    denom = np.clip(n_obs, 1, None).astype(np.float64)

    residual = np.where(mask, true_mat - pred_mat, 0.0)
    rmse = np.sqrt((residual ** 2 * mask).sum(axis=1) / denom)
    rmse[n_obs < 2] = np.nan

    mu_t = (true_mat * mask).sum(axis=1) / denom
    mu_p = (pred_mat * mask).sum(axis=1) / denom
    dt = np.where(mask, true_mat - mu_t[:, None], 0.0)
    dp = np.where(mask, pred_mat - mu_p[:, None], 0.0)
    cov = (dt * dp).sum(axis=1)
    std_t = np.sqrt((dt ** 2).sum(axis=1))
    std_p = np.sqrt((dp ** 2).sum(axis=1))
    r = np.full(len(site_keys), np.nan)
    valid = (n_obs >= 2) & (std_t * std_p > 1e-12)
    r[valid] = cov[valid] / (std_t[valid] * std_p[valid])

    out = pd.DataFrame({
        "Chromosome": [str(k[0]) for k in site_keys],
        "Position": [int(k[1]) for k in site_keys],
        "site_rmse": rmse,
        "site_pearson_r": r,
    })
    out["Chromosome"] = out["Chromosome"].astype(str)
    out["Position"] = out["Position"].astype(np.int64)
    return out


def load_annotated(preds_dir, sp, tis, split, exc_floor):
    """Prediction-cluster parquet + per-site RMSE/Pearson + RMSE-adjusted shape agreement."""
    d = pd.read_parquet(pred_clusters_path(preds_dir, sp, tis, split))
    d["Chromosome"] = d["Chromosome"].astype(str)
    m = _site_metrics_all(preds_dir, sp, tis, sites=list(zip(d["Chromosome"], d["Position"])))
    d = d.merge(m, on=["Chromosome", "Position"], how="left")
    rmse_ok = d["site_rmse"] < exc_floor
    d["same_shape_adj"] = d["same_shape"] | rmse_ok
    if "same_shape_ref" in d.columns:
        d["same_shape_ref_adj"] = d["same_shape_ref"] | rmse_ok
    return d


def compute_magnitude_metrics(preds_dir, sp, tis, sites, min_tp, exc_floor, median_win):
    """Pooled amplitude_r2 + rmse + trajectory_r2 over exc5-dynamic sites (numpy port of
    evaluate_splice.py's compute_trajectory_magnitude)."""
    site_keys, true_mat, pred_mat, mask = _traj_wide(preds_dir, sp, tis, sites=sites)
    if len(site_keys) == 0:
        return dict(n_sites_total=0, n_sites_eligible=0, rmse=float("nan"),
                    amplitude_r2=float("nan"), trajectory_r2=float("nan"))
    n_obs = mask.sum(axis=1)
    denom = np.clip(n_obs, 1, None).astype(np.float64)
    mu_t = (true_mat * mask).sum(axis=1) / denom
    mu_p = (pred_mat * mask).sum(axis=1) / denom
    dt = np.where(mask, true_mat - mu_t[:, None], 0.0)
    dp = np.where(mask, pred_mat - mu_p[:, None], 0.0)
    residual = np.where(mask, true_mat - pred_mat, 0.0)
    site_mse = (residual ** 2 * mask).sum(axis=1) / denom
    excursion_true = _trajectory_excursion(dt, mask, win=median_win)
    excursion_pred = _trajectory_excursion(dp, mask, win=median_win)
    eligible = (n_obs >= min_tp) & (excursion_true > exc_floor)

    rmse = float(np.sqrt(site_mse[eligible].mean())) if eligible.any() else float("nan")
    amp_r2 = _amplitude_r2(excursion_true[eligible], excursion_pred[eligible])

    eligible_pt_mask = mask & eligible[:, None]
    ss_tot = float(np.sum(dt[eligible_pt_mask] ** 2))
    traj_r2 = (float(1.0 - np.sum(residual[eligible_pt_mask] ** 2) / ss_tot)
               if ss_tot > 1e-12 else float("nan"))
    return dict(n_sites_total=int(len(site_keys)), n_sites_eligible=int(eligible.sum()),
                rmse=rmse, amplitude_r2=amp_r2, trajectory_r2=traj_r2)


# ── per-combo diagnostic plots (Sections 3 & 4) ─────────────────────────────────

def plot_shape_concordance(dsub, sp, tis, out_directory, split):
    shapes = [s for s in SHAPE_ORDER if s in set(dsub["obs_shape"]) | set(dsub["pred_shape"])]
    ct = pd.crosstab(dsub["obs_shape"], dsub["pred_shape"]).reindex(index=shapes, columns=shapes, fill_value=0)
    frac_rows = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(shapes) + 3), max(5, 0.55 * len(shapes) + 2)))
    im = ax.imshow(frac_rows.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(shapes))); ax.set_xticklabels(shapes, rotation=45, ha="right")
    ax.set_yticks(range(len(shapes))); ax.set_yticklabels(shapes)
    ax.set_xlabel("predicted shape"); ax.set_ylabel("observed shape")
    ax.set_title(f"{sp}/{tis}: shape concordance\n(color = row fraction, text = count; diagonal = agreement)")
    for i in range(len(shapes)):
        for j in range(len(shapes)):
            f = frac_rows.iloc[i, j]
            if ct.iloc[i, j]:
                ax.text(j, i, f"{int(ct.iloc[i, j]):,}\n{f:.2f}" if np.isfinite(f) else f"{int(ct.iloc[i, j]):,}",
                        ha="center", va="center", fontsize=7,
                        color="white" if (np.isfinite(f) and f > 0.5) else "#1a1a1a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fraction of observed-shape row", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_directory, f"{prefix(sp, tis, split)}_shape_concordance.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_agreement_breakdown(dsub, sp, tis, out_directory, split):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    by_shape = (dsub.groupby("obs_shape")["same_shape_adj"].agg(["mean", "size"])
                  .reindex(SHAPE_ORDER).dropna())
    ax = axes[0]
    ax.barh(range(len(by_shape)), by_shape["mean"],
            color=[SHAPE_COLORS.get(s, "#7f7f7f") for s in by_shape.index])
    ax.set_yticks(range(len(by_shape))); ax.set_yticklabels(by_shape.index)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("same shape (pred vs obs)")
    ax.set_title(f"agreement by shape\n(overall same_shape = {dsub['same_shape_adj'].mean():.2f})")
    for i, (mn, n) in enumerate(zip(by_shape["mean"], by_shape["size"])):
        ax.text(min(mn + 0.02, 0.98), i, f"{mn:.2f}  (n={int(n):,})", va="center", fontsize=8)

    ax = axes[1]
    if "pred_centroid_dist" in dsub.columns:
        for lab, sub, c in [("agree", dsub[dsub["same_shape_adj"]], "#2ca02c"),
                            ("disagree", dsub[~dsub["same_shape_adj"]], "#d62728")]:
            if len(sub):
                ax.hist(sub["pred_centroid_dist"], bins=40, alpha=0.5, label=lab, color=c, density=True)
        ax.set_xlabel("pred → nearest-centroid distance"); ax.set_ylabel("density")
        ax.set_title("centroid distance"); ax.legend(fontsize=8)
    else:
        ax.set_visible(False)

    ax = axes[2]
    if "n_obs" in dsub.columns:
        g = dsub.groupby("n_obs")["same_shape_adj"].mean()
        ax.plot(g.index, g.values, "o-", color="#1f77b4")
        ax.set_xlabel("# observed timepoints"); ax.set_ylabel("same_shape_adj"); ax.set_ylim(0, 1)
        ax.set_title("agreement vs coverage"); ax.grid(alpha=0.3)
    else:
        ax.set_visible(False)

    fig.suptitle(f"{sp}/{tis}", fontsize=11, y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(out_directory, f"{prefix(sp, tis, split)}_agreement_breakdown.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── cross-combo heatmaps (Section 9) ────────────────────────────────────────────

def _by_shape_matrix(per_combo, combos, value_fn):
    """(len(SHAPE_ORDER) x len(combos)) matrix of per-obs_shape values via `value_fn(d)`."""
    mat = np.full((len(SHAPE_ORDER), len(combos)), np.nan)
    for c, key in enumerate(combos):
        d = per_combo.get(key)
        if d is None or d.empty:
            continue
        by = value_fn(d).reindex(SHAPE_ORDER)
        mat[:, c] = by.to_numpy()
    return mat


def _species_separators(ax, combos):
    for c in range(1, len(combos)):
        if combos[c][0] != combos[c - 1][0]:
            ax.axvline(c - 0.5, color="white", lw=1.5)


def _heatmap(mat, combos, title, cbar_label, out_path, cmap, vmin, vmax, fmt, textrule):
    labels = [f"{sp}-{tis}" for sp, tis in combos]
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("#eeeeee")
    masked = np.ma.masked_invalid(mat)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(combos) + 2), 5))
    im = ax.imshow(masked, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(SHAPE_ORDER))); ax.set_yticklabels(SHAPE_ORDER, fontsize=12)
    ax.set_xticks(range(len(combos))); ax.set_xticklabels(labels, fontsize=11, rotation=90, ha="center")
    for r in range(len(SHAPE_ORDER)):
        for c in range(len(combos)):
            v = mat[r, c]
            if np.isfinite(v):
                ax.text(c, r, fmt(v), ha="center", va="center", fontsize=9, color=textrule(v))
    _species_separators(ax, combos)
    ax.set_xlabel("species / tissue", fontsize=13); ax.set_ylabel("observed shape", fontsize=13)
    ax.set_title(title, fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12); cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("heatmap -> %s", out_path)


def make_heatmaps(per_combo, combos, out_root):
    # agreement (higher better, [0,1])
    m = _by_shape_matrix(per_combo, combos, lambda d: d.groupby("obs_shape")["same_shape_adj"].mean())
    _heatmap(m, combos, "Agreement by shape across species × tissue",
             "trajectory shape agreement\n(pred vs obs)",
             os.path.join(out_root, "agreement_by_shape_species_tissue_heatmap.png"),
             "RdYlGn", 0, 1, lambda v: f"{v:.2f}",
             lambda v: "white" if (v < 0.35 or v > 0.65) else "#1a1a1a")

    # RMSE (lower better, [0, vmax])
    m = _by_shape_matrix(per_combo, combos, lambda d: d.groupby("obs_shape")["site_rmse"].mean())
    vmax = float(np.nanmax(m)) if np.isfinite(m).any() else 1.0
    _heatmap(m, combos, "RMSE by shape across species × tissue",
             "trajectory RMSE\n(pred vs obs, lower is better)",
             os.path.join(out_root, "rmse_by_shape_species_tissue_heatmap.png"),
             "RdYlGn_r", 0, vmax, lambda v: f"{v:.3f}",
             lambda v: "white" if (vmax > 0 and (v / vmax < 0.2 or v / vmax > 0.8)) else "#1a1a1a")

    # Pearson r (higher better, [-1, 1], diverging)
    m = _by_shape_matrix(per_combo, combos, lambda d: d.groupby("obs_shape")["site_pearson_r"].mean())
    _heatmap(m, combos, "Trajectory Pearson r by shape across species × tissue",
             "per-site trajectory Pearson r\n(pred vs obs, higher is better)",
             os.path.join(out_root, "pearson_by_shape_species_tissue_heatmap.png"),
             "RdYlGn", -1, 1, lambda v: f"{v:.2f}",
             lambda v: "white" if abs(v) > 0.6 else "#1a1a1a")


# ── discovery ───────────────────────────────────────────────────────────────────

def discover(preds_dir, species, organ, split):
    """(species, organ) pairs under preds_dir with a prediction-cluster parquet."""
    if species:
        sp_list = list(species)
    else:
        sp_list = sorted(d for d in os.listdir(preds_dir)
                         if os.path.isdir(os.path.join(preds_dir, d, "pred_gp_splice_usage")))
    combos = []
    for sp in sp_list:
        pg = os.path.join(preds_dir, sp, "pred_gp_splice_usage")
        if not os.path.isdir(pg):
            log.warning("skip %s: no pred_gp_splice_usage/ under %s", sp, preds_dir)
            continue
        if not os.path.exists(preds_parquet_path(preds_dir, sp)):
            log.warning("skip %s: no usage_%s.parquet (needed for per-site metrics)", sp, sp)
            continue
        if organ:
            organs = list(organ)
        else:
            organs = sorted(t for t in os.listdir(pg) if os.path.isdir(os.path.join(pg, t)))
        for tis in organs:
            if os.path.exists(pred_clusters_path(preds_dir, sp, tis, split)):
                combos.append((sp, tis))
            elif organ:  # only warn when the user explicitly asked for this organ
                log.warning("skip %s/%s: no prediction-cluster parquet", sp, tis)
    return combos


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds-dir", required=True,
                   help="Prediction root: <preds-dir>/<species>/usage_<species>.parquet and "
                        "<species>/pred_gp_splice_usage/<organ>/*_prediction_clusters.parquet.")
    p.add_argument("--species", nargs="+", default=None,
                   help="Species to analyze (default: all found under --preds-dir).")
    p.add_argument("--organ", nargs="+", default=None,
                   help="Organs/tissues to analyze (default: all found per species).")
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--output", default=None,
                   help="Directory for the summary CSV + heatmaps (default: --preds-dir).")
    p.add_argument("--no-per-combo-plots", action="store_true",
                   help="Skip the per-combo concordance / agreement-breakdown figures.")
    # eligibility / RMSE-adjust knobs (defaults match the notebook)
    p.add_argument("--min-tp", type=int, default=3)
    p.add_argument("--exc-floor", type=float, default=0.10)
    p.add_argument("--exc-median-win", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.preds_dir):
        sys.exit(f"--preds-dir not found: {args.preds_dir}")
    out_root = args.output or args.preds_dir
    os.makedirs(out_root, exist_ok=True)

    combos = discover(args.preds_dir, args.species, args.organ, args.split)
    if not combos:
        sys.exit("No (species, organ) pairs with prediction-cluster parquets found.")
    log.info("Analyzing %d (species, organ) combo(s): %s",
             len(combos), ", ".join(f"{s}/{t}" for s, t in combos))

    per_combo, rows = {}, []
    for sp, tis in combos:
        try:
            d = load_annotated(args.preds_dir, sp, tis, args.split, args.exc_floor)
        except Exception as e:
            log.exception("%s/%s: load failed (%s)", sp, tis, e)
            continue
        per_combo[(sp, tis)] = d
        odir = combo_out_dir(args.preds_dir, sp, tis)

        if not args.no_per_combo_plots and len(d):
            plot_shape_concordance(d, sp, tis, odir, args.split)
            plot_agreement_breakdown(d, sp, tis, odir, args.split)

        mag = compute_magnitude_metrics(
            args.preds_dir, sp, tis, sites=list(zip(d["Chromosome"], d["Position"])),
            min_tp=args.min_tp, exc_floor=args.exc_floor, median_win=args.exc_median_win)

        row = dict(
            species=sp, tissue=tis, n_sites=len(d),
            shape_pred_vs_obs=round(d["same_shape_adj"].mean(), 3) if len(d) else np.nan,
            shape_obs_vs_ref=round((d["obs_shape"] == d["ref_shape"]).mean(), 3)
                if len(d) and "ref_shape" in d.columns else np.nan,
            cluster_pred_vs_obs=round(d["same_cluster"].mean(), 3)
                if len(d) and "same_cluster" in d.columns else np.nan,
            median_site_pearson_r=round(float(d["site_pearson_r"].median()), 3) if len(d) else np.nan,
            mean_site_rmse=round(float(d["site_rmse"].mean()), 3) if len(d) else np.nan,
            **mag,
        )
        if "same_shape_ref_adj" in d.columns and len(d):
            row["shape_pred_vs_ref"] = round(d["same_shape_ref_adj"].mean(), 3)
        if "same_shape_site" in d.columns and len(d):        # per-site shape (newer outputs)
            row["shape_site_pred_vs_obs"] = round(d["same_shape_site"].mean(), 3)
        rows.append(row)
        log.info("  %s/%s: %s sites | shape(pred-vs-obs)=%.3f | traj_r2=%.3f | median r=%.3f",
                 sp, tis, f"{len(d):,}", row["shape_pred_vs_obs"], mag["trajectory_r2"],
                 row["median_site_pearson_r"])

    summary = pd.DataFrame(rows)
    csv_path = os.path.join(out_root, "prediction_analysis_summary.csv")
    summary.to_csv(csv_path, index=False)
    log.info("summary -> %s", csv_path)

    make_heatmaps(per_combo, combos, out_root)
    log.info("=== Done: %d combos ===", len(per_combo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
