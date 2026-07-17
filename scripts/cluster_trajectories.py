#!/usr/bin/env python3
"""
cluster_trajectories.py
-----------------------
Cluster developmental splice site SSE trajectories using Gaussian process
regression + Ward-linkage hierarchical clustering, then annotate each cluster
with a shape label (up_early, down_late, flat_high, …).

Outputs
-------
  <prefix>_clustering_metadata.parquet   — per-trajectory cluster + shape labels + GP stats
  <prefix>_gp_features.npy               — GP posterior means (n_trajectories × 15)
  <prefix>_linkage.npy                   — Ward linkage matrix (for further analysis)
  <prefix>_gp_diagnostics.parquet        — per-trajectory GP hyperparameter diagnostics
  <prefix>_heatmap.png                   — (optional) heatmap
  <prefix>_profiles.png                  — (optional) cluster mean trajectory grid

Usage examples
--------------
  # Human, all tissues, 30 clusters, 16 parallel workers
  python cluster_trajectories.py \\
      --species human --n-clusters 30 \\
      --n-jobs 16 --output results/ --save-plots

  # All species, Brain only, auto-select k, save plots
  python cluster_trajectories.py \\
      --species all --tissue Brain \\
      --auto-k --n-jobs -1 --output results/ --save-plots

  # Resume: skip GP (reuse saved features), just re-cluster with different k
  python cluster_trajectories.py \\
      --species human --tissue Brain --n-clusters 50 \\
      --load-features results/human_brain_gp_features.npy \\
      --load-sites    results/human_brain_sites.parquet \\
      --output results/ --save-plots
"""

import os
import sys
import glob
import time
import logging
import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    sys.exit("scikit-learn is required: pip install scikit-learn")

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Shape taxonomy ────────────────────────────────────────────────────────────

SHAPE_COLORS = {
    # Upward: dark = abrupt early, medium = abrupt late, light = gradual
    "up_early":      "#1a7a1a",   # dark green
    "up_late":       "#5dac5d",   # medium green
    "up_mid":        "#b3e0b3",   # light green  (gradual)
    # Downward
    "down_early":    "#a01010",   # dark red
    "down_late":     "#e05050",   # medium red
    "down_mid":      "#f0b0b0",   # light pink   (gradual)
    # Biphasic
    "up-down":       "#1f77b4",   # blue
    "down-up":       "#ff7f0e",   # orange
    # Flat
    "flat_high":     "#9467bd",
    "flat_mid_high": "#d8c8e8",
    "flat_mid":      "#e0d4eb",
    "flat_mid_low":  "#c6b2da",
    "flat_low":      "#c5b0d5",
    # No clear pattern
    "complex":       "#7f7f7f",
}

SHAPE_ORDER = [
    "up_early",   "up_late",   "up_mid",
    "down_early", "down_late", "down_mid",
    "up-down", "down-up",
    "flat_high", "flat_mid_high", "flat_mid", "flat_mid_low", "flat_low",
    "complex",
]

T_GRID = np.arange(1, 16, dtype=float)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_trajectories(parquet_path, species=None, tissue=None,
                          min_timepoints=5, min_reads=1,
                          max_sites=None, random_seed=42):
    """
    Load splice-site usage data, aggregate to per-(site, [tissue], [species])
    × timepoint matrix, and filter by data availability.

    Parameters
    ----------
    parquet_path : str
        Direct path or template with ``{species}`` placeholder.
    species : str or None
        If None (or 'all'), all matching species files are loaded and Species
        is retained as a trajectory-identity dimension.
    tissue : str or None
        If None, all tissues are kept and Tissue is a trajectory dimension.
    """
    if "{species}" in parquet_path:
        if species is None:
            paths = sorted(glob.glob(parquet_path.replace("{species}", "*")))
            if not paths:
                raise FileNotFoundError(
                    f"No parquet files match template: {parquet_path!r}")
            log.info("Loading %d species files: %s",
                     len(paths), [Path(p).stem for p in paths])
            df = pd.concat([pd.read_parquet(p) for p in paths],
                           ignore_index=True)
        else:
            df = pd.read_parquet(parquet_path.format(species=species))
    else:
        df = pd.read_parquet(parquet_path)

    if df.index.names[0] is not None and df.index.names[0] != 0:
        df = df.reset_index()

    if species is not None and "Species" in df.columns:
        df = df[df["Species"] == species]
    if tissue is not None and "Tissue" in df.columns:
        df = df[df["Tissue"] == tissue]

    if df.empty:
        raise ValueError(f"No data for species={species!r}, tissue={tissue!r}")

    if "Reads" not in df.columns and "Alpha" in df.columns and "Beta" in df.columns:
        df = df.copy()
        df["Reads"] = df["Alpha"] + df["Beta"]
    if min_reads > 0 and "Reads" in df.columns:
        df = df[df["Reads"] >= min_reads]

    group_cols = ["Chromosome", "Position", "Strand"]
    if species is None and "Species" in df.columns:
        group_cols = ["Species"] + group_cols
    if tissue is None and "Tissue" in df.columns:
        group_cols = group_cols + ["Tissue"]

    all_tps = list(range(1, 16))
    reads_col = "Reads" if "Reads" in df.columns else "SSE"

    agg = (
        df.groupby(group_cols + ["Timepoint"])
          .agg(SSE_mean=("SSE", "mean"), Reads_sum=(reads_col, "sum"))
          .reset_index()
    )

    sse_wide = (
        agg.pivot_table(index=group_cols, columns="Timepoint", values="SSE_mean")
           .reindex(columns=all_tps)
    )
    reads_wide = (
        agg.pivot_table(index=group_cols, columns="Timepoint",
                        values="Reads_sum", fill_value=0)
           .reindex(columns=all_tps, fill_value=0)
           .reindex(sse_wide.index, fill_value=0)
    )

    keep = (reads_wide.values > 0).sum(axis=1) >= min_timepoints
    sse_wide   = sse_wide[keep]
    reads_wide = reads_wide[keep]

    extra_dims = [c for c in group_cols
                  if c not in ("Chromosome", "Position", "Strand")]
    sp_label  = species or "all species"
    tis_label = tissue  or "all tissues"
    dim_note  = f"  [dims: {', '.join(extra_dims)}]" if extra_dims else ""
    log.info("%s/%s: %s trajectories with >=%d detected timepoints "
             "(from %s total)%s",
             sp_label, tis_label,
             f"{keep.sum():,}", min_timepoints, f"{len(keep):,}", dim_note)

    if max_sites is not None and len(sse_wide) > max_sites:
        rng = np.random.default_rng(random_seed)
        idx = np.sort(rng.choice(len(sse_wide), size=max_sites, replace=False))
        sse_wide   = sse_wide.iloc[idx]
        reads_wide = reads_wide.iloc[idx]
        log.info("Subsampled to %s trajectories", f"{max_sites:,}")

    sites = sse_wide.reset_index()[group_cols]
    return sites, sse_wide, reads_wide


# ── Split filtering ──────────────────────────────────────────────────────────

def filter_to_split(sites, data_config_path, species=None, split="test"):
    """
    Return a boolean mask selecting sites that fall inside the genomic windows
    of a given data split (test / val / train).

    The split boundaries come from BED files referenced in the JSON config
    (key: ``<split>_bed``).  Each BED row is a ~131 kb window; a site is
    'in the split' if its (Chromosome, Position) is contained in any window.

    Parameters
    ----------
    sites : pd.DataFrame
        Must contain 'Chromosome' and 'Position' columns.
        When ``species`` is None it must also contain a 'Species' column.
    data_config_path : str
        Path to the JSON config file.
    species : str or None
        Single species name (e.g. 'human'), or None to handle all species
        listed in sites['Species'] individually.
    split : str
        One of 'test', 'val', 'train'.

    Returns
    -------
    mask : np.ndarray, shape (len(sites),), dtype bool
    """
    import json

    with open(data_config_path) as _f:
        cfg = json.load(_f)

    bed_key = f"{split}_bed"

    def _load_intervals(sp):
        """Return dict {chrom: (sorted_starts, ends)} for one species."""
        path = os.path.expanduser(cfg[sp][bed_key])
        bed  = pd.read_csv(path, sep="\t", header=None,
                           usecols=[0, 1, 2],
                           names=["chrom", "start", "end"],
                           dtype={"chrom": str, "start": int, "end": int})
        ivs = {}
        for chrom, grp in bed.groupby("chrom"):
            g = grp.sort_values("start")
            ivs[chrom] = (g["start"].values, g["end"].values)
        return ivs

    def _apply_intervals(sub_sites, ivs):
        """Vectorised interval containment check."""
        chroms = sub_sites["Chromosome"].astype(str).values
        posns  = sub_sites["Position"].astype(int).values
        result = np.zeros(len(sub_sites), dtype=bool)
        for chrom in np.unique(chroms):
            if chrom not in ivs:
                continue
            starts, ends = ivs[chrom]
            sel = chroms == chrom
            pos = posns[sel]
            idx = np.searchsorted(starts, pos, side="right") - 1
            ok  = (idx >= 0) & (idx < len(ends))
            hit = np.zeros(len(pos), dtype=bool)
            hit[ok] = ends[idx[ok]] >= pos[ok]
            result[sel] = hit
        return result

    mask = np.zeros(len(sites), dtype=bool)

    if species is not None:
        if species not in cfg:
            raise KeyError(f"Species {species!r} not found in {data_config_path}")
        ivs  = _load_intervals(species)
        mask = _apply_intervals(sites, ivs)
    else:
        for sp in sites["Species"].unique():
            if sp not in cfg:
                log.warning("Species %r not in config — skipping split filter", sp)
                continue
            sp_sel = (sites["Species"] == sp).values
            ivs    = _load_intervals(sp)
            mask[sp_sel] = _apply_intervals(sites[sp_sel], ivs)

    n   = int(mask.sum())
    pct = 100.0 * n / max(len(mask), 1)
    log.info("Split '%s': %s / %s trajectories (%.1f%%) inside %s windows",
             split, f"{n:,}", f"{len(mask):,}", pct, split)
    return mask


# ── GP smoothing ──────────────────────────────────────────────────────────────

def _make_kernel(length_scale=0.20, noise_level=0.05):
    return (
        C(1.0, (1e-3, 5.0))
        * RBF(length_scale=length_scale, length_scale_bounds=(0.05, 2.0))
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-5, 0.5))
    )


def _smooth_one(sse_row, reads_row, col_tps, t_grid,
                length_scale, noise_level, n_restarts,
                bound_tol=0.01):
    """
    Smooth a single trajectory; falls back to linear interpolation.

    Returns
    -------
    y_mean : (n_grid,) GP posterior mean (or interpolated fallback)
    y_std  : (n_grid,) GP posterior std (zeros for fallback)
    lml    : float     log marginal likelihood (nan for fallback)
    diag   : dict       fitted hyperparameters + boundary flags, for
                        diagnosing ConvergenceWarning-type issues.
    """
    n_grid = len(t_grid)
    kernel = _make_kernel(length_scale, noise_level)
    ls_lo, ls_hi = kernel.k1.k2.length_scale_bounds
    nl_lo, nl_hi = kernel.k2.noise_level_bounds

    empty_diag = dict(
        fit_length_scale=np.nan, fit_noise_level=np.nan,
        length_scale_bound_lo=ls_lo, length_scale_bound_hi=ls_hi,
        noise_level_bound_lo=nl_lo, noise_level_bound_hi=nl_hi,
        hit_length_scale_lo=False, hit_length_scale_hi=False,
        hit_noise_level_lo=False, hit_noise_level_hi=False,
        n_observed=int((reads_row > 0).sum()), fallback=True,
    )

    detected = reads_row > 0
    if detected.sum() < 2:
        return np.full(n_grid, np.nan), np.zeros(n_grid), np.nan, empty_diag

    tps  = col_tps[detected]
    vals = np.nan_to_num(sse_row[detected], nan=0.0)

    t_min, t_max = float(t_grid[0]), float(t_grid[-1])
    norm = lambda t: (np.asarray(t, float) - t_min) / (t_max - t_min)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=ConvergenceWarning)
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restarts,
                normalize_y=True, alpha=1e-8, random_state=42,
            )
            gpr.fit(norm(tps).reshape(-1, 1), vals)
            saw_convergence_warning = any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )

        y_mean, y_std = gpr.predict(norm(t_grid).reshape(-1, 1), return_std=True)
        lml = gpr.log_marginal_likelihood_value_

        fit_ls = float(gpr.kernel_.k1.k2.length_scale)
        fit_nl = float(gpr.kernel_.k2.noise_level)

        def _near(v, lo, hi, tol=bound_tol):
            span = hi - lo
            return (v - lo) <= tol * span, (hi - v) <= tol * span

        ls_hit_lo, ls_hit_hi = _near(fit_ls, ls_lo, ls_hi)
        nl_hit_lo, nl_hit_hi = _near(fit_nl, nl_lo, nl_hi)

        diag = dict(
            fit_length_scale=fit_ls, fit_noise_level=fit_nl,
            length_scale_bound_lo=ls_lo, length_scale_bound_hi=ls_hi,
            noise_level_bound_lo=nl_lo, noise_level_bound_hi=nl_hi,
            hit_length_scale_lo=bool(ls_hit_lo), hit_length_scale_hi=bool(ls_hit_hi),
            hit_noise_level_lo=bool(nl_hit_lo), hit_noise_level_hi=bool(nl_hit_hi),
            n_observed=int(detected.sum()), fallback=False,
            convergence_warning=bool(saw_convergence_warning),
        )
    except Exception:
        f = interp1d(tps, vals, kind="linear", bounds_error=False,
                     fill_value=(vals[0], vals[-1]))
        y_mean = f(t_grid)
        y_std  = np.zeros(n_grid)
        lml    = np.nan
        diag   = empty_diag

    return np.clip(y_mean, 0.0, 1.0), y_std, lml, diag


def smooth_all_trajectories(sse_wide, reads_wide, t_grid,
                             length_scale=0.20, noise_level=0.05,
                             n_restarts=2, n_jobs=-1, bound_tol=0.01):
    """
    Fit a GP to every trajectory in sse_wide (parallelised).

    Returns
    -------
    features   : (n, 15) float32   GP posterior means at t_grid
    stds       : (n, 15) float32   GP posterior stds
    lmls       : (n,)    float64   log marginal likelihoods
    diagnostics: pd.DataFrame      per-trajectory fitted hyperparameters and
                                   boundary flags (see _smooth_one docstring)
    """
    sse_arr   = sse_wide.values.astype(float)
    reads_arr = reads_wide.values.astype(float)
    col_tps   = np.array([float(c) for c in sse_wide.columns])
    n_sites   = sse_arr.shape[0]

    log.info("Fitting GPs to %s trajectories  (n_jobs=%d, n_restarts=%d) ...",
             f"{n_sites:,}", n_jobs, n_restarts)
    t0 = time.time()

    kwargs = dict(col_tps=col_tps, t_grid=t_grid,
                  length_scale=length_scale, noise_level=noise_level,
                  n_restarts=n_restarts, bound_tol=bound_tol)

    if HAS_JOBLIB and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_smooth_one)(sse_arr[i], reads_arr[i], **kwargs)
            for i in tqdm(range(n_sites), desc="GP", unit="site")
        )
    else:
        results = [
            _smooth_one(sse_arr[i], reads_arr[i], **kwargs)
            for i in tqdm(range(n_sites), desc="GP", unit="site")
        ]

    features = np.array([r[0] for r in results], dtype=np.float32)
    stds     = np.array([r[1] for r in results], dtype=np.float32)
    lmls     = np.array([r[2] for r in results], dtype=np.float64)
    diagnostics = pd.DataFrame([r[3] for r in results])
    diagnostics.insert(0, "site_index", np.arange(n_sites))

    elapsed = time.time() - t0
    n_nan   = np.isnan(features).any(axis=1).sum()
    log.info("GP done in %.0fs.  Invalid: %s/%s", elapsed, f"{n_nan:,}", f"{n_sites:,}")

    # ── Boundary-hugging summary ───────────────────────────────────────────
    fitted = diagnostics[~diagnostics["fallback"]]
    if len(fitted):
        ls_lo_n = int(fitted["hit_length_scale_lo"].sum())
        ls_hi_n = int(fitted["hit_length_scale_hi"].sum())
        nl_lo_n = int(fitted["hit_noise_level_lo"].sum())
        nl_hi_n = int(fitted["hit_noise_level_hi"].sum())
        cw_n    = int(fitted.get("convergence_warning", pd.Series(dtype=bool)).sum())
        log.info(
            "Hyperparameter boundary check (tol=%.0f%% of range, n=%s fitted):",
            bound_tol * 100, f"{len(fitted):,}",
        )
        log.info("  length_scale at lower bound (%.3g): %s trajectories",
                 fitted["length_scale_bound_lo"].iloc[0], f"{ls_lo_n:,}")
        log.info("  length_scale at upper bound (%.3g): %s trajectories",
                 fitted["length_scale_bound_hi"].iloc[0], f"{ls_hi_n:,}")
        log.info("  noise_level  at lower bound (%.3g): %s trajectories",
                 fitted["noise_level_bound_lo"].iloc[0], f"{nl_lo_n:,}")
        log.info("  noise_level  at upper bound (%.3g): %s trajectories",
                 fitted["noise_level_bound_hi"].iloc[0], f"{nl_hi_n:,}")
        log.info("  sklearn ConvergenceWarning raised: %s trajectories", f"{cw_n:,}")
        if ls_lo_n:
            log.warning(
                "%s trajectories hit the length_scale lower bound (%.3g) — "
                "the GP wants to smooth less than allowed. Consider raising "
                "--gp-noise-level / its upper bound, or lowering "
                "--gp-length-scale's floor only if short-scale wiggles are "
                "real biological signal rather than noise.",
                f"{ls_lo_n:,}", fitted["length_scale_bound_lo"].iloc[0],
            )

    return features, stds, lmls, diagnostics


# ── Shape classification ───────────────────────────────────────────────────────

def classify_cluster_shape(mean_traj,
                             amplitude_threshold=0.08,
                             smooth_window=3,
                             monotone_frac=0.65,
                             peak_window=(0.25, 0.75),
                             reversal_fraction=0.30,
                             min_net_change=0.12,
                             knee_frac=0.65,
                             flat_high=0.80,
                             flat_low=0.20,
                             level_window=3):
    """
    Classify a cluster's mean trajectory into one of 14 shape labels.

    Decision tree
    -------------
    1. amplitude < ``amplitude_threshold`` → flat_*
    2. frac_up >= ``monotone_frac`` AND |net| >= ``min_net_change`` → up_{early/late/mid}
    3. frac_down >= ``monotone_frac`` AND |net| >= ``min_net_change`` → down_{early/late/mid}
    4. Interior peak with both legs >= ``reversal_fraction * amplitude`` → up-down
    5. Interior valley (same) → down-up
    6. |net| >= ``min_net_change`` (net-direction fallback) → up_*/down_*
    7. complex

    Timing suffix (early / late / mid)
    -----------------------------------
    Determined by the ``_knee()`` helper: if >= ``knee_frac`` of the net change
    is concentrated in the first half of development → 'early'; second half →
    'late'; otherwise → 'mid' (gradual, no clear knee).
    """
    y = uniform_filter1d(np.asarray(mean_traj, float),
                         size=smooth_window, mode="nearest")
    n   = len(y)
    win = max(1, min(level_window, n // 4))

    amplitude = float(np.max(y) - np.min(y))

    # ── Flat ────────────────────────────────────────────────────────────────
    if amplitude < amplitude_threshold:
        mv = float(np.mean(y))
        if mv >= flat_high: return "flat_high"
        if mv <= flat_low:  return "flat_low"
        if mv <= 0.4:       return "flat_mid_low"
        if mv >= 0.6:       return "flat_mid_high"
        return "flat_mid"

    # ── Timing helper ─────────────────────────────────────────────────────────
    def _knee():
        """'early' / 'late' / 'mid' based on where the change is concentrated."""
        total = abs(float(y[-1]) - float(y[0]))
        if total < 1e-6:
            return "mid"
        mid = n // 2
        first_half  = abs(float(y[mid]) - float(y[0]))
        second_half = abs(float(y[-1]) - float(y[mid]))
        if first_half  / total >= knee_frac: return "early"
        if second_half / total >= knee_frac: return "late"
        return "mid"

    # ── Net change ────────────────────────────────────────────────────────────
    start_val  = float(np.mean(y[:win]))
    end_val    = float(np.mean(y[-win:]))
    net_change = end_val - start_val
    abs_net    = abs(net_change)

    diffs     = np.diff(y)
    frac_up   = float((diffs > 0).mean())
    frac_down = float((diffs < 0).mean())

    # ── Monotone up ─────────────────────────────────────────────────────────
    if frac_up >= monotone_frac and abs_net >= min_net_change:
        return f"up_{_knee()}"

    # ── Monotone down ─────────────────────────────────────────────────────────
    if frac_down >= monotone_frac and abs_net >= min_net_change:
        return f"down_{_knee()}"

    # ── Biphasic: peak in the middle ──────────────────────────────────────────
    lo = int(np.floor(peak_window[0] * n))
    hi = int(np.ceil(peak_window[1] * n))
    argmax_idx = int(np.argmax(y))
    argmin_idx = int(np.argmin(y))
    min_leg    = reversal_fraction * amplitude

    if lo <= argmax_idx <= hi:
        rise_before = float(y[argmax_idx] - y[0])
        fall_after  = float(y[argmax_idx] - y[-1])
        if rise_before >= min_leg and fall_after >= min_leg:
            return "up-down"

    # ── Biphasic: valley in the middle ────────────────────────────────────────
    if lo <= argmin_idx <= hi:
        fall_before = float(y[0]  - y[argmin_idx])
        rise_after  = float(y[-1] - y[argmin_idx])
        if fall_before >= min_leg and rise_after >= min_leg:
            return "down-up"

    # ── Net-direction fallback ────────────────────────────────────────────────
    if abs_net >= min_net_change:
        return f'{"up" if net_change > 0 else "down"}_{_knee()}'

    return "complex"


# ── Automatic k selection ─────────────────────────────────────────────────────

def select_k_gap(Z, k_min=5, k_max=80):
    """
    Choose k by finding the largest gap between successive merge distances.

    This is a simple but robust heuristic: the natural grouping of the data
    tends to produce a large drop in merge distance at the 'right' k.
    """
    k_max = min(k_max, len(Z))
    # merge_dists[-k_max:][::-1][i] = merge distance when going from k=i+2 → i+1
    last_dists = Z[:, 2][-(k_max - 1):][::-1]
    gaps = np.diff(last_dists)
    # +1 because diff shifts index; +k_min to align with actual k values
    best_k = int(np.argmax(gaps)) + 1 + k_min
    return max(k_min, min(k_max, best_k))


# ── Plotting ───────────────────────────────────────────────────────────────────

def save_plots(features_v, cluster_labels, cluster_shapes,
               n_clusters, out_dir, prefix, label, random_seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    cc = [plt.cm.tab20(k / n_clusters) for k in range(n_clusters)]
    rng = np.random.default_rng(random_seed)

    # ── Heatmap ──────────────────────────────────────────────────────────────
    n_per = [(cluster_labels == k).sum() for k in range(1, n_clusters + 1)]
    order = []
    for k in range(1, n_clusters + 1):
        idx_k = np.where(cluster_labels == k)[0]
        ctr   = features_v[cluster_labels == k].mean(axis=0)
        d     = np.linalg.norm(features_v[idx_k] - ctr, axis=1)
        order.extend(idx_k[np.argsort(d)].tolist())

    order  = np.array(order)
    fsort  = features_v[order]
    GAP    = max(3, int(len(fsort) * 0.004))
    blocks = []
    bk     = []
    s      = 0
    for k, nc in enumerate(n_per, 1):
        blocks.append(fsort[s:s + nc])
        bk.append(k)
        if k != n_clusters:
            blocks.append(np.full((GAP, fsort.shape[1]), np.nan))
            bk.append(None)
        s += nc
    fsplit = np.vstack(blocks)

    row_ranges = []
    rc = 0
    for blk, k in zip(blocks, bk):
        nr = blk.shape[0]
        if k is not None:
            row_ranges.append((k, rc, rc + nr))
        rc += nr

    fig = plt.figure(figsize=(10.5, max(5, len(fsort) * 0.005)))
    gs  = gridspec.GridSpec(2, 3, width_ratios=[0.05, 0.02, 0.93],
                            height_ratios=[0.05, 0.95], hspace=0.15, wspace=0.02)
    ax_cb  = fig.add_subplot(gs[0, 2])
    ax_lbl = fig.add_subplot(gs[1, 0])
    ax_str = fig.add_subplot(gs[1, 1])
    ax_ht  = fig.add_subplot(gs[1, 2])

    masked = np.ma.masked_invalid(fsplit)
    cm_bad = plt.get_cmap("RdBu_r").copy()
    cm_bad.set_bad("white")
    im = ax_ht.imshow(masked, aspect="auto", cmap=cm_bad,
                      vmin=0, vmax=1, interpolation="nearest")
    ax_ht.set_xlabel("Developmental Timepoint")
    ax_ht.set_xticks(range(len(T_GRID)))
    ax_ht.set_xticklabels([str(int(t)) for t in T_GRID], fontsize=8)
    ax_ht.set_yticks([])
    ax_ht.set_title(f"GP SSE  |  {label}  |  {len(fsort):,} sites  |  k={n_clusters}",
                    fontsize=10)

    ax_str.set_xlim(0, 1); ax_str.set_ylim(len(fsplit), 0); ax_str.axis("off")
    ax_lbl.set_xlim(0, 1); ax_lbl.set_ylim(len(fsplit), 0); ax_lbl.axis("off")
    for k, y0, y1 in row_ranges:
        ax_str.add_patch(plt.Rectangle((0, y0), 1, y1 - y0,
                                        color=cc[k-1], ec="none"))
        ax_lbl.text(1.0, (y0+y1)/2, f"C{k}", ha="right", va="center",
                    fontsize=8, fontweight="bold", color=cc[k-1])

    cb = plt.colorbar(im, cax=ax_cb, orientation="horizontal")
    cb.set_label("SSE", fontsize=8)
    ax_cb.xaxis.set_ticks_position("top")
    ax_cb.xaxis.set_label_position("top")

    out_hm = os.path.join(out_dir, f"{prefix}_heatmap.png")
    fig.savefig(out_hm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Heatmap → %s", out_hm)

    # ── Cluster profiles (ordered by shape, then descending mean SSE) ─────────
    _shape_rank = {s: i for i, s in enumerate(SHAPE_ORDER)}
    _sorted_clusters = sorted(
        range(1, n_clusters + 1),
        key=lambda k: (
            _shape_rank.get(cluster_shapes[k], len(SHAPE_ORDER)),
            -float(features_v[cluster_labels == k].mean()),
        ),
    )

    n_cols = min(5, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig2, axes2 = plt.subplots(n_rows, n_cols,
                                figsize=(n_cols * 3, n_rows * 2.5),
                                squeeze=False, sharey=True)
    for plot_pos, k in enumerate(_sorted_clusters):
        r, c   = plot_pos // n_cols, plot_pos % n_cols
        ax     = axes2[r, c]
        mask_k = cluster_labels == k
        mean_k = features_v[mask_k].mean(axis=0)
        std_k  = features_v[mask_k].std(axis=0)
        shape_k = cluster_shapes[k]
        color  = SHAPE_COLORS.get(shape_k, "#7f7f7f")

        n_draw = min(80, mask_k.sum())
        for i in rng.choice(np.where(mask_k)[0], size=n_draw, replace=False):
            ax.plot(T_GRID, features_v[i], color=color, alpha=0.08, lw=0.5)
        ax.fill_between(T_GRID, mean_k - std_k, mean_k + std_k, alpha=0.3, color=color)
        ax.plot(T_GRID, mean_k, "-", color=color, lw=2.5)

        badge = shape_k.replace("_", "\n")
        ax.text(0.97, 0.97, badge, transform=ax.transAxes,
                ha="right", va="top", fontsize=6, fontweight="bold",
                color="white", linespacing=1.1,
                bbox=dict(facecolor=color, edgecolor="none",
                          boxstyle="round,pad=0.25", alpha=0.9))
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(T_GRID[0], T_GRID[-1])
        ax.set_title(f"C{k}  n={mask_k.sum():,}", fontsize=8, fontweight="bold")
        if c == 0:
            ax.set_ylabel("SSE", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for k in range(n_clusters, n_rows * n_cols):
        axes2[k // n_cols, k % n_cols].set_visible(False)

    fig2.suptitle(f"Cluster shapes (ordered by shape)  |  {label}", fontsize=11, y=1.01)
    plt.tight_layout()
    out_pr = os.path.join(out_dir, f"{prefix}_profiles.png")
    fig2.savefig(out_pr, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log.info("Profiles → %s", out_pr)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--parquet-template",
                   default=("/home/elek/sds/sd17d003/Anamaria/"
                            "alphagenome_genomicsxai/data/"
                            "combined_usage_data_{species}.parquet"),
                   help="Path template with {species} placeholder, or a direct path.")
    g.add_argument("--species", default="human",
                   help="Species (e.g. human) or 'all' for all species. "
                        "Default: human")
    g.add_argument("--tissue", default=None,
                   help="Tissue to filter (e.g. Brain). "
                        "Omit to use all tissues as separate trajectory dimensions.")
    g.add_argument("--data-config", default=None,
                   metavar="JSON",
                   help="Path to data_config.json. "
                        "When provided together with --split, sites are "
                        "filtered to those inside the split's genomic windows.")
    g.add_argument("--split", default="test",
                   choices=["test", "val", "train"],
                   help="Which split to keep (requires --data-config). "
                        "Default: test")
    g.add_argument("--min-timepoints", type=int, default=5,
                   help="Min detected timepoints per trajectory. Default: 5")
    g.add_argument("--min-reads", type=int, default=1)
    g.add_argument("--max-sites", type=int, default=None,
                   help="Random subsample limit applied *after* split filtering "
                        "(default: all sites).")
    g.add_argument("--random-seed", type=int, default=42)

    # GP
    g = p.add_argument_group("GP smoothing")
    g.add_argument("--gp-length-scale", type=float, default=0.20,
                   help="RBF kernel length scale in normalised time [0,1]. "
                        "0.20 ≈ 2-3 timepoint window. Default: 0.20")
    g.add_argument("--gp-noise-level", type=float, default=0.05)
    g.add_argument("--gp-n-restarts", type=int, default=2,
                   help="Hyperparameter optimiser restarts. Default: 2")
    g.add_argument("--gp-bound-tol", type=float, default=0.01,
                   help="Fraction of a bound's range within which a fitted "
                        "hyperparameter is flagged as 'hitting' that bound. "
                        "Default: 0.01 (1%%)")
    g.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel workers for GP fitting. "
                        "-1 = all cores (default). 1 = sequential.")
    g.add_argument("--load-features",
                   help="Path to previously saved .npy GP features array. "
                        "Skips GP fitting when provided.")
    g.add_argument("--load-sites",
                   help="Path to previously saved sites parquet (required with "
                        "--load-features).")

    # Clustering
    g = p.add_argument_group("Clustering")
    g.add_argument("--n-clusters", type=int, default=None,
                   help="Number of clusters. Omit to auto-select via gap statistic.")
    g.add_argument("--auto-k-min", type=int, default=5)
    g.add_argument("--auto-k-max", type=int, default=80)

    # Shape classification
    g = p.add_argument_group("Shape classification")
    g.add_argument("--flat-high", type=float, default=0.80,
                   help="Mean SSE >= this → flat_high. Default: 0.80")
    g.add_argument("--flat-low",  type=float, default=0.20,
                   help="Mean SSE <= this → flat_low. Default: 0.20")
    g.add_argument("--knee-frac", type=float, default=0.65,
                   help="Fraction of net change in one half → 'early'/'late'; "
                        "otherwise 'mid' (gradual). Default: 0.65")
    g.add_argument("--min-net-change", type=float, default=0.12,
                   help="Min absolute start-to-end SSE change for an up/down "
                        "label. Default: 0.12")
    g.add_argument("--amplitude-threshold", type=float, default=0.08,
                   help="Max amplitude to be considered flat. Default: 0.08")
    g.add_argument("--monotone-frac", type=float, default=0.65,
                   help="Min fraction of steps in one direction for monotone "
                        "classification. Default: 0.65")
    g.add_argument("--reversal-fraction", type=float, default=0.30,
                   help="Min size of each leg (as fraction of amplitude) for "
                        "up-down/down-up classification. Default: 0.30")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--output", required=True,
                   help="Output directory.")
    g.add_argument("--prefix", default="",
                   help="Optional file prefix. Auto-generated from species/tissue "
                        "if empty.")
    g.add_argument("--save-plots", action="store_true",
                   help="Save heatmap and cluster profile plots.")

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    species = None if args.species == "all" else args.species
    tissue  = args.tissue

    sp_part  = species or "all"
    tis_part = (tissue.replace(" ", "_").lower() if tissue else "all_tissues")
    label    = f"{sp_part}/{tis_part}"
    prefix   = args.prefix or f"{sp_part}_{tis_part}"

    # ── 1. Load / prepare data ─────────────────────────────────────────────
    log.info("=== Data ===")
    if args.load_features:
        # Resume mode: skip GP, reuse saved features
        if not args.load_sites:
            sys.exit("--load-sites is required when using --load-features")
        log.info("Loading saved GP features from %s", args.load_features)
        features_v = np.load(args.load_features)
        sites_v    = pd.read_parquet(args.load_sites)
        lmls_v     = np.full(len(features_v), np.nan)
        stds_v     = np.zeros_like(features_v)
        diagnostics_v = None
        log.info("Loaded %s trajectories × %d timepoints",
                 f"{len(features_v):,}", features_v.shape[1])
    else:
        sites, sse_wide, reads_wide = prepare_trajectories(
            args.parquet_template,
            species=species, tissue=tissue,
            min_timepoints=args.min_timepoints,
            min_reads=args.min_reads,
            max_sites=None,           # subsample *after* split filtering
            random_seed=args.random_seed,
        )

        # ── 1b. Filter to split ────────────────────────────────────────────
        if args.data_config is not None:
            split_mask = filter_to_split(
                sites, args.data_config,
                species=species, split=args.split,
            )
            sites      = sites[split_mask].reset_index(drop=True)
            sse_wide   = sse_wide.iloc[np.where(split_mask)[0]]
            reads_wide = reads_wide.iloc[np.where(split_mask)[0]]
            label      = f"{sp_part}/{tis_part} [{args.split}]"
            prefix     = args.prefix or f"{sp_part}_{tis_part}_{args.split}"

        # ── 1c. Subsample if requested ─────────────────────────────────────
        if args.max_sites is not None and len(sites) > args.max_sites:
            rng_ss = np.random.default_rng(args.random_seed)
            idx_ss = np.sort(rng_ss.choice(len(sites), size=args.max_sites,
                                           replace=False))
            sites      = sites.iloc[idx_ss].reset_index(drop=True)
            sse_wide   = sse_wide.iloc[idx_ss]
            reads_wide = reads_wide.iloc[idx_ss]
            log.info("Subsampled to %s trajectories", f"{args.max_sites:,}")

        log.info("Final dataset: %s trajectories", f"{len(sites):,}")

        # ── 2. GP smoothing ────────────────────────────────────────────────
        log.info("=== GP smoothing ===")
        features, stds, lmls, diagnostics = smooth_all_trajectories(
            sse_wide, reads_wide, T_GRID,
            length_scale=args.gp_length_scale,
            noise_level=args.gp_noise_level,
            n_restarts=args.gp_n_restarts,
            n_jobs=args.n_jobs,
            bound_tol=args.gp_bound_tol,
        )

        valid      = ~np.isnan(features).any(axis=1)
        features_v = features[valid].astype(np.float32)
        stds_v     = stds[valid].astype(np.float32)
        lmls_v     = lmls[valid]
        sites_v    = sites.iloc[np.where(valid)[0]].reset_index(drop=True)
        diagnostics_v = diagnostics.iloc[np.where(valid)[0]].reset_index(drop=True)

        # Save GP features (allows re-clustering without re-fitting)
        out_feats = os.path.join(args.output, f"{prefix}_gp_features.npy")
        out_sites = os.path.join(args.output, f"{prefix}_sites.parquet")
        out_diag  = os.path.join(args.output, f"{prefix}_gp_diagnostics.parquet")
        np.save(out_feats, features_v)
        sites_v.to_parquet(out_sites, index=False)
        diagnostics.to_parquet(out_diag, index=False)
        log.info("Saved GP features → %s", out_feats)
        log.info("Saved sites       → %s", out_sites)
        log.info("Saved GP diagnostics → %s", out_diag)

    n = len(features_v)
    log.info("Working with %s valid trajectories", f"{n:,}")

    # ── 3. Hierarchical clustering ─────────────────────────────────────────
    log.info("=== Ward-linkage clustering ===")
    Z = linkage(features_v, method="ward", metric="euclidean")

    # Save linkage matrix
    out_Z = os.path.join(args.output, f"{prefix}_linkage.npy")
    np.save(out_Z, Z)
    log.info("Saved linkage → %s", out_Z)

    if args.n_clusters is not None:
        n_clusters = args.n_clusters
        log.info("Using k=%d (user-specified)", n_clusters)
    else:
        n_clusters = select_k_gap(Z, k_min=args.auto_k_min, k_max=args.auto_k_max)
        log.info("Auto-selected k=%d  (range %d–%d)",
                 n_clusters, args.auto_k_min, args.auto_k_max)

    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # ── 4. Shape classification ────────────────────────────────────────────
    log.info("=== Shape classification ===")
    cluster_shapes = {}
    for k in range(1, n_clusters + 1):
        mean_k = features_v[cluster_labels == k].mean(axis=0)
        cluster_shapes[k] = classify_cluster_shape(
            mean_k,
            amplitude_threshold=args.amplitude_threshold,
            monotone_frac=args.monotone_frac,
            reversal_fraction=args.reversal_fraction,
            min_net_change=args.min_net_change,
            knee_frac=args.knee_frac,
            flat_high=args.flat_high,
            flat_low=args.flat_low,
        )

    log.info("%-8s  %-16s  %8s", "Cluster", "Shape", "N sites")
    log.info("-" * 38)
    for k, shape in cluster_shapes.items():
        nc = int((cluster_labels == k).sum())
        log.info("C%-7d  %-16s  %8s", k, shape, f"{nc:,}")

    # ── 5. Build metadata parquet ──────────────────────────────────────────
    log.info("=== Building metadata ===")
    meta = sites_v.copy()
    meta["Cluster"]      = cluster_labels
    meta["ClusterShape"] = meta["Cluster"].map(cluster_shapes)
    meta["GP_mean_SSE"]  = features_v.mean(axis=1)
    meta["GP_max_SSE"]   = features_v.max(axis=1)
    meta["GP_min_SSE"]   = features_v.min(axis=1)
    meta["GP_range"]     = meta["GP_max_SSE"] - meta["GP_min_SSE"]
    meta["GP_lml"]       = lmls_v

    if diagnostics_v is not None:
        meta["GP_fit_length_scale"]      = diagnostics_v["fit_length_scale"].values
        meta["GP_fit_noise_level"]       = diagnostics_v["fit_noise_level"].values
        meta["GP_hit_length_scale_lo"]   = diagnostics_v["hit_length_scale_lo"].values
        meta["GP_hit_length_scale_hi"]   = diagnostics_v["hit_length_scale_hi"].values
        meta["GP_hit_noise_level_lo"]    = diagnostics_v["hit_noise_level_lo"].values
        meta["GP_hit_noise_level_hi"]    = diagnostics_v["hit_noise_level_hi"].values

    _cmean  = {}
    _crange = {}
    for k in range(1, n_clusters + 1):
        mk = features_v[cluster_labels == k].mean(axis=0)
        _cmean[k]  = float(mk.mean())
        _crange[k] = float(mk.max() - mk.min())
    meta["ClusterMeanSSE"]  = meta["Cluster"].map(_cmean)
    meta["ClusterRangeSSE"] = meta["Cluster"].map(_crange)

    out_meta = os.path.join(args.output, f"{prefix}_clustering_metadata.parquet")
    meta.to_parquet(out_meta, index=False)
    log.info("Metadata → %s  (%s rows)", out_meta, f"{len(meta):,}")

    # Shape summary
    sc = Counter(cluster_shapes.values())
    ss = Counter()
    for k, sh in cluster_shapes.items():
        ss[sh] += int((cluster_labels == k).sum())
    log.info("Shape distribution:")
    for sh in SHAPE_ORDER:
        if ss.get(sh, 0) > 0:
            log.info("  %-16s  %s trajectories", sh, f"{ss[sh]:,}")

    # ── 6. Plots ───────────────────────────────────────────────────────────
    if args.save_plots:
        log.info("=== Saving plots ===")
        save_plots(features_v, cluster_labels, cluster_shapes,
                   n_clusters, args.output, prefix, label,
                   random_seed=args.random_seed)

    log.info("=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())