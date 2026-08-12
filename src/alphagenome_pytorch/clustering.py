"""Developmental-trajectory clustering building blocks.

Data preparation, GP smoothing, cluster-shape classification, k-selection, and
assignment of new (e.g. predicted) trajectories to an existing reference
clustering. Extracted from ``alphagenome_pytorch.plotting.splicing`` (which now
re-exports these names for backwards compatibility) so clustering logic lives
independently of the plotting utilities.
"""

import os
import warnings
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Developmental-trajectory clustering: shared constants ───────────────────────

T_GRID = np.arange(1, 16, dtype=float)  # developmental timepoints 1..15

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
    # High-baseline dynamics (stays in the high band but still moves)
    "high_up":       "#1b9e77",   # teal-green
    "high_down":     "#d95f02",   # burnt orange
    "high_var":      "#7570b3",   # slate purple
    # Low-baseline dynamics (stays in the low band but still moves)
    "low_up":        "#66c2a5",   # light teal
    "low_down":      "#fc8d62",   # light orange
    "low_var":       "#b3a2c7",   # light purple
    # Flat
    "flat_high":     "#9467bd",
    "flat_mid_high": "#d8c8e8",
    "flat_mid":      "#e0d4eb",
    "flat_mid_low":  "#c6b2da",
    "flat_low":      "#c5b0d5",
    # No clear pattern
    "noisy":         "#7f7f7f",
    "complex":       "#7f7f7f",   # legacy alias (kept so old outputs still colour)
}

SHAPE_ORDER = [
    "up_early",   "up_late",   "up_mid",
    "down_early", "down_late", "down_mid",
    "up-down", "down-up",
    "high_up", "high_down", "high_var",
    "low_up", "low_down", "low_var",
    "flat_high", "flat_mid_high", "flat_mid", "flat_mid_low", "flat_low",
    "noisy",
]


def prepare_trajectories(parquet_path, species=None, tissue=None,
                         min_timepoints=5, min_reads=1,
                         max_sites=None, random_seed=42):
    """
    Load splice-site usage data, aggregate to a per-(site, [tissue], [species])
    x timepoint matrix, and filter by data availability.

    Parameters
    ----------
    parquet_path : str
        Direct path or template with ``{species}`` placeholder.
    species : str or None
        If None (or 'all'), all matching species files are loaded and Species
        is retained as a trajectory-identity dimension.
    tissue : str or None
        If None, all tissues are kept and Tissue is a trajectory dimension.

    Returns
    -------
    sites : pd.DataFrame        trajectory identifier columns
    sse_wide : pd.DataFrame     SSE values (rows = trajectories, cols = timepoints 1-15)
    reads_wide : pd.DataFrame   read counts, same shape (0 where undetected)
    """
    import glob
    from pathlib import Path

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


def filter_to_split(sites, data_config_path, species=None, split="test"):
    """
    Return a boolean mask selecting sites inside the genomic windows of a data
    split (test / val / train). Boundaries come from BED files referenced in the
    JSON config (key ``<split>_bed``); a site is in the split if its
    (Chromosome, Position) falls in any ~131 kb window.
    """
    import json
    import os

    with open(data_config_path) as _f:
        cfg = json.load(_f)

    bed_key = f"{split}_bed"

    def _load_intervals(sp):
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


def _make_kernel(length_scale=0.20, noise_level=0.05):
    from sklearn.gaussian_process.kernels import (
        RBF, ConstantKernel as C, WhiteKernel,
    )
    return (
        C(1.0, (1e-3, 5.0))
        * RBF(length_scale=length_scale, length_scale_bounds=(0.05, 2.0))
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-5, 0.5))
    )


def smooth_trajectory_gp(tps_obs, sse_obs, t_grid,
                         length_scale=0.20, noise_level=0.05, n_restarts=2):
    """Fit a GP to one trajectory's observed points; return (mean, std, lml).

    The single-trajectory smoother used for illustration in
    ``splice_trajectory_clustering.ipynb`` (see also :func:`_smooth_one`, which
    adds hyperparameter diagnostics for the batch pipeline).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor

    t_min, t_max = float(t_grid[0]), float(t_grid[-1])
    norm = lambda t: (np.asarray(t, float) - t_min) / (t_max - t_min)

    gpr = GaussianProcessRegressor(
        kernel=_make_kernel(length_scale, noise_level),
        n_restarts_optimizer=n_restarts,
        normalize_y=True, alpha=1e-8, random_state=42,
    )
    gpr.fit(norm(tps_obs).reshape(-1, 1), np.asarray(sse_obs, float))
    y_mean, y_std = gpr.predict(norm(t_grid).reshape(-1, 1), return_std=True)
    return np.clip(y_mean, 0.0, 1.0), y_std, gpr.log_marginal_likelihood_value_


def _smooth_one(sse_row, reads_row, col_tps, t_grid,
                length_scale, noise_level, n_restarts, bound_tol=0.01,
                optimize=True):
    """Smooth a single trajectory (GP, with hyperparameter diagnostics); falls
    back to linear interpolation. Returns (y_mean, y_std, lml, diag).

    ``optimize=True`` (default) fits the kernel hyper-parameters per trajectory by
    maximising the marginal likelihood — the clustering-pipeline behaviour.
    ``optimize=False`` holds the length-scale fixed at ``length_scale``, which is
    far cheaper and only mildly smoother; useful for large ad-hoc batches.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.exceptions import ConvergenceWarning
    from scipy.interpolate import interp1d

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
                n_restarts_optimizer=(n_restarts if optimize else 0),
                optimizer=("fmin_l_bfgs_b" if optimize else None),
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


def _smooth_all_fixed(sse_arr, reads_arr, col_tps, t_grid,
                      length_scale=0.20, noise_level=0.05, bound_tol=0.01):
    """Closed-form, vectorised equivalent of ``_smooth_one(..., optimize=False)``.

    With the kernel hyper-parameters held fixed, the GP posterior mean is a linear
    smoother whose matrix depends only on *which* timepoints are observed — not on
    their values. So trajectories are grouped by observed-timepoint pattern, the
    smoother is built once per pattern, and every trajectory sharing that pattern is
    smoothed with a single matrix product. Same numbers as the per-trajectory sklearn
    path, but orders of magnitude faster (no per-fit estimator overhead).

    Returns (features, stds, lmls, diagnostics) — the same contract as
    :func:`smooth_all_trajectories`.
    """
    n, G = sse_arr.shape
    n_grid = len(t_grid)
    t_min, t_max = float(t_grid[0]), float(t_grid[-1])
    span = (t_max - t_min) or 1.0
    tn = (np.asarray(col_tps, float) - t_min) / span
    gn = (np.asarray(t_grid, float) - t_min) / span

    features = np.full((n, n_grid), np.nan)
    stds     = np.zeros((n, n_grid))
    lmls     = np.full(n, np.nan)

    detected  = reads_arr > 0
    n_obs_all = detected.sum(axis=1)

    # Group rows by their observed-timepoint pattern
    codes = detected.astype(np.int64) @ (1 << np.arange(G)).astype(np.int64)
    for code in np.unique(codes):
        rows = np.where(codes == code)[0]
        m = detected[rows[0]]
        n_obs = int(m.sum())
        if n_obs < 2:
            continue  # leave NaN, as _smooth_one does

        X  = tn[m]
        K  = np.exp(-0.5 * ((X[:, None] - X[None, :]) / length_scale) ** 2)
        K[np.diag_indices_from(K)] += noise_level + 1e-8          # WhiteKernel + alpha
        Ks = np.exp(-0.5 * ((gn[:, None] - X[None, :]) / length_scale) ** 2)
        Kinv = np.linalg.inv(K)
        S = Ks @ Kinv

        Y  = np.nan_to_num(sse_arr[np.ix_(rows, np.where(m)[0])], nan=0.0)
        mu = Y.mean(axis=1, keepdims=True)
        sd = Y.std(axis=1, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)      # sklearn's _handle_zeros_in_scale
        Yn = (Y - mu) / sd                      # normalize_y=True

        features[rows] = (Yn @ S.T) * sd + mu

        var = (1.0 + noise_level) - np.einsum("ij,ij->i", Ks, S)
        stds[rows] = np.sqrt(np.clip(var, 0.0, None))[None, :] * sd

        quad = np.einsum("ij,ji->i", Yn, Kinv @ Yn.T)
        _, logdet = np.linalg.slogdet(K)
        lmls[rows] = -0.5 * quad - 0.5 * logdet - 0.5 * n_obs * np.log(2 * np.pi)

    features = np.clip(features, 0.0, 1.0)

    ls_lo, ls_hi = 0.05, 2.0
    nl_lo, nl_hi = 1e-5, 0.5
    fitted = n_obs_all >= 2
    diagnostics = pd.DataFrame({
        "fit_length_scale": np.where(fitted, length_scale, np.nan),
        "fit_noise_level":  np.where(fitted, noise_level, np.nan),
        "length_scale_bound_lo": ls_lo, "length_scale_bound_hi": ls_hi,
        "noise_level_bound_lo":  nl_lo, "noise_level_bound_hi":  nl_hi,
        "hit_length_scale_lo": bool((length_scale - ls_lo) <= bound_tol * (ls_hi - ls_lo)),
        "hit_length_scale_hi": bool((ls_hi - length_scale) <= bound_tol * (ls_hi - ls_lo)),
        "hit_noise_level_lo":  bool((noise_level - nl_lo) <= bound_tol * (nl_hi - nl_lo)),
        "hit_noise_level_hi":  bool((nl_hi - noise_level) <= bound_tol * (nl_hi - nl_lo)),
        "n_observed": n_obs_all.astype(int),
        "fallback": ~fitted,
        "convergence_warning": False,
    })
    return (features.astype(np.float32), stds.astype(np.float32),
            lmls.astype(np.float64), diagnostics)


def smooth_all_trajectories(sse_wide, reads_wide, t_grid,
                            length_scale=0.20, noise_level=0.05,
                            n_restarts=2, n_jobs=-1, bound_tol=0.01,
                            optimize=True):
    """Fit a GP to every trajectory in ``sse_wide`` (parallelised).

    Each trajectory is smoothed individually (interpolating its missing
    timepoints); a cluster profile is then a plain average of these smooth
    curves. A timepoint counts as observed where ``reads_wide > 0`` — for data
    without read counts, pass ``sse_wide.notna().astype(int)``.

    Parameters
    ----------
    sse_wide, reads_wide : pd.DataFrame
        Values and read counts, rows = trajectories, columns = ``t_grid``.
    optimize : bool
        True (default) fits kernel hyper-parameters per trajectory (the
        clustering-pipeline behaviour). False holds the length-scale fixed at
        ``length_scale`` — much faster for large batches, only mildly smoother.

    Returns (features, stds, lmls, diagnostics) where ``features`` are the GP
    posterior means at ``t_grid`` (n x len(t_grid)).
    """
    import time
    try:
        from joblib import Parallel, delayed
        has_joblib = True
    except ImportError:
        has_joblib = False
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    sse_arr   = sse_wide.values.astype(float)
    reads_arr = reads_wide.values.astype(float)
    col_tps   = np.array([float(c) for c in sse_wide.columns])
    n_sites   = sse_arr.shape[0]

    t0 = time.time()

    if not optimize:
        # Fixed hyper-parameters -> the posterior mean is a linear smoother, so the
        # whole batch is done in closed form (grouped by observed-timepoint pattern).
        log.info("Smoothing %s trajectories with fixed hyper-parameters "
                 "(vectorised closed form) ...", f"{n_sites:,}")
        features, stds, lmls, diagnostics = _smooth_all_fixed(
            sse_arr, reads_arr, col_tps, t_grid,
            length_scale=length_scale, noise_level=noise_level,
            bound_tol=bound_tol,
        )
    else:
        log.info("Fitting GPs to %s trajectories  (n_jobs=%d, n_restarts=%d) ...",
                 f"{n_sites:,}", n_jobs, n_restarts)

        kwargs = dict(col_tps=col_tps, t_grid=t_grid,
                      length_scale=length_scale, noise_level=noise_level,
                      n_restarts=n_restarts, bound_tol=bound_tol,
                      optimize=optimize)

        if has_joblib and n_jobs != 1:
            # The per-trajectory optimiser is CPU-bound and holds the GIL, so a thread
            # pool oversubscribes and runs *slower* than sequential. Use processes
            # (loky) instead — same results, ~10x faster on a many-core machine.
            results = Parallel(n_jobs=n_jobs, backend="loky")(
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

    fitted = diagnostics[~diagnostics["fallback"]]
    if len(fitted):
        ls_lo_n = int(fitted["hit_length_scale_lo"].sum())
        ls_hi_n = int(fitted["hit_length_scale_hi"].sum())
        nl_lo_n = int(fitted["hit_noise_level_lo"].sum())
        nl_hi_n = int(fitted["hit_noise_level_hi"].sum())
        cw_n    = int(fitted.get("convergence_warning", pd.Series(dtype=bool)).sum())
        log.info("Hyperparameter boundary check (tol=%.0f%% of range, n=%s fitted):",
                 bound_tol * 100, f"{len(fitted):,}")
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
                "%s trajectories hit the length_scale lower bound (%.3g) — the GP "
                "wants to smooth less than allowed.",
                f"{ls_lo_n:,}", fitted["length_scale_bound_lo"].iloc[0],
            )

    return features, stds, lmls, diagnostics


def classify_cluster_shape(mean_traj,
                           amplitude_threshold=0.08,
                           smooth_window=3,
                           monotone_frac=0.65,
                           peak_window=(0.25, 0.75),
                           reversal_fraction=0.30,
                           biphasic_abs_leg=0.15,
                           min_net_change=0.12,
                           knee_frac=0.60,
                           flat_high=0.80,
                           flat_low=0.20,
                           level_window=3,
                           strict_updown=True,
                           updown_low=0.35,
                           updown_high=0.65,
                           updown_min_change=0.30,
                           label_high_range=True,
                           high_base_min=0.55,
                           low_base_max=0.45,
                           high_dir_change=0.15):
    """Classify a cluster's mean trajectory into one of 14 shape labels
    (see ``SHAPE_ORDER``).

    Strict decision tree (``strict_updown=True``, default), by *baseline* then change:
      1. ``flat_*`` — amplitude < ``amplitude_threshold`` (sub-labelled by mean level).
      2. **High baseline** (``min(y) >= high_base_min``, stays in the high band):
         ``high_up`` / ``high_down`` if the net change exceeds ``high_dir_change``,
         else ``high_var``.
      3. **Low baseline** (``max(y) <= low_base_max``, stays in the low band):
         ``low_up`` / ``low_down`` / ``low_var`` (symmetric to high).
      4. **Mid range**, only when the change is large (amplitude *and* ``|net|`` ≥
         ``updown_min_change``): biphasic ``up-down`` / ``down-up``, else a directional
         ``up_{early|late|mid}`` / ``down_{early|late|mid}`` (knee timing from
         ``knee_frac``).
      5. Otherwise ``noisy`` — weak, small, or ambiguous changes are deliberately not
         over-called as trends (historically ``up_early`` was heavily over-annotated).

    ``label_high_range`` toggles the high/low baseline categories (steps 2–3). Set
    ``strict_updown=False`` for the legacy permissive behaviour (any
    ``monotone_frac``-consistent direction with ``|net| >= min_net_change``, plus a
    net-direction fallback)."""
    from scipy.ndimage import uniform_filter1d

    y = uniform_filter1d(np.asarray(mean_traj, float),
                         size=smooth_window, mode="nearest")
    n   = len(y)
    win = max(1, min(level_window, n // 4))

    amplitude = float(np.max(y) - np.min(y))

    if amplitude < amplitude_threshold:
        mv = float(np.mean(y))
        if mv >= flat_high: return "flat_high"
        if mv <= flat_low:  return "flat_low"
        if mv <= 0.4:       return "flat_mid_low"
        if mv >= 0.6:       return "flat_mid_high"
        return "flat_mid"

    def _knee():
        total = abs(float(y[-1]) - float(y[0]))
        if total < 1e-6:
            return "mid"
        mid = n // 2
        first_half  = abs(float(y[mid]) - float(y[0]))
        second_half = abs(float(y[-1]) - float(y[mid]))
        if first_half  / total >= knee_frac: return "early"
        if second_half / total >= knee_frac: return "late"
        return "mid"

    start_val  = float(np.mean(y[:win]))
    end_val    = float(np.mean(y[-win:]))
    net_change = end_val - start_val
    abs_net    = abs(net_change)

    diffs     = np.diff(y)
    frac_up   = float((diffs > 0).mean())
    frac_down = float((diffs < 0).mean())

    # Biphasic detection (shared): a mid-trajectory peak (up-down) or valley (down-up)
    # with both legs large enough. Each leg must clear BOTH a relative floor
    # (reversal_fraction x amplitude) and an absolute floor (biphasic_abs_leg) — the
    # absolute floor separates a clean reversal (both legs substantial) from a shallow
    # one-sided wiggle (one tiny leg), which stays 'noisy'.
    lo = int(np.floor(peak_window[0] * n))
    hi = int(np.ceil(peak_window[1] * n))
    argmax_idx = int(np.argmax(y))
    argmin_idx = int(np.argmin(y))
    min_leg    = max(reversal_fraction * amplitude, biphasic_abs_leg)

    def _is_up_down():
        return (lo <= argmax_idx <= hi
                and float(y[argmax_idx] - y[0]) >= min_leg
                and float(y[argmax_idx] - y[-1]) >= min_leg)

    def _is_down_up():
        return (lo <= argmin_idx <= hi
                and float(y[0]  - y[argmin_idx]) >= min_leg
                and float(y[-1] - y[argmin_idx]) >= min_leg)

    if strict_updown:
        ymin, ymax = float(np.min(y)), float(np.max(y))

        # 1) Biphasic reversal — a clean mid peak / valley (both legs clear the relative
        #    AND absolute leg floors). Checked first so a real reversal is labelled
        #    up-down / down-up even when it sits inside the high or low band.
        if _is_up_down(): return "up-down"
        if _is_down_up(): return "down-up"

        # 2) Baseline dynamics — the trajectory is confined to (mostly) one band but
        #    still moves. Labelled by net direction: <name>_up / _down when |net| >=
        #    high_dir_change, else <name>_var. (label_high_range gates the naming.)
        if label_high_range and ymin >= high_base_min:              # stays high
            if net_change >= high_dir_change:  return "high_up"
            if net_change <= -high_dir_change: return "high_down"
            return "high_var"
        if label_high_range and ymax <= low_base_max:               # stays low
            if net_change >= high_dir_change:  return "low_up"
            if net_change <= -high_dir_change: return "low_down"
            return "low_var"

        # 3) Mid-range directional trend — only when the change is large (amplitude AND
        #    |net| >= updown_min_change). Weak or ambiguous mid changes -> noisy.
        if amplitude >= updown_min_change:
            if net_change >=  updown_min_change: return f"up_{_knee()}"
            if net_change <= -updown_min_change: return f"down_{_knee()}"
        return "noisy"

    # ── Permissive (legacy) behaviour ────────────────────────────────────────────
    if frac_up >= monotone_frac and abs_net >= min_net_change:
        return f"up_{_knee()}"
    if frac_down >= monotone_frac and abs_net >= min_net_change:
        return f"down_{_knee()}"
    if _is_up_down(): return "up-down"
    if _is_down_up(): return "down-up"
    if abs_net >= min_net_change:
        return f'{"up" if net_change > 0 else "down"}_{_knee()}'
    return "noisy"


def select_k_gap(Z, k_min=5, k_max=80):
    """Choose k by the largest gap between successive Ward merge distances."""
    k_max = min(k_max, len(Z))
    last_dists = Z[:, 2][-(k_max - 1):][::-1]
    gaps = np.diff(last_dists)
    best_k = int(np.argmax(gaps)) + 1 + k_min
    return max(k_min, min(k_max, best_k))

# ── Assigning new trajectories to an existing reference clustering ────────────────

def cluster_centroids(features, labels):
    """Mean feature vector per cluster.

    Returns ``(centroids, cluster_ids)`` where ``centroids[i]`` is the mean of
    ``features`` over rows with ``labels == cluster_ids[i]`` (ids sorted).
    """
    features = np.asarray(features, float)
    labels = np.asarray(labels)
    cluster_ids = np.array(sorted(np.unique(labels)))
    centroids = np.vstack([features[labels == k].mean(axis=0) for k in cluster_ids])
    return centroids, cluster_ids


def assign_to_centroids(features, centroids, cluster_ids):
    """Assign each row of ``features`` to its nearest centroid (Euclidean).

    Returns ``(assigned, dist)``: the cluster id and distance to the chosen
    centroid for every row.
    """
    features = np.asarray(features, float)
    centroids = np.asarray(centroids, float)
    # (n, k) pairwise Euclidean distances
    d = np.sqrt(((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    nearest = d.argmin(axis=1)
    return np.asarray(cluster_ids)[nearest], d[np.arange(len(features)), nearest]


def reference_prefix(species, tissue, split="test"):
    """Filename prefix used by ``cluster_trajectories.py`` (e.g. ``human_Brain_test``)."""
    tis = tissue.replace(" ", "_") if tissue else "all_tissues"
    return f"{species}_{tis}_{split}"


def load_reference(ref_dir, species, tissue, split="test", prefix=None):
    """Load a saved true-trajectory clustering as a fixed reference.

    Expects ``cluster_trajectories.py`` outputs in ``ref_dir``:
    ``<prefix>_gp_features.npy`` and ``<prefix>_clustering_metadata.parquet``
    (row-aligned). Returns a dict with the reference sites/labels/shapes, the
    per-cluster centroids, and the cluster->shape map.
    """
    prefix = prefix or reference_prefix(species, tissue, split)
    feats = np.load(os.path.join(ref_dir, f"{prefix}_gp_features.npy"))
    meta = pd.read_parquet(os.path.join(ref_dir, f"{prefix}_clustering_metadata.parquet"))
    if len(feats) != len(meta):
        raise ValueError(f"features ({len(feats)}) and metadata ({len(meta)}) not aligned "
                         f"for {prefix}")
    labels = meta["Cluster"].to_numpy()
    centroids, cluster_ids = cluster_centroids(feats, labels)
    cluster_to_shape = (meta.drop_duplicates("Cluster")
                            .set_index("Cluster")["ClusterShape"].to_dict())
    return dict(prefix=prefix, meta=meta, features=feats, labels=labels,
                centroids=centroids, cluster_ids=cluster_ids,
                cluster_to_shape=cluster_to_shape)


def self_assignment_accuracy(ref):
    """Fraction of reference trajectories whose nearest centroid equals their Ward
    cluster label. This is the ceiling for the nearest-centroid classifier — how
    well it reproduces the reference partition (Ward is not centroid-based)."""
    assigned, _ = assign_to_centroids(ref["features"], ref["centroids"], ref["cluster_ids"])
    return float(np.mean(assigned == ref["labels"]))


def load_pred_true_trajectories(preds_parquet, usage_parquet, species, tissue,
                                min_timepoints=5, min_reads=1):
    """Build predicted *and* observed per-(site x timepoint) trajectories, aligned.

    ``preds_parquet`` supplies ``SSE_pred`` and ``SSE_true`` per (Chromosome,
    Position, Tissue, Timepoint); ``usage_parquet`` (the combined true usage)
    supplies ``Strand`` and ``Reads`` for the *same* observed conditions. Both
    trajectories are meant to be smoothed with the same true reads as GP weights,
    so predicted and observed are processed through an identical transform (which
    also matches the reference clustering).

    Returns ``(sites, true_wide, pred_wide, reads_wide)`` indexed by
    (Chromosome, Position, Strand), columns = timepoints 1..15. ``reads_wide`` is
    the observed read count (0 where undetected).
    """
    preds = pd.read_parquet(preds_parquet, columns=["Chromosome", "Position", "Tissue",
                                                    "Timepoint", "SSE_pred", "SSE_true"])
    preds = preds[preds["Tissue"] == tissue].copy()
    preds["Chromosome"] = preds["Chromosome"].astype(str)

    usage = pd.read_parquet(usage_parquet, columns=["Chromosome", "Position", "Strand",
                                                    "Tissue", "Timepoint", "Reads"])
    usage = usage[usage["Tissue"] == tissue].copy()
    usage["Chromosome"] = usage["Chromosome"].astype(str)
    if min_reads > 0:
        usage = usage[usage["Reads"] >= min_reads]

    # attach Strand + true Reads to the predicted/observed conditions
    df = preds.merge(usage[["Chromosome", "Position", "Strand", "Timepoint", "Reads"]],
                     on=["Chromosome", "Position", "Timepoint"], how="inner")
    if df.empty:
        raise ValueError(f"No overlapping predicted/observed conditions for "
                         f"species={species!r}, tissue={tissue!r}")

    group_cols = ["Chromosome", "Position", "Strand"]
    all_tps = list(range(1, 16))
    agg = (df.groupby(group_cols + ["Timepoint"])
             .agg(SSE_true=("SSE_true", "mean"), SSE_pred=("SSE_pred", "mean"),
                  Reads=("Reads", "sum"))
             .reset_index())

    def _wide(col, fill=None):
        w = agg.pivot_table(index=group_cols, columns="Timepoint", values=col,
                            **({"fill_value": fill} if fill is not None else {}))
        return w.reindex(columns=all_tps, **({"fill_value": fill} if fill is not None else {}))

    true_wide = _wide("SSE_true")
    pred_wide = _wide("SSE_pred").reindex(true_wide.index)
    reads_wide = _wide("Reads", fill=0).reindex(true_wide.index, fill_value=0)

    keep = (reads_wide.values > 0).sum(axis=1) >= min_timepoints
    true_wide, pred_wide, reads_wide = true_wide[keep], pred_wide[keep], reads_wide[keep]
    sites = true_wide.reset_index()[group_cols]
    return sites, true_wide, pred_wide, reads_wide


def load_prediction_trajectories(preds_parquet, usage_parquet, species, tissue,
                                 min_timepoints=5, min_reads=1):
    """Predicted trajectories only — thin wrapper over
    :func:`load_pred_true_trajectories` returning ``(sites, pred_wide, reads_wide)``."""
    sites, _true_wide, pred_wide, reads_wide = load_pred_true_trajectories(
        preds_parquet, usage_parquet, species, tissue, min_timepoints, min_reads)
    return sites, pred_wide, reads_wide
