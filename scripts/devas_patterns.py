"""Mazin et al. 2021 (Nat Genet, doi:10.1038/s41588-021-00851-w) developmental
AS pattern classification -- up / down / up-down / down-up -- ported to this
project's splice-site usage trajectories.

Their procedure (Methods, "Developmental PSI and devAS pattern definition"):
  1. approximate PSI as a function of log developmental age with a cubic spline
     at 4 degrees of freedom;
  2. dPSI = max - min of the spline-predicted values at the sample ages;
  3. call devAS when adjusted P < 0.05 AND dPSI > 0.2;
  4. interpolate the fit onto 1,000 evenly spaced age points, take consecutive
     differences, and accumulate
        up          = sum of positive changes
        down        = |sum of negative changes|
        up_timing   = sum(positive change * age) / up
        down_timing = |sum(negative change * age)| / down
  5. ratio = up / (up + down);  ratio < 0.3 -> down, > 0.7 -> up; otherwise
     up_timing < down_timing -> up-down, else down-up.

Two properties worth knowing before you port it:
  * `up`, `down` and therefore `ratio` are invariant to how the x-axis is
    parameterised -- they are the total upward / downward variation of the
    curve. Only the two timing statistics depend on the x variable, so the
    up-vs-down call is safe on timepoint index while up-down vs down-up is not.
  * both statistics count *every* reversal in the fitted curve, however small.
    A looser smoother inflates `up` and `down` together, pulls `ratio` toward
    0.5, and so over-assigns the two biphasic classes. `min_prominence`
    below is the guard: reversals smaller than that (in usage units) are
    pruned before the sums are taken.
"""
import numpy as np
from patsy import dmatrix
import statsmodels.api as sm

N_GRID = 1000


def spline_predict(x, y, x_out, df=4):
    """Cubic spline of y on x at `df` degrees of freedom, evaluated at x_out.

    Pass x already on the scale the fit should be linear-in-basis on (i.e.
    log age, as in the paper).
    """
    X = dmatrix("cr(v, df=%d)" % df, {"v": np.asarray(x, float)}, return_type="dataframe")
    fit = sm.OLS(np.asarray(y, float), X).fit()
    Xo = dmatrix(X.design_info, {"v": np.asarray(x_out, float)}, return_type="dataframe")
    return np.asarray(fit.predict(Xo), float)


def _turning_points(y):
    """Indices of the curve's endpoints and interior local extrema."""
    d = np.diff(y)
    s = np.sign(d)
    nz = s[s != 0]
    if len(nz) == 0:
        return [0, len(y) - 1]
    idx = [0]
    last = None
    for i, si in enumerate(s):
        if si == 0:
            continue
        if last is not None and si != last:
            idx.append(i)
        last = si
    idx.append(len(y) - 1)
    return idx


def _prune(y, idx, tau):
    """Drop reversals whose amplitude is below tau, smallest first."""
    idx = list(idx)
    while len(idx) > 2:
        amps = [abs(y[idx[k + 1]] - y[idx[k]]) for k in range(len(idx) - 1)]
        interior = [(a, k) for k, a in enumerate(amps) if 0 < k < len(amps) - 1] or \
                   [(a, k) for k, a in enumerate(amps)]
        a_min, k_min = min(interior)
        if a_min >= tau:
            break
        # remove the segment by deleting the extremum that ends it, then
        # collapse the now-monotone neighbours
        drop = idx[k_min + 1] if 0 < k_min + 1 < len(idx) - 1 else idx[k_min]
        idx = [i for i in idx if i != drop]
        # merge consecutive same-direction segments
        keep = [idx[0]]
        for k in range(1, len(idx) - 1):
            before = np.sign(y[idx[k]] - y[keep[-1]])
            after = np.sign(y[idx[k + 1]] - y[idx[k]])
            if before != 0 and before == after:
                continue
            keep.append(idx[k])
        keep.append(idx[-1])
        if keep == idx:
            break
        idx = keep
    return idx


def pattern_stats(x, y, min_prominence=0.0):
    """Mazin's four statistics on a fitted curve sampled at x (dense, ordered).

    min_prominence prunes reversals below that amplitude in usage units before
    accumulating; 0.0 reproduces the paper exactly.
    Returns dict with up, down, ratio, up_timing, down_timing, dpsi, n_turns.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < 3:
        return dict(up=np.nan, down=np.nan, ratio=np.nan, up_timing=np.nan,
                    down_timing=np.nan, dpsi=np.nan, n_turns=np.nan)
    idx = _turning_points(y)
    if min_prominence > 0:
        idx = _prune(y, idx, min_prominence)
    dy, xm = [], []
    for k in range(len(idx) - 1):
        a, b = idx[k], idx[k + 1]
        seg = np.diff(y[a:b + 1])
        sgn = np.sign(y[b] - y[a])
        dy.append(np.abs(seg) * sgn)          # sign-consistent within a segment
        xm.append((x[a:b] + x[a + 1:b + 1]) / 2)
    dy = np.concatenate(dy); xm = np.concatenate(xm)
    pos, neg = dy > 0, dy < 0
    up = dy[pos].sum(); down = -dy[neg].sum()
    tot = up + down
    return dict(
        up=up, down=down,
        ratio=up / tot if tot > 0 else np.nan,
        up_timing=(dy[pos] * xm[pos]).sum() / up if up > 0 else np.nan,
        down_timing=-(dy[neg] * xm[neg]).sum() / down if down > 0 else np.nan,
        dpsi=y.max() - y.min(), n_turns=len(idx) - 2)


def classify(st, lo=0.3, hi=0.7):
    """up / down / up-down / down-up from pattern_stats output."""
    r = st["ratio"]
    if not np.isfinite(r):
        return "flat"
    if r < lo:
        return "down"
    if r > hi:
        return "up"
    ut, dt = st["up_timing"], st["down_timing"]
    if not (np.isfinite(ut) and np.isfinite(dt)):
        return "up" if r >= 0.5 else "down"
    return "up-down" if ut < dt else "down-up"


def classify_series(x_meas, y_meas, x_grid=None, curve=None, method="spline",
                    df=4, min_prominence=0.0, dpsi_min=0.2, log_x=True):
    """Classify one trajectory.

    method="spline": fit the paper's spline to (x_meas, y_meas).
    method="curve" : use an already-smoothed curve given on x_grid (e.g. the
                     GP mean), interpolated onto the dense grid.
    x_meas/x_grid are ages (or stage indices); log_x applies log() first, as in
    the paper.
    """
    f = np.log if log_x else (lambda v: np.asarray(v, float))
    if method == "spline":
        xs = f(x_meas)
        g = np.linspace(xs.min(), xs.max(), N_GRID)
        yg = spline_predict(xs, y_meas, g)
    else:
        xs = f(np.asarray(x_grid, float)[np.isfinite(curve)])
        cv = np.asarray(curve, float)[np.isfinite(curve)]
        g = np.linspace(xs.min(), xs.max(), N_GRID)
        yg = np.interp(g, xs, cv)
    st = pattern_stats(g, yg, min_prominence=min_prominence)
    st["pattern"] = classify(st)
    st["devAS_amplitude_pass"] = bool(st["dpsi"] > dpsi_min)
    return st
