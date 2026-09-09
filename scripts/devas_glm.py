"""Mazin et al. 2021 (Nat Genet, doi:10.1038/s41588-021-00851-w) developmental
alternative-splicing (devAS) test, vectorised over sites.

The paper's test (Methods, "devAS segment definition"):

    quasi-binomial GLM   (i, e) ~ a + a^2 + a^3

where i and e are the inclusion and exclusion read counts of a segment in one
sample and `a` is the logarithm of the number of days from conception. Each
term is tested by a quasi-likelihood ratio test, P values are adjusted with
Benjamini-Hochberg, and a segment is significant if *any* term has adjusted
P < 0.05. Amplitude and pattern come from a separate cubic-spline step at 4
degrees of freedom (see devas_patterns.py).

Everything here is vectorised across sites that share one species-organ design:
the design matrix is common, per-site differences enter only as observation
weights (a sample whose coverage is too low gets weight 0), so the whole IRLS
runs as batched 4x4 solves. `verify_against_statsmodels` checks the batched
fit against statsmodels' GLM site by site.

Conventions kept identical to the paper:
  * SSE (=PSI) is undefined where Alpha+Beta < 10; those samples are dropped
    from the site's series (weight 0).
  * a site is tested only where Alpha+Beta > 9 in >= 60% of the organ's
    samples and >= 4 samples have SSE in [0.1, 0.9].
  * the quasi-likelihood ratio test is a drop-one analysis-of-deviance F test
    with the dispersion taken from the full model (R's drop1(..., test="F")).
"""
import numpy as np
from scipy import stats
from patsy import dmatrix

TERMS = ("a", "a2", "a3")
N_GRID = 1000
EPS = 1e-9


# ---------------------------------------------------------------- design ----

def cubic_design(a, orthogonal=True):
    """[1, a, a^2, a^3] for log-age vector a.

    With orthogonal=True the raw powers are replaced by an orthonormal basis
    of the same nested column spans (QR, i.e. R's poly()). Sequential
    (analysis-of-deviance) tests are *identical* under the two bases because
    they only depend on the nested spans; conditioning is far better, which
    matters when the ages span three orders of magnitude and a^3 does not.
    Drop-one tests are basis-dependent, so they use the raw powers.
    """
    a = np.asarray(a, float)
    X = np.column_stack([np.ones_like(a), a, a ** 2, a ** 3])
    if not orthogonal:
        return X
    Q, R = np.linalg.qr(X)
    return Q * np.sign(np.diag(R))


def spline_design(a, df=4):
    """The paper's 4-df cubic spline basis (patsy `cr`), plus intercept.

    Returns (B, design_info) so the identical basis can be evaluated on a
    dense grid with `dmatrix(design_info, ...)`.
    """
    a = np.asarray(a, float)
    B = dmatrix("cr(v, df=%d)" % df, {"v": a}, return_type="dataframe")
    return np.asarray(B, float), B.design_info


def spline_eval_design(design_info, a):
    return np.asarray(dmatrix(design_info, {"v": np.asarray(a, float)},
                              return_type="dataframe"), float)


# ------------------------------------------------------------------ IRLS ----

def _batched_solve(XtWX, XtWy):
    """Solve a stack of small symmetric systems, NaN where singular."""
    p = XtWX.shape[-1]
    out = np.full(XtWy.shape, np.nan)
    # jitter only the scale-free direction; keeps well-conditioned sites exact
    ok = np.isfinite(XtWX).all((1, 2)) & np.isfinite(XtWy).all(1)
    if ok.any():
        A = XtWX[ok]; b = XtWy[ok][..., None]
        try:
            out[ok] = np.linalg.solve(A, b)[..., 0]
        except np.linalg.LinAlgError:
            sub = np.full((int(ok.sum()), p), np.nan)
            for k in range(A.shape[0]):
                try:
                    sub[k] = np.linalg.solve(A[k], b[k])[:, 0]
                except np.linalg.LinAlgError:
                    sub[k] = np.linalg.lstsq(A[k], b[k], rcond=None)[0][:, 0]
            out[ok] = sub
        bad = ~np.isfinite(out[ok]).all(1)
        if bad.any():                       # fall back to least squares
            idx = np.where(ok)[0][bad]
            for k in idx:
                out[k] = np.linalg.lstsq(XtWX[k], XtWy[k], rcond=None)[0]
    return out


def binomial_deviance(I, N, mu):
    """Per-site binomial deviance; zero-weight samples contribute nothing."""
    E = N - I
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(I > 0, I * np.log(np.maximum(I, EPS) / np.maximum(N * mu, EPS)), 0.0)
        t2 = np.where(E > 0, E * np.log(np.maximum(E, EPS) / np.maximum(N * (1 - mu), EPS)), 0.0)
    d = 2.0 * (t1 + t2)
    return np.where(N > 0, d, 0.0).sum(1)


def pearson_chi2(I, N, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (I - N * mu) ** 2 / np.maximum(N * mu * (1 - mu), EPS)
    return np.where(N > 0, r, 0.0).sum(1)


def irls_binomial(I, N, X, maxiter=60, tol=1e-10):
    """Batched binomial IRLS with prior weights N (N=0 drops the sample).

    I, N : (S, n) integer-valued arrays of inclusion counts and totals.
    X    : (n, p) shared design.
    Returns (beta (S,p), mu (S,n), deviance (S,), n_iter).
    """
    I = np.asarray(I, float); N = np.asarray(N, float)
    S, n = I.shape
    y = np.divide(I, N, out=np.full_like(I, 0.5), where=N > 0)
    mu = np.clip((I + 0.5) / (N + 1.0), 1e-6, 1 - 1e-6)
    beta = np.zeros((S, X.shape[1]))
    for it in range(1, maxiter + 1):
        v = mu * (1 - mu)
        W = N * v
        eta = np.log(mu / (1 - mu))
        z = eta + (y - mu) / v
        XtWX = np.einsum("ni,nj,sn->sij", X, X, W, optimize=True)
        XtWz = np.einsum("ni,sn->si", X, W * z, optimize=True)
        new = _batched_solve(XtWX, XtWz)
        step = np.nanmax(np.abs(new - beta)) if np.isfinite(new).any() else 0.0
        beta = new
        eta = np.clip(beta @ X.T, -30, 30)
        mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-10, 1 - 1e-10)
        if step < tol:
            break
    return beta, mu, binomial_deviance(I, N, mu), it


# ------------------------------------------------------------- the test -----

def bh_adjust(p):
    """Benjamini-Hochberg, NaN-safe, over the finite entries of a 1-D array."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return out
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def devas_test(I, N, log_age, alpha=0.05, mode="sequential", phi=None):
    """The paper's quasi-binomial cubic-polynomial test for a batch of sites.

    I, N    : (S, n) inclusion counts and totals, already coverage-masked
              (N == 0 marks a sample the site does not contribute).
    log_age : (n,) log(days from conception) per sample slot.

    Returns a dict of per-site arrays: p_a/p_a2/p_a3, padj_*, devAS_p (any
    term adjusted P < alpha), dispersion, resid_df, n_used, beta, deviance.
    """
    X = cubic_design(log_age, orthogonal=(mode == "sequential"))
    n_used = (N > 0).sum(1)
    beta, mu, dev, nit = irls_binomial(I, N, X)
    resid_df = n_used - X.shape[1]
    chi2 = pearson_chi2(I, N, mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_hat = np.where(resid_df > 0, chi2 / resid_df, np.nan)
    phi_hat = np.where(phi_hat > 0, phi_hat, np.nan)

    phi_own = phi_hat.copy()
    if phi is not None:
        # Test against an externally supplied dispersion (used to test a
        # noiseless predicted trajectory against the observed noise level).
        phi = np.asarray(phi, float).copy()
        phi[~np.isfinite(phi) | (phi <= 0)] = np.nan
    else:
        phi = phi_hat
    res = {"dispersion": phi, "dispersion_own": phi_own, "resid_df": resid_df.astype(float), "mode": mode,
           "n_used": n_used, "deviance_full": dev, "beta": beta, "irls_iter": nit}

    if mode == "sequential":
        # R's anova(fit, test="F"): each term against the model with all
        # lower-order terms only. Dispersion from the full model.
        devs = [irls_binomial(I, N, X[:, :j])[2] for j in range(1, X.shape[1])] + [dev]
        reduced = {t: devs[j - 1] for j, t in enumerate(TERMS, start=1)}
    elif mode == "drop1":
        # R's drop1(fit, test="F"): each term against the model without it.
        reduced = {t: irls_binomial(I, N, np.delete(X, j, axis=1))[2]
                   for j, t in enumerate(TERMS, start=1)}
    else:
        raise ValueError("mode must be 'sequential' or 'drop1'")

    for j, term in enumerate(TERMS, start=1):
        dev_r = reduced[term]
        with np.errstate(divide="ignore", invalid="ignore"):
            F = (dev_r - dev) / phi
        F = np.where(np.isfinite(F) & (F > 0), F, np.where(np.isfinite(F), 0.0, np.nan))
        p = np.where(np.isfinite(F) & (resid_df > 0),
                     stats.f.sf(F, 1, np.maximum(resid_df, 1)), np.nan)
        res["F_" + term] = F
        res["p_" + term] = p
        res["padj_" + term] = bh_adjust(p)
    res["devAS_p"] = np.any(np.column_stack(
        [res["padj_" + t] < alpha for t in TERMS]), axis=1)
    return res


# --------------------------------------------------- amplitude & pattern ----

def spline_curves(log_age, Y, Wmask, df=4, n_grid=N_GRID):
    """Batched weighted 4-df cubic-spline fit; returns (curves, grid, fitted).

    Y, Wmask : (S, n) usage values and 0/1 masks (weight 0 = sample dropped).
    curves   : (S, n_grid) spline evaluated on a dense log-age grid.
    fitted   : (S, n) spline evaluated at the sample ages (for dPSI).
    """
    B, di = spline_design(log_age, df=df)
    Yf = np.where(Wmask > 0, np.nan_to_num(Y), 0.0)
    W = Wmask.astype(float)
    BtWB = np.einsum("ni,nj,sn->sij", B, B, W, optimize=True)
    BtWy = np.einsum("ni,sn->si", B, W * Yf, optimize=True)
    coef = _batched_solve(BtWB, BtWy)
    lo = np.min(log_age); hi = np.max(log_age)
    grid = np.linspace(lo, hi, n_grid)
    Bg = spline_eval_design(di, grid)
    return coef @ Bg.T, grid, coef @ B.T


def pattern_stats_matrix(grid, curves):
    """Mazin's four statistics, vectorised (their exact version: no pruning)."""
    d = np.diff(curves, axis=1)
    xm = (grid[:-1] + grid[1:]) / 2.0
    pos = np.clip(d, 0, None); neg = np.clip(-d, 0, None)
    up = pos.sum(1); down = neg.sum(1)
    tot = up + down
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tot > 0, up / tot, np.nan)
        up_t = np.where(up > 0, (pos * xm).sum(1) / up, np.nan)
        down_t = np.where(down > 0, (neg * xm).sum(1) / down, np.nan)
    return {"up": up, "down": down, "ratio": ratio,
            "up_timing": up_t, "down_timing": down_t,
            "n_turns": (np.diff(np.sign(d), axis=1) != 0).sum(1)}


def classify_matrix(ratio, up_timing, down_timing, lo=0.3, hi=0.7):
    """up / down / up-down / down-up, same rules as devas_patterns.classify."""
    out = np.full(ratio.shape, "flat", dtype=object)
    fin = np.isfinite(ratio)
    out[fin & (ratio < lo)] = "down"
    out[fin & (ratio > hi)] = "up"
    mid = fin & (ratio >= lo) & (ratio <= hi)
    tfin = mid & np.isfinite(up_timing) & np.isfinite(down_timing)
    out[tfin & (up_timing < down_timing)] = "up-down"
    out[tfin & (up_timing >= down_timing)] = "down-up"
    # timings unavailable (a monotone curve inside the middle band): fall back
    nod = mid & ~tfin
    out[nod & (ratio >= 0.5)] = "up"
    out[nod & (ratio < 0.5)] = "down"
    return out


def amplitude_and_pattern(log_age, Y, Wmask, df=4, dpsi_min=0.2):
    """dPSI at the sample ages plus the pattern call, for a batch of sites."""
    curves, grid, fitted = spline_curves(log_age, Y, Wmask, df=df)
    fit_masked = np.where(Wmask > 0, fitted, np.nan)
    with np.errstate(invalid="ignore"):
        dpsi = np.nanmax(fit_masked, axis=1) - np.nanmin(fit_masked, axis=1)
    st = pattern_stats_matrix(grid, curves)
    st["dpsi"] = dpsi
    st["dpsi_pass"] = dpsi > dpsi_min
    st["pattern"] = classify_matrix(st["ratio"], st["up_timing"], st["down_timing"])
    return st, curves, grid


# ---------------------------------------------------------------- filters ---

def coverage_filters(N, SSE, min_total=10, frac_covered=0.6,
                     n_intermediate=4, lo=0.1, hi=0.9):
    """The paper's gating filters.

    Returns (mask, testable) where `mask` is the per-sample usable flag
    (Alpha+Beta >= min_total) and `testable` the per-site flag
    (covered in >= frac_covered of samples AND >= n_intermediate samples with
    SSE in [lo, hi]).
    """
    mask = N >= min_total
    frac = mask.mean(1)
    inter = (mask & (SSE >= lo) & (SSE <= hi)).sum(1)
    return mask, (frac >= frac_covered) & (inter >= n_intermediate)


# ----------------------------------------------------------- verification ---

def verify_against_statsmodels(I, N, log_age, n_check=200, seed=0, mode="sequential"):
    """Refit `n_check` random sites one at a time with statsmodels and report
    the largest absolute discrepancy in coefficients, dispersion and P values.
    """
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)
    S = I.shape[0]
    idx = rng.choice(S, size=min(n_check, S), replace=False)
    ours = devas_test(I[idx], N[idx], log_age, mode=mode)
    X = cubic_design(log_age, orthogonal=(mode == "sequential"))
    dif = {k: 0.0 for k in ["beta", "dispersion", "F_a", "F_a2", "F_a3",
                            "p_a", "p_a2", "p_a3"]}
    n_ok = 0
    for k in range(len(idx)):
        Ni, Ii = N[idx[k]], I[idx[k]]
        use = Ni > 0
        if use.sum() <= X.shape[1]:
            continue
        endog = np.column_stack([Ii[use], (Ni - Ii)[use]])
        try:
            fit = sm.GLM(endog, X[use], family=sm.families.Binomial()).fit(scale="X2")
        except Exception:
            continue
        n_ok += 1
        # R's quasi-binomial dispersion: Pearson X^2 / residual df, with the
        # N-weighted binomial variance. (statsmodels' .scale drops the
        # n_trials factor from the variance function, so it is not the
        # quantity R reports and not the one the paper's test uses.)
        phi = fit.pearson_chi2 / fit.df_resid
        dif["beta"] = max(dif["beta"], np.max(np.abs(fit.params - ours["beta"][k])))
        dif["dispersion"] = max(dif["dispersion"], abs(phi - ours["dispersion"][k]))
        for j, term in enumerate(TERMS, start=1):
            Xr = X[:, :j] if mode == "sequential" else np.delete(X, j, axis=1)
            fr = sm.GLM(endog, Xr[use], family=sm.families.Binomial()).fit()
            F = (fr.deviance - fit.deviance) / phi
            p = stats.f.sf(F, 1, int(use.sum()) - X.shape[1])
            dif["F_" + term] = max(dif["F_" + term], abs(F - ours["F_" + term][k]))
            dif["p_" + term] = max(dif["p_" + term], abs(p - ours["p_" + term][k]))
    dif["n_sites_checked"] = n_ok
    return dif
