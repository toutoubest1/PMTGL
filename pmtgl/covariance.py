"""Covariance estimators for graphical lasso.

Each estimator returns the covariance on the original data scale and a
metadata dictionary. Use pmtgl.psd when a positive-definite input is needed."""
import numpy as np
from numba import njit
from scipy.stats import norm, rankdata

MAD_C = 1.482602218505602


def robust_center_scale(X):
    m = np.median(X, axis=0)
    s = MAD_C * np.median(np.abs(X - m), axis=0)
    s = np.where(s <= 1e-12, 1.0, s)
    return m, s


def standardize(X):
    m, s = robust_center_scale(X)
    return (X - m) / s, m, s


# classical
def sample_cov(X, **kw):
    return np.cov(X, rowvar=False), {}


# rank based
def kendall_matrix(X, chunk=40):
    n, p = X.shape
    R = np.zeros((p, p))
    for i0 in range(0, n, chunk):
        D = np.sign(X[i0:i0 + chunk, None, :] - X[None, :, :]).reshape(-1, p)
        R += D.T @ D
    R /= n * (n - 1)
    np.fill_diagonal(R, 1.0)
    return R


def kendall_cov(X, **kw):
    _, s = robust_center_scale(X)
    R = np.sin(np.pi / 2 * kendall_matrix(X))
    return R * np.outer(s, s), {}


def spearman_cov(X, **kw):
    _, s = robust_center_scale(X)
    Rk = np.apply_along_axis(rankdata, 0, X)
    R = np.corrcoef(Rk, rowvar=False)
    R = 2 * np.sin(np.pi / 6 * R)
    np.fill_diagonal(R, 1.0)
    return R * np.outer(s, s), {}


def gaussrank_cov(X, **kw):
    n = X.shape[0]
    _, s = robust_center_scale(X)
    Q = norm.ppf(np.apply_along_axis(rankdata, 0, X) / (n + 1))
    R = (Q.T @ Q) / np.sum(norm.ppf(np.arange(1, n + 1) / (n + 1)) ** 2)
    np.fill_diagonal(R, 1.0)
    return R * np.outer(s, s), {}


# wrapping
_B, _C, _Q1, _Q2 = 1.5, 4.0, 1.540793, 0.8622731


def wrap_psi(z):
    az = np.abs(z)
    out = np.where(az <= _B, z, _Q1 * np.tanh(_Q2 * (_C - az)) * np.sign(z))
    return np.where(az > _C, 0.0, out)


def wrap_cov(X, **kw):
    Z, m, s = standardize(X)
    W = wrap_psi(Z)
    R = np.nan_to_num(np.corrcoef(W, rowvar=False))
    np.fill_diagonal(R, 1.0)
    return R * np.outer(s, s), {}


# Winsorization
@njit(cache=True)
def _winsor_pairs(Z, c1, cbiv):
    n, p = Z.shape
    R = np.eye(p)
    for j in range(p):
        for k in range(j + 1, p):
            u = Z[:, j]
            v = Z[:, k]
            same = (u * v) >= 0
            n1 = same.sum()
            n2 = n - n1
            if n1 >= n2:
                cmaj, cmin = c1, c1 * n2 / max(n1, 1)
                maj = same
            else:
                cmaj, cmin = c1, c1 * n1 / max(n2, 1)
                maj = ~same
            uu = np.empty(n)
            vv = np.empty(n)
            for i in range(n):
                c = cmaj if maj[i] else cmin
                uu[i] = min(max(u[i], -c), c)
                vv[i] = min(max(v[i], -c), c)
            r0 = np.corrcoef(uu, vv)[0, 1]
            r0 = min(max(r0, -0.95), 0.95)
            det = 1 - r0 * r0
            for i in range(n):
                d2 = (u[i] ** 2 - 2 * r0 * u[i] * v[i] + v[i] ** 2) / det
                f = 1.0 if d2 <= cbiv else np.sqrt(cbiv / d2)
                uu[i] = u[i] * f
                vv[i] = v[i] * f
            r = np.corrcoef(uu, vv)[0, 1]
            R[j, k] = r
            R[k, j] = r
    return R


def winsor_cov(X, c1=2.0, cbiv=5.99, **kw):
    Z, m, s = standardize(X)
    R = _winsor_pairs(np.ascontiguousarray(Z), c1, cbiv)
    return R * np.outer(s, s), {}


# PMT
def eff_factor(a):
    """g(a) = ((1+a)^2/(1+2a))^{3/2}: per-variable asymptotic-variance factor
    of the PMT correlation at rho = 0 (Proposition 6)."""
    a = np.asarray(a, float)
    return ((1 + a) ** 2 / (1 + 2 * a)) ** 1.5


def ges_factor(a):
    """gamma(a)^2 = (1+a)^3/(a e): per-variable factor of the squared
    gross-error sensitivity of the PMT correlation at rho = 0 (Proposition 7)."""
    a = np.asarray(a, float)
    return (1 + a) ** 3 / (a * np.e)


DEFAULT_GRID = np.round(np.geomspace(0.05, 3.0, 25), 4)


def excess_tail_rate(Z, cut=3.5):
    return np.clip(np.mean(np.abs(Z) > cut, axis=0) - 2 * norm.sf(cut), 0.0, 1.0)


def select_strengths(Z, grid=DEFAULT_GRID, cut=3.5):
    """a_j = argmin_{a in grid} g(a)/n + eps_j^2 gamma(a)^2 (Sec. 3.3)."""
    n = Z.shape[0]
    eps = excess_tail_rate(Z, cut)
    risk = eff_factor(grid)[None, :] / n + eps[:, None] ** 2 * ges_factor(grid)[None, :]
    return grid[np.argmin(risk, axis=1)], eps


def mt_location(X, a, m0, s, iters=100, tol=1e-8):
    """Estimate each location by Gaussian mean shift from the median.

    Keep the median if the iterate fails to converge or leaves the interval
    |m - m0| <= s / (2 sqrt(2 a))."""
    m = m0.copy()
    conv = np.zeros(len(m0), bool)
    for _ in range(iters):
        W = np.exp(-0.5 * a[None, :] * ((X - m) / s) ** 2)
        m_new = (W * X).sum(0) / W.sum(0)
        conv = np.abs(m_new - m) / s < tol
        m = m_new
        if conv.all():
            break
    bad = ~conv | (np.abs(m - m0) / s > 1.0 / (2.0 * np.sqrt(2.0 * a)))
    return np.where(bad, m0, m)


def pmt_from_standardized(Z, a, row_weight=None, clip=0.999):
    """Core PMT map on standardized data Z (n x p) with strengths a (p,).

    Returns (R_hat, v_hat, n_fail): pairwise back-transformed correlations,
    univariate back-transformed variances, number of failed pairs.
    If row_weight (n,) is given, every moment is additionally weighted by it
    (used by the composite estimator)."""
    E = np.exp(-0.5 * a[None, :] * Z ** 2)
    if row_weight is not None:
        Er = E * row_weight[:, None]
    else:
        Er = E
    EZ, EZ2 = E * Z, E * Z ** 2
    S0 = Er.T @ E
    c11 = ((EZ2 * (row_weight[:, None] if row_weight is not None else 1)).T @ E) / S0
    c22 = c11.T
    c12 = ((EZ * (row_weight[:, None] if row_weight is not None else 1)).T @ EZ) / S0
    D = np.maximum(c11 * c22 - c12 ** 2, 1e-300)
    m11 = c22 / D - a[:, None]
    m22 = c11 / D - a[None, :]
    m12 = -c12 / D
    ok = (m11 > 1e-10) & (m22 > 1e-10)
    with np.errstate(all='ignore'):
        R = np.where(ok, -m12 / np.sqrt(np.abs(m11 * m22)), 0.0)
    R = np.clip(R, -clip, clip)
    np.fill_diagonal(R, 1.0)
    # univariate back-transform
    if row_weight is not None:
        num = (row_weight[:, None] * E * Z ** 2).sum(0)
        den = (row_weight[:, None] * E).sum(0)
    else:
        num, den = (E * Z ** 2).sum(0), E.sum(0)
    c = num / den
    inv = 1.0 / np.maximum(c, 1e-300) - a
    v = 1.0 / np.maximum(inv, 1e-2)
    nf = int((~ok).sum() - (~np.diag(ok)).sum()) // 2
    return R, v, nf


def pmt_cov(X, a=None, adaptive=True, a_const=0.5, centre="median",
            grid=DEFAULT_GRID, cut=3.5, **kw):
    """Estimate PMT covariance on the original scale.

    Use centre="median" for median centering or centre="mt" for MT location."""
    Z0, m, s = standardize(X)
    p = X.shape[1]
    eps = None
    if a is None:
        if adaptive:
            a, eps = select_strengths(Z0, grid, cut)
        else:
            a = np.full(p, float(a_const))
    a = np.asarray(a, float)
    if centre == "mt":
        m = mt_location(X, a, m, s)
    Z = (X - m) / s
    R, v, nf = pmt_from_standardized(Z, a)
    sd = s * np.sqrt(v)
    return R * np.outer(sd, sd), {"a": a, "eps": eps, "n_fail": nf, "m": m, "s": s}


# full MT
def mt_weighted_second_moment(Z, a_vec):
    w = np.exp(-0.5 * (Z ** 2 @ a_vec))
    w = w / w.sum()
    return (Z * w[:, None]).T @ Z, w


def fullmt_th(X, c=5.0, iters=100, **kw):
    """Full-dimensional Gaussian MT-GLASSO input with the Fisher-information
    rule of Todros & Hero (2015, Sec. V), translated to real data.

    MT function u(z) = exp(-||z||^2 / (2 tau^2)) on median/MAD-standardized z.
    tau^2 solves tau^2 = (c+1) lambda_max(C(tau)), C the empirical MT second
    moment; at the Gaussian model this gives tau^2 = c lambda_max(R) and a
    transform-domain Fisher-information ratio >= (c/(1+c))^2. Back-transform
    Sigma_hat = tau^2 C (tau^2 I - C)^{-1}, well defined for p > n.
    NOTE: reconstructed from Todros & Hero (2015); the SSP 2025 MT-GLASSO paper
    describes its rule only as limiting Fisher-information loss."""
    Z, m, s = standardize(X)
    p = Z.shape[1]
    tau2 = 25.0 * np.mean(np.ones(p))  # tau_0 = 5 * rms(MAD scale) = 5 on the standardized scale
    for _ in range(iters):
        C, w = mt_weighted_second_moment(Z, np.full(p, 1.0 / tau2))
        new = (c + 1) * np.linalg.eigvalsh(C)[-1]
        if abs(new - tau2) < 1e-6 * tau2:
            tau2 = new
            break
        tau2 = new
    C, w = mt_weighted_second_moment(Z, np.full(p, 1.0 / tau2))
    Sz = tau2 * C @ np.linalg.inv(tau2 * np.eye(p) - C)
    Sz = (Sz + Sz.T) / 2
    return Sz * np.outer(s, s), {"tau2": tau2, "a": 1.0 / tau2, "ess": 1.0 / np.sum(w ** 2)}


def fullmt_mass(X, mass=0.5, **kw):
    """Scalar full MT with strength a = mass^{-2/p} - 1 (retained clean-model
    mass `mass` under independence), back-transformed as in fullmt_th."""
    Z, m, s = standardize(X)
    p = Z.shape[1]
    a = mass ** (-2.0 / p) - 1.0
    C, w = mt_weighted_second_moment(Z, np.full(p, a))
    tau2 = 1.0 / a
    Sz = tau2 * C @ np.linalg.inv(tau2 * np.eye(p) - C)
    Sz = (Sz + Sz.T) / 2
    return Sz * np.outer(s, s), {"a": a, "ess": 1.0 / np.sum(w ** 2)}


# composite
def composite_cov(X, a0=None, mass=0.5, trunc=None, **kw):
    """Composite row x cell measure transform (Sec. 7, extension).

    Row tilt v_i = exp(-a0/2 sum_l z_il^2 [|z_il| <= trunc]) times the pairwise
    PMT tilt. With trunc=None the row tilt is Gaussian and, at the clean model,
    the pairwise back-transform returns Sigma' = (Omega_z + a0 I)^{-1} exactly
    (Proposition 12); the graph is recovered by glasso on Sigma' followed by
    subtracting a0 from the diagonal. Returns the standardized Sigma' and meta
    needed for that post-processing."""
    Z0, m, s = standardize(X)
    n, p = Z0.shape
    a, eps = select_strengths(Z0)
    if a0 is None:
        a0 = mass ** (-2.0 / p) - 1.0
    Q = Z0 ** 2 if trunc is None else np.where(np.abs(Z0) <= trunc, Z0 ** 2, 0.0)
    v = np.exp(-0.5 * a0 * Q.sum(1))
    v = v / v.max()
    R, var, nf = pmt_from_standardized(Z0, a, row_weight=v)
    sd = np.sqrt(var)
    return R * np.outer(sd, sd), {"a": a, "a0": a0, "s": s, "m": m}
