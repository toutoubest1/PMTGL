"""Row-weighting baselines that re-estimate weights for every lambda.
All operate on median/MAD-standardized data, return precision matrices on the
original scale, and use warm-started glasso."""
import numpy as np
from .glasso import glasso_path, glasso_warm
from .covariance import standardize
from .psd import eigclip


def _path_iter(X, lams, weight_fn, scale=1.0, iters=5):
    Z, m, s = standardize(X)
    n, p = Z.shape
    out = []
    mu = np.zeros(p)
    S = np.cov(Z, rowvar=False)
    Th = np.linalg.inv(eigclip(S, 1e-2))
    state = None
    for lam in sorted(lams, reverse=True):
        for _ in range(iters):
            D = Z - mu
            d2 = np.einsum("ij,jk,ik->i", D, Th, D)
            w = weight_fn(d2, p)
            mu = (w @ Z) / w.sum()
            D = Z - mu
            S = scale * (D * w[:, None]).T @ D / w.sum()
            Th_new, state = glasso_warm(eigclip(S, 1e-3), lam, state)
            if not np.all(np.isfinite(Th_new)):
                break
            if np.abs(Th_new - Th).max() < 1e-4 * np.abs(Th).max():
                Th = Th_new
                break
            Th = Th_new
        out.append((lam, Th / np.outer(s, s)))
    d = dict(out)
    return [d[l] for l in lams], S


def tlasso_path(X, lams, nu=3.0, iters=5):
    """Finegold & Drton (2011) EM for the multivariate t (nu fixed); E-step
    weights (nu+p)/(nu+d^2), M-step glasso on the weighted covariance."""
    f = lambda d2, p: (nu + p) / (nu + d2)
    return _path_iter(X, lams, f, 1.0, iters)[0]


def trimmed_path(X, lams, h=0.8, iters=5):
    """Trimmed graphical lasso (Yang & Lozano 2015): alternate glasso on the
    h-fraction of rows with smallest Mahalanobis distance."""
    def f(d2, p):
        k = int(np.ceil(h * len(d2)))
        w = np.zeros(len(d2))
        w[np.argsort(d2)[:k]] = 1.0
        return w
    return _path_iter(X, lams, f, 1.0, iters)[0]


def dpd_path(X, lams, beta=None, mass=0.5, iters=5):
    """Density-power-divergence glasso (Sun & Li 2012 type): weights
    exp(-beta d^2/2), Fisher-consistency factor (1+beta). beta set by a
    retained-mass budget (1+beta)^{-p/2} = mass unless given (our choice)."""
    p = X.shape[1]
    b = beta if beta is not None else mass ** (-2.0 / p) - 1.0
    f = lambda d2, p: np.exp(-0.5 * b * (d2 - d2.min()))
    return _path_iter(X, lams, f, 1.0 + b, iters)[0]
