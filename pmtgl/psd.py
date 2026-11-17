"""Positive-semidefinite corrections of a symmetric covariance input."""
import numpy as np


def min_eig(S):
    return float(np.linalg.eigvalsh((S + S.T) / 2)[0])


def eigclip(S, rel=1e-3):
    """Frobenius-nearest matrix with eigenvalues >= rel * mean(diag S)."""
    S = (S + S.T) / 2
    w, V = np.linalg.eigh(S)
    floor = rel * np.mean(np.diag(S))
    if w[0] >= floor:
        return S
    return (V * np.maximum(w, floor)) @ V.T


def _proj_l1_ball(v, r):
    """Euclidean projection of vector v onto {x: ||x||_1 <= r} (Duchi et al. 2008)."""
    if np.abs(v).sum() <= r:
        return v
    u = np.sort(np.abs(v))[::-1]
    cs = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, len(u) + 1) > (cs - r))[0][-1]
    theta = (cs[k] - r) / (k + 1.0)
    return np.sign(v) * np.maximum(np.abs(v) - theta, 0)


def maxnorm_proj(S, rel=1e-3, mu=None, max_it=500, tol=1e-6):
    """argmin_{W >= eps I} max_ij |W_ij - S_ij| by ADMM (cf. Datta & Zou 2017).

    Splitting: min ||Y||_max + I(W >= eps I)  s.t.  W - Y = S."""
    S = (S + S.T) / 2
    p = S.shape[0]
    eps = rel * np.mean(np.diag(S))
    if min_eig(S) >= eps:
        return S
    mu = mu or 1.0 / max(np.abs(S).max(), 1e-8) * 10
    W = eigclip(S, rel)
    Y = W - S
    L = np.zeros((p, p))
    for it in range(max_it):
        # W-step: projection of S + Y - L/mu onto {W >= eps I}
        A = S + Y - L / mu
        A = (A + A.T) / 2
        w, V = np.linalg.eigh(A)
        W = (V * np.maximum(w, eps)) @ V.T
        # Y-step: prox_{||.||_max / mu}(W - S + L/mu) = v - proj_{l1 ball(1/mu)}(v)
        B = W - S + L / mu
        v = B.ravel()
        Yn = (v - _proj_l1_ball(v, 1.0 / mu)).reshape(p, p)
        Yn = (Yn + Yn.T) / 2
        L = L + mu * (W - S - Yn)
        if np.abs(W - S - Yn).max() < tol and np.abs(Yn - Y).max() < tol:
            Y = Yn
            break
        Y = Yn
    return (W + W.T) / 2


def nearcorr(S, max_it=200, tol=1e-7, rel=1e-3):
    """Higham (2002) alternating projections to the nearest correlation matrix
    of D^{-1/2} S D^{-1/2}, rescaled back; eigenvalues floored at rel."""
    d = np.sqrt(np.diag(S))
    R = S / np.outer(d, d)
    Y = R.copy()
    dS = np.zeros_like(R)
    for _ in range(max_it):
        Rk = Y - dS
        w, V = np.linalg.eigh((Rk + Rk.T) / 2)
        X = (V * np.maximum(w, rel)) @ V.T
        dS = X - Rk
        Yn = X.copy()
        np.fill_diagonal(Yn, 1.0)
        if np.linalg.norm(Yn - Y) < tol * max(1, np.linalg.norm(Y)):
            Y = Yn
            break
        Y = Yn
    w, V = np.linalg.eigh(Y)
    Y = (V * np.maximum(w, rel)) @ V.T
    return Y * np.outer(d, d)


def correct(S, method="eigclip"):
    if method == "none":
        return (S + S.T) / 2
    if method == "eigclip":
        return eigclip(S, 1e-3)
    if method == "eigclip05":
        return eigclip(S, 0.05)
    if method == "maxnorm":
        return maxnorm_proj(S, 1e-3)
    if method == "nearcorr":
        return nearcorr(S)
    raise ValueError(method)
