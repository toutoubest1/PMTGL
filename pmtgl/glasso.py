"""Graphical lasso with an off-diagonal L1 penalty.

Block coordinate descent (Friedman et al., 2008) uses warm starts and a
positive-definite input. ADMM also accepts indefinite symmetric inputs."""
import numpy as np
from numba import njit


@njit(cache=True)
def _lasso_cd(W11, s12, beta, lam, tol, max_it):
    q = beta.shape[0]
    r = W11 @ beta
    for it in range(max_it):
        dmax = 0.0
        for k in range(q):
            old = beta[k]
            z = s12[k] - (r[k] - W11[k, k] * old)
            if z > lam:
                new = (z - lam) / W11[k, k]
            elif z < -lam:
                new = (z + lam) / W11[k, k]
            else:
                new = 0.0
            if new != old:
                d = new - old
                for l in range(q):
                    r[l] += W11[l, k] * d
                beta[k] = new
                if abs(d) > dmax:
                    dmax = abs(d)
        if dmax < tol:
            break
    return beta


@njit(cache=True)
def _glasso_bcd(S, lam, W, B, tol, max_sweeps):
    p = S.shape[0]
    idx = np.empty(p - 1, np.int64)
    thr = tol * np.mean(np.abs(S - np.diag(np.diag(S)))) + 1e-12
    for sweep in range(max_sweeps):
        W_old = W.copy()
        for j in range(p):
            c = 0
            for l in range(p):
                if l != j:
                    idx[c] = l
                    c += 1
            W11 = np.empty((p - 1, p - 1))
            s12 = np.empty(p - 1)
            for a in range(p - 1):
                s12[a] = S[idx[a], j]
                for b in range(p - 1):
                    W11[a, b] = W[idx[a], idx[b]]
            beta = B[:, j][idx].copy()
            beta = _lasso_cd(W11, s12, beta, lam, 1e-6, 1000)
            w12 = W11 @ beta
            for a in range(p - 1):
                W[idx[a], j] = w12[a]
                W[j, idx[a]] = w12[a]
                B[idx[a], j] = beta[a]
        if np.mean(np.abs(W - W_old)) < thr:
            break
    Th = np.zeros((p, p))
    for j in range(p):
        c = 0
        for l in range(p):
            if l != j:
                idx[c] = l
                c += 1
        bj = B[:, j][idx]
        wj = W[:, j][idx]
        t22 = 1.0 / (W[j, j] - wj @ bj)
        Th[j, j] = t22
        for a in range(p - 1):
            Th[idx[a], j] = -bj[a] * t22
    return (Th + Th.T) / 2.0, W, B


def glasso_path(S, lams, tol=1e-4, max_sweeps=100):
    """Solve glasso for each lambda in `lams` (any order; solved in decreasing
    order with warm starts). Returns list of precision matrices in input order."""
    S = np.ascontiguousarray((S + S.T) / 2, dtype=np.float64)
    p = S.shape[0]
    order = np.argsort(-np.asarray(lams))
    W = np.diag(np.diag(S)).astype(np.float64)
    B = np.zeros((p, p))
    out = [None] * len(lams)
    for i in order:
        lam = float(lams[i])
        W = W.copy()
        np.fill_diagonal(W, np.diag(S))
        Th, W, B = _glasso_bcd(S, lam, W, B, tol, max_sweeps)
        if not np.all(np.isfinite(Th)) or np.min(np.linalg.eigvalsh(Th)) <= 0:
            Th = glasso_admm(S, lam)
            if Th is None:
                Th = np.full((p, p), np.nan)
            W = np.diag(np.diag(S)).astype(np.float64)
            B = np.zeros((p, p))
        out[i] = Th
    return out


def glasso(S, lam):
    return glasso_path(S, [lam])[0]


def glasso_warm(S, lam, state=None, tol=1e-4, max_sweeps=100):
    """Single-lambda glasso with a warm start; state = (W, B) from a previous call
    on a similar S. Returns (Theta, state)."""
    S = np.ascontiguousarray((S + S.T) / 2, dtype=np.float64)
    p = S.shape[0]
    if state is None:
        W, B = np.diag(np.diag(S)).astype(np.float64), np.zeros((p, p))
    else:
        W, B = state[0].copy(), state[1].copy()
        np.fill_diagonal(W, np.diag(S))
    Th, W, B = _glasso_bcd(S, float(lam), W, B, tol, max_sweeps)
    if not np.all(np.isfinite(Th)) or np.linalg.eigvalsh(Th)[0] <= 0:
        return glasso_path(S, [lam])[0], None
    return Th, (W, B)


def glasso_admm(S, lam, rho=1.0, max_it=2000, tol=1e-6, Z0=None):
    """ADMM for min tr(S T) - logdet T + lam * sum_{i!=j} |T_ij|.
    Works for indefinite S whenever the minimum exists (Lemma S.9)."""
    S = (S + S.T) / 2
    p = S.shape[0]
    Z = np.diag(1.0 / np.diag(S)) if Z0 is None else Z0.copy()
    U = np.zeros((p, p))
    off = ~np.eye(p, dtype=bool)
    for it in range(max_it):
        w, Q = np.linalg.eigh(rho * (Z - U) - S)
        th = (w + np.sqrt(w ** 2 + 4 * rho)) / (2 * rho)
        T = (Q * th) @ Q.T
        V = T + U
        Zn = V.copy()
        Zn[off] = np.sign(V[off]) * np.maximum(np.abs(V[off]) - lam / rho, 0.0)
        U = U + T - Zn
        r = np.linalg.norm(T - Zn)
        s = rho * np.linalg.norm(Zn - Z)
        Z = Zn
        if not np.isfinite(r) or np.abs(Z).max() > 1e8:
            return None
        if r < tol * max(1, np.linalg.norm(T)) and s < tol * max(1, np.linalg.norm(U * rho)):
            break
    if it == max_it - 1:
        return None  # not converged: treated as non-existence (unbounded problem)
    Z = (Z + Z.T) / 2
    return Z


def glasso_path_admm(S, lams, max_it=3000):
    """ADMM along a decreasing path with warm starts; NaN matrix where the
    problem has no solution (ADMM fails to converge)."""
    S = (S + S.T) / 2
    p = S.shape[0]
    order = np.argsort(-np.asarray(lams))
    out = [None] * len(lams)
    Z0 = None
    for i in order:
        T = glasso_admm(S, float(lams[i]), max_it=max_it, Z0=Z0)
        if T is None:
            out[i] = np.full((p, p), np.nan)
        else:
            out[i] = T
            Z0 = T
    return out
