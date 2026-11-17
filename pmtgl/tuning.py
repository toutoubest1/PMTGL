"""Penalty selection using the training sample.

Plug-in rules use the PSD-corrected correlation matrix supplied to glasso."""
import numpy as np
from . import covariance as C
from .psd import correct
from .glasso import glasso_path
from .sim import support

LAMF = np.round(np.geomspace(0.8, 0.05, 41), 5)      # fine grid (ratio 0.933)

COV = {
    "GLASSO": (C.sample_cov, {}),
    "Kendall-GL": (C.kendall_cov, {}),
    "GaussRank-GL": (C.gaussrank_cov, {}),
    "Wrap-GL": (C.wrap_cov, {}),
    "Winsor-GL": (C.winsor_cov, {}),
    "MT-GLASSO(TH)": (C.fullmt_th, {}),
    "MT-GLASSO(mass)": (C.fullmt_mass, {}),
    "PMT-fixed": (C.pmt_cov, dict(adaptive=False, a_const=0.5)),
    "PMT": (C.pmt_cov, {}),
    "PMT-mt": (C.pmt_cov, dict(centre="mt")),
}


def plugin_input(name, X, psd="eigclip"):
    """Return (R, d): glasso input on the correlation scale and the scales d
    such that Theta_original = Theta_R / outer(d, d)."""
    fn, kw = COV[name]
    S, _ = fn(X, **kw)
    S = correct(S, psd)
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d), d


def off_max(R):
    iu = np.triu_indices(len(R), 1)
    return float(np.max(np.abs(R[iu])))


# permutation null
def perm_null(name, X, B=100, rng=None):
    """Null statistics from independent column permutations.

    Permuting each column separately keeps every marginal (outlying cells
    included) and destroys all dependence. Returns the observed input R, the
    B null maxima T*_b = max_{j<k} |R*_jk| and the pooled null |R*_jk|."""
    rng = np.random.default_rng(rng)
    R, _ = plugin_input(name, X)
    n, p = X.shape
    iu = np.triu_indices(p, 1)
    T, pooled = np.empty(B), []
    for b in range(B):
        Xp = np.column_stack([X[rng.permutation(n), j] for j in range(p)])
        Rb, _ = plugin_input(name, Xp)
        a = np.abs(Rb[iu])
        T[b] = a.max()
        pooled.append(np.sort(a))
    return R, T, np.concatenate(pooled)


def lam_perm_fwer(R, T, alpha=0.05):
    """Smallest lambda that is >= the permutation (1-alpha) quantile of the
    null maximum, computed with the observed statistic included (so that the
    global-null guarantee of Proposition S.x holds for finite B)."""
    allT = np.sort(np.concatenate([[off_max(R)], T]))
    k = int(np.ceil((1 - alpha) * len(allT))) - 1
    return float(allT[k])


def lam_perm_fdr(R, pooled, B, q=0.1, grid=None):
    """Permutation FDR estimate for marginal screening |R_jk| > lambda:
    FDR(lam) = mean_b #{|R*_b,jk| > lam} / max(1, #{|R_jk| > lam});
    returns the smallest lambda (largest graph) with FDR(lam) <= q."""
    iu = np.triu_indices(len(R), 1)
    obs = np.sort(np.abs(R[iu]))
    grid = np.sort(obs)[::-1] if grid is None else np.sort(grid)[::-1]
    best = None
    for lam in grid:                       # decreasing lambda
        nnull = (len(pooled) - np.searchsorted(pooled, lam, side="right")) / B
        nobs = len(obs) - np.searchsorted(obs, lam, side="right")
        if nobs == 0:
            continue
        if nnull / nobs <= q:
            best = lam
        else:
            break                          # first crossing (monotone scan)
    return float(best) if best is not None else float(obs[-1])


# StARS
def stars_lambda(name, X, lams=LAMF, B=20, b=None, beta=0.05, rng=None, fit=None):
    """StARS (Liu, Roeder & Wasserman 2010). `fit(X, lams)` returns a list of
    precision matrices; by default the plug-in path of `name`."""
    rng = np.random.default_rng(rng)
    n = len(X)
    b = b or (int(10 * np.sqrt(n)) if n > 144 else int(0.8 * n))
    if fit is None:
        def fit(Xs, ls):
            R, _ = plugin_input(name, Xs)
            return glasso_path(R, ls)
    freq = None
    npair = X.shape[1] * (X.shape[1] - 1) // 2
    for _ in range(B):
        idx = rng.choice(n, b, replace=False)
        E = np.array([support(T) if T is not None and np.all(np.isfinite(T))
                      else np.zeros(npair, bool)
                      for T in fit(X[idx], lams)], float)
        freq = E if freq is None else freq + E
    th = freq / B
    D = (2 * th * (1 - th)).mean(1)        # lams decreasing
    Dm = np.maximum.accumulate(D)
    ok = np.nonzero(Dm <= beta)[0]
    return float(lams[ok[-1]] if len(ok) else lams[0]), D


# information criteria and CV
def ic_lambda(R, n, path, lams, gamma=0.0):
    vals = []
    for T in path:
        sign, ld = np.linalg.slogdet(T)
        k = int(support(T).sum())
        vals.append(np.inf if sign <= 0 else
                    n * (np.trace(R @ T) - ld) + k * np.log(n) + 4 * gamma * k * np.log(len(T)))
    return float(lams[int(np.argmin(vals))])


def cv_lambda(name, X, lams=LAMF, K=5, rng=None):
    """K-fold CV with the Gaussian loss tr(R_val Theta) - logdet Theta, where
    R_val is the method's own (robust) input computed on the validation fold."""
    rng = np.random.default_rng(rng)
    n = len(X)
    folds = np.array_split(rng.permutation(n), K)
    loss = np.zeros(len(lams))
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        R, _ = plugin_input(name, X[tr])
        Rv, _ = plugin_input(name, X[f])
        for i, T in enumerate(glasso_path(R, lams)):
            sign, ld = np.linalg.slogdet(T)
            loss[i] += np.trace(Rv @ T) - ld if sign > 0 else np.inf
    return float(lams[int(np.argmin(loss))])
