"""Method registry. Every method maps (X, lams) to a list over lams of
(Theta_original_scale, Theta_fit_scale, S_fit_scale): the last two are used
for the eBIC criterion."""
import time
import numpy as np
from . import covariance as C
from .glasso import glasso_path
from .psd import correct
from . import iterative as I

LAMS = np.round(np.geomspace(0.9, 0.02, 20), 5)


def plugin(cov_fn, psd="eigclip", **kw):
    def run(X, lams=LAMS):
        S, meta = cov_fn(X, **kw)
        S = correct(S, psd)
        d = np.sqrt(np.diag(S))
        R = S / np.outer(d, d)
        path = glasso_path(R, lams)
        return [(T / np.outer(d, d), T, R) for T in path]
    return run


def structure_preserving_mt(a_rule="th"):
    """Full MT, glasso on the transformed second moment (target Omega_z + A),
    then A subtracted from the diagonal (Proposition 1). Works for p > n."""
    def run(X, lams=LAMS):
        Z, m, s = C.standardize(X)
        p = Z.shape[1]
        if a_rule == "th":
            a = C.fullmt_th(X)[1]["a"]
        else:
            a = 0.5 ** (-2.0 / p) - 1.0
        M, _ = C.mt_weighted_second_moment(Z, np.full(p, a))
        M = correct(M, "eigclip")
        path = glasso_path(M, lams)
        out = []
        for T in path:
            Oz = T - a * np.eye(p)
            out.append((Oz / np.outer(s, s), T, M))
        return out
    return run


def composite(trunc=None, mass=0.5):
    def run(X, lams=LAMS):
        Sp, meta = C.composite_cov(X, trunc=trunc, mass=mass)
        Sp = correct(Sp, "eigclip")
        a0, s = meta["a0"], meta["s"]
        path = glasso_path(Sp, lams)
        return [((T - a0 * np.eye(len(T))) / np.outer(s, s), T, Sp) for T in path]
    return run


def iterative(fn, **kw):
    def run(X, lams=LAMS):
        Z, m, s = C.standardize(X)
        path = fn(X, lams, **kw)
        Sz = np.corrcoef(Z, rowvar=False)  # eBIC proxy for row-weighting methods
        return [(T, T * np.outer(s, s), Sz) for T in path]
    return run


METHODS = {
    "GLASSO": plugin(C.sample_cov),
    "Kendall-GL": plugin(C.kendall_cov),
    "Spearman-GL": plugin(C.spearman_cov),
    "GaussRank-GL": plugin(C.gaussrank_cov),
    "Wrap-GL": plugin(C.wrap_cov),
    "Winsor-GL": plugin(C.winsor_cov),
    "MT-GLASSO(TH)": plugin(C.fullmt_th),
    "MT-GLASSO(mass)": plugin(C.fullmt_mass),
    "MT-SP(TH)": structure_preserving_mt("th"),
    "tlasso": iterative(I.tlasso_path),
    "Trimmed-GL": iterative(I.trimmed_path),
    "DPD-GL": iterative(I.dpd_path),
    "PMT-fixed": plugin(C.pmt_cov, adaptive=False, a_const=0.5),
    "PMT": plugin(C.pmt_cov),
    "PMT-mt": plugin(C.pmt_cov, centre="mt"),
    "Composite": composite(),
    "Composite-trunc": composite(trunc=4.0),
}


def timed(name, X, lams=LAMS):
    t0 = time.perf_counter()
    out = METHODS[name](X, lams)
    return out, time.perf_counter() - t0
