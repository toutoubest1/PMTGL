"""Checks for dual feasibility, MT location, adaptive strength and permutation tuning."""
import numpy as np
from pmtgl.covariance import DEFAULT_GRID, eff_factor, ges_factor, select_strengths, mt_location, standardize, pmt_cov
from pmtgl.glasso import glasso_path, glasso_admm
from pmtgl.psd import correct
from pmtgl.sim import make_precision, generate, support


def test_rule_never_above_0459():
    g, G2 = eff_factor(DEFAULT_GRID), ges_factor(DEFAULT_GRID)
    i = int(np.argmin(G2))
    assert np.isclose(DEFAULT_GRID[i], 0.4593)
    for e in np.geomspace(1e-4, 1, 200):
        for n in (50, 200, 5000):
            assert DEFAULT_GRID[np.argmin(g / n + e ** 2 * G2)] <= 0.4593
    a0 = DEFAULT_GRID[0]
    kap = min((eff_factor(a) - eff_factor(a0)) / (ges_factor(a0) - ges_factor(a))
              for a in DEFAULT_GRID[1:] if ges_factor(a) < ges_factor(a0))
    assert abs(kap - 1.144e-3) < 1e-6


def test_meanshift_ascent_and_root():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=80), rng.uniform(5, 9, 20)])
    s, a = 1.3, 0.3
    c = a / (2 * s * s)
    f = lambda m: np.exp(-c * (x - m) ** 2).sum()
    m, fs = np.median(x), []
    for _ in range(200):
        w = np.exp(-c * (x - m) ** 2)
        mn = (w * x).sum() / w.sum()
        assert f(mn) - f(m) >= c * f(m) * (mn - m) ** 2 - 1e-12
        m = mn
    g = np.mean(((x - m) / s) * np.exp(-0.5 * a * ((x - m) / s) ** 2))
    assert abs(g) < 1e-10


def test_mt_location_safeguard_keeps_median_when_far():
    X = np.array([[0.0], [0.1], [-0.1], [0.05], [10.0], [10.1], [9.9], [10.05], [10.2]])
    m0 = np.median(X, 0)
    m = mt_location(X, np.array([0.05]), m0, np.array([0.1]))
    assert np.all(np.abs(m - m0) <= 0.1 / (2 * np.sqrt(0.1)) + 1e-12)


def test_permutation_rule_part_i():
    rng = np.random.default_rng(1)
    Om, Sig = make_precision(30, "er2", rng=rng)
    X = generate(60, Sig, "hetcell", rng)
    S, _ = pmt_cov(X)
    S = correct(S, "eigclip")
    d = np.sqrt(np.diag(S)); R = S / np.outer(d, d)
    T = np.abs(R - np.eye(30)).max()
    above, below = glasso_path(R, [T * 1.0001, T * 0.98])
    assert support(above).sum() == 0 and support(below).sum() > 0


def test_existence_certificate():
    rng = np.random.default_rng(2)
    Om, Sig = make_precision(60, "er2", rng=rng)
    X = generate(40, Sig, "homcell", rng)
    S, _ = pmt_cov(X)
    d = np.sqrt(np.diag(S)); R = S / np.outer(d, d)
    Sc = correct(S, "eigclip"); dc = np.sqrt(np.diag(Sc)); Rc = Sc / np.outer(dc, dc)
    eta = np.abs(Rc - R)[~np.eye(60, dtype=bool)].max()
    assert np.linalg.eigvalsh(R)[0] < 0          # indefinite input
    Th = glasso_admm(R, eta * 1.05 + 1e-3, max_it=8000)
    assert Th is not None and np.linalg.eigvalsh(Th)[0] > 0


def test_remainder_bound_lemma():
    rng = np.random.default_rng(3)
    Om, Sig = make_precision(25, "er2", rng=rng)
    d = Om.copy()
    S = np.abs(Om) > 1e-12
    deg = S.sum(1).max()
    kR = np.abs(Sig).sum(1).max()
    for _ in range(50):
        D = rng.normal(size=Om.shape) * S
        D = (D + D.T) / 2
        D *= (1 / (3 * kR * deg)) / np.abs(D).max() * rng.uniform(0.1, 1)
        Rm = np.linalg.inv(Om + D) - Sig + Sig @ D @ Sig
        assert np.abs(Rm).max() <= 1.5 * deg * kR ** 3 * np.abs(D).max() ** 2 + 1e-12
