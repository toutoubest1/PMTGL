"""Numerical checks of population identities and solver behavior."""
import numpy as np
import pytest
from pmtgl import covariance as C
from pmtgl.sim import make_precision
from pmtgl.glasso import glasso_path, glasso_admm

rng = np.random.default_rng(2026)
N = 400_000


def _gauss(p, kind="er2"):
    Om, Sig = make_precision(p, kind, rng=1)
    X = rng.normal(size=(N, p)) @ np.linalg.cholesky(Sig).T
    return Om, Sig, X


def test_prop1_full_transform():
    Om, Sig, X = _gauss(5)
    a = np.array([0.3, 0.0, 1.0, 0.5, 0.2])
    w = np.exp(-0.5 * (X ** 2 @ a))
    Cw = (X * w[:, None]).T @ X / w.sum()
    assert np.abs(np.linalg.inv(Cw) - (Om + np.diag(a))).max() < 0.03 * np.abs(Om).max()
    assert abs(w.mean() - np.linalg.det(np.eye(5) + Sig @ np.diag(a)) ** -0.5) < 3e-3


def test_prop3_pairwise_backtransform_known_centre():
    Om, Sig, X = _gauss(6)
    a = np.array([0.5, 1.0, 0.2, 0.7, 0.3, 0.1])
    R, v, nf = C.pmt_from_standardized(X, a)
    assert nf == 0
    assert np.abs(R - Sig).max() < 0.01
    assert np.abs(v - 1).max() < 0.02


def test_prop4_scale_equivariance():
    Om, Sig, X = _gauss(4)
    a = np.full(4, 0.5)
    for c in (0.5, 2.0):
        R, v, _ = C.pmt_from_standardized(X / c, a)
        assert np.abs(R - Sig).max() < 0.01
        assert np.abs(v * c ** 2 - 1).max() < 0.03


def test_prop5_centre_error_formula():
    # bivariate: C(mu + delta) back-transformed = Sigma + delta delta^T / (1 - kappa + kappa_A)
    rho = 0.6
    S2 = np.array([[1, rho], [rho, 1]])
    X = rng.normal(size=(2 * N, 2)) @ np.linalg.cholesky(S2).T
    a = np.array([0.8, 0.4])
    delta = np.array([0.3, -0.2])
    Z = X - delta
    w = np.exp(-0.5 * (Z ** 2 @ a))
    Cw = (Z * w[:, None]).T @ Z / w.sum()
    T = np.linalg.inv(np.linalg.inv(Cw) - np.diag(a))
    O = np.linalg.inv(S2)
    SA = np.linalg.inv(O + np.diag(a))
    k, kA = delta @ O @ delta, delta @ O @ SA @ O @ delta
    pred = S2 + np.outer(delta, delta) / (1 - k + kA)
    assert np.abs(T - pred).max() < 0.01


def test_prop6_efficiency_closed_form():
    from pmtgl.theory import asy_var_corr
    for rho, aj, ak in [(0.0, 0.5, 1.0), (0.5, 0.3, 0.3), (-0.7, 1.0, 0.2)]:
        S2 = np.array([[1, rho], [rho, 1]])
        L = np.linalg.cholesky(S2)
        reps, n = 3000, 300
        rs = []
        for _ in range(reps):
            Z = rng.normal(size=(n, 2)) @ L.T
            R, _, _ = C.pmt_from_standardized(Z, np.array([aj, ak]))
            rs.append(R[0, 1])
        mc = n * np.var(rs)
        th = asy_var_corr(rho, aj, ak)
        assert abs(mc / th - 1) < 0.1, (rho, mc, th)


def test_composite_identity():
    Om, Sig, X = _gauss(6)
    a = np.full(6, 0.5)
    a0 = 0.2
    v = np.exp(-0.5 * a0 * (X ** 2).sum(1))
    R, var, _ = C.pmt_from_standardized(X, a, row_weight=v)
    Sp = R * np.sqrt(np.outer(var, var))
    target = np.linalg.inv(Om + a0 * np.eye(6))
    assert np.abs(Sp - target).max() < 0.01


def test_glasso_indefinite_input():
    # Lemma S.9: glasso exists for indefinite S iff a PD W with diag(W)=diag(S)
    # lies within lambda of S off the diagonal.
    S = np.array([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1.0]])
    assert np.linalg.eigvalsh(S)[0] < 0
    assert glasso_admm(S, 0.05) is None
    T = glasso_admm(S, 0.6)
    assert T is not None and np.linalg.eigvalsh(T)[0] > 0
