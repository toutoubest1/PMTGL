"""Gaussian-model variance, influence and sensitivity for PMT."""
import numpy as np


def _T_and_G(rho, aj, ak):
    R = np.array([[1.0, rho], [rho, 1.0]])
    A = np.diag([aj, ak])
    Ci = np.linalg.inv(R) + A                 # C^{-1}
    C = np.linalg.inv(Ci)
    T = R
    r = rho
    H = np.array([[-r / 2, 0.5], [0.5, -r / 2]])  # dr = tr(H dT) at T = R (unit variances)
    G = Ci @ T @ H @ T @ Ci                   # dr = tr(G dC)
    return R, A, C, G


def asy_var_corr(rho, aj, ak):
    """n * asymptotic variance of the PMT correlation at the Gaussian model.

    V = det(I+RA) / det(I+2RA)^{1/2} * [ 2 tr((G C2)^2) + tr(G (C2 - C))^2 ],
    C2 = (R^{-1} + 2A)^{-1}; valid with median/MAD plug-ins (Proposition 7)."""
    R, A, C, G = _T_and_G(rho, aj, ak)
    C2 = np.linalg.inv(np.linalg.inv(R) + 2 * A)
    pref = np.linalg.det(np.eye(2) + R @ A) / np.sqrt(np.linalg.det(np.eye(2) + 2 * R @ A))
    M = G @ C2
    return pref * (2 * np.trace(M @ M) + np.trace(G @ (C2 - C)) ** 2)


def are_vs_pearson(rho, aj, ak):
    return (1 - rho ** 2) ** 2 / asy_var_corr(rho, aj, ak)


def influence_corr(z, rho, aj, ak):
    """Influence function of the PMT correlation at z (2-vector), Gaussian model."""
    R, A, C, G = _T_and_G(rho, aj, ak)
    D = np.linalg.det(np.eye(2) + R @ A) ** -0.5
    w = np.exp(-0.5 * (aj * z[0] ** 2 + ak * z[1] ** 2))
    return w * np.trace(G @ (np.outer(z, z) - C)) / D


def ges_corr(rho, aj, ak, L=12.0, m=601):
    g = np.linspace(-L, L, m)
    best = 0.0
    for x in g:
        vals = [abs(influence_corr(np.array([x, y]), rho, aj, ak)) for y in g[::4]]
        best = max(best, max(vals))
    return best
