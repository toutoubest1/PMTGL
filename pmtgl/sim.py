"""Graphs, contamination scenarios and evaluation metrics."""
import numpy as np


# graphs
def make_precision(p, kind="er2", rng=None):
    """Sparse precision matrix rescaled so that Sigma has unit diagonal.
    er2/er6: Erdos-Renyi with expected degree 2 / 6; band: bandwidth 2;
    hub: hubs every 10 nodes, each linked to the next 9 nodes."""
    rng = np.random.default_rng(rng)
    mask = np.zeros((p, p), bool)
    if kind.startswith("er"):
        deg = float(kind[2:])
        mask = np.triu(rng.random((p, p)) < deg / (p - 1), 1)
    elif kind == "band":
        for b in (1, 2):
            mask |= np.eye(p, k=b, dtype=bool)
    elif kind == "hub":
        for h in range(0, p, 10):
            mask[h, h + 1:min(h + 10, p)] = True
    else:
        raise ValueError(kind)
    vals = rng.uniform(0.4, 0.8, (p, p)) * rng.choice([-1, 1], (p, p))
    Om = np.where(mask, vals, 0.0)
    Om = Om + Om.T
    Om += (abs(np.linalg.eigvalsh(Om)[0]) + 0.2) * np.eye(p)
    Sig = np.linalg.inv(Om)
    d = np.sqrt(np.diag(Sig))
    return Om * np.outer(d, d), Sig / np.outer(d, d)


# data
SCENARIOS = ["clean", "t3", "row", "row_struct", "homcell", "hetcell",
             "hetcell_mod", "hetcell_asym"]


def generate(n, Sig, scenario, rng, eps=None, mag=None):
    """Returns X (n x p). Contamination follows the independent contamination
    model X = (I-B)Y + BV with B, Y, V independent (cellwise scenarios)."""
    p = Sig.shape[0]
    L = np.linalg.cholesky(Sig)
    Y = rng.normal(size=(n, p)) @ L.T
    if scenario == "clean":
        return Y
    if scenario == "t3":
        g = rng.chisquare(3, n) / 3
        return Y / np.sqrt(g)[:, None]
    X = Y.copy()
    if scenario == "row":
        r = rng.random(n) < (eps or 0.10)
        X[r] = rng.normal(size=(r.sum(), p)) + 3.0 * rng.choice([-1, 1], p)
        return X
    if scenario == "row_struct":
        # rows with unremarkable margins but a different dependence structure
        r = rng.random(n) < (eps or 0.10)
        J = 0.9 * np.ones((p, p)) + 0.1 * np.eye(p)
        X[r] = rng.normal(size=(r.sum(), p)) @ np.linalg.cholesky(J).T
        return X
    if scenario == "homcell":
        B = rng.random((n, p)) < (eps or 0.05)
        X[B] = rng.choice([-1, 1], B.sum()) * rng.uniform(6, 10, B.sum())
        return X
    if scenario in ("hetcell", "hetcell_mod", "hetcell_asym", "sweep"):
        cols = rng.choice(p, max(1, int(round(0.2 * p))), replace=False)
        e = eps or (0.15 if scenario == "hetcell_asym" else 0.10)
        for j in cols:
            b = rng.random(n) < e
            k = b.sum()
            if scenario == "hetcell":
                X[b, j] = rng.choice([-1, 1], k) * rng.uniform(6, 10, k)
            elif scenario == "hetcell_mod":
                X[b, j] = rng.choice([-1, 1], k) * rng.uniform(2.5, 4, k)
            elif scenario == "hetcell_asym":
                X[b, j] = rng.uniform(5, 9, k)
            else:  # sweep: symmetric, fixed magnitude mag
                X[b, j] = rng.choice([-1, 1], k) * mag
        return X
    raise ValueError(scenario)


# metrics
def support(Th, rel=1e-6):
    p = len(Th)
    iu = np.triu_indices(p, 1)
    d = np.max(np.abs(np.diag(Th)))
    return np.abs(Th[iu]) > rel * d


def graph_metrics(Th, Om):
    if not np.all(np.isfinite(Th)):
        return dict(tp=0, fp=0, prec=np.nan, rec=np.nan, f1=np.nan, fdr=np.nan,
                    tpr=np.nan, fpr=np.nan, frob=np.nan, spec=np.nan, ne=np.nan)
    est, tru = support(Th), support(Om)
    tp, fp, fn = np.sum(est & tru), np.sum(est & ~tru), np.sum(~est & tru)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    E = Th - Om
    return dict(tp=int(tp), fp=int(fp), prec=prec, rec=rec, f1=f1, fdr=1 - prec,
                tpr=rec, fpr=fp / max(np.sum(~tru), 1),
                frob=np.linalg.norm(E) / np.linalg.norm(Om),
                spec=np.linalg.norm(E, 2) / np.linalg.norm(Om, 2), ne=int(tp + fp))


def roc_auc(tpr, fpr):
    """Area under the ROC curve traced by the lambda path, completed by the
    points (0,0) and (1,1) and taking the upper monotone envelope."""
    ok = np.isfinite(tpr) & np.isfinite(fpr)
    f = np.concatenate([[0.0], np.asarray(fpr)[ok], [1.0]])
    t = np.concatenate([[0.0], np.asarray(tpr)[ok], [1.0]])
    o = np.lexsort((t, f))
    f, t = f[o], np.maximum.accumulate(t[o])
    return float(np.trapezoid(t, f))


def ebic(S, Th, n, gamma=0.5):
    """Extended BIC of Foygel & Drton (2010) for a glasso fit."""
    if not np.all(np.isfinite(Th)):
        return np.inf
    sign, ld = np.linalg.slogdet(Th)
    if sign <= 0:
        return np.inf
    p = len(Th)
    k = int(support(Th).sum())
    return n * (np.trace(S @ Th) - ld) + k * np.log(n) + 4 * gamma * k * np.log(p)
