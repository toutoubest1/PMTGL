"""Fit graphs to Sachs et al. (2005) flow cytometry data.

Inject outliers into the training split. Select penalties on clean validation
cells and report likelihood on clean test cells. Compare 18-edge graphs with
the clean-training, full-data and published consensus graphs."""
import sys, os, zlib, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from pmtgl.methods import METHODS, LAMS
from pmtgl.sim import support
from pmtgl.glasso import glasso_path

D = os.path.join(os.path.dirname(__file__), "..", "data")
X_all = np.log(pd.read_csv(os.path.join(D, "cyto_full_data.csv")).values)
names = list(pd.read_csv(os.path.join(D, "cyto_full_data.csv")).columns)
tgt = pd.read_csv(os.path.join(D, "cyto_full_target.csv"))
p = X_all.shape[1]
CONS = np.zeros((p, p), bool)
for a, b in zip(tgt.Cause, tgt.Effect):
    i, j = names.index(a), names.index(b); CONS[i, j] = CONS[j, i] = True
iu = np.triu_indices(p, 1)
METH = ["GLASSO", "Kendall-GL", "GaussRank-GL", "Wrap-GL", "Winsor-GL", "MT-GLASSO(TH)",
        "tlasso", "Trimmed-GL", "DPD-GL", "PMT-fixed", "PMT", "PMT-mt"]
FINE = np.round(np.geomspace(0.9, 0.005, 40), 5)


def at_k(path, k=18):
    ne = [support(T).sum() if np.all(np.isfinite(T)) else -999 for T in path]
    i = int(np.argmin([abs(e - k) for e in ne]))
    return support(path[i])


def ref_graph():
    S = np.corrcoef(X_all, rowvar=False)
    return at_k(glasso_path(S, FINE))


REF = ref_graph()


def nll(Th, X):
    if not np.all(np.isfinite(Th)):
        return np.inf
    S = np.cov(X, rowvar=False)
    sgn, ld = np.linalg.slogdet(Th)
    return np.trace(S @ Th) - ld if sgn > 0 else np.inf


def contaminate(X, scen, rng):
    X = X.copy(); n = len(X)
    med = np.median(X, 0); mad = 1.4826 * np.median(np.abs(X - med), 0)
    if scen == "clean":
        return X
    if scen == "hetcell":
        for j in rng.choice(p, 2, replace=False):
            b = rng.random(n) < 0.10
            X[b, j] = med[j] + rng.choice([-1, 1], b.sum()) * rng.uniform(6, 10, b.sum()) * mad[j]
    elif scen == "homcell":
        B = rng.random(X.shape) < 0.05
        sgn = rng.choice([-1, 1], B.sum()); mag = rng.uniform(6, 10, B.sum())
        X[B] = (med + 0 * X)[B] + sgn * mag * (mad + 0 * X)[B]
    return X


def jacc(a, b):
    u = np.sum(a | b)
    return np.sum(a & b) / u if u else 1.0


def one(job):
    scen, n_tr, b = job
    rng = np.random.default_rng([zlib.crc32(f"sachs-{scen}-{n_tr}".encode()), b])
    idx = rng.permutation(len(X_all))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + 2000], idx[n_tr + 2000:n_tr + 4000]
    Xtr0 = X_all[tr]; Xtr = contaminate(Xtr0, scen, rng)
    rows = []
    for m in METH:
        path = [T for (T, _, _) in METHODS[m](Xtr, FINE)]
        path0 = [T for (T, _, _) in METHODS[m](Xtr0, FINE)]
        v = [nll(T, X_all[va]) for T in path]
        i = int(np.argmin(v))
        g, g0 = at_k(path), at_k(path0)
        rows.append(dict(scenario=scen, n=n_tr, rep=b, method=m, test_nll=nll(path[i], X_all[te]),
                         jacc_self=jacc(g, g0), jacc_ref=jacc(g, REF),
                         prec_cons=np.sum(g & CONS[iu]) / max(g.sum(), 1)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="results/sachs.csv"); a = ap.parse_args()
    jobs = [(s, n, b) for s in ("clean", "hetcell", "homcell") for n in (100, 300) for b in range(a.reps)]
    with Pool(2) as pool:
        res = pool.map(one, jobs)
    df = pd.DataFrame([r for rr in res for r in rr]); df.to_csv(a.out, index=False)
    print("reference GLASSO graph (n=7466) precision vs consensus:", np.sum(REF & CONS[iu]) / REF.sum())
    print(df.groupby(["scenario", "n", "method"])[["test_nll", "jacc_self", "jacc_ref", "prec_cons"]].mean().round(3).to_string())
