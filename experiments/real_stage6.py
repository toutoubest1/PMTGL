"""Compare graphs for the Wille isoprenoid and Scheetz eye data.

Match edge counts across methods. Use Jaccard similarity to assess
half-sample stability and sensitivity to added cellwise outliers."""
import sys, os, json, zlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from pmtgl.methods import METHODS
from pmtgl.sim import support
from pmtgl.covariance import standardize, select_strengths

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
LAMR = {"isoprenoid": np.round(np.geomspace(0.95, 0.01, 61), 5), "eye": np.round(np.geomspace(0.95, 0.05, 45), 5)}
REPS = {"isoprenoid": (30, 20), "eye": (20, 10)}   # (half-samples, contaminated copies per type)
M_ISO = ["GLASSO", "Kendall-GL", "GaussRank-GL", "Wrap-GL", "Winsor-GL", "MT-GLASSO(TH)", "tlasso", "PMT", "PMT-mt"]
M_EYE = ["GLASSO", "Kendall-GL", "GaussRank-GL", "Wrap-GL", "Winsor-GL", "MT-GLASSO(TH)", "PMT", "PMT-mt"]

# Compartments from Wille et al. (2004), Table 1 (C: cytosol / MVA side, P: plastid / MEP side, M: mitochondrion).
# Exclude genes with unknown compartment ("?") from compartment shares.
COMPARTMENT = {
    "AACT1": "C", "AACT2": "C", "DPPS1": "C", "DPPS3": "C", "FPPS1": "C", "FPPS2": "C", "GGPPS3": "C", "GGPPS4": "C",
    "GGPPS8": "C", "HMGR1": "C", "HMGR2": "C", "HMGS": "C", "IPPI2": "C", "MK": "C", "MPDC1": "C", "MPDC2": "C",
    "CMK": "P", "DXPS1": "P", "DXPS2(cla1)": "P", "DXPS3": "P", "DXR": "P", "GGPPS2": "P", "GGPPS6": "P",
    "GGPPS10": "P", "GGPPS11": "P", "GGPPS12": "P", "GPPS": "P", "HDR": "P", "HDS": "P", "IPPI1": "P", "MCT": "P",
    "MECPS": "P", "PPDS1": "P",
    "DPPS2": "M", "GGPPS1mt": "M", "GGPPS5": "M", "GGPPS9": "M", "UPPS1": "M",
    "PPDS2mt": "?",
}


def load(name):
    if name == "isoprenoid":
        df = pd.read_csv(os.path.join(DATA, "isoprenoid_wille2004.csv"))
    else:
        df = pd.read_csv(os.path.join(DATA, "eye_scheetz2006.csv"))
    return df.columns.tolist(), df.to_numpy(float)


def graph_at(m, X, target, lams):
    """Graph (upper-triangular support) of method m with the edge count closest to target."""
    path = METHODS[m](X, lams)
    ne = np.array([support(T).sum() if np.all(np.isfinite(T)) else -10 ** 9 for (T, _, _) in path])
    i = int(np.argmin(np.abs(ne - target)))
    best = (support(path[i][0]), float(lams[i]), int(ne[i]))
    # refine by bisection between the grid points that bracket the target (lams decreasing, ne increasing)
    above = np.nonzero(ne >= target)[0]
    if len(above) and above[0] > 0 and ne[above[0]] != target:
        hi, lo = lams[above[0] - 1], lams[above[0]]          # ne(hi) < target <= ne(lo)
        for _ in range(8):
            mid = float(np.sqrt(hi * lo))
            T = METHODS[m](X, np.array([mid]))[0][0]
            if not np.all(np.isfinite(T)):
                break
            G = support(T); k = int(G.sum())
            if abs(k - target) < abs(best[2] - target):
                best = (G, mid, k)
            if k == target:
                break
            hi, lo = (mid, lo) if k < target else (hi, mid)
    return best


def jacc(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def contaminate(X, kind, rng):
    Z, m, s = standardize(X)
    Xc = X.copy()
    n, p = X.shape
    if kind == "homcell":
        B = rng.random((n, p)) < 0.05
    else:
        B = np.zeros((n, p), bool)
        cols = rng.choice(p, max(1, round(0.2 * p)), replace=False)
        B[:, cols] = rng.random((n, len(cols))) < 0.10
    V = rng.choice([-1, 1], (n, p)) * rng.uniform(6, 10, (n, p))
    Xc[B] = (m + s * V)[B]
    return Xc


def job(args):
    name, m, task, b = args
    cols, X = load(name)
    p = X.shape[1]
    G0, lam0, ne0 = graph_at(m, X, p, LAMR[name])
    rng = np.random.default_rng([zlib.crc32(name.encode()), b, {"half": 1, "homcell": 2, "hetcell": 3}.get(task, 0)])
    if task == "full":
        return dict(data=name, method=m, task=task, b=b, lam=lam0, ne=ne0, edges=json.dumps(np.nonzero(G0)[0].tolist()))
    if task == "half":
        idx = rng.choice(len(X), len(X) // 2, replace=False)
        G, lam, ne = graph_at(m, X[idx], p, LAMR[name])
    else:
        G, lam, ne = graph_at(m, contaminate(X, task, rng), p, LAMR[name])
    return dict(data=name, method=m, task=task, b=b, lam=lam, ne=ne, jacc=jacc(G, G0))


if __name__ == "__main__":
    out = "results/real_stage6.csv"
    jobs = []
    for name, M in (("isoprenoid", M_ISO), ("eye", M_EYE)):
        for m in M:
            jobs.append((name, m, "full", 0))
            jobs += [(name, m, "half", b) for b in range(REPS[name][0])]
            jobs += [(name, m, t, b) for t in ("homcell", "hetcell") for b in range(REPS[name][1])]
    done = set()
    if os.path.exists(out):
        o = pd.read_csv(out); done = set(zip(o.data, o.method, o.task, o.b))
    jobs = [j for j in jobs if j not in done]
    print(len(jobs), "jobs", flush=True)
    rows = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=4):
            pd.DataFrame([r]).reindex(columns=["data", "method", "task", "b", "lam", "ne", "jacc", "edges"]).to_csv(
                out, mode="a", header=not os.path.exists(out), index=False)
    # marginal diagnostics
    diag = []
    for name in ("isoprenoid", "eye"):
        cols, X = load(name)
        Z, _, _ = standardize(X)
        a, eps = select_strengths(Z)
        flag = np.abs(Z) > 3.5
        diag.append(dict(data=name, n=X.shape[0], p=X.shape[1], cells_flagged=float(flag.mean()),
                         rows_with_flag=float(flag.any(1).mean()), genes_with_excess=int((eps > 0).sum()),
                         genes_a_gt_min=int((a > a.min()).sum()), a_max=float(a.max())))
    pd.DataFrame(diag).to_csv("results/real_stage6_diag.csv", index=False)
    print(pd.DataFrame(diag).to_string())
