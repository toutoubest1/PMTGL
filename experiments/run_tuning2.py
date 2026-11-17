"""Compare StARS, BIC, eBIC, permutation calibration and cross-validation.

Data-driven rules use the training sample. Held-out and oracle rules use
the true graph as benchmarks.

Usage: python experiments/run_tuning2.py OUT.csv REPS N P GRAPH SCENARIOS METHODS
Pass scenarios and methods as comma-separated lists."""
import sys, os, zlib, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from pmtgl.sim import make_precision, generate, graph_metrics
from pmtgl.glasso import glasso_path
from pmtgl.methods import METHODS
from pmtgl import tuning as T

LAMF = T.LAMF
KEEP = ("prec", "rec", "f1", "fdr", "frob", "ne")
COLS = ["n", "p", "graph", "scenario", "rep", "method", "rule", "lam", *KEEP, "secs", "error"]


def b_half(n):
    return min(int(10 * np.sqrt(n)), n // 2)


def run_method(m, X, Xtu, Om, rep):
    n = len(X)
    plug = m in T.COV
    if plug:
        R, d = T.plugin_input(m, X)
        path = glasso_path(R, LAMF)
        fit_te = [Th / np.outer(d, d) for Th in path]
        Rt, dt = T.plugin_input(m, Xtu)
        fit_tu = [Th / np.outer(dt, dt) for Th in glasso_path(Rt, LAMF)]
        Rfit, pfit = R, path
        fitfun = None
    else:  # iterative (row-weighting) methods: own path; eBIC proxy as in runner.py
        out = METHODS[m](X, LAMF)
        fit_te = [o[0] for o in out]
        fit_tu = [o[0] for o in METHODS[m](Xtu, LAMF)]
        Rfit, pfit = out[0][2], [o[1] for o in out]
        fitfun = lambda Xs, ls: [o[0] for o in METHODS[m](Xs, ls)]
    mets = [graph_metrics(Th, Om) for Th in fit_te]
    mtu = [graph_metrics(Th, Om) for Th in fit_tu]
    lam = {"heldout": LAMF[int(np.nanargmax([q["f1"] for q in mtu]))],
           "heldout_frob": LAMF[int(np.nanargmin([q["frob"] for q in mtu]))],
           "oracle": LAMF[int(np.nanargmax([q["f1"] for q in mets]))],
           "oracle_frob": LAMF[int(np.nanargmin([q["frob"] for q in mets]))]}
    ok = [np.all(np.isfinite(T_)) for T_ in pfit]
    lam["bic"] = T.ic_lambda(Rfit, n, [T_ if o else np.full_like(Rfit, np.nan) for T_, o in zip(pfit, ok)], LAMF, 0.0)
    lam["ebic"] = T.ic_lambda(Rfit, n, [T_ if o else np.full_like(Rfit, np.nan) for T_, o in zip(pfit, ok)], LAMF, 0.5)
    r2 = np.random.default_rng([rep, 7])
    lam["stars08"], _ = T.stars_lambda(m, X, LAMF, B=20, b=int(0.8 * n), rng=r2, fit=fitfun)
    lam["starsH"], _ = T.stars_lambda(m, X, LAMF, B=20, b=b_half(n), rng=r2, fit=fitfun)
    rows = []
    if plug:
        lam["cv"] = T.cv_lambda(m, X, LAMF, K=5, rng=r2)
        R0, Tn, _ = T.perm_null(m, X, B=100, rng=r2)
        lam["pfwer0.1"] = T.lam_perm_fwer(R0, Tn, 0.1)
    for rule, l in lam.items():
        if rule == "pfwer0.1":   # exact (off-grid) lambda
            Th = glasso_path(R, [l])[0] / np.outer(d, d)
            g = graph_metrics(Th, Om)
        else:
            g = mets[int(np.argmin(np.abs(LAMF - l)))]
        rows.append(dict(method=m, rule=rule, lam=l, **{k: g[k] for k in KEEP}))
    return rows


def one(job):
    n, p, graph, scen, rep, methods = job
    rng = np.random.default_rng([zlib.crc32(f"tune2-{n}-{p}-{graph}-{scen}".encode()), rep])
    Om, Sig = make_precision(p, graph, rng=rng)
    X = generate(n, Sig, scen, rng)
    Xtu = generate(n, Sig, scen, rng)
    rows = []
    for m in methods:
        t0 = time.perf_counter()
        try:
            rr = run_method(m, X, Xtu, Om, rep)
        except Exception as e:  # recorded, never silently dropped
            rr = [dict(method=m, rule="error", lam=np.nan, error=repr(e))]
        for r in rr:
            r.update(n=n, p=p, graph=graph, scenario=scen, rep=rep, secs=time.perf_counter() - t0)
        rows += rr
    return rows


if __name__ == "__main__":
    out, reps, n, p, graph = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    scens, methods = sys.argv[6].split(","), sys.argv[7].split(",")
    jobs = [(n, p, graph, s, r, methods) for s in scens for r in range(reps)]
    done = set()
    if os.path.exists(out):
        o = pd.read_csv(out)
        o = o[o.method.isin(methods)]
        cnt = o.groupby(["n", "p", "graph", "scenario", "rep"]).method.nunique()
        done = {k for k, v in cnt.items() if v == len(methods)}
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3], j[4]) not in done]
    print(len(jobs), "jobs ->", out, flush=True)
    with Pool(2) as pool:
        for rr in pool.imap_unordered(one, jobs):
            pd.DataFrame(rr).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print("done", rr[0]["scenario"], rr[0]["rep"], flush=True)
