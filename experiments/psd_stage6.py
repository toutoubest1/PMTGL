"""Compare ADMM on raw PMT correlations with coordinate descent after clipping.

The clipped correlation matrix certifies a unique solution on the raw input
when lambda exceeds the largest off-diagonal change caused by clipping."""
import sys, os, zlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from pmtgl import covariance as C
from pmtgl.psd import correct, min_eig
from pmtgl.glasso import glasso_path, glasso_admm
from pmtgl.sim import make_precision, generate, graph_metrics, support

COLS = ["n", "p", "scenario", "rep", "lam", "eta", "mineig", "maxerr", "certified", "admm_conv", "f1_clip", "frob_clip",
        "ne_clip", "kkt_clip", "f1_raw", "frob_raw", "ne_raw", "kkt_raw", "jacc", "reldiff"]
LAMS6 = np.array([0.6, 0.45, 0.35, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02])


def kkt(S, Th, lam):
    W = np.linalg.inv(Th)
    G = S - W
    off = ~np.eye(len(S), dtype=bool)
    nz = np.abs(Th) > 1e-8 * np.abs(np.diag(Th)).max()
    r = np.where(nz, np.abs(G + lam * np.sign(Th)), np.maximum(np.abs(G) - lam, 0.0))
    return float(max(np.abs(np.diag(G)).max(), r[off].max()))


def one(job):
    (n, p, scen), rep = job
    rng = np.random.default_rng([zlib.crc32(f"psd6-{n}-{p}-{scen}".encode()), rep])
    Om, Sig = make_precision(p, "er2", rng=rng)
    X = generate(n, Sig, scen, rng)
    S, _ = C.pmt_cov(X)
    d = np.sqrt(np.diag(S)); R = S / np.outer(d, d)
    Sc = correct(S, "eigclip"); dc = np.sqrt(np.diag(Sc)); Rc = Sc / np.outer(dc, dc)
    off = ~np.eye(p, dtype=bool)
    eta = float(np.abs(Rc - R)[off].max())
    clip_path = glasso_path(Rc, LAMS6)
    rows, Z0 = [], None
    for lam, Tc in zip(LAMS6, clip_path):
        Tr = glasso_admm(R, float(lam), max_it=5000, Z0=Z0)
        conv = Tr is not None
        row = dict(n=n, p=p, scenario=scen, rep=rep, lam=lam, eta=eta, mineig=min_eig(R),
                   maxerr=float(np.abs(R - Sig)[off].max()), certified=bool(lam > eta), admm_conv=conv)
        gc = graph_metrics(Tc / np.outer(dc, dc), Om)   # Theta = Theta_Rc / (dc dc^T), Rc = Sc/(dc dc^T)
        row.update(f1_clip=gc["f1"], frob_clip=gc["frob"], ne_clip=gc["ne"], kkt_clip=kkt(Rc, Tc, lam))
        if conv:
            Z0 = Tr
            gr = graph_metrics(Tr / np.outer(d, d), Om)
            a, b = support(Tr), support(Tc)
            row.update(f1_raw=gr["f1"], frob_raw=gr["frob"], ne_raw=gr["ne"], kkt_raw=kkt(R, Tr, lam),
                       jacc=float((a & b).sum() / max((a | b).sum(), 1)),
                       reldiff=float(np.linalg.norm(Tr - Tc) / np.linalg.norm(Tr)))
        rows.append(row)
    return rows


if __name__ == "__main__":
    out = "results/psd_stage6.csv"
    settings = [(100, 100, s) for s in ("clean", "homcell", "hetcell")]
    jobs = [(s, r) for s in settings for r in range(10)]
    settings = [(100, 200, s) for s in ("clean", "homcell", "hetcell")]
    jobs += [(s, r) for s in settings for r in range(5)]
    done = set()
    if os.path.exists(out):
        o = pd.read_csv(out); done = set(zip(o.n, o.p, o.scenario, o.rep))
    jobs = [j for j in jobs if (j[0][0], j[0][1], j[0][2], j[1]) not in done]
    print(len(jobs), "jobs", flush=True)
    with Pool(2) as pool:
        for rr in pool.imap_unordered(one, jobs):
            pd.DataFrame(rr).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print("done", rr[0]["p"], rr[0]["scenario"], rr[0]["rep"], flush=True)
