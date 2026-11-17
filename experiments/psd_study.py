"""PSD-projection study (Sec. 5.3): effect of the correction applied to the
PMT input before the graphical lasso."""
import sys, os, zlib, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from pmtgl import covariance as C
from pmtgl.psd import correct, min_eig
from pmtgl.glasso import glasso_path, glasso_path_admm
from pmtgl.methods import LAMS
from pmtgl.sim import make_precision, generate, graph_metrics, roc_auc

VARIANTS = ["none", "eigclip", "eigclip05", "maxnorm", "nearcorr"]


def paths(X, Om):
    S, _ = C.pmt_cov(X)
    d = np.sqrt(np.diag(S)); R = S / np.outer(d, d)
    res = {}
    for v in VARIANTS:
        Rv = correct(R, v)
        dv = np.sqrt(np.diag(Rv)); Rv = Rv / np.outer(dv, dv)
        path = glasso_path_admm(R, LAMS) if v == "none" else glasso_path(Rv, LAMS)
        mets = [graph_metrics(T / np.outer(d * dv, d * dv), Om) for T in path]
        res[v] = (Rv, mets)
    return R, res


def one(job):
    (n, p, scen), rep = job
    rng = np.random.default_rng([zlib.crc32(f"psd-{n}-{p}-{scen}".encode()), rep])
    Om, Sig = make_precision(p, "er2", rng=rng)
    Xtu, Xte = generate(n, Sig, scen, rng), generate(n, Sig, scen, rng)
    _, tu = paths(Xtu, Om)
    R, te = paths(Xte, Om)
    rows = []
    for v in VARIANTS:
        Rv, mte = te[v]; mtu = tu[v][1]
        f1tu = np.array([m["f1"] for m in mtu]); frtu = np.array([m["frob"] for m in mtu])
        iF, iR = int(np.nanargmax(f1tu)), int(np.nanargmin(frtu))
        E = Rv - Sig
        rows.append(dict(n=n, p=p, scenario=scen, rep=rep, variant=v,
                         mineig_raw=min_eig(R), maxerr=np.abs(E).max(),
                         maxerr_off=np.abs(E - np.diag(np.diag(E))).max(),
                         froerr=np.linalg.norm(E) / np.linalg.norm(Sig),
                         specerr=np.linalg.norm(E, 2),
                         infeasible=int(np.sum([not np.isfinite(m["f1"]) for m in mte])),
                         lam_min_feasible=min([l for l, m in zip(LAMS, mte) if np.isfinite(m["f1"])], default=np.nan),
                         auc=roc_auc(np.array([m["tpr"] for m in mte]), np.array([m["fpr"] for m in mte])),
                         f1=mte[iF]["f1"], frob=mte[iR]["frob"],
                         or_f1=np.nanmax([m["f1"] for m in mte]), or_frob=np.nanmin([m["frob"] for m in mte])))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", default="results/psd_study.csv"); a = ap.parse_args()
    settings = [(n, p, s) for (n, p) in [(200, 50), (100, 100), (100, 200)]
                for s in ["clean", "hetcell", "homcell"]]
    jobs = [(s, r) for s in settings for r in range(a.reps)]
    with Pool(2) as pool:
        res = pool.map(one, jobs)
    df = pd.DataFrame([r for rr in res for r in rr]); df.to_csv(a.out, index=False)
    print(df.groupby(["n", "p", "scenario", "variant"])[["mineig_raw", "maxerr_off", "froerr", "infeasible", "auc", "f1", "frob"]].mean().round(3).to_string())
