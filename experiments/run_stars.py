"""Select penalties by StARS on the evaluation sample using 20 subsamples of size 0.8n."""
import sys, os, zlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from pmtgl.methods import METHODS, LAMS
from pmtgl.sim import make_precision, generate, graph_metrics, support

M = ["GLASSO", "Kendall-GL", "GaussRank-GL", "Wrap-GL", "Winsor-GL", "MT-GLASSO(TH)", "tlasso", "PMT", "PMT-mt"]


def stars_index(X, m, B=20, beta=0.05, rng=None):
    n = len(X); b = int(10 * np.sqrt(n)) if n > 144 else int(0.8 * n)
    freq = None
    for _ in range(B):
        idx = rng.choice(n, b, replace=False)
        path = METHODS[m](X[idx], LAMS)
        E = np.array([support(T) if np.all(np.isfinite(T)) else np.zeros(len(support(path[0][0])), bool) for (T, _, _) in path], float)
        freq = E if freq is None else freq + E
    th = freq / B
    D = (2 * th * (1 - th)).mean(1)          # instability per lambda (LAMS decreasing)
    Dm = np.maximum.accumulate(D)            # monotonized from large to small lambda
    ok = np.nonzero(Dm <= beta)[0]
    return int(ok[-1]) if len(ok) else 0


def one(job):
    scen, rep = job
    rng = np.random.default_rng([zlib.crc32(f"stars-{scen}".encode()), rep])
    Om, Sig = make_precision(100, "er2", rng=rng)
    X = generate(100, Sig, scen, rng)
    rows = []
    for m in M:
        i = stars_index(X, m, rng=np.random.default_rng(rep))
        T = METHODS[m](X, LAMS)[i][0]
        g = graph_metrics(T, Om)
        rows.append(dict(scenario=scen, rep=rep, method=m, lam=LAMS[i], **{k: g[k] for k in ("prec", "rec", "f1", "fdr", "frob")}))
    return rows


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    jobs = [(s, r) for s in ("clean", "homcell", "hetcell", "row") for r in range(reps)]
    out = os.path.join(os.path.dirname(__file__), "..", "results", "stars.csv")
    part = out.replace("stars.csv", "stars_partial.csv")   # resume after interruption
    prev = pd.read_csv(part) if os.path.exists(part) else pd.DataFrame()
    done = set(zip(prev.scenario, prev.rep)) if len(prev) else set()
    jobs = [j for j in jobs if j not in done]
    print(len(jobs), "StARS jobs to run", flush=True)
    rows = prev.to_dict("records")
    with Pool(2) as pool:
        for rr in pool.imap_unordered(one, jobs):
            rows += rr
            pd.DataFrame(rows).to_csv(part, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.groupby(["scenario", "method"])[["prec", "rec", "f1", "fdr", "frob"]].mean().round(3).to_string())
