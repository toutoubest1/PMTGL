"""Runtime scaling (Sec. 6.6): wall-clock seconds for the covariance input and for the
full 20-point path, n = 200, clean er2 data, single core, median of 3 runs."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from pmtgl import covariance as C
from pmtgl.methods import METHODS, LAMS
from pmtgl.sim import make_precision, generate
COV = {"GLASSO": C.sample_cov, "Kendall-GL": C.kendall_cov, "GaussRank-GL": C.gaussrank_cov,
       "Wrap-GL": C.wrap_cov, "Winsor-GL": C.winsor_cov, "MT-GLASSO(TH)": C.fullmt_th, "PMT": C.pmt_cov}
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "runtime.csv")
PART = OUT.replace("runtime.csv", "runtime_partial.csv")  # resume after interruption
rows = pd.read_csv(PART).to_dict("records") if os.path.exists(PART) else []
done = {(r["p"], r["method"]) for r in rows}
rng = np.random.default_rng(0)
METHODS["GLASSO"](rng.normal(size=(50, 10)), LAMS)  # numba warm-up
C.winsor_cov(rng.normal(size=(50, 10)))
for p in (50, 100, 200, 400, 800):
    Om, Sig = make_precision(p, "er2", rng=rng)
    X = generate(200, Sig, "clean", rng)
    for m in list(COV) + (["tlasso"] if p <= 200 else []):
        if (p, m) in done:
            continue
        tc, tp = [], []
        for _ in range(3):
            if m in COV:
                t0 = time.perf_counter(); COV[m](X); tc.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); METHODS[m](X, LAMS); tp.append(time.perf_counter() - t0)
        rows.append(dict(p=p, method=m, t_cov=np.median(tc) if tc else np.nan, t_path=np.median(tp)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(PART, index=False)
pd.DataFrame(rows).to_csv(OUT, index=False)
