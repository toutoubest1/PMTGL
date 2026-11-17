"""Save paired differences between PMT methods and the best competing method."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from make_tables import RES

parts = []
for f in ("main_a.csv", "main_b.csv", "main_c.csv", "main_d.csv"):
    fp = os.path.join(RES, f)
    if os.path.exists(fp):
        parts.append(pd.read_csv(fp))
d = pd.concat(parts, ignore_index=True)
d = d.drop_duplicates(subset=["n", "p", "graph", "scenario", "rep", "method"], keep="last")
OWN = {"PMT", "PMT-mt", "PMT-fixed"}
rows = []
for (n, p, g, sc), x in d.groupby(["n", "p", "graph", "scenario"]):
    for metric, better in (("f1", "max"), ("frob", "min"), ("bic_f1", "max")):
        if metric not in x or x[metric].isna().all():
            continue
        m = x.groupby("method")[metric].mean()
        comp = m.drop([k for k in m.index if k in OWN])
        best = comp.idxmax() if better == "max" else comp.idxmin()
        for own in ("PMT", "PMT-mt"):
            a = x[x.method == own].set_index("rep")[metric]
            b = x[x.method == best].set_index("rep")[metric]
            z = (a - b).dropna()
            if len(z) < 2:
                continue
            rows.append(dict(n=n, p=p, graph=g, scenario=sc, metric=metric, own=own, best=best,
                             own_mean=a.mean(), best_mean=b.mean(), diff=z.mean(),
                             se=z.std(ddof=1) / np.sqrt(len(z)), reps=len(z), reps_best=int(b.notna().sum())))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(RES, "paired_main.csv"), index=False)
if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(S[S.own == "PMT"].round(4).to_string())
