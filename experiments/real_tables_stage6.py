"""Summarize the isoprenoid and eye network analyses."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from make_tables import LABEL, TAB, RES
from real_stage6 import COMPARTMENT, load

d = pd.read_csv(os.path.join(RES, "real_stage6.csv"))
rows = []
for (name, m), g in d.groupby(["data", "method"]):
    full = g[g.task == "full"].iloc[0]
    r = dict(data=name, method=m, lam=full["lam"], ne=int(full["ne"]))
    for t in ("half", "homcell", "hetcell"):
        x = g[g.task == t].jacc
        r[t] = x.mean(); r[t + "_se"] = x.std(ddof=1) / np.sqrt(len(x)); r[t + "_n"] = len(x)
    if name == "isoprenoid":
        cols, _ = load(name)
        p = len(cols)
        iu = np.triu_indices(p, 1)
        E = json.loads(full.edges)
        comp = np.array([COMPARTMENT[c] for c in cols])
        a, b = comp[iu[0][E]], comp[iu[1][E]]
        ok = (a != "?") & (b != "?")
        r["within"] = float(np.mean(a[ok] == b[ok]))
        A, B = comp[iu[0]], comp[iu[1]]
        okall = (A != "?") & (B != "?")
        r["within_random"] = float(np.mean(A[okall] == B[okall]))
        r["cross_PC"] = int(np.sum(((a == "P") & (b == "C")) | ((a == "C") & (b == "P"))))
    rows.append(r)
S = pd.DataFrame(rows)
S.to_csv(os.path.join(RES, "real_stage6_summary.csv"), index=False)
print(S.round(3).to_string())

order = [m for m in LABEL if m in set(S.method)]
for name in ("isoprenoid", "eye"):
    T = S[S.data == name].set_index("method").reindex(order).dropna(how="all")
    head = "Method & $\\lambda$ & Half-sample & Hom.\\ cells & Het.\\ cells" + (" & Within comp. & P--C edges" if name == "isoprenoid" else "") + "\\\\"
    lines = ["\\begin{tabular}{l" + "c" * (6 if name == "isoprenoid" else 4) + "}", "\\toprule", head, "\\midrule"]
    best = {t: T[t].max() for t in ("half", "homcell", "hetcell")}
    for m, r in T.iterrows():
        cells = [f"{r.lam:.3f}"]
        for t in ("half", "homcell", "hetcell"):
            v = f"{r[t]:.3f}"
            if r[t] >= best[t] - 2 * r[t + '_se']:
                v = f"\\underline{{{v}}}" if r[t] < best[t] else f"\\textbf{{{v}}}"
            cells.append(v)
        if name == "isoprenoid":
            cells += [f"{r.within:.2f}", f"{int(r.cross_PC)}"]
        lines.append(LABEL[m] + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, f"real_{name}.tex"), "w").write("\n".join(lines) + "\n")
dg = pd.read_csv(os.path.join(RES, "real_stage6_diag.csv"))
print(dg.to_string())
