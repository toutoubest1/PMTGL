"""Table for the PSD study: means and paired differences vs. no correction."""
import os, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(__file__), "..")
d = pd.read_csv(os.path.join(R, "results", "psd_study.csv"))
d = d[d.p >= 100]
V = {"none": "None (ADMM)", "eigclip": "Eigen-clip $10^{-3}$", "eigclip05": "Eigen-clip $0.05$",
     "maxnorm": "Max-norm proj.", "nearcorr": "Nearest corr."}
S = [(100, 100, "clean"), (100, 100, "homcell"), (100, 200, "clean"), (100, 200, "hetcell"), (100, 200, "homcell")]
SN = {"clean": "clean", "hetcell": "cell het.", "homcell": "cell 5\\%"}
base = d[d.variant == "none"].set_index(["n", "p", "scenario", "rep"])
lines = ["\\begin{tabular}{l" + "c" * len(S) + "}", "\\toprule",
         " & " + " & ".join(f"$p={p}$, {SN[s]}" for (n, p, s) in S) + "\\\\", "\\midrule",
         "\\multicolumn{%d}{l}{\\emph{Minimum eigenvalue of the raw PMT correlation matrix}}\\\\" % (len(S) + 1)]
row = []
for (n, p, s) in S:
    x = d[(d.n == n) & (d.p == p) & (d.scenario == s) & (d.variant == "none")].mineig_raw
    row.append(f"{x.mean():.3f}")
lines.append(" & " + " & ".join(row) + "\\\\")
for k, title in [("maxerr_off", "Max-norm error of the corrected input (off-diagonal)"),
                 ("frob", "Relative Frobenius error of $\\widehat\\Omega$ (held-out tuned)"),
                 ("f1", "F1 (held-out tuned)"), ("auc", "ROC AUC")]:
    lines.append("\\midrule\\multicolumn{%d}{l}{\\emph{%s}}\\\\" % (len(S) + 1, title))
    for v in V:
        cells = []
        for (n, p, s) in S:
            x = d[(d.n == n) & (d.p == p) & (d.scenario == s) & (d.variant == v)].set_index("rep")[k]
            b = base.loc[(n, p, s)][k]
            diff = (x - b).dropna()
            z = diff.mean() / (diff.std() / np.sqrt(len(diff))) if v != "none" and diff.std() > 0 else 0
            mark = "$^{*}$" if abs(z) > 2 else ""
            cells.append(f"{x.mean():.3f}{mark}")
        lines.append(V[v] + " & " + " & ".join(cells) + "\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(R, "paper", "tables", "psd.tex"), "w").write("\n".join(lines))
print("\n".join(lines))
