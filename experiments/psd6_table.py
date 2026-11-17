"""Summarize solver convergence and the effect of eigenvalue clipping."""
import os, numpy as np, pandas as pd
RES = os.path.join(os.path.dirname(__file__), "..", "results")
TAB = os.path.join(os.path.dirname(__file__), "..", "paper", "tables")
d = pd.read_csv(os.path.join(RES, "psd_stage6.csv"))
SC = {"clean": "clean", "homcell": "cell 5\\%", "hetcell": "cell het."}
lines = ["\\begin{tabular}{llcccccccc}", "\\toprule",
         "$p$ & scenario & $\\lambda_{\\min}(\\widehat R)$ & $\\hat\\eta$ & $\\maxnorm{\\widehat R-R}$ & ADMM fails & Jaccard & $\\Delta$F1 & $\\Delta$Frob\\\\",
         "\\midrule"]
rows = []
for (p, sc), g in d.groupby(["p", "scenario"], sort=False):
    f = g.groupby("rep").first()
    fail = g[g.admm_conv == 0]
    failtxt = "none" if not len(fail) else f"{len(fail)}/{f.shape[0]} at $\\lambda={fail.lam.max():g}$"
    jac = g[g.lam == 0.3].jacc
    b = g.groupby("rep").agg(frc=("frob_clip", "min"), frr=("frob_raw", "min"), f1c=("f1_clip", "max"), f1r=("f1_raw", "max"))
    df1, dfr = b.f1r - b.f1c, b.frr - b.frc
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    rows.append(dict(p=p, scenario=sc, mineig=f.mineig.mean(), eta=f.eta.mean(), eta_max=f.eta.max(), maxerr=f.maxerr.mean(),
                     fails=len(fail), jacc03=jac.mean(), df1=df1.mean(), df1_se=se(df1), dfr=dfr.mean(), dfr_se=se(dfr), reps=len(f)))
    lines.append(f"{p} & {SC[sc]} & ${f.mineig.mean():.3f}$ & {f.eta.mean():.3f} ({f.eta.max():.3f}) & {f.maxerr.mean():.2f} & {failtxt} & "
                 f"{jac.mean():.2f} & ${df1.mean():+.3f}\\pm{se(df1):.3f}$ & ${dfr.mean():+.4f}\\pm{se(dfr):.4f}$\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "psd6.tex"), "w").write("\n".join(lines) + "\n")
pd.DataFrame(rows).to_csv(os.path.join(RES, "psd_stage6_summary.csv"), index=False)
print("\n".join(lines))
