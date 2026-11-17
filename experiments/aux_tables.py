"""Summarize contamination sweeps, StARS, runtime and Sachs experiments."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from make_tables import LABEL, SCEN, TAB, RES, fmt_block
from style import plt, PAL, MARK, LS

FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")


def _mse(d, keys, col):
    g = d.groupby(keys)[col]
    return g.mean(), g.std() / np.sqrt(g.size())


def sweeps():
    f = os.path.join(RES, "sweeps.csv")
    if not os.path.exists(f):
        return print("sweeps.csv missing")
    d = pd.read_csv(f)
    meths = list(dict.fromkeys(d.method))
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey="row")
    for c, (lab, sel, x) in enumerate([("contaminated-cell fraction $\\varepsilon$ (magnitude 8)", d.mag == 8.0, "eps"),
                                       ("outlier magnitude (in s.d.; $\\varepsilon=0.1$)", d.eps == 0.1, "mag")]):
        ds = d[sel]
        for r, met in enumerate(["f1", "frob"]):
            a = ax[r, c]
            for i, m in enumerate(meths):
                mu, se = _mse(ds[ds.method == m], [x], met)
                a.errorbar(mu.index, mu.values, yerr=2 * se.values, color=PAL[i % 8], marker=MARK[i % 8],
                           ls=LS[i % 8], capsize=1.5, label=LABEL[m])
            if x == "mag":
                a.set_xscale("log"); a.set_xticks([2, 3, 4, 6, 10, 15]); a.set_xticklabels([2, 3, 4, 6, 10, 15])
            a.set_xlabel(lab)
            a.set_ylabel({"f1": "F1 (held-out tuned)", "frob": "relative Frobenius error"}[met] if c == 0 else "")
    ax[0, 0].legend(ncol=4, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sweeps.pdf")); plt.close(fig)
    reps = d.groupby(["eps", "mag", "method"]).size()
    print("sweeps reps min/max", reps.min(), reps.max())


def stars():
    f = os.path.join(RES, "stars.csv")
    if not os.path.exists(f):
        return print("stars.csv missing")
    d = pd.read_csv(f)
    scen = [s for s in ("clean", "homcell", "hetcell", "row") if s in set(d.scenario)]
    meths = list(dict.fromkeys(d.method))
    cells = {m: [] for m in meths}
    for s in scen:
        ds = d[d.scenario == s]
        for k in ("f1", "fdr"):
            mu, se = _mse(ds, ["method"], k)
            blk = fmt_block(mu.to_frame(k), se.to_frame(k), [k], meths)
            for m in meths:
                cells[m].append(blk[m][0])
    hdr = " & ".join(f"\\multicolumn{{2}}{{c}}{{{SCEN[s]}}}" for s in scen)
    sub = " & ".join(["F1 & FDR"] * len(scen))
    L = ["\\begin{tabular}{l" + "cc" * len(scen) + "}", "\\toprule", f" & {hdr}\\\\", f" & {sub}\\\\", "\\midrule"]
    L += [f"{LABEL[m]} & " + " & ".join(cells[m]) + "\\\\" for m in meths]
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "stars.tex"), "w").write("\n".join(L))
    print("stars reps", d.groupby(["scenario", "method"]).size().min())


def runtime():
    f = os.path.join(RES, "runtime.csv")
    if not os.path.exists(f):
        return print("runtime.csv missing")
    d = pd.read_csv(f)
    ps = sorted(set(d.p)); meths = list(dict.fromkeys(d.method))
    L = ["\\begin{tabular}{l" + "r" * len(ps) + "}", "\\toprule",
         " & " + " & ".join(f"$p={p}$" for p in ps) + "\\\\", "\\midrule"]
    for col, title in (("t_cov", "Input matrix only"), ("t_path", "Input and 20-point graphical-lasso path")):
        L.append(f"\\multicolumn{{{len(ps)+1}}}{{l}}{{\\emph{{{title} (seconds)}}}}\\\\")
        for m in meths:
            row = []
            for p in ps:
                v = d[(d.p == p) & (d.method == m)][col]
                row.append("--" if v.empty or not np.isfinite(v.iloc[0]) else f"{v.iloc[0]:.3g}")
            if all(r == "--" for r in row):
                continue
            L.append(f"{LABEL[m]} & " + " & ".join(row) + "\\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"; L.append("\\end{tabular}")
    open(os.path.join(TAB, "runtime.tex"), "w").write("\n".join(L))
    fig, a = plt.subplots(figsize=(3.6, 2.8))
    for i, m in enumerate(meths):
        dm = d[d.method == m].dropna(subset=["t_cov"])
        if len(dm):
            a.loglog(dm.p, dm.t_cov, color=PAL[i % 8], marker=MARK[i % 8], ls=LS[i % 8], label=LABEL[m])
    a.set_xlabel("$p$ ($n=200$)"); a.set_ylabel("seconds (input matrix)")
    a.legend(fontsize=6.5, ncol=2)
    fig.savefig(os.path.join(FIG, "runtime.pdf")); plt.close(fig)


def sachs():
    f = os.path.join(RES, "sachs.csv")
    if not os.path.exists(f):
        return print("sachs.csv missing")
    d = pd.read_csv(f)
    d = d.replace([np.inf], np.nan)
    meths = list(dict.fromkeys(d.method))
    for n in sorted(set(d.n)):
        dn = d[d.n == n]
        scen = [s for s in ("clean", "hetcell", "homcell") if s in set(dn.scenario)]
        cells = {m: [] for m in meths}
        cols = []
        for s in scen:
            ds = dn[dn.scenario == s]
            for k in ("test_nll", "jacc_self", "jacc_ref", "prec_cons"):
                if s == "clean" and k == "jacc_self":
                    continue
                cols.append((s, k))
                mu, se = _mse(ds, ["method"], k)
                from make_tables import HIGHER
                HIGHER.update({"test_nll": False, "jacc_self": True, "jacc_ref": True, "prec_cons": True})
                blk = fmt_block(mu.to_frame(k), se.to_frame(k), [k], meths)
                for m in meths:
                    cells[m].append(blk[m][0])
        KN = {"test_nll": "NLL", "jacc_self": "$J_{\\rm self}$", "jacc_ref": "$J_{\\rm ref}$", "prec_cons": "Prec."}
        hdr = []
        for s in scen:
            k = sum(1 for c in cols if c[0] == s)
            hdr.append(f"\\multicolumn{{{k}}}{{c}}{{{SCEN[s]}}}")
        L = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", " & " + " & ".join(hdr) + "\\\\",
             " & " + " & ".join(KN[c[1]] for c in cols) + "\\\\", "\\midrule"]
        L += [f"{LABEL[m]} & " + " & ".join(cells[m]) + "\\\\" for m in meths]
        L += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(TAB, f"sachs_n{n}.tex"), "w").write("\n".join(L))
    print("sachs reps", d.groupby(["scenario", "n", "method"]).size().min())


def sachs_full():
    f = os.path.join(RES, "sachs_fullgraph_jaccard.csv")
    if not os.path.exists(f):
        return print("sachs_fullgraph_jaccard.csv missing")
    J = pd.read_csv(f, index_col=0)
    P = pd.read_csv(os.path.join(RES, "sachs_fullgraph_prec.csv"), index_col=0).iloc[:, 0]
    ms = list(J.index)
    ab = {m: f"({i+1})" for i, m in enumerate(ms)}
    L = ["\\begin{tabular}{l" + "c" * len(ms) + "c}", "\\toprule",
         " & " + " & ".join(ab[m] for m in ms) + " & Prec.\\\\", "\\midrule"]
    for m in ms:
        L.append(f"{ab[m]} {LABEL[m]} & " + " & ".join("" if a == m else f"{J.loc[m, a]:.2f}" for a in ms)
                 + f" & {P[m]:.2f}\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "sachs_full.tex"), "w").write("\n".join(L))


if __name__ == "__main__":
    for fn in (sweeps, stars, runtime, sachs, sachs_full):
        fn()
