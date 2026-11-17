"""Aggregate raw Monte Carlo CSVs into LaTeX tables (mean and Monte Carlo SE).
Best mean per column in bold; entries within 2 SE of the best are underlined."""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, TAB = os.path.join(ROOT, "results"), os.path.join(ROOT, "paper", "tables")
LABEL = {"GLASSO": "GLASSO", "Kendall-GL": "Kendall", "Spearman-GL": "Spearman", "GaussRank-GL": "Gauss-rank",
         "Wrap-GL": "Wrapping", "Winsor-GL": "Winsor", "MT-GLASSO(TH)": "MT-GLASSO (TH)",
         "MT-GLASSO(mass)": "MT-GLASSO (mass)", "MT-SP(TH)": "MT-SP", "tlasso": "$t$-lasso",
         "Trimmed-GL": "Trimmed", "DPD-GL": "DPD", "PMT-fixed": "PMT ($a=0.5$)", "PMT": "PMT",
         "PMT-mt": "PMT-mt", "Composite": "Composite", "Composite-trunc": "Composite-trunc"}
SCEN = {"clean": "Clean", "t3": "$t_3$", "row": "Row shift", "row_struct": "Row structural",
        "homcell": "Cell 5\\%", "hetcell": "Cell het.", "hetcell_mod": "Cell het. moderate",
        "hetcell_asym": "Cell het. one-sided", "sweep": "sweep"}
HIGHER = {"auc": True, "f1": True, "frob": False, "fdr": False, "prec": True, "rec": True,
          "eb_f1": True, "bic_f1": True, "or_f1": True, "time": False, "bic_fdr": False}


def summarize(df, keys, metrics):
    g = df.groupby(keys + ["method"])
    m = g[metrics].mean()
    s = g[metrics].std() / np.sqrt(g.size().values[:, None])
    return m, s, g.size()


def fmt_block(m, s, metrics, methods, digits=3):
    """m, s indexed by method only. Returns dict method -> list of cell strings."""
    out = {mt: [] for mt in methods}
    for k in metrics:
        col = m[k].reindex(methods)
        se = s[k].reindex(methods)
        best = col.max() if HIGHER.get(k, True) else col.min()
        ib = col.idxmax() if HIGHER.get(k, True) else col.idxmin()
        for mt in methods:
            v = col[mt]
            if not np.isfinite(v):
                out[mt].append("--"); continue
            txt = f"{v:.{digits}f}"
            if mt == ib:
                txt = f"\\textbf{{{txt}}}"
            elif abs(v - best) <= 2 * np.sqrt(se[mt] ** 2 + se[ib] ** 2):
                txt = f"\\underline{{{txt}}}"
            out[mt].append(txt)
    return out


def scenario_table(df, n, p, graph, scenarios, methods, metrics, name, caption_note=""):
    d = df[(df.n == n) & (df.p == p) & (df.graph == graph)]
    hdr = " & ".join(SCEN[s] for s in scenarios)
    lines = []
    for k in metrics:
        lines.append(f"\\multicolumn{{{len(scenarios)+1}}}{{l}}{{\\emph{{{METRIC_NAME[k]}}}}}\\\\")
        cells = {}
        maxse = 0
        for s in scenarios:
            ds = d[d.scenario == s]
            if ds.empty:
                for mt in methods: cells.setdefault(mt, []).append("")
                continue
            m, se, cnt = summarize(ds, [], [k])
            m.index = m.index.get_level_values(-1); se.index = se.index.get_level_values(-1)
            maxse = max(maxse, np.nanmax(se[k].values))
            blk = fmt_block(m, se, [k], methods)
            for mt in methods:
                cells.setdefault(mt, []).append(blk[mt][0])
        for mt in methods:
            lines.append(f"{LABEL[mt]} & " + " & ".join(cells[mt]) + "\\\\")
        lines.append(f"\\multicolumn{{{len(scenarios)+1}}}{{r}}{{\\footnotesize max.\\ MC s.e.\\ {maxse:.3f}}}\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    reps = int(d.groupby(["scenario", "method"]).size().min()) if not d.empty else 0
    body = ["\\begin{tabular}{l" + "c" * len(scenarios) + "}", "\\toprule",
            f" & {hdr}\\\\", "\\midrule"] + lines + ["\\end{tabular}"]
    open(os.path.join(TAB, name), "w").write("\n".join(body))
    return reps


METRIC_NAME = {"auc": "ROC AUC along the path (higher is better)",
               "f1": "F1 at held-out-tuned $\\lambda$ (higher is better)",
               "fdr": "FDR at held-out-tuned $\\lambda$ (lower is better)",
               "prec": "Precision at held-out-tuned $\\lambda$", "rec": "Recall at held-out-tuned $\\lambda$",
               "frob": "Relative Frobenius error of $\\widehat\\Omega$ at held-out-tuned $\\lambda$ (lower is better)",
               "bic_f1": "F1 at BIC-selected $\\lambda$", "bic_fdr": "FDR at BIC-selected $\\lambda$",
               "eb_f1": "F1 at eBIC-selected $\\lambda$", "time": "Runtime per path (s)"}
