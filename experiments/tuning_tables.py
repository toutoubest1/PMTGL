"""Summarize data-driven penalty selection and benchmark comparisons."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from make_tables import LABEL, SCEN, TAB, RES

PMT = ["PMT", "PMT-mt"]
RULES_F1 = ["heldout", "starsH", "stars08", "bic", "pfwer0.1", "ebic"]
RNAME = {"heldout": "Held-out (benchmark)", "oracle": "Best on path", "starsH": "StARS, $b=n/2$",
         "stars08": "StARS, $b=0.8n$", "bic": "BIC", "ebic": "eBIC ($\\gamma=0.5$)", "pfwer0.1": "Permutation ($\\alpha=0.1$)",
         "cv": "5-fold CV", "heldout_frob": "Held-out Frobenius (benchmark)", "oracle_frob": "Best on path"}


def load(f):
    f = os.path.join(RES, f)
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d[d.rule != "error"]


def paired(d, metric, rule, better="max"):
    """PMT minus the best non-PMT method (best by mean) for each scenario: mean, se, competitor."""
    out = {}
    for sc, g in d[d.rule == rule].groupby("scenario"):
        m = g.groupby("method")[metric].mean()
        comp = m.drop([x for x in m.index if x in PMT or x == "PMT-fixed"], errors="ignore")
        best = comp.idxmax() if better == "max" else comp.idxmin()
        a = g[g.method == "PMT"].set_index("rep")[metric]
        b = g[g.method == best].set_index("rep")[metric]
        x = (a - b).dropna()
        out[sc] = (x.mean(), x.std(ddof=1) / np.sqrt(len(x)), best, len(x))
    return out


def fmt(v, se):
    s = f"{v:+.3f}"
    star = "$^{\\dagger}$" if abs(v) > 2 * se else ""
    return f"{s}{star}"


def rule_table(d, scens, fname, caption_n):
    """Rows: rules. Columns: scenarios. Cells: PMT F1 / paired diff vs best competitor (F1)."""
    lines = ["\\begin{tabular}{l" + "c" * len(scens) + "}", "\\toprule",
             " & " + " & ".join(SCEN[s] for s in scens) + "\\\\", "\\midrule"]
    lines.append("\\multicolumn{%d}{l}{\\emph{F1 of PMT, and PMT minus the best competitor (paired)}}\\\\" % (len(scens) + 1))
    for r in RULES_F1:
        if r not in set(d.rule):
            continue
        pdiff = paired(d, "f1", r)
        cells = []
        for s in scens:
            g = d[(d.rule == r) & (d.scenario == s) & (d.method == "PMT")]
            if s not in pdiff or not len(g):
                cells.append("--"); continue
            v, se, best, k = pdiff[s]
            cells.append(f"{g.f1.mean():.3f} ({fmt(v, se)})")
        lines.append(RNAME[r] + " & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{%d}{l}{\\emph{FDR of PMT / of the best competitor}}\\\\" % (len(scens) + 1))
    for r in ["heldout", "starsH", "stars08", "bic", "pfwer0.1"]:
        if r not in set(d.rule):
            continue
        pdiff = paired(d, "f1", r)
        cells = []
        for s in scens:
            g = d[(d.rule == r) & (d.scenario == s)]
            if s not in pdiff or not len(g):
                cells.append("--"); continue
            best = pdiff[s][2]
            cells.append(f"{g[g.method == 'PMT'].fdr.mean():.2f} / {g[g.method == best].fdr.mean():.2f}")
        lines.append(RNAME[r] + " & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{%d}{l}{\\emph{Relative Frobenius error of PMT, and PMT minus the best competitor (paired)}}\\\\" % (len(scens) + 1))
    for r in ["heldout_frob", "cv", "starsH", "bic"]:
        if r not in set(d.rule):
            continue
        pdiff = paired(d, "frob", r, better="min")
        cells = []
        for s in scens:
            g = d[(d.rule == r) & (d.scenario == s) & (d.method == "PMT")]
            if s not in pdiff or not len(g):
                cells.append("--"); continue
            v, se, best, k = pdiff[s]
            cells.append(f"{g.frob.mean():.3f} ({fmt(v, se)})")
        lines.append(RNAME[r] + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, fname), "w").write("\n".join(lines) + "\n")


def full_table(d, scens, rule, metrics, fname):
    """All methods x scenarios for one rule (Supplement)."""
    meths = [m for m in LABEL if m in set(d.method)]
    lines = ["\\begin{tabular}{l" + "c" * (len(scens) * len(metrics)) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{{len(metrics)}}}{{c}}{{{SCEN[s]}}}" for s in scens) + "\\\\",
             " & " + " & ".join(" & ".join(metrics) for _ in scens) + "\\\\", "\\midrule"]
    g = d[d.rule == rule]
    for m in meths:
        cells = []
        for s in scens:
            x = g[(g.method == m) & (g.scenario == s)]
            for k in metrics:
                cells.append(f"{x[k].mean():.3f}" if len(x) else "--")
        lines.append(LABEL[m] + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, fname), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    summ = []
    for f, scens, tag in [("tuning_p100.csv", ["clean", "t3", "row", "homcell", "hetcell", "hetcell_asym", "hetcell_mod"], "p100"),
                          ("tuning_p200.csv", ["clean", "homcell", "hetcell"], "p200")]:
        d = load(f)
        if d is None:
            continue
        scens = [s for s in scens if s in set(d.scenario)]
        rule_table(d, scens, f"tuning_{tag}.tex", tag)
        for r in ["heldout", "starsH", "stars08", "bic", "pfwer0.1"]:
            full_table(d, scens, r, ["f1", "fdr"], f"tuning_{tag}_{r.replace('.', '')}.tex")
        full_table(d, scens, "cv", ["frob"], f"tuning_{tag}_cv.tex")
        for r in sorted(set(d.rule)):
            for sc, (v, se, best, k) in paired(d, "f1", r).items():
                summ.append(dict(file=tag, rule=r, scenario=sc, metric="f1", diff=v, se=se, best=best, reps=k))
            for sc, (v, se, best, k) in paired(d, "frob", r, "min").items():
                summ.append(dict(file=tag, rule=r, scenario=sc, metric="frob", diff=v, se=se, best=best, reps=k))
        print(f, d.groupby("scenario").rep.nunique().to_dict())
    dg = load("tuning_graphs.csv")
    if dg is not None:
        for gname in sorted(set(dg.graph)):
            d = dg[dg.graph == gname]
            scens = [s for s in ["clean", "homcell", "hetcell"] if s in set(d.scenario)]
            rule_table(d, scens, f"tuning_{gname}.tex", gname)
            full_table(d, scens, "starsH", ["f1", "fdr"], f"tuning_{gname}_starsH.tex")
            for r in sorted(set(d.rule)):
                for sc, (v, se, best, k) in paired(d, "f1", r).items():
                    summ.append(dict(file=gname, rule=r, scenario=sc, metric="f1", diff=v, se=se, best=best, reps=k))
                for sc, (v, se, best, k) in paired(d, "frob", r, "min").items():
                    summ.append(dict(file=gname, rule=r, scenario=sc, metric="frob", diff=v, se=se, best=best, reps=k))
    di, dp = load("tuning_p100_iter.csv"), load("tuning_p100.csv")
    if di is not None and dp is not None:
        # iterative reweighting methods next to PMT and MT-GLASSO from the main tuning study (same seeds)
        scens = [s for s in ["clean", "row", "homcell", "hetcell"] if s in set(di.scenario)]
        reps = di.groupby(["scenario"]).rep.unique()
        dpp = pd.concat([dp[(dp.scenario == s) & dp.rep.isin(reps[s])] for s in scens])
        dd = pd.concat([di, dpp[dpp.method.isin(["PMT", "MT-GLASSO(TH)", "Wrap-GL", "Kendall-GL"])]])
        meths = [m for m in ["tlasso", "DPD-GL", "MT-GLASSO(TH)", "Kendall-GL", "Wrap-GL", "PMT"] if m in set(dd.method)]
        lines = ["\\begin{tabular}{ll" + "c" * len(scens) + "}", "\\toprule",
                 "Rule & Method & " + " & ".join(SCEN[s] + f" ({len(reps[s])})" for s in scens) + "\\\\", "\\midrule"]
        for r in ["heldout", "starsH", "bic"]:
            for i, m in enumerate(meths):
                cells = []
                for s in scens:
                    x = dd[(dd.rule == r) & (dd.method == m) & (dd.scenario == s)]
                    cells.append(f"{x.f1.mean():.3f} / {x.fdr.mean():.2f}" if len(x) else "--")
                lines.append((RNAME[r] if i == 0 else "") + " & " + LABEL[m] + " & " + " & ".join(cells) + "\\\\")
            lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines.append("\\end{tabular}")
        open(os.path.join(TAB, "tuning_iter.tex"), "w").write("\n".join(lines) + "\n")
    for f in ("tuning_graphs.csv", "tuning_p100_iter.csv"):
        d = load(f)
        if d is None:
            continue
        print(f)
        print(d.pivot_table(index=["graph", "scenario", "rule"] if "graph" in d else ["scenario", "rule"],
                            columns="method", values="f1").round(3).to_string())
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(RES, "tuning_summary.csv"), index=False)
    if len(s):
        print(s[s.rule.isin(["heldout", "starsH", "stars08", "bic", "pfwer0.1", "cv", "heldout_frob"])].round(4).to_string())
