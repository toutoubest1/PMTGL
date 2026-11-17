"""Summarize the main simulation grid and plot held-out F1."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from make_tables import scenario_table, LABEL, SCEN, TAB, RES
from style import plt, PAL, MARK, LS
FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
ALL = ["GLASSO", "Kendall-GL", "GaussRank-GL", "Wrap-GL", "Winsor-GL", "MT-GLASSO(TH)",
       "MT-GLASSO(mass)", "tlasso", "Trimmed-GL", "DPD-GL", "PMT-fixed", "PMT", "PMT-mt"]
FAST = [m for m in ALL if m not in ("tlasso", "Trimmed-GL", "DPD-GL")]
SC8 = ["clean", "t3", "row", "row_struct", "homcell", "hetcell", "hetcell_mod", "hetcell_asym"]
SC5 = ["clean", "t3", "row", "homcell", "hetcell"]
out = {}
# Pool parts a and c with part d, which uses the same replicate seed stream.
D = pd.read_csv(os.path.join(RES, "main_d.csv")) if os.path.exists(os.path.join(RES, "main_d.csv")) else None
if os.path.exists(os.path.join(RES, "main_a.csv")):
    a = pd.read_csv(os.path.join(RES, "main_a.csv"))
    if D is not None:
        a = pd.concat([a, D[D.p < 400]], ignore_index=True)
    for (n, p) in [(200, 50), (100, 100), (100, 200)]:
        for sc, tag in [(SC8[:4], "A"), (SC8[4:], "B")]:
            out[f"a_{p}_{tag}"] = scenario_table(a, n, p, "er2", sc, ALL, ["auc", "f1", "fdr", "frob"], f"main_er2_p{p}_{tag}.tex")
        out[f"a_{p}_bic"] = scenario_table(a, n, p, "er2", SC8, ALL, ["bic_f1", "bic_fdr"], f"main_er2_p{p}_bic.tex")
if os.path.exists(os.path.join(RES, "main_b.csv")):
    b = pd.read_csv(os.path.join(RES, "main_b.csv"))
    for g in ("er6", "band", "hub"):
        for (n, p) in [(200, 50), (100, 200)]:
            out[f"b_{g}_{p}"] = scenario_table(b, n, p, g, SC5, ALL, ["auc", "f1", "frob"], f"main_{g}_p{p}.tex")
if os.path.exists(os.path.join(RES, "main_c.csv")):
    c = pd.read_csv(os.path.join(RES, "main_c.csv"))
    if D is not None:
        c = pd.concat([c, D[D.p == 400]], ignore_index=True)
    out["c"] = scenario_table(c, 200, 400, "er2", ["clean", "row", "homcell", "hetcell"], FAST, ["auc", "f1", "fdr", "frob", "time"], "main_er2_p400.tex")
print(out)

# Summary figure: held-out F1 by scenario and size (er2), selected methods
if os.path.exists(os.path.join(RES, "main_a.csv")):
    sel = ["GLASSO", "Kendall-GL", "Wrap-GL", "Winsor-GL", "MT-GLASSO(TH)", "tlasso", "PMT", "PMT-mt"]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=True)
    for ax, (n, p) in zip(axs, [(200, 50), (100, 100), (100, 200)]):
        d = a[(a.p == p)]
        m = d.groupby(["scenario", "method"]).f1.mean().unstack().reindex(index=SC8, columns=sel)
        se = (d.groupby(["scenario", "method"]).f1.std().unstack()
              / np.sqrt(d.groupby(["scenario", "method"]).size().unstack())).reindex(index=SC8, columns=sel)
        y = np.arange(len(SC8))
        for i, mt in enumerate(sel):
            ax.errorbar(m.loc[SC8, mt], y + (i - 3.5) * 0.09, xerr=2 * se.loc[SC8, mt], fmt=MARK[i % 8],
                        color=PAL[i % 8], ms=3.5, lw=0.8, label=LABEL[mt])
        ax.set_yticks(y); ax.set_yticklabels([SCEN[s].replace("\\%", "%") for s in SC8])
        ax.invert_yaxis(); ax.set_title(f"n={n}, p={p}"); ax.set_xlabel("held-out F1")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.6, -0.22), ncol=4, fontsize=7)
    fig.savefig(os.path.join(FIG, "f1_er2.pdf"))
