"""Generate the efficiency and sensitivity table and the population bias plot."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from style import plt, PAL, MARK, LS
from pmtgl.theory import are_vs_pearson, ges_corr
from pmtgl import covariance as C
OUT = os.path.join(os.path.dirname(__file__), "..", "paper")

# ARE and GES
A = [0.05, 0.1, 0.25, 0.5, 1.0]
R = [0.0, 0.3, 0.6, 0.9]
lines = [r"\begin{tabular}{l" + "c" * len(R) + "c}", r"\toprule",
         r"$a$ & " + " & ".join([f"ARE, $\\rho={r}$" for r in R]) + r" & GES ($\rho=0$)\\", r"\midrule"]
for a in A:
    are = [are_vs_pearson(r, a, a) for r in R]
    ges = (1 + a) ** 3 / (a * np.e)
    lines.append(f"{a} & " + " & ".join(f"{x:.3f}" for x in are) + f" & {ges:.2f}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tables", "are.tex"), "w").write("\n".join(lines))
# numerical cross-check of GES closed form
assert abs(ges_corr(0, 0.5, 0.5, m=201) - 1.5 ** 3 / (0.5 * np.e)) < 0.02

# Bias versus outlier magnitude
rng = np.random.default_rng(7); n = 200_000; rho = 0.6
Ms = [2, 3, 4, 5, 6, 8, 10, 14, 20]
rows = []
for eps in (0.1, 0.2):
    for M in Ms:
        Z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
        b = rng.random(n) < eps
        Z[b, 0] = M * rng.choice([-1, 1], b.sum())
        corr = lambda S: S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])
        rows += [dict(eps=eps, M=M, method="Pearson", bias=np.corrcoef(Z.T)[0, 1] - rho),
                 dict(eps=eps, M=M, method="Gaussian rank", bias=corr(C.gaussrank_cov(Z)[0]) - rho),
                 dict(eps=eps, M=M, method="Wrapping", bias=corr(C.wrap_cov(Z)[0]) - rho),
                 dict(eps=eps, M=M, method="Winsorization", bias=corr(C.winsor_cov(Z[:20000])[0]) - rho),
                 dict(eps=eps, M=M, method="Kendall", bias=corr(C.kendall_cov(Z[:4000])[0]) - rho),
                 dict(eps=eps, M=M, method="PMT, a=0.25", bias=corr(C.pmt_cov(Z, a=[.25, .25])[0]) - rho),
                 dict(eps=eps, M=M, method="PMT, a=0.5", bias=corr(C.pmt_cov(Z, a=[.5, .5])[0]) - rho)]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "bias_curve.csv"), index=False)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
meths = ["Pearson", "Gaussian rank", "Kendall", "Winsorization", "Wrapping", "PMT, a=0.25", "PMT, a=0.5"]
for ax, eps in zip(axs, (0.1, 0.2)):
    for i, m in enumerate(meths):
        d = df[(df.eps == eps) & (df.method == m)]
        ax.plot(d.M, d.bias, color=PAL[i], marker=MARK[i], ls=LS[i], label=m)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_title(f"$\\varepsilon={eps}$"); ax.set_xlabel("outlier magnitude $M$ (in SD)")
axs[0].set_ylabel("bias of correlation ($\\rho=0.6$)")
axs[1].legend(loc="lower right", fontsize=7, ncol=2)
fig.savefig(os.path.join(OUT, "figures", "bias_curve.pdf"))
print(df.pivot_table(index=["eps", "M"], columns="method", values="bias").round(3).to_string())
