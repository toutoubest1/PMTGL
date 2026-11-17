"""Summarize marginal diagnostics and 18-edge graphs for the full Sachs data."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import stats
import real_sachs as R
from pmtgl.covariance import robust_center_scale
X = R.X_all
m, s = robust_center_scale(X)
Z = (X - m) / s
diag = pd.DataFrame(dict(var=R.names, skew=stats.skew(X), exkurt=stats.kurtosis(X),
                         tail35=(np.abs(Z) > 3.5).mean(0)))
print(diag.round(3).to_string())
print("Gaussian P(|Z|>3.5) =", round(2 * stats.norm.sf(3.5), 5))
G = {}
for meth in R.METH:
    G[meth] = R.at_k([T for (T, _, _) in R.METHODS[meth](X, R.FINE)])
J = pd.DataFrame({a: {b: R.jacc(G[a], G[b]) for b in R.METH} for a in R.METH})
prec = {a: np.sum(G[a] & R.CONS[R.iu]) / G[a].sum() for a in R.METH}
print(J.round(2).to_string()); print(pd.Series(prec).round(3).to_string())
out = os.path.join(os.path.dirname(__file__), "..", "results")
diag.to_csv(os.path.join(out, "sachs_marginals.csv"), index=False)
J.to_csv(os.path.join(out, "sachs_fullgraph_jaccard.csv"))
pd.Series(prec, name="prec_cons").to_csv(os.path.join(out, "sachs_fullgraph_prec.csv"))
