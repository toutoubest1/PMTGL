"""Contamination-rate and outlier-magnitude sweeps (Sec. 6.4), n=200, p=50, er2.
20% of variables contaminated, symmetric outliers of fixed magnitude."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run
M = ["GLASSO", "Kendall-GL", "GaussRank-GL", "Wrap-GL", "Winsor-GL", "MT-GLASSO(TH)", "PMT-fixed", "PMT"]
S = [(200, 50, "er2", "sweep", e, 8.0) for e in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4)]
S += [(200, 50, "er2", "sweep", 0.1, m) for m in (2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 15.0)]
if __name__ == "__main__":
    run(S, int(sys.argv[1]), M, "results/sweeps.csv", tag="sweep")
