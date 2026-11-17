"""Compare PMT with composite row and cell weighting."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run
M = ["GLASSO", "Kendall-GL", "MT-GLASSO(TH)", "PMT", "PMT-mt", "Composite", "Composite-trunc"]
S = [(200, 50, "er2", s, None, None) for s in
     ["clean", "t3", "row", "row_struct", "homcell", "hetcell", "hetcell_mod", "hetcell_asym"]]
S += [(100, 200, "er2", s, None, None) for s in ["clean", "t3", "row", "homcell", "hetcell"]]
if __name__ == "__main__":
    run(S, int(sys.argv[1]) if len(sys.argv) > 1 else 10, M, "results/composite.csv", tag="comp")
