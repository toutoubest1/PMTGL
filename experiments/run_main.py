"""Run the main simulation grid.

Usage: python experiments/run_main.py PART REPS
Parts a, b and c cover ER2, other graph families and p=400, respectively.
Part d uses fixed replicate indices that extend the grids in a and c."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run
ALL = ["GLASSO", "Kendall-GL", "GaussRank-GL", "Wrap-GL", "Winsor-GL", "MT-GLASSO(TH)",
       "MT-GLASSO(mass)", "tlasso", "Trimmed-GL", "DPD-GL", "PMT-fixed", "PMT", "PMT-mt"]
FAST = [m for m in ALL if m not in ("tlasso", "Trimmed-GL", "DPD-GL")]
SIZES = [(200, 50), (100, 100), (100, 200)]
SC8 = ["clean", "t3", "row", "row_struct", "homcell", "hetcell", "hetcell_mod", "hetcell_asym"]
SC5 = ["clean", "t3", "row", "homcell", "hetcell"]
if __name__ == "__main__":
    part, reps = sys.argv[1], int(sys.argv[2])
    if part == "a":
        # Cap replicates at 10 for the larger row-structural and cellwise settings.
        S = [(n, p, "er2", s, None, None) for (n, p) in SIZES for s in SC8
             if p < 200 or s in ("clean", "t3", "row")]
        run(S, reps, ALL, "results/main_a.csv", tag="main")
        S = [(100, 200, "er2", s, None, None) for s in SC8 if s not in ("clean", "t3", "row")]
        run(S, min(reps, 10), ALL, "results/main_a.csv", tag="main")
    elif part == "b":
        # Fit t-lasso only at p = 50 in this grid.
        S = [(200, 50, g, s, None, None) for g in ("er6", "band", "hub") for s in SC5]
        run(S, reps, ALL, "results/main_b.csv", tag="main")
        S = [(100, 200, g, s, None, None) for g in ("er6", "band", "hub") for s in SC5]
        run(S, reps, [m for m in ALL if m != "tlasso"], "results/main_b.csv", tag="main")
    elif part == "d":
        out = "results/main_d.csv"
        S = [(n, p, "er2", s, None, None) for (n, p) in SIZES[:2] for s in SC8]
        run(S, range(20, 30), ALL, out, tag="main")
        S = [(100, 200, "er2", s, None, None) for s in ("clean", "t3", "row")]
        run(S, range(20, 30), FAST, out, tag="main")
        S = [(100, 200, "er2", s, None, None) for s in SC8 if s not in ("clean", "t3", "row")]
        run(S, range(10, 30), FAST, out, tag="main")
        S = [(200, 400, "er2", s, None, None) for s in ("homcell", "hetcell")]
        run(S, range(5, 20), FAST, out, tag="main")
        S = [(200, 400, "er2", s, None, None) for s in ("clean", "row")]
        run(S, range(5, 10), FAST, out, tag="main")
    elif part == "c":
        S = [(200, 400, "er2", s, None, None) for s in ("clean", "homcell", "hetcell", "row")]
        run(S, reps, FAST, "results/main_c.csv", tag="main")
