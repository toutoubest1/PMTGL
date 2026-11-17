import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from make_tables import scenario_table
R = os.path.join(os.path.dirname(__file__), "..")
d = pd.read_csv(os.path.join(R, "results", "composite.csv"))
M = ["GLASSO", "Kendall-GL", "MT-GLASSO(TH)", "PMT", "PMT-mt", "Composite", "Composite-trunc"]
r1 = scenario_table(d, 200, 50, "er2", ["clean", "t3", "row", "row_struct", "homcell", "hetcell", "hetcell_mod", "hetcell_asym"],
                    M, ["auc", "f1", "frob"], "composite_p50.tex")
r2 = scenario_table(d, 100, 200, "er2", ["clean", "t3", "row", "homcell", "hetcell"], M, ["auc", "f1", "frob"], "composite_p200.tex")
print(r1, r2)
print(d.groupby(["p", "scenario", "method"])[["auc", "f1", "frob"]].mean().round(3).unstack("method").to_string())
