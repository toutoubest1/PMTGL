"""Shared plot colors, line styles and markers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), "-", "--"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#888888", "lines.linewidth": 1.6, "lines.markersize": 4.5,
                     "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
