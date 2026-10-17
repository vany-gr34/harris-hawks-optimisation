"""Turn results/raw.json into results/summary.md, summary.csv and convergence.png.

Usage:  python report.py [results/raw.json]
"""

import csv
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = sys.argv[1] if len(sys.argv) > 1 else "results/raw.json"
OUT = os.path.dirname(RAW) or "."

TABLE_ORDER = ["HHO", "HHO (no dive)", "HHO (rank dec)", "HHO (rand key)", "GA", "SA", "Random"]
PLOT_ORDER = ["HHO", "GA", "SA", "Random"]
# categorical slots 1-4 of the reference palette (validated, light mode)
COLORS = {"HHO": "#2a78d6", "GA": "#eb6834", "SA": "#1baf7a", "Random": "#eda100"}
STYLES = {"HHO": "-", "GA": "--", "SA": "-.", "Random": ":"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

data = json.load(open(RAW))
grid = np.array(data["grid"])
results = data["results"]
n_runs = data["config"]["runs"]


def gap(cost, best_known):
    return 100.0 * (np.asarray(cost, dtype=float) - best_known) / best_known


# ---------------------------------------------------------------- table
rows, md = [], []
md.append(f"Results over {n_runs} independent runs per instance "
          f"(population {data['config']['pop']}, {data['config']['iters']} iterations). "
          "Gap = (cost - best known) / best known. "
          "GA, SA and Random get the same number of objective evaluations as the "
          "full HHO run with the same seed.\n")

for name, block in results.items():
    bk = block["best_known"]
    md.append(f"\n**{name}** (best known {bk})\n")
    md.append("| Method | Best | Mean | Std | Worst | Mean gap % | Evals | Time (s) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in TABLE_ORDER:
        costs = np.array([r["methods"][m]["cost"] for r in block["runs"]])
        evals = np.mean([r["methods"][m]["evals"] for r in block["runs"]])
        secs = np.mean([r["methods"][m]["time"] for r in block["runs"]])
        row = dict(instance=name, method=m, best=costs.min(), mean=costs.mean(),
                   std=costs.std(), worst=costs.max(), mean_gap=gap(costs.mean(), bk),
                   best_gap=gap(costs.min(), bk), evals=evals, time=secs)
        rows.append(row)
        md.append(f"| {m} | {row['best']:.0f} | {row['mean']:.0f} | {row['std']:.0f} | "
                  f"{row['worst']:.0f} | {row['mean_gap']:.1f} | {evals:.0f} | {secs:.1f} |")

# compact gap summary
md.append("\n**Mean gap to best known (%), all instances**\n")
names = list(results)
md.append("| Method | " + " | ".join(names) + " |")
md.append("|---|" + "---:|" * len(names))
for m in TABLE_ORDER:
    vals = [next(r["mean_gap"] for r in rows if r["instance"] == n and r["method"] == m) for n in names]
    md.append(f"| {m} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")

open(os.path.join(OUT, "summary.md"), "w").write("\n".join(md) + "\n")
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# ---------------------------------------------------------------- plot
show = [n for n in ["eil51", "berlin52", "kroA100", "A-n32-k5"] if n in results] or names[:4]
ncols = 2
nrows = int(np.ceil(len(show) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.2 * nrows), facecolor=SURFACE)
axes = np.atleast_1d(axes).ravel()

for ax, name in zip(axes, show):
    ax.set_facecolor(SURFACE)
    bk = results[name]["best_known"]
    ends = {}
    for m in PLOT_ORDER:
        curves = np.array([r["methods"][m]["curve"] for r in results[name]["runs"]])
        mean = gap(curves, bk).mean(axis=0)
        ax.plot(grid, mean, color=COLORS[m], ls=STYLES[m], lw=2, label=m)
        ends[m] = mean[-1]
    # direct labels at the line ends, nudged apart in log space so they never collide
    order = sorted(ends, key=lambda k: ends[k])
    pos, min_sep = {}, 0.09
    for m in order:
        y = np.log10(ends[m])
        if pos and y - pos[prev] < min_sep:
            y = pos[prev] + min_sep
        pos[m], prev = y, m
    for m in order:
        ax.annotate(f"{m} {ends[m]:.0f}%", (grid[-1], ends[m]), xytext=(1.02, 10 ** pos[m]),
                    textcoords="data", va="center",
                    fontsize=9, color=INK2, annotation_clip=False)
    ax.set_yscale("log")
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(name, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("fraction of evaluation budget", color=INK2, fontsize=9)
    ax.set_ylabel("mean gap to best known (%)", color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
for ax in axes[len(show):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.suptitle("Convergence at equal evaluation budget (mean of %d runs)" % n_runs,
             y=0.995, x=0.01, ha="left", fontsize=12, color=INK)
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=len(labels),
           frameon=False, fontsize=10, labelcolor=INK)
fig.tight_layout(rect=(0, 0, 0.97, 0.91))
fig.savefig(os.path.join(OUT, "convergence.png"), dpi=150, facecolor=SURFACE)
print("wrote summary.md, summary.csv, convergence.png in", OUT)
