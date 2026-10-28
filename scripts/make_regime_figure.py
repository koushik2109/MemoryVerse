#!/usr/bin/env python3
"""
Generate publication-quality Figure 2: Retrieval effectiveness across architectures and query formulations.
Synchronized with exact frozen metrics from Table III, Table XII, and colqwen_paired_v10.csv.
Formatted for IEEE/IJCDS single-column width (3.45 inches).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]

# Set style for publication (single column: 3.45 in)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(3.45, 2.5), dpi=300)

formulations = ["Verbatim\nLookup", "Light\nParaphrase", "Answerable\nQuestions"]

# Table III & verified Kaggle GPU metrics
char_mrr    = [0.9436, 0.2427, 0.9471]
word_mrr    = [0.8371, 0.4334, 0.9014]
dense_mrr   = [0.3022, 0.1601, 0.5128]
colqwen_mrr = [0.8529, 0.4962, 0.8595]

x = np.arange(len(formulations))
width = 0.20

rects1 = ax.bar(x - 1.5*width, char_mrr, width, label="Char BM25 (CPU)", color="#1f4e79", edgecolor="black", linewidth=0.4)
rects2 = ax.bar(x - 0.5*width, word_mrr, width, label="Word BM25 (CPU)", color="#2e75b6", edgecolor="black", linewidth=0.4)
rects3 = ax.bar(x + 0.5*width, dense_mrr, width, label="Dense BGE (CPU)", color="#d55e00", edgecolor="black", linewidth=0.4)
rects4 = ax.bar(x + 1.5*width, colqwen_mrr, width, label="ColQwen2 (GPU)", color="#009e73", edgecolor="black", linewidth=0.4)

for rects in [rects1, rects2, rects3, rects4]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 1.5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.8, rotation=0)

ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(formulations)
ax.set_ylim(0, 1.10)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="#cccccc", loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, handletextpad=0.3, columnspacing=0.8)
ax.grid(axis="y", linestyle=":", alpha=0.5)
plt.tight_layout(pad=0.3)

out_fig1 = repo_root / "ijcds" / "figures" / "fig_mrr_regime_systems.png"
out_fig2 = repo_root / "figures" / "fig_mrr_regime_systems.png"
out_fig3 = repo_root / "results" / "v10" / "fig_crossover_repro_v10.png"

out_fig1.parent.mkdir(parents=True, exist_ok=True)
out_fig2.parent.mkdir(parents=True, exist_ok=True)
out_fig3.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(out_fig1, dpi=300)
plt.savefig(out_fig2, dpi=300)
plt.savefig(out_fig3, dpi=300)
plt.close()
print("Updated scripts/make_regime_figure.py and regenerated images.")
