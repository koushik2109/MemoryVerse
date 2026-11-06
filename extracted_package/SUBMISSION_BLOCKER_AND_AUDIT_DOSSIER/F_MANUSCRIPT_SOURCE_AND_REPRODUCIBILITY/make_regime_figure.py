#!/usr/bin/env python3
"""Generate the publication Figure 2 regime comparison.

The line chart makes the query-formulation crossover visible without the
rotated value labels used by the earlier grouped-bar version.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]

# Set style for publication (single column: 3.45 in).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(3.45, 2.45), dpi=300)

formulations = ["Verbatim\nLookup", "Light\nParaphrase", "Answerable\nQuestions"]

# Table III & verified Kaggle GPU metrics (Apple Clean Real-OCR condition across all formulations)
char_mrr    = [0.9436, 0.2427, 0.9328]
word_mrr    = [0.8371, 0.4334, 0.8841]
dense_mrr   = [0.3022, 0.1601, 0.5128]
colqwen_mrr = [0.8417, 0.4962, 0.8595]

x = np.arange(len(formulations))
series = [
    ("Char BM25", char_mrr, "#1f4e79", "o", "-"),
    ("Word BM25", word_mrr, "#2e75b6", "s", "--"),
    ("Dense BGE", dense_mrr, "#d55e00", "^", "-"),
    ("ColQwen2", colqwen_mrr, "#009e73", "D", "-"),
]

# Specific vertical offsets for the right-side value labels to prevent collision (0.884 vs 0.860)
label_offsets = {
    "Char BM25": 0.000,
    "Word BM25": +0.012,
    "ColQwen2": -0.012,
    "Dense BGE": 0.000,
}

for label, values, color, marker, linestyle in series:
    ax.plot(
        x,
        values,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.6,
        markersize=4.2,
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=3,
    )
    offset = label_offsets.get(label, 0.0)
    ax.text(
        x[-1] + 0.07,
        values[-1] + offset,
        f"{values[-1]:.3f}",
        color=color,
        fontsize=6.5,
        va="center",
        ha="left",
    )

ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontweight="bold")
ax.set_xlabel("Query formulation")
ax.set_xticks(x)
ax.set_xticklabels(formulations)
ax.set_xlim(-0.15, 2.52)
ax.set_ylim(0, 1.02)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.axvspan(0.55, 1.45, color="#f3f4f6", zorder=0)
ax.text(1.0, 0.985, "semantic variation", ha="center", va="top", fontsize=6.5, color="#6b7280")
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
ax.grid(axis="x", visible=False)
ax.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.47, 1.13),
    ncol=4,
    handletextpad=0.25,
    columnspacing=0.55,
    borderaxespad=0,
)
fig.subplots_adjust(left=0.16, right=0.88, bottom=0.18, top=0.80)

out_fig1 = repo_root / "ijcds" / "figures" / "fig_mrr_regime_systems.png"
out_fig2 = repo_root / "figures" / "fig_mrr_regime_systems.png"
out_fig3 = repo_root / "results" / "v10" / "fig_crossover_repro_v10.png"
out_fig4 = repo_root / "paper_v4" / "figures" / "fig_mrr_regime_systems.png"

out_fig1.parent.mkdir(parents=True, exist_ok=True)
out_fig2.parent.mkdir(parents=True, exist_ok=True)
out_fig3.parent.mkdir(parents=True, exist_ok=True)
out_fig4.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(out_fig1, dpi=300, bbox_inches="tight")
plt.savefig(out_fig2, dpi=300, bbox_inches="tight")
plt.savefig(out_fig3, dpi=300, bbox_inches="tight")
plt.savefig(out_fig4, dpi=300, bbox_inches="tight")
plt.close()
print("Updated scripts/make_regime_figure.py and regenerated images including paper_v4.")
