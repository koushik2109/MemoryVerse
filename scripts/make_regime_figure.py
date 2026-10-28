#!/usr/bin/env python3
"""
Generate publication-quality Figure 2: Retrieval effectiveness across architectures and query formulations.
Synchronized with exact frozen metrics from Table III, Table XII, and colqwen_paired_v10.csv.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]

# Set style for publication
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "figure.autolayout": True,
})

fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)

formulations = ["Verbatim Lookups", "Light Paraphrase\n(Entity Shift)", "Answerable Questions\n(Topical Anchors)"]

# Table III & verified Kaggle GPU metrics
char_mrr    = [0.9436, 0.2427, 0.9471]
word_mrr    = [0.8371, 0.4334, 0.9014]
dense_mrr   = [0.3022, 0.1601, 0.5128]
colqwen_mrr = [0.8529, 0.4962, 0.8595]

x = np.arange(len(formulations))
width = 0.18

rects1 = ax.bar(x - 1.5*width, char_mrr, width, label='Char 3-5g BM25 (CPU)', color='#1f4e79', edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x - 0.5*width, word_mrr, width, label='Word BM25 (CPU)', color='#2e75b6', edgecolor='black', linewidth=0.5)
rects3 = ax.bar(x + 0.5*width, dense_mrr, width, label='Dense BGE-Base (CPU)', color='#d55e00', edgecolor='black', linewidth=0.5)
rects4 = ax.bar(x + 1.5*width, colqwen_mrr, width, label='ColQwen2-v1.0 (Tesla T4 GPU)', color='#009e73', edgecolor='black', linewidth=0.5)

# Add exact value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2.5),  # 2.5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6.8, rotation=0)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

ax.set_ylabel('Mean Reciprocal Rank (MRR)', fontweight='bold')
ax.set_title('Sparse vs. Dense vs. Native-Visual Retrieval Across Query Formulations (Clean OCR)', fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(formulations, fontweight='medium')
ax.set_ylim(0, 1.08)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
ax.legend(frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc', loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)
ax.grid(axis='y', linestyle=':', alpha=0.5)

# Save to both target locations
out_fig1 = repo_root / 'ijcds' / 'figures' / 'fig_mrr_regime_systems.png'
out_fig2 = repo_root / 'results' / 'v10' / 'fig_crossover_repro_v10.png'

out_fig1.parent.mkdir(parents=True, exist_ok=True)
out_fig2.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(out_fig1, dpi=300)
plt.savefig(out_fig2, dpi=300)
plt.close()
print(f"Successfully generated:\n  {out_fig1}\n  {out_fig2}")

