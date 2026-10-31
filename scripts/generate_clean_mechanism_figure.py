import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set high-quality typography
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.titlesize": 11,
    "mathtext.fontset": "stix",
    "figure.autolayout": False,
})

repo_root = Path("/Users/anuragkrishnavennela/Projects/MemoryVerse")
v2_dir = repo_root / "results" / "v2"

# 1. Load data
frag_df = pd.read_csv(v2_dir / "mechanism_tokenization_fragmentation.csv")
drift_df = pd.read_csv(v2_dir / "mechanism_embedding_drift.csv")
attrib_df = pd.read_csv(v2_dir / "mechanism_error_attribution.csv")

# Filter synthetic rates
frag_syn = frag_df[frag_df["Noise_Rate"] >= 0].sort_values("Noise_Rate")
drift_syn = drift_df[drift_df["Noise_Rate"] >= 0].sort_values("Noise_Rate")

x_rates = [0, 10, 20, 40]
x_rate_pct = [r * 100 for r in frag_syn["Noise_Rate"]]

# Create 1x3 composite figure
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3,
    figsize=(11.5, 3.2),
    gridspec_kw={"width_ratios": [1.0, 1.0, 1.55]},
    dpi=300
)

# ----------------- Panel (a): Tokenization Fragmentation -----------------
ax1.plot(x_rates, frag_syn["MiniLM_Subwords_Per_Word"], marker="o", color="#d62728", lw=2, ms=5.5, label="MiniLM (WordPiece)")
ax1.plot(x_rates, frag_syn["BGE_Subwords_Per_Word"], marker="s", color="#ff7f0e", lw=2, ms=5.5, label="BGE-base (WordPiece)")
ax1.set_title("(a) Subword Fragmentation", fontweight="bold", pad=8)
ax1.set_xlabel("Synthetic Noise Rate (%)")
ax1.set_ylabel("Mean Subwords / Word")
ax1.set_xticks(x_rates)
ax1.set_xticklabels([f"{x}%" for x in x_rates])
ax1.set_ylim(1.2, 3.15)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#cccccc")

# ----------------- Panel (b): Embedding Drift -----------------
ax2.plot(x_rates, drift_syn["MiniLM_Cosine_Sim"], marker="o", color="#d62728", lw=2, ms=5.5, label="MiniLM Cosine Sim")
ax2.plot(x_rates, drift_syn["BGE_Cosine_Sim"], marker="s", color="#ff7f0e", lw=2, ms=5.5, label="BGE-base Cosine Sim")
ax2.set_title("(b) Embedding Cosine Drift", fontweight="bold", pad=8)
ax2.set_xlabel("Synthetic Noise Rate (%)")
ax2.set_ylabel("Cosine Sim to Clean Embedding")
ax2.set_xticks(x_rates)
ax2.set_xticklabels([f"{x}%" for x in x_rates])
ax2.set_ylim(0.2, 1.05)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cccccc")

# ----------------- Panel (c): Error Attribution -----------------
buckets = ["Numeric-Heavy", "Short (<=4)", "Rare-Token", "Named Entity", "Long (>=8)"]
bucket_keys = ["Numeric-Heavy", "Short (<=4 tokens)", "Rare-Token", "Named Entity / Acronym", "Long (>=8 tokens)"]

plot_systems = [
    ("BM25", "BM25 (Word)", "#2b5c8f"),
    ("BM25 (Char 3-5gram)", "Char 3-5g BM25", "#6baed6"),
    ("Dense MiniLM", "Dense MiniLM", "#d62728"),
    ("Dense BGE-base", "Dense BGE", "#ff7f0e"),
    ("Hybrid (Tuned)", "Tuned Hybrid", "#2ca02c"),
]

x = np.arange(len(buckets))
width = 0.155

for idx, (sys_raw, sys_label, col) in enumerate(plot_systems):
    vals = []
    for b_key in bucket_keys:
        row = attrib_df[(attrib_df["Bucket"] == b_key) & (attrib_df["System"] == sys_raw)]
        vals.append(row["MRR_20pct_Noise"].values[0] if len(row) > 0 else 0.0)
    ax3.bar(x + idx * width - (len(plot_systems) - 1) * width / 2, vals, width, label=sys_label, color=col, edgecolor="black", linewidth=0.4)

ax3.set_title("(c) Probe Property Error Attribution (20% Noise)", fontweight="bold", pad=8)
ax3.set_xticks(x)
ax3.set_xticklabels(buckets, fontsize=8.2, fontweight="medium")
ax3.set_ylabel("MRR at 20% Noise")
ax3.set_ylim(0, 0.88)
ax3.grid(True, linestyle=":", alpha=0.6, axis="y")
ax3.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cccccc", ncol=2, fontsize=7.2)

fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.88, wspace=0.28)

out_composite = repo_root / "figures" / "fig_mechanism_composite.png"
fig.savefig(out_composite, dpi=300)
out_ijcds = repo_root / "ijcds" / "figures" / "fig_mechanism_composite.png"
out_ijcds.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_ijcds, dpi=300)
print(f"Generated {out_composite} and {out_ijcds}")
