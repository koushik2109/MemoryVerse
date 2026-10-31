import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Set publication typography matching IEEE/IJCDS Times font
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11,
    "mathtext.fontset": "stix",
    "figure.autolayout": False,
})

repo_root = Path("/Users/anuragkrishnavennela/Projects/MemoryVerse")
v6_dir = repo_root / "results" / "v6"

df_overlap = pd.read_csv(v6_dir / "dose_response_overlap_v6.csv")
df_deg = pd.read_csv(v6_dir / "dose_response_degradation_v6.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), dpi=300)

colors = {
    "Word BM25": "#1f77b4",
    "Char 3-5g BM25": "#ff7f0e",
    "Dense BGE-Base": "#2ca02c",
    "Tuned Hybrid RRF": "#d62728"
}

# ----------------- Axis 1: MRR vs Query Overlap -----------------
# x_ov is token overlap %: [15, 40, 70, 100]
df_ov_sorted = df_overlap.iloc[::-1].reset_index(drop=True)
x_ov = [15, 40, 70, 100]

l1, = ax1.plot(x_ov, df_ov_sorted["Word_BM25_MRR"], "o-", label="Word BM25", color=colors["Word BM25"], lw=2, ms=6)
l2, = ax1.plot(x_ov, df_ov_sorted["Char_BM25_MRR"], "s-", label="Char 3-5g BM25", color=colors["Char 3-5g BM25"], lw=2, ms=6)
l3, = ax1.plot(x_ov, df_ov_sorted["BGE_Base_MRR"], "^-", label="Dense BGE-Base", color=colors["Dense BGE-Base"], lw=2, ms=6)
l4, = ax1.plot(x_ov, df_ov_sorted["Tuned_Hybrid_RRF_MRR"], "D-", label="Tuned Hybrid RRF", color=colors["Tuned Hybrid RRF"], lw=2, ms=6)
l5 = ax1.axhline(0.0090, color="gray", linestyle="--", alpha=0.7, lw=1.5, label="Uniform Random (0.009)")

ax1.set_xlabel("Query-Document Token Overlap (%)", fontweight="bold")
ax1.set_ylabel("Retrieval MRR", fontweight="bold")
ax1.set_title("(a) Dose-Response: Query Overlap Axis\n(OCR Fixed Clean)", fontweight="bold", pad=8)
ax1.set_xticks(x_ov)
ax1.set_xticklabels(["15%", "40%", "70%", "100%"])
ax1.set_ylim(-0.02, 1.05)
ax1.grid(True, linestyle=":", alpha=0.6)

# ----------------- Axis 2: MRR vs Degradation Severity -----------------
x_deg = [0, 1, 2, 3, 4]
deg_labels = ["L0 Clean\n(CER 29%)", "L1 Mild\n(CER 34%)", "L2 Mod\n(CER 45%)", "L3 Sev\n(CER 64%)", "L4 Ext\n(CER 78%)"]

ax2.plot(x_deg, df_deg["Word_BM25_MRR"], "o-", label="Word BM25", color=colors["Word BM25"], lw=2, ms=6)
ax2.plot(x_deg, df_deg["Char_BM25_MRR"], "s-", label="Char 3-5g BM25", color=colors["Char 3-5g BM25"], lw=2, ms=6)
ax2.plot(x_deg, df_deg["BGE_Base_MRR"], "^-", label="Dense BGE-Base", color=colors["Dense BGE-Base"], lw=2, ms=6)
ax2.plot(x_deg, df_deg["Tuned_Hybrid_RRF_MRR"], "D-", label="Tuned Hybrid RRF", color=colors["Tuned Hybrid RRF"], lw=2, ms=6)
ax2.axhline(0.0090, color="gray", linestyle="--", alpha=0.7, lw=1.5, label="Uniform Random (0.009)")

ax2.set_xlabel("Physical Optical Degradation Severity", fontweight="bold")
ax2.set_ylabel("Retrieval MRR", fontweight="bold")
ax2.set_title("(b) Dose-Response: Degradation Severity Axis\n(Queries Fixed Verbatim)", fontweight="bold", pad=8)
ax2.set_xticks(x_deg)
ax2.set_xticklabels(deg_labels, fontsize=8.0)
ax2.set_ylim(-0.02, 1.05)
ax2.grid(True, linestyle=":", alpha=0.6)

# ----------------- Shared Top Legend -----------------
handles = [l1, l2, l3, l4, l5]
labels = ["Word BM25", "Char 3-5g BM25", "Dense BGE-Base", "Tuned Hybrid RRF", "Uniform Random (0.009)"]
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=5,
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
    fontsize=8.5,
    columnspacing=1.8,
    handletextpad=0.5
)

fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.82, wspace=0.22)

# Save to destination paths
out_paths = [
    repo_root / "figures" / "fig_dose_response_dual_v6.png",
    repo_root / "ijcds" / "figures" / "fig_dose_response_dual_v6.png",
    v6_dir / "fig_dose_response_dual_v6.png",
]

for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=300)
    print(f"Saved: {p}")

plt.close()
