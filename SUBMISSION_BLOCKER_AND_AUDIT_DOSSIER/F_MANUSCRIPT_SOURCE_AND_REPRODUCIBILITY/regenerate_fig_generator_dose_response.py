#!/usr/bin/env python3
"""scripts/regenerate_fig_generator_dose_response.py

Regenerates figures/fig_generator_dose_response.png with exact two-column alignment
matching the IEEEtran column geometry on Page 7:
- Left Subplot (a) spans across Column 1 (x in [56.69, 288.67] pt)
- Right Subplot (b) spans across Column 2 (x in [306.60, 538.58] pt)
- The central whitespace between Subplots (a) and (b) aligns with the column gutter (17.93 pt)
- Spacing, legend box, and typography are clean, spacious, and breathable.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "results" / "v4" / "G-10" / "generator_dose_response_summary.csv"
OUT_FIG_PATH = REPO_ROOT / "figures" / "fig_generator_dose_response.png"

def main():
    if not SUMMARY_PATH.exists():
        # Fallback to gapfill_v1 path
        summary_path = REPO_ROOT / "results" / "gapfill_v1" / "T3-02" / "generator_dose_response_summary.csv"
    else:
        summary_path = SUMMARY_PATH

    df_dose = pd.read_csv(summary_path)

    # Styling matching IEEE publication typography
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    # Figure dimensions: 13.2 in x 4.2 in (aspect ratio ~3.14)
    # When rendered at width=\textwidth (481.89 pt), height is ~153 pt, fitting Page 7 perfectly.
    fig = plt.figure(figsize=(13.2, 4.2), dpi=300)

    # Column-aligned geometry:
    # Column 1 span: [0.072, 0.472], Column 2 span: [0.578, 0.978]
    # Gutter span [0.472, 0.578] contains Subplot (b) Y-axis labels and tick numbers
    ax1 = fig.add_axes([0.072, 0.145, 0.400, 0.735])
    ax2 = fig.add_axes([0.578, 0.145, 0.400, 0.735])

    colors = {
        "GPT-4o-mini": "#1f77b4",
        "Human-Authored": "#2ca02c",
        "Qwen2.5-7B": "#ff7f0e",
        "All Generators Combined": "#d62728"
    }

    # ----------------- Panel (a): Dose-Response Curve -----------------
    df_all = df_dose[df_dose["generator"] == "All Generators Combined"].sort_values("mean_token_overlap")
    ax1.plot(df_all["mean_token_overlap"], df_all["char_bm25_mrr"], marker="o", linewidth=2.4, ms=6, color="#d62728", label="Char 3–5g BM25 (All Data)")
    ax1.plot(df_all["mean_token_overlap"], df_all["word_bm25_mrr"], marker="s", linewidth=2.4, ms=6, color="#1f77b4", label="Word BM25 (All Data)")

    # Crossover and Regimes
    ax1.axvline(x=0.52, color="#6c757d", linestyle="--", linewidth=1.5, alpha=0.85, label=r"Empirical Crossover ($o^* \approx 0.52$)")
    ax1.axvspan(0.0, 0.52, color="#1f77b4", alpha=0.08, label="Word BM25 Advantage Regime")
    ax1.axvspan(0.52, 1.0, color="#d62728", alpha=0.08, label="Char BM25 Dominance Regime")

    ax1.set_xlabel(r"Query-Document Token Overlap $o$", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Retrieval MRR", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) Dose-Response Curve: MRR vs. Token Overlap", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#ced4da", fontsize=8.5)

    # ----------------- Panel (b): Invariance across Generators -----------------
    for gen in ["GPT-4o-mini", "Human-Authored", "Qwen2.5-7B"]:
        gdf = df_dose[df_dose["generator"] == gen].sort_values("mean_token_overlap")
        if len(gdf) > 0:
            n_tot = gdf["n_queries"].sum()
            label_str = f"{gen} ($N={n_tot:,}$)"
            ax2.plot(gdf["mean_token_overlap"], gdf["char_minus_word"], marker="o", linewidth=2.0, ms=6, color=colors[gen], label=label_str)

    ax2.axhline(y=0.0, color="black", linestyle="-", linewidth=1.2, alpha=0.9)
    ax2.axvline(x=0.52, color="#6c757d", linestyle="--", linewidth=1.5, alpha=0.85)

    ax2.set_xlabel(r"Query-Document Token Overlap $o$", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel(r"Advantage Lift $\Delta$ (Char MRR - Word MRR)", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) Advantage Invariance Across LLM Generators & Humans", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.25, 0.25)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#ced4da", fontsize=9.0)

    # Save to all required destination paths
    targets = [
        OUT_FIG_PATH,
        REPO_ROOT / "paper_v4" / "figures" / "fig_generator_dose_response.png",
        REPO_ROOT / "paper_v4" / "ijcds" / "figures" / "fig_generator_dose_response.png",
        REPO_ROOT / "submission_package_ijcds" / "source" / "figures" / "fig_generator_dose_response.png",
        REPO_ROOT / "FINAL_IJCDS_MANUSCRIPT_SOURCE" / "figures" / "fig_generator_dose_response.png",
        REPO_ROOT / "ijcds" / "figures" / "fig_generator_dose_response.png",
        REPO_ROOT / "results" / "v4" / "G-10" / "fig_generator_dose_response.png",
    ]

    for p in targets:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)

    plt.close()
    print("Successfully regenerated fig_generator_dose_response.png across all targets!")

if __name__ == "__main__":
    main()

