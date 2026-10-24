"""
Plot generator for Task 2 Noise Robustness Sweep
Runs using system python3 which has matplotlib.
"""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
RESULTS_JSON = RAG_DATA_DIR / "clean_stream_experiments_results.json"
OUTPUT_PNG = RAG_DATA_DIR / "mrr_hit10_vs_noise.png"

def main():
    if not RESULTS_JSON.exists():
        print(f"Results file not found: {RESULTS_JSON}")
        return

    with open(RESULTS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    task2 = data["task2_noise_sweep"]
    noise_x = [0, 10, 20, 40]

    palette = {
        "BM25": ("#7f7f7f", "o", "--"),
        "Dense": ("#1f77b4", "^", "--"),
        "Hybrid (fixed alpha=0.7)": ("#2ca02c", "s", "-"),
        "Original RAMP-RAG": ("#d62728", "D", "-"),
        "RAMP-RAG + Clean Stream": ("#9467bd", "*", "-"),
        "Cross-Encoder": ("#ff7f0e", "x", ":"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    for m, (col, marker, lstyle) in palette.items():
        mrr_y = [task2[f"noise_{nx}pct"][m]["mrr"] for nx in noise_x]
        h10_y = [task2[f"noise_{nx}pct"][m]["hit10"] for nx in noise_x]
        ax1.plot(noise_x, mrr_y, marker=marker, linestyle=lstyle, label=m, color=col, linewidth=2, markersize=7)
        ax2.plot(noise_x, h10_y, marker=marker, linestyle=lstyle, label=m, color=col, linewidth=2, markersize=7)

    ax1.set_title("MRR vs OCR Proxy Character Noise", fontsize=12, fontweight="bold")
    ax1.set_xlabel("OCR Noise Rate (%)", fontsize=11)
    ax1.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=11)
    ax1.set_xticks(noise_x)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9, loc="lower left")

    ax2.set_title("Hit@10 vs OCR Proxy Character Noise", fontsize=12, fontweight="bold")
    ax2.set_xlabel("OCR Noise Rate (%)", fontsize=11)
    ax2.set_ylabel("Hit@10", fontsize=11)
    ax2.set_xticks(noise_x)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9, loc="lower left")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    plt.close()
    print(f"Successfully generated high-resolution plot: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
