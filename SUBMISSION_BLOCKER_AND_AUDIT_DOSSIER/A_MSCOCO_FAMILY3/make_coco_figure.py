#!/usr/bin/env python3
"""Generate the manuscript's MS-COCO native-pixel comparison figure.

The values and confidence intervals mirror ``tab:coco_benchmarks_v3`` in the
IJCDS manuscript.  The figure deliberately distinguishes machine captions,
human gold captions, and native pixels rather than mixing them into one
undifferentiated bar series.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
    }
)


ROWS = [
    ("BLIP + MiniLM", "Machine caption", 0.3728, (0.3487, 0.3969), 0.6530, (0.6240, 0.6820)),
    ("BLIP + BGE", "Machine caption", 0.3695, (0.3458, 0.3931), 0.6460, (0.6160, 0.6760)),
    ("BLIP-2*", "Machine caption", 0.4185, (0.3950, 0.4420), 0.7020, (0.6740, 0.7300)),
    ("Florence-2*", "Machine caption", 0.4420, (0.4180, 0.4660), 0.7380, (0.7100, 0.7650)),
    ("Single Gold", "Human gold", 0.4077, (0.3827, 0.4321), 0.6720, (0.6430, 0.7010)),
    ("4-ref Gold", "Human gold", 0.5835, (0.5577, 0.6086), 0.8320, (0.8090, 0.8540)),
    ("CLIP ViT-B/32", "Native pixels", 0.4709, (0.4462, 0.4965), 0.7570, (0.7300, 0.7840)),
    ("SigLIP Base-P16", "Native pixels", 0.6071, (0.5828, 0.6317), 0.8670, (0.8460, 0.8880)),
]

COLORS = {
    "Machine caption": "#4c78a8",
    "Human gold": "#e6a23c",
    "Native pixels": "#159a83",
}


def add_panel(ax, values, cis, title, x_max, show_y_labels):
    y = np.arange(len(ROWS))[::-1]
    colors = [COLORS[row[1]] for row in ROWS]
    lower = np.array([value - ci[0] for value, ci in zip(values, cis)])
    upper = np.array([ci[1] - value for value, ci in zip(values, cis)])
    xerr = np.vstack([lower, upper])

    ax.barh(
        y,
        values,
        height=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        xerr=xerr,
        error_kw={"ecolor": "#374151", "elinewidth": 0.8, "capsize": 2.0, "capthick": 0.8},
        zorder=3,
    )

    for ypos, value, ci in zip(y, values, cis):
        ax.annotate(
            f"{value:.3f}",
            xy=(ci[1], ypos),
            xytext=(3.0, 0),
            textcoords="offset points",
            va="center_baseline",
            ha="left",
            fontsize=6.7,
        )

    # Separators make the three representation families visible without adding
    # a second legend block to the plotting area.
    ax.axhline(1.5, color="#d1d5db", linewidth=0.7)
    ax.axhline(3.5, color="#d1d5db", linewidth=0.7)
    ax.set_title(title, fontweight="bold", pad=5)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_xlabel("Score")
    if show_y_labels:
        ax.set_yticks(y)
        ax.set_yticklabels([row[0] for row in ROWS])
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    else:
        ax.tick_params(left=True, labelleft=False)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.65, zorder=0)
    ax.grid(axis="y", visible=False)


def main() -> None:
    fig, (ax_mrr, ax_hit10) = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.35),
        dpi=300,
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.08},
    )

    mrr_values = [row[2] for row in ROWS]
    mrr_cis = [row[3] for row in ROWS]
    hit10_values = [row[4] for row in ROWS]
    hit10_cis = [row[5] for row in ROWS]
    add_panel(ax_mrr, mrr_values, mrr_cis, "(a) Mean Reciprocal Rank", 0.80, True)
    add_panel(ax_hit10, hit10_values, hit10_cis, "(b) Hit@10", 1.05, False)

    handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in COLORS.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.99), ncol=3, frameon=False, fontsize=7.2)
    fig.text(
        0.53,
        0.015,
        r"Bars show means; whiskers show 95% bootstrap CIs. SigLIP vs BLIP: +0.2344 MRR, $p_{\mathrm{Holm}}<10^{-4}$; * indicates historical projections.",
        ha="center",
        fontsize=7.0,
        color="#374151",
    )
    fig.subplots_adjust(left=0.21, right=0.985, bottom=0.15, top=0.86)

    outputs = [
        REPO_ROOT / "figures" / "fig_coco_native_vs_proxy.png",
        REPO_ROOT / "ijcds" / "figures" / "fig_coco_native_vs_proxy.png",
        REPO_ROOT / "paper_v4" / "figures" / "fig_coco_native_vs_proxy.png",
        REPO_ROOT / "paper_v4" / "ijcds" / "figures" / "fig_coco_native_vs_proxy.png",
        REPO_ROOT / "submission_package_ijcds" / "source" / "figures" / "fig_coco_native_vs_proxy.png",
    ]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Generated:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
