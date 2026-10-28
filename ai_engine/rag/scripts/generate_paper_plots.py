"""Generate the IEEE figures for the implemented multimodal RAG benchmark.

The alpha plot deliberately reads its values from the checked-in benchmark
artifact.  This keeps the figure synchronized with the benchmark output when
the evaluation is rerun instead of maintaining a second set of metric values
in this plotting script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_RESULTS_PATH = (
    PROJECT_ROOT / "ai_engine" / "rag" / "data" / "real_benchmark_results.json"
)
ALPHA_FIGURE_PATH = PROJECT_ROOT / "fig_alpha_sensitivity.png"
ARCHITECTURE_FIGURE_PATH = PROJECT_ROOT / "fig_architecture.png"


# Compact serif typography and restrained line weights survive scaling into an
# IEEE single-column figure.  Mathtext is used instead of usetex so generation
# only requires the Python plotting dependencies.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "savefig.facecolor": "white",
    }
)


def _metric_value(metrics: Any, key: str, alpha: float, path: Path) -> float:
    """Read and validate one numeric metric from an alpha-sweep entry."""

    if not isinstance(metrics, dict) or key not in metrics:
        raise ValueError(f"Missing {key!r} for alpha={alpha:g} in {path}")

    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Non-numeric {key!r} for alpha={alpha:g} in {path}")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{key!r} outside [0, 1] for alpha={alpha:g} in {path}")
    return float(value)


def load_alpha_sweep(
    results_path: Path = BENCHMARK_RESULTS_PATH,
) -> tuple[list[float], list[float], list[float]]:
    """Load sorted alpha, HitRate@10, and MRR values from the benchmark JSON."""

    try:
        with results_path.open(encoding="utf-8") as results_file:
            results = json.load(results_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Benchmark artifact not found: {results_path}. "
            "Run ai_engine/rag/scripts/run_real_benchmarks.py first."
        ) from exc

    alpha_sweep = results.get("alpha_sweep") if isinstance(results, dict) else None
    if not isinstance(alpha_sweep, dict) or not alpha_sweep:
        raise ValueError(f"Missing non-empty 'alpha_sweep' in {results_path}")

    rows: list[tuple[float, float, float]] = []
    for alpha_text, metrics in alpha_sweep.items():
        try:
            alpha = float(alpha_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid alpha key {alpha_text!r} in {results_path}") from exc
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha={alpha:g} outside [0, 1] in {results_path}")
        rows.append(
            (
                alpha,
                _metric_value(metrics, "HitRate@10", alpha, results_path),
                _metric_value(metrics, "MRR", alpha, results_path),
            )
        )

    rows.sort(key=lambda row: row[0])
    alphas, hit_rate10, mrr = zip(*rows)
    return list(alphas), list(hit_rate10), list(mrr)


def generate_alpha_curve(output_path: Path = ALPHA_FIGURE_PATH) -> Path:
    """Generate the dense-weight sensitivity plot from benchmark results."""

    alphas, hit_rate10, mrr = load_alpha_sweep()
    metric_min = min(*hit_rate10, *mrr)
    metric_max = max(*hit_rate10, *mrr)
    lower = max(0.0, metric_min - 0.01)
    upper = min(1.01, metric_max + 0.005)

    fig, ax1 = plt.subplots(figsize=(3.5, 2.55), dpi=300)
    color_hit_rate = "#0072B2"
    color_mrr = "#D55E00"

    ax1.set_xlabel(r"Dense weight coefficient $\alpha$")
    ax1.set_ylabel("HitRate@10", color=color_hit_rate)
    hit_rate_line = ax1.plot(
        alphas,
        hit_rate10,
        marker="o",
        markersize=3.8,
        color=color_hit_rate,
        label="HitRate@10",
    )
    ax1.tick_params(axis="y", labelcolor=color_hit_rate, width=0.7)
    ax1.tick_params(axis="x", width=0.7)
    ax1.set_xticks(alphas)
    ax1.set_xticklabels([f"{alpha:g}" for alpha in alphas])
    ax1.set_ylim(lower, upper)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.grid(True, axis="both", linestyle="--", linewidth=0.55, alpha=0.45)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean reciprocal rank (MRR)", color=color_mrr)
    mrr_line = ax2.plot(
        alphas,
        mrr,
        marker="s",
        markersize=3.8,
        color=color_mrr,
        linestyle="--",
        label="MRR",
    )
    ax2.tick_params(axis="y", labelcolor=color_mrr, width=0.7)
    ax2.set_ylim(lower, upper)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    lines = hit_rate_line + mrr_line
    ax1.legend(
        lines,
        [line.get_label() for line in lines],
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        borderpad=0.35,
        handlelength=2.0,
    )
    ax1.set_title(r"Sensitivity to dense weight $\alpha$", pad=5)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def _add_box(
    ax: plt.Axes,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    face_color: str,
    edge_color: str,
) -> None:
    """Add a consistently styled pipeline stage to the architecture figure."""

    rect = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=1.1,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#202020",
        linespacing=1.15,
    )


def generate_architecture_diagram(output_path: Path = ARCHITECTURE_FIGURE_PATH) -> Path:
    """Generate the architecture diagram using the implemented retrieval stages."""

    fig, ax = plt.subplots(figsize=(7.16, 3.55), dpi=300)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    boxes = [
        (
            "Optional deployment inputs\n(PDF, image, audio,\nvideo text proxies)",
            0.02,
            0.59,
            0.18,
            0.27,
            "#ECEFF1",
            "#78909C",
        ),
        (
            "Optional proxy extraction\nOCR · image captions\nWhisper ASR",
            0.23,
            0.59,
            0.18,
            0.27,
            "#ECEFF1",
            "#78909C",
        ),
        (
            "Dual-stream index\nDense cosine search\nall-MiniLM-L6-v2\nBM25 inverted index",
            0.44,
            0.59,
            0.20,
            0.27,
            "#FFF3E0",
            "#F57C00",
        ),
        (
            "User query\n(Natural language /\nmultihop)",
            0.02,
            0.12,
            0.18,
            0.27,
            "#F3E5F5",
            "#7B1FA2",
        ),
        (
            "Multi-hop decomposition\nQuery analyzer\nEntity bridging",
            0.23,
            0.12,
            0.18,
            0.27,
            "#EDE7F6",
            "#512DA8",
        ),
        (
            "Stage 1: Hybrid Retrieval\nConvex Fusion ($\\alpha$) / RRF\nTop-$K_1$ Candidates ($K_1=20$)",
            0.44,
            0.12,
            0.20,
            0.27,
            "#E0F2F1",
            "#00796B",
        ),
        (
            "Stage 2: Cascade Reranker\nNeural Cross-Encoder\n(BGE / MiniLM Reranker)\nTop-$K_1$ Rescored",
            0.68,
            0.27,
            0.15,
            0.45,
            "#FCE4EC",
            "#C2185B",
        ),
        (
            "Generator\n(GPT-4o-mini)\nGrounded multimodal\nresponse",
            0.86,
            0.27,
            0.12,
            0.45,
            "#FFF8E1",
            "#FFA000",
        ),
    ]

    for box in boxes:
        _add_box(ax, *box)

    arrow_props = {
        "arrowstyle": "-|>",
        "lw": 1.0,
        "color": "#4A4A4A",
        "mutation_scale": 10,
    }

    # Ingestion flow.
    ax.annotate("", xy=(0.23, 0.725), xytext=(0.20, 0.725), arrowprops=arrow_props)
    ax.annotate("", xy=(0.44, 0.725), xytext=(0.41, 0.725), arrowprops=arrow_props)

    # The dual index feeds the first-stage hybrid candidate pool.
    ax.annotate("", xy=(0.54, 0.39), xytext=(0.54, 0.59), arrowprops=arrow_props)

    # Query decomposition flow.
    ax.annotate("", xy=(0.23, 0.255), xytext=(0.20, 0.255), arrowprops=arrow_props)
    ax.annotate("", xy=(0.44, 0.255), xytext=(0.41, 0.255), arrowprops=arrow_props)

    # Candidate reranking and grounded generation.
    ax.annotate("", xy=(0.68, 0.50), xytext=(0.64, 0.255), arrowprops=arrow_props)
    ax.annotate("", xy=(0.86, 0.50), xytext=(0.83, 0.50), arrowprops=arrow_props)

    fig.tight_layout(pad=0.25)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


if __name__ == "__main__":
    generate_alpha_curve()
    generate_architecture_diagram()
