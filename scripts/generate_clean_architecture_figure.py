import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Configure publication typography matching IEEE / IJCDS
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "figure.autolayout": False,
})

repo_root = Path("/Users/anuragkrishnavennela/Projects/MemoryVerse")

# Clean balanced aspect ratio: 12.0 in x 4.2 in
fig, ax = plt.subplots(figsize=(12.0, 4.2), dpi=300)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.axis("off")

# Helper function to draw a modern card with header
def draw_card(ax, x, y, w, h, title, subtitle_lines, bg_color="#ffffff", border_color="#1f2937", header_color="#f1f5f9", title_color="#000000", header_fsize=8.6, body_fsize=8.8, body_lspace=1.30):
    # Main card container
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.018",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.15,
        zorder=2
    )
    ax.add_patch(box)
    
    # Header banner box
    hh = 0.17 * h if h > 0.5 else 0.24 * h
    header = patches.FancyBboxPatch(
        (x, y + h - hh), w, hh,
        boxstyle="round,pad=0.005,rounding_size=0.018",
        facecolor=header_color,
        edgecolor="none",
        zorder=3
    )
    ax.add_patch(header)
    
    # Square bottom of header
    rect_flat = patches.Rectangle((x, y + h - hh), w, hh * 0.45, facecolor=header_color, edgecolor="none", zorder=3)
    ax.add_patch(rect_flat)
    
    # Header divider line
    ax.plot([x, x + w], [y + h - hh, y + h - hh], color=border_color, lw=0.9, zorder=4)
    
    # Header text
    ax.text(x + w / 2, y + h - hh / 2, title, ha="center", va="center", fontsize=header_fsize, fontweight="bold", color=title_color, zorder=5)
    
    # Body text
    body_y = y + (h - hh) / 2
    body_text = "\n".join(subtitle_lines)
    ax.text(x + w / 2, body_y, body_text, ha="center", va="center", fontsize=body_fsize, color="#000000", linespacing=body_lspace, zorder=5)

# 5-Column Coordinates
c_w = 0.176
gap = 0.020
xs = [0.015 + i * (c_w + gap) for i in range(5)]

top_y, top_h = 0.53, 0.43
bot_y, bot_h = 0.04, 0.43
full_y, full_h = 0.04, 0.92

# ==================== COLUMN 1: INPUT SOURCES ====================
draw_card(ax, xs[0], top_y, c_w, top_h,
          "MODAL INPUT PROXIES",
          ["Scanned Receipts (SROIE)", "Administrative Forms (FUNSD)", "Natural Scenes (MS-COCO)", "Acoustic Audio (ESC-50)"],
          bg_color="#f8fafc", border_color="#64748b", header_color="#e2e8f0", title_color="#0f172a",
          header_fsize=8.5, body_fsize=9.2, body_lspace=1.45)

draw_card(ax, xs[0], bot_y, c_w, bot_h,
          "QUERY FORMULATIONS",
          ["Verbatim Lookup (Exact Spans)", "Light Paraphrase (Entity Sub)", "Answerable Factual Queries", "Relational Multi-Hop Queries"],
          bg_color="#faf5ff", border_color="#9333ea", header_color="#f3e8ff", title_color="#581c87",
          header_fsize=8.5, body_fsize=9.0, body_lspace=1.45)

# ==================== COLUMN 2: EXTRACTION & ANALYSIS ====================
draw_card(ax, xs[1], top_y, c_w, top_h,
          "EXTRACTION & STORAGE",
          ["Apple Vision / Tesseract OCR", "BLIP / Florence-2 Captions", "Word & Char 3–5g Inverted Index", "Dense BGE-base / MiniLM Index"],
          bg_color="#f0f9ff", border_color="#0284c7", header_color="#e0f2fe", title_color="#075985",
          header_fsize=8.4, body_fsize=9.0, body_lspace=1.45)

draw_card(ax, xs[1], bot_y, c_w, bot_h,
          "QUERY PARSING & BRIDGING",
          ["Topical Anchor Extraction", "Surface Feature Router", "Grounded PRF Entity Bridging", "Confidence Delegation"],
          bg_color="#faf5ff", border_color="#9333ea", header_color="#f3e8ff", title_color="#581c87",
          header_fsize=8.1, body_fsize=9.0, body_lspace=1.45)

# ==================== COLUMN 3: STAGE 1 HYBRID FUSION ====================
draw_card(ax, xs[2], full_y, c_w, full_h,
          "STAGE 1: HYBRID RETRIEVAL",
          ["Dual-Stream Scoring:",
           r"$S_{\mathrm{lex}}(q, d)$ & $S_{\mathrm{dense}}(q, d)$",
           "",
           "Subword Lexical Indexing:",
           "Character 3–5gram BM25",
           r"(+10.6% lift on real OCR)",
           "",
           "Fusion Mechanics:",
           r"Convex Sweep ($\alpha^* = 1.00$)",
           "Reciprocal Rank Fusion",
           "",
           "Candidate Extraction:",
           r"Top-$K_1$ Pool ($K_1 = 20$)",
           r"$\geq 94$% Verbatim Hit@10"],
          bg_color="#f0fdf4", border_color="#16a34a", header_color="#dcfce7", title_color="#14532d",
          header_fsize=8.4, body_fsize=8.6, body_lspace=1.22)

# ==================== COLUMN 4: STAGE 2 CASCADE RERANK ====================
draw_card(ax, xs[3], full_y, c_w, full_h,
          "STAGE 2: CASCADE RERANK",
          ["Neural Cross-Encoder:",
           "MiniLM-L6-v2 Cross-Encoder",
           "BGE-Reranker-Base",
           "",
           "Cascade Calibration:",
           r"Score Margin $\Delta_{1,2} < \tau^*$",
           "Conformal RCPS Gating",
           r"Marginal Miss Risk $\leq 0.05$",
           "",
           "Rescored Candidates:",
           r"Top-$K_2$ Ranked ($K_2 = 10$)",
           "+4.7% MRR on ambiguous queries",
           "Zero latency on high-margin"],
          bg_color="#fffbeb", border_color="#d97706", header_color="#fef3c7", title_color="#78350f",
          header_fsize=8.4, body_fsize=8.6, body_lspace=1.22)

# ==================== COLUMN 5: GROUNDED GENERATION ====================
draw_card(ax, xs[4], full_y, c_w, full_h,
          "GROUNDED GENERATION",
          ["Context Assembly:",
           "Attributed Multimodal Media",
           "Factual Evidence Passages",
           "",
           "Generation Models:",
           "GPT-4o-mini",
           "Qwen2.5-7B Instruct",
           "",
           "Benchmark Evaluation:",
           "Exact Match & ANLS",
           "Dual-Reviewer Human Audit",
           r"$\kappa = 0.8716$ Agreement",
           "100% Zero-Leakage Tested"],
          bg_color="#fff1f2", border_color="#e11d48", header_color="#ffe4e6", title_color="#881337",
          header_fsize=8.4, body_fsize=8.6, body_lspace=1.22)

# Arrow helper
def draw_arrow(ax, x1, y1, x2, y2, color="#1e293b", lw=1.3):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=9.0,
            shrinkA=1,
            shrinkB=1,
            joinstyle="miter"
        ),
        zorder=10
    )

# Top lane horizontal connections (Ingestion -> Index -> Stage 1)
draw_arrow(ax, xs[0] + c_w, top_y + top_h / 2, xs[1], top_y + top_h / 2)
draw_arrow(ax, xs[1] + c_w, top_y + top_h / 2, xs[2], top_y + top_h / 2)

# Bottom lane horizontal connections (Query -> Parsing -> Stage 1)
draw_arrow(ax, xs[0] + c_w, bot_y + bot_h / 2, xs[1], bot_y + bot_h / 2)
draw_arrow(ax, xs[1] + c_w, bot_y + bot_h / 2, xs[2], bot_y + bot_h / 2)

# Inter-stage horizontal connections (Stage 1 -> Stage 2 -> Generation)
draw_arrow(ax, xs[2] + c_w, 0.50, xs[3], 0.50)
draw_arrow(ax, xs[3] + c_w, 0.50, xs[4], 0.50)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)

# Save to all destination paths
out_paths = [
    repo_root / "figures" / "fig_architecture.png",
    repo_root / "ijcds" / "figures" / "fig_architecture.png",
    repo_root / "fig_architecture.png",
    repo_root / "ijcds" / "fig_architecture.png",
]

for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=300)
    print(f"Saved: {p}")

plt.close()
