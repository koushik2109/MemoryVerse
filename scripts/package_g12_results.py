#!/usr/bin/env python3
"""scripts/package_g12_results.py

Packages and consolidates Query-Side Noise Robustness & Failure Mode Attribution (G-12):
- Evaluates Query-Side Typos & Corruptions (T3-04):
  * Keyboard-adjacent QWERTY typos, transpositions, and omissions across error rates r in [0%, 30%]
  * Char 3-5g BM25 retains 77.7% of clean performance at 20% typo rate (0.9436 -> 0.7328 MRR)
  * Word BM25 collapses to 0.4820 (57.6% retention)
  * Levenshtein fuzzy expansion (dist<=1) lifts Word to 0.6012, still lagging Char BM25 by -0.1316 MRR
- Evaluates Diagnostic Error Taxonomy & Attribution (T3-10):
  * Exhaustive root-cause failure attribution across all N=306 probes
  * Explains dense collapse: 96.9% of BGE verbatim failures are caused by 'dense_numeric_short_dispersion' (mean-pooling loses specificity on numbers, dates, and short SKU codes)
  * Explains verbatim Word failures: 76.9% are 'distractor_collision' from OCR token fragmentation
  * Explains Char BM25 near-perfection: only 5 failures (98.37% Hit@10), 4 of which are total OCR target absence

Outputs:
- results/v4/G-12/config.json
- results/v4/G-12/query_noise_summary.csv
- results/v4/G-12/query_noise_results.csv
- results/v4/G-12/fig_query_noise_robustness.png
- results/v4/G-12/error_taxonomy_breakdown.csv
- results/v4/G-12/per_query_failure_attribution.csv
- results/v4/G-12/table_query_noise_and_error_taxonomy.tex
- results/v4/G-12/README.md
"""

import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "v4" / "G-12"
OUT_DIR.mkdir(parents=True, exist_ok=True)

src_dir_noise = REPO_ROOT / "results" / "gapfill_v1" / "T3-04"
shutil.copy(src_dir_noise / "query_noise_summary.csv", OUT_DIR / "query_noise_summary.csv")
shutil.copy(src_dir_noise / "query_noise_results.csv", OUT_DIR / "query_noise_results.csv")
shutil.copy(src_dir_noise / "fig_query_noise_robustness.png", OUT_DIR / "fig_query_noise_robustness.png")

src_dir_tax = REPO_ROOT / "results" / "gapfill_v1" / "T3-10"
shutil.copy(src_dir_tax / "error_taxonomy_breakdown.csv", OUT_DIR / "error_taxonomy_breakdown.csv")
shutil.copy(src_dir_tax / "per_query_failure_attribution.csv", OUT_DIR / "per_query_failure_attribution.csv")

git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

config = {
    "item_id": "G-12",
    "title": "Query-Side Noise Robustness and Diagnostic Failure Mode Attribution",
    "git_commit": git_hash,
    "typo_rates_evaluated": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "failure_categories": [
        "ocr_target_absence",
        "distractor_collision",
        "dense_numeric_short_dispersion",
        "dense_subword_drift",
        "paraphrase_entity_gap",
        "partial_mismatch"
    ],
    "source_files": [
        "results/gapfill_v1/T3-04/query_noise_summary.csv",
        "results/gapfill_v1/T3-10/error_taxonomy_breakdown.csv"
    ]
}

with open(OUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

df_noise = pd.read_csv(OUT_DIR / "query_noise_summary.csv")
df_tax = pd.read_csv(OUT_DIR / "error_taxonomy_breakdown.csv")

# Generate LaTeX table combining query noise and failure taxonomy
latex_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Diagnostic Failure Attribution and Query-Side Noise Robustness Across Systems ($N=306$).}",
    r"\label{tab:query_noise_and_error_taxonomy}",
    r"\small",
    r"\begin{subtable}[t]{\textwidth}",
    r"\centering",
    r"\caption{Query-side typo dose-response: MRR degradation under keyboard/character error rates $r \in [0\%, 30\%]$.}",
    r"\begin{tabular}{lccccccc}",
    r"\toprule",
    r"\textbf{System} & \textbf{0\% (Clean)} & \textbf{5\%} & \textbf{10\%} & \textbf{15\%} & \textbf{20\%} & \textbf{30\%} & \textbf{Retention at 20\%} \\",
    r"\midrule"
]

noise_systems = [
    ("Char 3-5g BM25", "Char 3--5g BM25"),
    ("Word + Fuzzy BM25", "Word + Fuzzy BM25"),
    ("Word BM25", "Word BM25"),
    ("Dense MiniLM", "Dense MiniLM")
]

for sys_key, sys_label in noise_systems:
    sub = df_noise[df_noise["System"] == sys_key]
    if not sub.empty:
        m0 = sub[sub["Noise_Rate"] == 0.0]["MRR"].values[0]
        m5 = sub[sub["Noise_Rate"] == 0.05]["MRR"].values[0]
        m10 = sub[sub["Noise_Rate"] == 0.10]["MRR"].values[0]
        m15 = sub[sub["Noise_Rate"] == 0.15]["MRR"].values[0]
        m20 = sub[sub["Noise_Rate"] == 0.20]["MRR"].values[0]
        m30 = sub[sub["Noise_Rate"] == 0.30]["MRR"].values[0]
        ret = (m20 / m0) * 100

        bold = sys_key == "Char 3-5g BM25"
        name_str = f"\\textbf{{{sys_label}}}" if bold else sys_label
        m0_str = f"\\textbf{{{m0:.4f}}}" if bold else f"{m0:.4f}"
        m20_str = f"\\textbf{{{m20:.4f}}}" if bold else f"{m20:.4f}"

        latex_lines.append(
            f"{name_str} & {m0_str} & {m5:.4f} & {m10:.4f} & {m15:.4f} & {m20_str} & {m30:.4f} & {ret:.1f}\\% \\\\"
        )

latex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{subtable}",
    r"\vspace{0.25cm}",
    r"\begin{subtable}[t]{\textwidth}",
    r"\centering",
    r"\caption{Diagnostic failure attribution under Real OCR verbatim queries ($N=306$, failures out of Hit@10).}",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"\textbf{System} & \textbf{Hit@10} & \textbf{Failures} & \textbf{Numeric/Short Dispersion} & \textbf{Distractor Collision} & \textbf{OCR Target Absence} & \textbf{Other Drift} \\",
    r"\midrule"
])

real_verb = df_tax[(df_tax["ocr_regime"] == "real_ocr") & (df_tax["query_slice"] == "verbatim")]
order_tax = [
    ("char_bm25", "Char 3--5g BM25"),
    ("word_bm25", "Word BM25"),
    ("bge-small-en-v1.5", "Dense BGE-small"),
    ("all-MiniLM-L6-v2", "Dense MiniLM")
]

for r_key, r_label in order_tax:
    row = real_verb[real_verb["retriever"] == r_key].iloc[0]
    h10 = row["hit10_rate"]
    fail = int(row["total_failures"])
    num_disp = int(row["dense_numeric_short_dispersion"])
    coll = int(row["distractor_collision"])
    absent = int(row["ocr_target_absence"])
    other = int(row["dense_subword_drift"] + row["partial_mismatch"])

    bold = r_key == "char_bm25"
    name_str = f"\\textbf{{{r_label}}}" if bold else r_label

    latex_lines.append(
        f"{name_str} & {h10:.4f} & {fail} & {num_disp} & {coll} & {absent} & {other} \\\\"
    )

latex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{subtable}",
    r"\end{table*}"
])

with open(OUT_DIR / "table_query_noise_and_error_taxonomy.tex", "w") as f:
    f.write("\n".join(latex_lines) + "\n")

readme_content = f"""# G-12: Query-Side Noise Robustness & Diagnostic Failure Attribution

## Executive Summary & Statistical Findings
This study addresses Reviewer Objection **G-12** by evaluating:
1. **Query-Side Corruptions (T3-04):** Robustness against realistic keyboard-adjacent QWERTY typos, transpositions, and character omissions ($r \\in [0\\%, 30\\%]$).
2. **Diagnostic Error Taxonomy (T3-10):** Mechanistic failure attribution of every retrieval miss across $N=306$ held-out probes.

### 1. Asymmetric Query-Side Noise Tolerance
- **Char 3–5g BM25:** Retains **0.7328 MRR** at 20% query typo rate (**77.7% retention** relative to clean 0.9436). Contiguous $n$-grams preserve matching mass when internal characters are transposed or substituted.
- **Word BM25:** Collapses to **0.4820 MRR** at 20% typo rate (**57.6% retention**). Single character typos invalidate entire word tokens in the inverted dictionary.
- **Word + Fuzzy Levenshtein Expansion (dist $\\le 1$):** Recovers Word BM25 to **0.6012 MRR** (75.1% retention), but still trails Char BM25 (0.7328) by over $-0.13$ MRR while multiplying query evaluation latency.
- **Dense MiniLM:** Starts from a crippled baseline (0.2040) and degrades to 0.1582.

### 2. Mechanistic Root Cause of Dense Retrieval Collapse
Diagnostic analysis of 306 queries reveals why dense bi-encoders collapse on document text:
- Under Real OCR verbatim queries:
  * **BGE-small** suffers 192 failures (Hit@10 = 37.25%). **186 of 192 failures (96.9%)** are attributable to **`dense_numeric_short_dispersion`**. Dense pooling maps short numeric strings (dates, totals, invoice IDs) to diffuse, overlapping embedding spaces that collide with hundreds of unrelated documents.
  * **MiniLM** suffers 210 failures (Hit@10 = 31.37%), with **202 of 210 failures (96.2%)** driven by numeric/short dispersion.
- **Why Character BM25 Dominates:**
  * Suffers only **5 failures out of 306** (98.37% Hit@10).
  * 4 of the 5 failures are caused by **total OCR omission** (the scanner dropped the line entirely), representing an insurmountable physical target ceiling rather than an IR failure.
- **Why Whole-Word BM25 Fails on Verbatim Text:**
  * Suffers 26 failures (91.50% Hit@10). **20 of 26 failures (76.9%)** are caused by **`distractor_collision`** resulting from OCR word fragmentation (e.g. `SDN BHD` split into `S DN BH D`).

## Provenance
- **Git Commit:** `{git_hash}`
- **Source Files:** `results/gapfill_v1/T3-04/query_noise_summary.csv`, `results/gapfill_v1/T3-10/error_taxonomy_breakdown.csv`
- **Output Artifacts:** `results/v4/G-12/table_query_noise_and_error_taxonomy.tex`, `results/v4/G-12/fig_query_noise_robustness.png`
"""

with open(OUT_DIR / "README.md", "w") as f:
    f.write(readme_content)

print(f"Successfully packaged G-12 results to {OUT_DIR}")
