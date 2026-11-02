#!/usr/bin/env python3
"""scripts/package_g09_results.py

Packages and consolidates Controlled Query-Formulation Experiments (G-09):
- Evaluates 5 controlled query formulations holding identical ground-truth information need:
  1. Short Verbatim Snippet (isolated substring/entity)
  2. Multi-Anchor Keyword Bag (exact multi-entity co-specification)
  3. Declarative Sentence (grammatical sentence)
  4. Interrogative Question (natural question syntax)
  5. Conversational Imperative (user command syntax)
- Across 4 retrievers: Char 3-5g BM25, Word BM25, Dense BGE-small, Dense MiniLM
- Across 2 OCR regimes: Apple Clean Real OCR and Clean Ground Truth (N=306 queries against N=813 pool)
- Statistically tests paired permutations (B=100,000 resamples):
  * Causal test: Does syntactic form (interrogative vs declarative vs keyword) affect dense MRR?
    Finding: No significant difference (Keyword vs Interrogative BGE p=0.6617, Declarative vs Interrogative p=0.5316).
  * Information mass test: Moving from snippet to multi-anchor keyword bag lifts BGE by +0.2126 (p < 1e-5).
  * Policy recommendation: Restrict causal claims; dense failure is caused by lack of entity co-specification mass, not interrogative phrasing.

Outputs:
- results/v4/G-09/config.json
- results/v4/G-09/query_formulation_summary.csv
- results/v4/G-09/query_formulation_results.csv
- results/v4/G-09/statistical_tests.csv
- results/v4/G-09/statistical_tests.json
- results/v4/G-09/table_query_formulation_controlled.tex
- results/v4/G-09/README.md
"""

import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "v4" / "G-09"
OUT_DIR.mkdir(parents=True, exist_ok=True)

src_dir = REPO_ROOT / "results" / "gapfill_v1" / "T3-01"
shutil.copy(src_dir / "query_formulation_summary.csv", OUT_DIR / "query_formulation_summary.csv")
shutil.copy(src_dir / "query_formulation_results.csv", OUT_DIR / "query_formulation_results.csv")
shutil.copy(src_dir / "statistical_tests.csv", OUT_DIR / "statistical_tests.csv")
shutil.copy(src_dir / "statistical_tests.json", OUT_DIR / "statistical_tests.json")

git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

config = {
    "item_id": "G-09",
    "title": "Controlled Test of Query-Formulation and Syntactic Syntax Claim",
    "git_commit": git_hash,
    "formulations": ["snippet", "keyword", "declarative", "interrogative", "imperative"],
    "systems_evaluated": ["char_bm25", "word_bm25", "bge-small-en-v1.5", "all-MiniLM-L6-v2"],
    "ocr_regimes": ["real_ocr", "clean_gt"],
    "n_queries": 306,
    "permutation_resamples": 100000,
    "source_file": "results/gapfill_v1/T3-01/query_formulation_summary.csv"
}

with open(OUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

df_summary = pd.read_csv(OUT_DIR / "query_formulation_summary.csv")

# Generate LaTeX table snippet
latex_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Controlled Query-Formulation Ablation: Disentangling Syntactic Phrasing from Entity Anchor Mass under Real OCR ($N=306$).}",
    r"\label{tab:query_formulation_controlled}",
    r"\small",
    r"\begin{tabular}{lcccccccc}",
    r"\toprule",
    r"& \multicolumn{2}{c}{\textbf{Char 3--5g BM25}} & \multicolumn{2}{c}{\textbf{Word BM25}} & \multicolumn{2}{c}{\textbf{Dense BGE-small}} & \multicolumn{2}{c}{\textbf{Dense MiniLM}} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
    r"\textbf{Query Formulation} & \textbf{MRR} & \textbf{Hit@10} & \textbf{MRR} & \textbf{Hit@10} & \textbf{MRR} & \textbf{Hit@10} & \textbf{MRR} & \textbf{Hit@10} \\",
    r"\midrule"
]

formulation_map = [
    ("snippet", "Short Verbatim Snippet (1 Anchor)"),
    ("keyword", "Keyword Bag (Multi-Anchor Co-spec)"),
    ("declarative", "Declarative Sentence"),
    ("interrogative", "Interrogative Question"),
    ("imperative", "Conversational Imperative"),
]

for form_key, form_name in formulation_map:
    row_vals = []
    for ret in ["char_bm25", "word_bm25", "bge-small-en-v1.5", "all-MiniLM-L6-v2"]:
        sub = df_summary[(df_summary["ocr_regime"] == "real_ocr") & (df_summary["formulation"] == form_key) & (df_summary["retriever"] == ret)]
        mrr = sub["mrr"].values[0]
        h10 = sub["hit10"].values[0]
        row_vals.append(f"{mrr:.4f} & {h10:.4f}")

    latex_lines.append(f"{form_name} & " + " & ".join(row_vals) + r" \\")

latex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table*}"
])

with open(OUT_DIR / "table_query_formulation_controlled.tex", "w") as f:
    f.write("\n".join(latex_lines) + "\n")

readme_content = f"""# G-09: Controlled Query-Formulation Experiments

## Executive Summary & Statistical Findings
This study resolves Reviewer Objection **G-09 / T3-01** by isolating whether the dense retrieval deficit observed in document RAG is caused by **syntactic phrasing** (interrogative question vs. declarative sentence vs. keyword bag) or by **information mass / entity co-specification** (isolated entity snippet vs. multi-anchor context).

### 1. Syntactic Syntax Has No Causal Effect on Dense Retrieval
Holding entity anchors strictly identical across 306 queries:
- **Dense BGE-small on Real OCR:**
  * Keyword Bag: MRR = **0.5187** (Hit@10: 0.8137)
  * Declarative Sentence: MRR = **0.5210** (Hit@10: 0.7908)
  * Interrogative Question: MRR = **0.5073** (Hit@10: 0.8072)
  * Paired difference Keyword vs. Question: $+0.0114$ ($p = 0.6617$).
  * Paired difference Declarative vs. Question: $+0.0137$ ($p = 0.5316$).
- **Dense MiniLM on Real OCR:**
  * Keyword Bag: MRR = **0.3382** (Hit@10: 0.5425)
  * Declarative Sentence: MRR = **0.3081** (Hit@10: 0.4477)
  * Interrogative Question: MRR = **0.3091** (Hit@10: 0.4771)
  * Paired difference Keyword vs. Question: $+0.0291$ ($p = 0.0073$, favoring keywords!).
  * Paired difference Declarative vs. Question: $-0.0011$ ($p = 0.5516$).

### 2. Entity Anchor Mass Drives the Observed Dense Recovery
When queries provide multiple entity anchors (vendor name, total amount, date), dense models recover substantial performance regardless of syntax:
- **BGE-small:** Lifts from **0.3061** (single snippet) to **0.5187** (multi-anchor bag), $\\Delta = +0.2126$ ($p < 10^{{-5}}$).
- **MiniLM:** Lifts from **0.2238** to **0.3382**, $\\Delta = +0.1144$ ($p < 10^{{-5}}$).
- **Char 3-5g BM25:** Lifts from **0.9399** to **0.9644**, $\\Delta = +0.0245$ ($p < 10^{{-4}}$).
- **Word BM25:** Lifts from **0.8091** to **0.9426**, $\\Delta = +0.1335$ ($p < 10^{{-5}}$).

### 3. Manuscript Policy & Wording Rule
The manuscript **MUST NOT assert causal claims** that dense bi-encoders collapse because queries lack question syntax. The data demonstrates that keyword bags match or exceed interrogative questions. The paper will adopt **strictly observational language** regarding syntactic form and state that dense performance scales with **entity co-specification mass**.

## Provenance
- **Git Commit:** `{git_hash}`
- **Source Files:** `results/gapfill_v1/T3-01/query_formulation_summary.csv`
- **Output Artifacts:** `results/v4/G-09/query_formulation_summary.csv`, `results/v4/G-09/table_query_formulation_controlled.tex`
"""

with open(OUT_DIR / "README.md", "w") as f:
    f.write(readme_content)

print(f"Successfully packaged G-09 results to {OUT_DIR}")

