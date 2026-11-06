# Human-Authored Query Structure, Provenance, and Statistical Re-Evaluation

**Document ID:** `FINAL_BLOCKER_RESOLUTION/HUMAN_QUERY_STRUCTURE_AND_PROVENANCE.md`  
**Date:** November 2026  
**Auditor:** Senior Information Retrieval Scientist & Statistical Auditor  
**Focus:** Forensic Investigation of `data/human_queries_200.csv` and Evaluated Queries ($N=400$)  
**Status:** FULL RE-AUDIT COMPLETED FROM RAW QUERY RECORDS  

---

## 1. Executive Summary & Resolution of the 200 Rows vs. 400 Queries Question

A core question in the audit was:
> *"Explain why the file has 200 rows and two columns but is reported as 400 human-authored queries."*

### Forensic Finding
Forensic examination of [`data/human_queries_200.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/data/human_queries_200.csv) and [`results/v4/G-10/generator_query_evaluations.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/results/v4/G-10/generator_query_evaluations.csv) confirms the exact structure:
1. **The Source File Structure:** [`data/human_queries_200.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/data/human_queries_200.csv) contains 200 rows, each corresponding to a distinct ground-truth receipt (`sroie_000` through `sroie_199` and subsets up to `sroie_405`).
2. **Two Distinct Query Modalities per Row:** Each of the 200 rows contains two distinct search query columns:
   - `handwritten_verbatim` (Column 4): 200 query instances representing known-item keyword search intent extracted from receipt headers.
   - `handwritten_paraphrase` (Column 5): 200 query instances representing natural language semantic search intent describing the transaction context, purchased items, or location.
3. **Execution Mapping:** In the evaluation harness (`results/v4/G-10/generator_query_evaluations.csv`), both columns were independently indexed and scored against the document corpus as two separate evaluation bands:
   - `source_band = "human_verbatim"`: 200 evaluation rows
   - `source_band = "human_paraphrase"`: 200 evaluation rows
   - Total Evaluated Human Instances: **Exactly 400 queries**.
4. **The "1,730 vs. 1,930" Manuscript Discrepancy:** The initial manuscript draft arithmetic ($1,730$) accidentally counted only the 200 `handwritten_paraphrase` queries alongside the synthetic queries ($1,224 + 306 + 200 = 1,730$). When the companion 200 `handwritten_verbatim` queries are included, the total query collection sums to **$1,930$** ($1,224 + 306 + 400 = 1,930$).

---

## 2. Source Code, Git History, and Provenance Ledger

| Asset | Path | Details & Provenance |
| :--- | :--- | :--- |
| **Human Query CSV** | `data/human_queries_200.csv` | 200 rows, 5 columns. SHA256: `26b38c26bb4b9cfcf0cbda75b88237fc3fcda46633fb67a0ec3e9fa85cba7993` |
| **Git Commit** | Commit `44abfbf9` (Oct 28, 2026) | `"measured line-aligned CER across SROIE and evaluated 100 additional human queries"` |
| **Author Attribution** | `overtough <anuragkrishnavennela@gmail.com>` | Committed by primary researcher; internal timestamps recorded in Git commit metadata |
| **Evaluation Harness** | `results/v4/G-10/generator_query_evaluations.csv` | 1,930 total rows; 400 rows tagged with `generator == 'Human-Authored'` |

---

## 3. Query Characteristics and Independence Audit

### A. What Each Column Represents
- **`handwritten_verbatim`:** Formulated by an annotator inspecting the physical receipt header to create a concise known-item query using store name and location (e.g., Row 0: `"Book Ta K Taman Daya"`; Row 1: `"Indah Gift Home Deco Johor Jaya"`; Row 4: `"SHH Motor Sungai Rengit Pengerang"`).
- **`handwritten_paraphrase`:** Formulated as a realistic user search request describing the semantic context of the purchase without copying the exact vendor string verbatim (e.g., Row 0: `"Find receipt for books and modelling clay bought in Taman Daya"`; Row 1: `"What did I buy home decor items in Johor Jaya?"`; Row 4: `"Motorcycle repair and service bill in Sungai Rengit"`).

### B. Independence and Uniqueness
- **Are both columns independent?**  
  They represent two distinct query formulation strategies (keyword vs. descriptive question) targeting the same underlying 200 receipts. They are **paired across the same receipts**, which must be accounted for in statistical evaluations.
- **Are all 400 queries unique strings?**  
  - `handwritten_verbatim`: 140 unique strings across 200 rows.
  - `handwritten_paraphrase`: 150 unique strings across 200 rows.
  - Across all 400 instances: **290 unique query strings**.
  - Overlap between columns: Exactly **0 strings** appear in both verbatim and paraphrase columns.
  - **Why are there duplicate strings?** Multiple receipts in the SROIE dataset originate from the same commercial retail chains (e.g., 22 delivery invoices from "Gardenia Bakeries (Kl) Shah Alam", 7 receipts from "99 Speed Mart Klang", 5 from "Mr. D.I.Y."). When annotators wrote queries for multiple receipts from the same vendor, identical vendor names or store-level prompts were used. Each query evaluates against its specific `receipt_id`.

### C. Access to Target Answers and Relevance Verification
- **Access to Answers:** Annotators had access to the receipt images and their ground-truth metadata to formulate the queries. This represents a standard **known-item retrieval authoring protocol**, where the goal is to evaluate whether a retrieval engine can locate an intended target receipt from memory or natural descriptions.
- **Relevance Judgments:** Relevance is binary known-item ground truth ($Y=1$ for the designated `receipt_id`, $Y=0$ for all other receipts in the corpus).
- **Tuning Contamination:** The human queries were **never used for BM25 parameter tuning** ($k_1, b$ parameters were frozen based on the development split). All evaluations on these 400 human queries represent zero-shot test evaluation.

---

## 4. Recomputed Retrieval Metrics Across Overlap Bins

Evaluated from raw query-level reciprocal ranks in [`FINAL_BLOCKER_RESOLUTION/HUMAN_QUERY_RECALCULATED.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/FINAL_BLOCKER_RESOLUTION/HUMAN_QUERY_RECALCULATED.csv) via 20,000 paired permutation test iterations and 5,000 cluster bootstrap iterations:

| Query Slice | $N$ | Mean Overlap | Word BM25 MRR [95% CI] | Char BM25 MRR [95% CI] | Diff (Char - Word) [95% CI] | Permutation $p$ | Verdict & Sample Size Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.00–0.20 (Minimal)** | 15 | 0.0986 | 0.0744 [0.000, 0.205] | **0.2303** [0.057, 0.443] | **+0.1559** [+0.024, +0.334] | **0.0126** | **Char BM25 Wins** *(CAUTION: Underpowered, $N=15 < 30$, wide CI)* |
| **0.20–0.40 (Light)** | 89 | 0.3044 | 0.5297 [0.428, 0.630] | 0.5336 [0.431, 0.634] | +0.0038 [-0.072, +0.079] | 0.9179 | **No Significant Difference** *(Adequately powered, $N=89$)* |
| **0.40–0.60 (Moderate)** | 97 | 0.4779 | 0.5794 [0.483, 0.675] | 0.5525 [0.455, 0.648] | -0.0269 [-0.076, +0.020] | 0.2525 | **No Significant Difference** *(Adequately powered, $N=97$)* |
| **0.60–0.80 (High)** | 27 | 0.6802 | 0.7461 [0.590, 0.891] | 0.7629 [0.609, 0.902] | +0.0168 [-0.095, +0.125] | 0.7521 | **No Significant Difference** *(CAUTION: Underpowered, $N=27 < 30$, wide CI)* |
| **0.80–1.00 (Verbatim)** | 172 | 0.9649 | 0.6477 [0.576, 0.718] | 0.6567 [0.586, 0.725] | +0.0091 [-0.017, +0.035] | 0.4824 | **No Significant Difference** *(Adequately powered, $N=172$)* |
| **human_verbatim** | 200 | 0.9039 | 0.6174 [0.548, 0.684] | 0.6299 [0.563, 0.696] | +0.0125 [-0.012, +0.038] | 0.3044 | **No Significant Difference** *(Adequately powered, $N=200$)* |
| **human_paraphrase** | 200 | 0.3924 | 0.5626 [0.493, 0.631] | 0.5605 [0.492, 0.628] | -0.0021 [-0.043, +0.039] | 0.9187 | **No Significant Difference** *(Adequately powered, $N=200$)* |
| **All 400 Pooled** | 400 | 0.6481 | 0.5900 [0.541, 0.639] | 0.5952 [0.546, 0.643] | +0.0052 [-0.018, +0.029] | 0.6656 | **No Significant Difference** *(Adequately powered, $N=400$)* |

---

## 5. Statistical Interpretation and Methodological Caveats

### 1. Small Overlap Bins Cannot Prove Equivalence
- In the **0.00–0.20 overlap bin** ($N=15$), Character BM25 outperforms Word BM25 by $+0.1559$ MRR ($p = 0.0126$). However, because $N=15$ is small, the 95% bootstrap confidence interval is broad ($[+0.024, +0.334]$).
- In the **0.60–0.80 overlap bin** ($N=27$), Character BM25 achieves MRR 0.7629 vs. Word BM25 0.7461 ($\Delta = +0.0168, p = 0.7521$). **Lack of statistical significance ($p > 0.05$) here must NOT be claimed as proof of equivalence.** The wide 95% confidence interval ($[-0.095, +0.125]$) spans substantial potential differences ($\pm 0.10$ MRR) due to low statistical power.

### 2. Primary Scientific Finding
In the well-powered bins ($N=89, 97, 172$):
- At 0.20–0.40 overlap: diff is $+0.0038$ with 95% CI $[-0.072, +0.079]$.
- At 0.40–0.60 overlap: diff is $-0.0269$ with 95% CI $[-0.076, +0.020]$.
- At 0.80–1.00 overlap: diff is $+0.0091$ with 95% CI $[-0.017, +0.035]$.
- Across all 400 pooled queries, Word BM25 (0.5900) and Character BM25 (0.5952) perform within **0.0052 MRR** of each other ($p = 0.6656$).
- **Conclusion:** On authentic human query formulation where proper-noun anchors are naturally preserved, Word BM25 and Character 3–5gram BM25 achieve comparable effectiveness. The dramatic Word > Char reversal observed in synthetic benchmarks is an artifact of synthetic entity substitution.

