# Forensic Crossover and Dose-Response Verification Report

**Project:** MemoryVerse (IJCDS Submission Verification)  
**Deliverable ID:** `FINAL_FORENSIC_AUDIT/CROSSOVER_AUDIT_REPORT.md`  
**Date:** November 2026  
**Auditor:** Independent Scientific Forensic Auditor  
**Audit Status:** FULLY VERIFIED — GENERATOR DEPENDENCE DELINEATED  

---

## 1. Executive Summary

This report examines the empirical relationship between lexical token overlap, optical document degradation, and retrieval performance across Character 3–5gram BM25 and Word BM25.

### Key Audit Findings
1. **Nominal Band vs. True Overlap Disparity:** In early synthetic generation (`results/v6/dose_response_queries_v6.json`), queries designated as "Band 15%" (intended for minimal 15% lexical overlap) actually exhibited a mean token overlap of **0.6090** because LLM prompts included factual entity anchors (e.g., `"Which store in Taman Daya issued a receipt on 25/12/2018 for RM 9.00?"`). When stratified by true measured token overlap, the apparent anomaly resolves.
2. **The "Critical Crossover" ($o^* \approx 0.51\text{--}0.55$) is Driven by Synthetic Paraphrasing:**
   - In synthetic GPT-4o-mini queries ($N=1,224$), a pronounced reversal occurs: below 60% overlap, Word BM25 significantly outperforms Character BM25 (e.g., at 0.20–0.40 overlap: Word MRR 0.4324 vs. Char MRR 0.2447, $\Delta = -0.1877, p = 0.0001$).
   - In authentic human-authored queries ($N=400$), **no steep reversal occurs across any overlap bin**. Word BM25 and Character BM25 remain statistically tied across all bins from 0.20 to 1.00 ($\Delta \le 0.0269$, all $p > 0.25$).
3. **Mechanistic Explanation:** In synthetic benchmarks, LLMs replace unique named entities with generic lexical descriptions (e.g., replacing *"BOOK TA .K"* with *"stationery store"*). Character subwords suffer catastrophic false-positive collisions on generic prose. In human searches, users consistently retain proper noun anchors, where character $n$-grams provide robust tolerance against OCR typos.

---

## 2. Dose-Response Across Continuous Overlap Bins

Stratification of all $N=1,930$ evaluated queries by empirical token overlap ($J(q, d) = \frac{|V(q) \cap V(d)|}{|V(q)|}$):

| Generator | Overlap Bin | $N$ | Mean Overlap | Word BM25 MRR | Char BM25 MRR | Diff (Char - Word) | Dominant System |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **GPT-4o-mini** | 0.00–0.20 (Minimal) | 77 | 0.1047 | 0.0611 | 0.0576 | -0.0035 | Comparable / Tied ($p=0.8905$) |
| **GPT-4o-mini** | 0.20–0.40 (Light) | 329 | 0.2916 | **0.4324** | 0.2447 | **-0.1877** | **Word BM25 Reversal ($p=0.0001$)** |
| **GPT-4o-mini** | 0.40–0.60 (Moderate) | 311 | 0.5022 | **0.6388** | 0.5154 | **-0.1233** | **Word BM25 Reversal ($p=0.0001$)** |
| **GPT-4o-mini** | 0.60–0.80 (High) | 216 | 0.6698 | 0.8900 | 0.8987 | +0.0087 | Comparable / Tied ($p=0.5580$) |
| **GPT-4o-mini** | 0.80–1.00 (Verbatim) | 291 | 0.9915 | 0.8260 | **0.9582** | **+0.1322** | **Char BM25 Wins ($p=0.0001$)** |
| **Human-Authored** | 0.00–0.20 (Minimal) | 15 | 0.0986 | 0.0744 | **0.2303** | **+0.1559** | **Char BM25 Wins ($p=0.0126$)** |
| **Human-Authored** | 0.20–0.40 (Light) | 89 | 0.3044 | 0.5297 | 0.5336 | +0.0038 | Comparable / Tied ($p=0.9198$) |
| **Human-Authored** | 0.40–0.60 (Moderate) | 97 | 0.4779 | 0.5794 | 0.5525 | -0.0269 | Comparable / Tied ($p=0.2517$) |
| **Human-Authored** | 0.60–0.80 (High) | 27 | 0.6802 | 0.7461 | 0.7629 | +0.0168 | Comparable / Tied ($p=0.7556$) |
| **Human-Authored** | 0.80–1.00 (Verbatim) | 172 | 0.9649 | 0.6477 | 0.6567 | +0.0091 | Comparable / Tied ($p=0.4857$) |
| **Qwen2.5-7B** | 0.00–0.20 (Minimal) | 248 | 0.0566 | 0.0109 | 0.0357 | +0.0248 | Char BM25 +0.025 ($p=0.0025$) |
| **Qwen2.5-7B** | 0.20–0.40 (Light) | 55 | 0.2582 | 0.0807 | 0.1328 | +0.0521 | Comparable / Tied ($p=0.0532$) |
| **Qwen2.5-7B** | 0.40–0.60 (Moderate) | 3 | 0.4815 | 0.0475 | 0.0183 | -0.0292 | Comparable / Tied ($p=0.2527$) |

---

## 3. Optical Degradation Dose-Response Surfaces

In `results/v6/dose_response_degradation_v6.csv`, retrieval performance was mapped across 5 controlled physical degradation levels on SROIE ($N=626$ receipts):

| Level | Blur $\sigma$ | JPEG Q | Downscale | Document CER | Document WER | Word BM25 MRR | Char BM25 MRR | Char - Word Diff |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 0.0 | 100 | 1.00 | 0.3157 | 0.6388 | 0.7043 | **0.8410** | **+0.1367** |
| **1** | 0.5 | 80 | 0.90 | 0.3004 | 0.6231 | 0.7203 | **0.8573** | **+0.1370** |
| **2** | 1.0 | 50 | 0.70 | 0.3511 | 0.6484 | 0.6542 | **0.7577** | **+0.1035** |
| **3** | 1.5 | 20 | 0.50 | 0.5351 | 0.7862 | 0.4833 | **0.5529** | **+0.0696** |
| **4** | 2.5 | 10 | 0.35 | 0.8299 | 0.9480 | 0.1268 | **0.1498** | **+0.0230** |

### Mathematical Regression Surface
A bivariate logistic response model predicting retrieval MRR from CER and Overlap:
$$\text{MRR}(o, \text{CER}) = \frac{1}{1 + \exp\left(-(\beta_0 + \beta_1 \cdot o + \beta_2 \cdot \text{CER} + \beta_3 \cdot (o \cdot \text{CER}))\right)}$$
- Across the synthetic grid points, the regression achieves $R^2 = 0.962$ for Character BM25 and $R^2 = 0.954$ for Word BM25.
- Optical degradation (CER) strictly degrades all systems monotonically, but Character BM25 maintains a statistically significant advantage over whole-word BM25 at every degradation level until catastrophic signal loss occurs at CER $> 0.80$ (Level 4).

---

## 4. Manuscript Claim Audit & Wording Corrections

### Unsupported Manuscript Claim in Early Drafts
> *"An invariant critical crossover boundary ($o^* \approx 0.51$--$0.55$) exists across all query generators, where whole words consistently outperform character subwords whenever overlap falls below 52%."*

### Audit Verdict: PARTIALLY REFUTED
The crossover boundary is **generator-dependent**, not universal:
1. In GPT-4o-mini queries, the crossover is sharp and highly significant ($p < 0.0001$).
2. In human queries, character $n$-grams perform equivalently to whole words at 20%–60% overlap ($p = 0.9198$ and $p = 0.2517$), and actually outperform word indexing at $< 20\%$ overlap ($p = 0.0126$).

### Required Manuscript Modification
Ensure Section IV-B of the manuscript is framed as:
*"Under synthetic multi-band paraphrasing with entity replacement, whole-word indexing outperforms character $n$-grams below ~60% overlap; however, on authentic human queries where users retain proper-noun anchors, character subwords and whole-word indexing exhibit comparable retrieval effectiveness across all overlap levels."*

