# SCIENTIFIC REANALYSIS: QUERY OVERLAP DOSE-RESPONSE & CROSSOVER DYNAMICS
**Manuscript:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Auditor:** Independent Senior Information Retrieval & Statistical Reviewer  
**Date:** November 2026  
**Audit Directory:** `IJCDS_SECOND_AUDIT/`  

---

## 1. Executive Summary & Core Defects Identified

The crossover boundary between Word BM25 and Character $n$-gram BM25 is one of the paper's central theoretical contributions. The manuscript claims:
1. Retrieval effectiveness undergoes a sharp crossover at lexical overlap $\theta \in [0.51, 0.55]$, below which Character BM25 dominates and above which Word BM25 dominates.
2. This crossover boundary is statistically invariant across query generators ($F = 0.413, p = 0.6617$).
3. The evaluation evaluated $N = 1{,}730$ queries.

### Independent Second-Stage Audit Findings:
1. **Sample Size Reporting Error:** The evaluation actually scored **$1{,}930$ queries** (1,224 GPT-4o-mini + 400 Human-Authored + 306 Qwen2.5-7B). The text states $1{,}730$ because 200 human query rows (each containing both a verbatim and a paraphrase query, totaling 400 queries) were miscounted as 200 queries.
2. **Severe Overfitting of the 2D Response Surface:** In `scripts/gapfill_t2_01_regime_map.py`, the 6-parameter quadratic logistic model was fitted to only **12 aggregated summary points** using fixed categorical midpoints ($0.35, 0.65, 1.00$) rather than continuous per-query overlaps. The claimed crossover range of $[0.51, 0.55]$ is a mathematical artifact of solving two overfitted curves on 12 points.
3. **Absence of Crossover on Human Queries:** Query-level analysis of the 400 authentic human-authored queries reveals that **Character BM25 and Word BM25 are virtually tied across all overlap tiers** ($\Delta \text{MRR} \le 0.027$ across all bins from $0.20$ to $1.00$). The steep crossover effect is exclusively an artifact of synthetic LLM query generation.
4. **Invalid Invariance Claim:** Concluding "generator invariance" from an interaction ANOVA with $p = 0.6617$ is a classic statistical fallacy (absence of evidence $\neq$ evidence of absence). Equivalence was never tested, and human queries exhibit a flat response surface.

---

## 2. Dataset Reconstruction & Overlap Verification

### 2.1 Query Sources & Breakdown (`generator_query_evaluations.csv`)
- Total rows: $1{,}930$ unique evaluations across test corpus receipts.
- Breakdown by generator:
  - **GPT-4o-mini:** 1,224 queries ($306 \text{ receipts} \times 4 \text{ overlap bands}$: `band_100`, `band_70`, `band_40`, `band_15`).
  - **Human-Authored:** 400 queries (200 receipts from `data/human_queries_200.csv`, each providing `handwritten_verbatim` and `handwritten_paraphrase`).
  - **Qwen2.5-7B:** 306 queries (frozen test probes heavy paraphrase).

### 2.2 Overlap Metric Verification
- **Formula:**
  $$\text{Overlap}(Q, D) = \frac{|T(Q) \cap T(D)|}{|T(Q)|}$$
  where $T(\cdot)$ extracts lowercased alphanumeric words via regex `\w+`.
- **Target Document:** Real OCR text (Tesseract 5.5) from the SROIE test corpus.
- **Verification:** Overlap calculation verified against raw text; strictly bounded in $[0.0, 1.0]$.

---

## 3. Deconstruction of the 6-Parameter Model

In `scripts/gapfill_t2_01_regime_map.py` (lines 59–65):
$$\text{MRR}_m(\text{CER}, \text{Overlap}) = \sigma\left(\beta_0 + \beta_1 c + \beta_2 o + \beta_3 c^2 + \beta_4 o^2 + \beta_5 c o\right)$$
- The model has 6 free parameters.
- It was trained on `results/v3/engine_replication.csv`, which has only:
  - 4 Optical Conditions: Ground Truth ($\text{CER}=0$), Apple Vision ($\text{CER}=0.2848$), Tesseract Clean ($\text{CER}=0.2939$), Tesseract Degraded ($\text{CER}=0.6377$).
  - 3 Query Overlaps: Verbatim ($o=1.00$), Light ($o=0.35$), Heavy ($o=0.65$).
  - Total Training Points: $4 \times 3 = \mathbf{12\text{ points}}$!
- **Statistical Critique:** Fitting a 6-parameter nonlinear response surface to 12 aggregated data points leaves only $6$ degrees of freedom. Interpolating continuous crossover roots from this surface is methodologically indefensible when 1,930 query-level evaluations are available.

---

## 4. Query-Level Empirical Reanalysis

When evaluating the actual query-level reciprocal ranks in `generator_query_evaluations.csv`:

### 4.1 Binned Performance Table Across All 1,930 Queries

| Overlap Tier | Overlap Range | Mean Overlap | Sample Size $N$ | Word BM25 MRR | Char BM25 MRR | Paired Diff ($\text{Char} - \text{Word}$) | Status |
|---|---|---|---|---|---|---|---|
| **Tier 1 (Minimal)** | $[0.00, 0.20)$ | 0.0693 | 340 | 0.0251 | 0.0492 | $+0.0241$ | Char leads |
| **Tier 2 (Light)** | $[0.20, 0.40)$ | 0.2902 | 473 | 0.4098 | 0.2861 | $-0.1238$ | Word leads |
| **Tier 3 (Moderate)**| $[0.40, 0.60)$ | 0.4963 | 411 | 0.6205 | 0.5206 | $-0.0999$ | Word leads |
| **Tier 4 (High)** | $[0.60, 0.80)$ | 0.6709 | 243 | 0.8740 | 0.8836 | $+0.0096$ | Virtually tied |
| **Tier 5 (Verbatim)**| $[0.80, 1.00]$ | 0.9816 | 463 | 0.7598 | 0.8462 | $+0.0864$ | Char leads |

### 4.2 Generator-Stratified Behavior

#### A. Synthetic LLM Queries (GPT-4o-mini, $N = 1{,}224$)
- $[0.00, 0.20)$: Word = $0.0611$, Char = $0.0576$ (Diff: $-0.0035$)
- $[0.20, 0.40)$: Word = $0.4324$, Char = $0.2447$ (Diff: $-0.1877$, Word leads)
- $[0.40, 0.60)$: Word = $0.6388$, Char = $0.5154$ (Diff: $-0.1233$, Word leads)
- $[0.60, 0.80)$: Word = $0.8900$, Char = $0.8987$ (Diff: $+0.0087$, Tied)
- $[0.80, 1.00]$: Word = $0.8260$, Char = $0.9582$ (Diff: $+0.1322$, Char leads)
- **Empirical Crossover:** Crossover occurs at $\theta \approx 0.65\text{--}0.70$, NOT $0.51\text{--}0.55$.

#### B. Authentic Human Queries ($N = 400$)
- $[0.00, 0.20)$: Word = $0.0744$, Char = $0.2303$ (Diff: $+0.1559$, Char leads, $N=15$)
- $[0.20, 0.40)$: Word = $0.5297$, Char = $0.5336$ (Diff: $+0.0038$, Tied, $N=89$)
- $[0.40, 0.60)$: Word = $0.5794$, Char = $0.5525$ (Diff: $-0.0269$, Tied, $N=97$)
- $[0.60, 0.80)$: Word = $0.7461$, Char = $0.7629$ (Diff: $+0.0168$, Tied, $N=27$)
- $[0.80, 1.00]$: Word = $0.6477$, Char = $0.6567$ (Diff: $+0.0091$, Tied, $N=172$)
- **Empirical Finding:** On authentic human queries, **there is NO steep crossover boundary**. The difference between Word and Character BM25 remains within $\pm 0.027$ across all tiers above $0.20$.

#### C. Open-Weight Model (Qwen2.5-7B, $N = 306$)
- $248$ out of $306$ queries ($81.0\%$) have overlap in $[0.0, 0.20)$.
- ZERO queries have overlap $\ge 0.60$.
- Qwen2.5-7B queries are heavily concentrated in low-overlap semantic paraphrases; they cannot inform crossover dynamics in the $0.50\text{--}1.00$ range.

---

## 5. Critique of the Invariance Claim

The manuscript asserts:
> "A one-way ANOVA across generator slopes confirms that the crossover location is statistically invariant across generators ($F=0.413, p=0.6617$)."

### Methodological Critique:
1. **Misinterpretation of Null Hypothesis:** An ANOVA tests $H_0: \text{slopes are equal}$. A $p$-value of $0.6617$ indicates failure to reject $H_0$. It does **NOT** prove that the slopes are equal (or equivalent).
2. **Distributional Divergence:** The three generators occupy radically different overlap supports:
   - Qwen2.5-7B has zero queries above $0.60$.
   - Human queries are clustered at verbatim ($1.00$) and natural paraphrase ($0.30\text{--}0.50$).
   - Fitting an interaction slope across unequal domains yields high standard errors, trivially inflating the $p$-value.
3. **Substantive Difference:** The slope of $(\text{Char} - \text{Word})$ against overlap is $+0.3589$ for GPT-4o-mini, but $-0.0181$ for human authors. Claiming these are "invariant" is scientifically misleading.

---

## 6. Required Manuscript Replacement Text

### Section 5.B Replacement (`ijcds_body.tex`, lines 496–515):
```latex
To characterize retrieval behavior across varying lexical similarity, we evaluated $N=1{,}930$ queries spanning three sources: GPT-4o-mini ($N=1{,}224$), open-weight Qwen2.5-7B ($N=306$), and authentic human annotators ($N=400$). Fig.~\ref{fig:dose_response} displays retrieval effectiveness as a function of query-document token overlap. For synthetic LLM-generated queries, word BM25 degrades sharply at low overlap while character BM25 degrades gracefully, producing a modeled crossover boundary at higher overlap tiers. However, query-level analysis reveals that the crossover dynamic is heavily modulated by query source: on authentic human queries, character BM25 and word BM25 perform comparably across all overlap ranges ($\Delta \text{MRR} \le 0.027$ across bins from 0.20 to 1.00). While a linear interaction model showed no statistically significant difference in slope ($F=0.413, p=0.6617$), this reflects variance across divergent overlap distributions rather than strict equivalence. Consequently, lexical overlap serves as a diagnostic indicator rather than an invariant universal threshold.
```

---
**Auditor Signature:** Independent Statistical & Reproducibility Auditor  
**Date:** November 4, 2026

