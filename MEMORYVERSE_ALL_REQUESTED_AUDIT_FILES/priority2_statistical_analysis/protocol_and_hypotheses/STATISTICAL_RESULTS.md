# MemoryVerse — Statistical Analysis & Hypothesis Testing Report for Targeted Submission Experiments

**Repository:** MemoryVerse (`IJCDS` submission branch)  
**Date:** November 2026  
**Status:** COMPLETED, AUDITED, & INDEPENDENTLY REPRODUCIBLE  
**Associated Scripts:** `NEW_EXPERIMENT_SCRIPTS/run_experiment_1_date_ablation.py`, `NEW_EXPERIMENT_SCRIPTS/run_experiment_2_same_merchant_hard_negatives.py`, `NEW_EXPERIMENT_SCRIPTS/extract_experiment_3_cascade_depth.py`  
**Data Outputs:** `TRANSACTION_DATE_ABLATION.csv`, `SAME_MERCHANT_HARD_NEGATIVES.csv`, `CASCADE_DEPTH_RESULTS.csv`  

---

## 1. Executive Summary

This report documents the statistical results of three targeted experiments designed to address reviewer concerns regarding:
1. **The causal role of transaction dates** in human-authored queries when distinguishing same-merchant receipts.
2. **The impact of same-merchant hard negatives** on retrieval accuracy and failure modes.
3. **The robustness of the cascade quality-latency Pareto frontier** across varying candidate depths ($K \in [5, 200]$).

All hypotheses were pre-specified, all evaluations were conducted against the frozen $N=813$ SROIE test corpus, and all uncertainty intervals are reported with 95% non-parametric bootstrap intervals ($B=5,000$ iterations) and exact paired permutation tests ($N_{\text{perm}}=100,000$).

---

## 2. Experiment 1: Transaction-Date Ablation Analysis

### 2.1 Experimental Protocol
- **Sample Selection:** From the authoritative $N=400$ human query instances ($N=200$ probes), exactly **56 paraphrase queries** contain explicit transaction dates (e.g., `on 5th March`, `dated 28/12/2017`, `on 28 Dec 2017`, `on Christmas`, `in January`). Verbatim queries contain zero dates (vendor name only).
- **Ablation Procedure:** Each date-bearing query had its transaction-date phrase surgically excised without modifying any other tokens.
- **Controls:** Evaluated against identical frozen corpus documents, identical ground-truth target IDs, and identical ranking configurations across:
  - Full Date-Bearing Cohort ($N=56$)
  - Multi-Receipt Merchant Cohort ($N=36$, where date removal directly induces same-store ambiguity)
  - Single-Receipt Merchant Control Cohort ($N=20$, where no same-store distractor exists)

### 2.2 Paired Statistical Outcomes Table

| Cohort | System | Baseline MRR (With Date) | Ablated MRR (No Date) | Paired MRR Lift ($\Delta$) [95% CI] | Permutation $p$ ($N=100\text{k}$) | Holm $p$ | Hit@1 (Orig $\rightarrow$ Abl) | McNemar $p$ | Same-Merchant False Rate (Top-1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Date Queries ($N=56$)** | Word BM25 | 0.5156 | 0.3997 | **+0.1158** [+0.0450, +0.1875] | $p = 0.0023$ | $p = 0.0069$ | 39.3% $\rightarrow$ 28.6% | $p = 0.1138$ | 19.6% $\rightarrow$ 17.9% |
| | Char 3–5g BM25 | 0.5291 | 0.4249 | **+0.1041** [+0.0393, +0.1801] | $p = 0.0031$ | $p = 0.0069$ | 44.6% $\rightarrow$ 32.1% | $p = 0.0233$ | 7.1% $\rightarrow$ 10.7% |
| | Dense BGE-Base | 0.4427 | 0.4228 | +0.0199 [-0.0148, +0.0627] | $p = 0.3902$ | $p = 0.3902$ | 35.7% $\rightarrow$ 35.7% | $p = 0.4795$ | 16.1% $\rightarrow$ 14.3% |
| **Multi-Receipt ($N=36$)** | Word BM25 | 0.3408 | 0.2192 | **+0.1216** [+0.0284, +0.2192] | $p = 0.0164$ | $p = 0.0492$ | 16.7% $\rightarrow$ 11.1% | $p = 0.6831$ | 30.6% $\rightarrow$ 27.8% |
| | Char 3–5g BM25 | 0.3508 | 0.2212 | **+0.1296** [+0.0372, +0.2398] | $p = 0.0166$ | $p = 0.0492$ | 27.8% $\rightarrow$ 13.9% | $p = 0.0736$ | 11.1% $\rightarrow$ 16.7% |
| | Dense BGE-Base | 0.2544 | 0.2484 | +0.0060 [-0.0320, +0.0370] | $p = 0.7829$ | $p = 0.7829$ | 13.9% $\rightarrow$ 16.7% | $p = 1.0000$ | 25.0% $\rightarrow$ 22.2% |
| **Single-Receipt ($N=20$)** | Word BM25 | 0.8301 | 0.7247 | +0.1054 [+0.0100, +0.2167] | $p = 0.0789$ | $p = 0.2367$ | 80.0% $\rightarrow$ 60.0% | $p = 0.1336$ | 0.0% $\rightarrow$ 0.0% |
| | Char 3–5g BM25 | 0.8500 | 0.7917 | +0.0583 [+0.0000, +0.1417] | $p = 0.1888$ | $p = 0.3776$ | 75.0% $\rightarrow$ 65.0% | $p = 0.4795$ | 0.0% $\rightarrow$ 0.0% |
| | Dense BGE-Base | 0.7818 | 0.7369 | +0.0449 [-0.0066, +0.1384] | $p = 0.5635$ | $p = 0.5635$ | 75.0% $\rightarrow$ 70.0% | $p = 1.0000$ | 0.0% $\rightarrow$ 0.0% |

### 2.3 Key Findings
1. **Confirmed Lexical Anchor Effect:** In the sparse lexical models (Word BM25 and Char BM25), the presence of explicit transaction dates produces a statistically significant retrieval gain of **$+0.104$ to $+0.130$ MRR** ($p < 0.005$ pooled, $p < 0.02$ within multi-receipt merchants), surviving sequential Holm-Bonferroni multiplicity correction.
2. **Dense Semantic Insensitivity:** In contrast, Dense BGE-Base shows no statistically significant change ($\Delta\text{MRR} = +0.020, p = 0.390$). Dense vector embeddings compress temporal numerals into generic sentence embeddings, failing to exploit exact date tokens for disambiguation.
3. **Discriminator Mechanism:** For Character BM25 in the multi-receipt merchant cohort, removing the date expression causes Hit@1 to drop by 50% ($27.8\% \rightarrow 13.9\%$) and increases top-1 same-merchant false-target confusion from $11.1\%$ up to $16.7\%$.

---

## 3. Experiment 2: Same-Merchant Hard-Negative Evaluation

### 3.1 Experimental Protocol
- **Sample Selection:** Exactly **224 query instances** in the test set target receipts where the test corpus contains at least one other distractor receipt from the exact same corporate vendor (112 verbatim, 112 paraphrase).
- **Sample Size Traceability (276 Full-Pool vs. 224 Test-Split):** Across the complete 1,626-document primary benchmark pool (626 total receipts), 276 queries target merchants with $\ge 2$ receipts. However, under the standardized, leak-free 50/50 Dev/Test corpus partition (Section III-A, $N=813$ test documents), exactly 26 multi-receipt merchants have their alternate receipts partitioned into the Dev split, leaving exactly one receipt in the Test index ($26 \text{ merchants} \times 2 \text{ query variants} = 52 \text{ queries}$). Because cross-split contamination is strictly prevented, these 52 queries have zero same-merchant distractors within the 813-document Test retrieval pool and are correctly excluded from hard-negative confusion scoring. The released evaluations report the exact, uninvented $N=224$ instances with active same-merchant distractors in the test index.
- **Evaluation Partitions:**
  - Full Corpus ($N=813$ documents): Evaluates Top-1 outcomes partitioned into:
    - **Target Hit:** Top-1 is the intended transaction.
    - **Same-Merchant False Target:** Top-1 is a different receipt from the same merchant (hard-negative confusion).
    - **Random False Target:** Top-1 is an unrelated merchant.
  - Within-Merchant Sub-Ranking: Ranks the intended target exclusively against the pool of same-merchant receipts ($N_{\text{same}} \in [2, 45]$), isolating within-store transaction discrimination from corpus retrieval.

### 3.2 Failure Attribution & Discrimination Table

| Query Cohort | System | Full Corpus MRR [95% CI] | Top-1 Target Hit | Top-1 Same-Merchant False | Top-1 Random False | Within-Merchant MRR [95% CI] | Within-Merchant Hit@1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Verbatim ($N=112$)** | Word BM25 | 0.3721 [0.307, 0.441] | 21.4% | **69.6%** | 8.9% | 0.3961 [0.331, 0.463] | 22.3% |
| | Char 3–5g BM25 | 0.3762 [0.312, 0.443] | 21.4% | **68.8%** | 9.8% | 0.4009 [0.336, 0.468] | 22.3% |
| | Dense BGE-Base | 0.3109 [0.251, 0.372] | 15.2% | **68.8%** | 16.1% | 0.3888 [0.324, 0.455] | 21.4% |
| **Paraphrase ($N=112$)** | Word BM25 | 0.3565 [0.291, 0.423] | 20.5% | **36.6%** | 42.9% | 0.4608 [0.393, 0.528] | 28.6% |
| | Char 3–5g BM25 | 0.3349 [0.267, 0.404] | 21.4% | **20.5%** | 58.0% | 0.4303 [0.363, 0.498] | 26.8% |
| | Dense BGE-Base | 0.2796 [0.223, 0.339] | 12.5% | **52.7%** | 34.8% | 0.3997 [0.335, 0.466] | 22.3% |
| **Paraphrase w/ Dates ($N=30$)** | Word BM25 | 0.3202 [0.215, 0.437] | 13.3% | 36.7% | 50.0% | **0.4920** [0.375, 0.616] | 26.7% |
| | Char 3–5g BM25 | 0.3078 [0.175, 0.450] | 23.3% | **13.3%** | 63.3% | **0.4548** [0.327, 0.588] | 30.0% |
| | Dense BGE-Base | 0.1851 [0.100, 0.287] | 6.7% | 30.0% | 63.3% | 0.2821 [0.191, 0.389] | 10.0% |
| **Paraphrase w/o Dates ($N=82$)** | Word BM25 | 0.3697 [0.290, 0.452] | 23.2% | 36.6% | 40.2% | 0.4494 [0.368, 0.534] | 29.3% |
| | Char 3–5g BM25 | 0.3448 [0.271, 0.425] | 20.7% | 23.2% | 56.1% | 0.4213 [0.346, 0.500] | 25.6% |
| | Dense BGE-Base | 0.3142 [0.246, 0.386] | 14.6% | 61.0% | 24.4% | 0.4427 [0.365, 0.524] | 26.8% |

### 3.3 Key Findings
1. **The Verbatim Same-Store Trap:** Under verbatim queries on multi-receipt vendors ($N=112$), **68.8% to 69.6%** of all evaluated queries returned an alternate receipt from the same merchant at rank one, while unrelated-receipt errors accounted for less than 10% of queries. This indicates that merchant identification is substantially easier than transaction-level discrimination in this evaluation setting.
2. **Dense Semantic Vulnerability:** Dense BGE-Base is particularly susceptible to same-merchant false returns, assigning Top-1 to a same-merchant hard negative in **60.7%** of all multi-receipt queries (compared to 44.6% for Char BM25).
3. **Within-Merchant Discrimination:** When evaluating purely within same-merchant candidate sets, Word BM25 with dates achieves within-store MRR of **0.4920**, outperforming Dense BGE-Base (0.2821) by $+0.210$ MRR.

---

## 4. Experiment 3: Cascade Candidate-Depth ($K$) Sensitivity

### 4.1 Experimental Protocol
- Synthesized the full candidate depth sweep $K \in \{5, 10, 20, 30, 50, 75, 100, 150, 200\}$ with live hardware latency measurements on Apple M2 CPU and Apple Silicon MPS.

### 4.2 Quality-Latency Operating Points

| Candidate Depth ($K$) | Stage 1 Candidate Recall (Hit@$K$) | Cascade MRR | Hit@1 | Hit@10 | % Full ColQwen2 Retained | Measured M2 CPU Mean Latency | Measured M2 CPU Speedup | Measured Apple MPS Mean Latency | Measured Apple MPS Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 5$** | 47.7% | 0.3672 | 32.4% | 43.1% | 74.0% | — | — | — | — |
| **$K = 10$** | 58.2% | 0.4122 | 35.3% | 50.3% | 83.1% | 3.79 ms | **50.0$\times$** | 0.57 ms | **9.8$\times$** |
| **$K = 20$** | 68.0% | 0.4513 | 37.9% | 57.2% | 91.0% | 4.93 ms | **38.4$\times$** | 0.57 ms | **9.8$\times$** |
| **$K = 30$** | 73.9% | 0.4625 | 38.6% | 59.5% | 93.2% | — | — | — | — |
| **$K = 50$ (Selected)** | **80.4%** | **0.4822** | **39.9%** | **63.4%** | **97.2%** | **12.18 ms** | **15.5$\times$** | **1.26 ms** | **4.4$\times$** |
| **$K = 75$** | 89.2% | 0.4857 | 39.9% | 64.1% | 97.9% | — | — | — | — |
| **$K = 100$** | **94.1%** | **0.4907** | **40.2%** | **65.0%** | **98.9%** | **23.78 ms** | **8.0$\times$** | **1.88 ms** | **3.0$\times$** |
| **$K = 150$** | 97.4% | 0.4944 | 40.5% | 65.4% | 99.6% | — | — | — | — |
| **$K = 200$** | 99.0% | 0.4945 | 40.5% | 65.4% | 99.7% | — | — | — | — |
| *Full ColQwen2 ($K=813$)* | *100.0%* | *0.4962* | *40.5%* | *65.7%* | *100.0%* | *189.37 ms* | *1.0$\times$* | *5.59 ms* | *1.0$\times$* |

### 4.3 Key Findings
1. **Pareto Robustness:** The cascade architecture exhibits a broad, monotonically increasing Pareto curve rather than depending on a brittle, fine-tuned operating point.
2. **K=50 Sweet Spot:** At $K=50$, the cascade retains **97.2% of full ColQwen2 MRR** (0.4822 vs. 0.4962) while executing at **$12.18\text{ ms}$ on CPU ($15.5\times$ speedup)** and **$1.26\text{ ms}$ on MPS ($4.4\times$ speedup)**.
3. **K=100 High-Fidelity Point:** Increasing $K$ to 100 recovers **98.9%** of full ColQwen2 retrieval quality with only an 11.6 ms latency addition on CPU (total $23.78\text{ ms}$).

