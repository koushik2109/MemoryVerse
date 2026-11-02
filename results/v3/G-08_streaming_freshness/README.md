# G-08: Temporal / Streaming Corpus Updates and Index Freshness

## Overview
This directory contains experimental results for **G-08**, systematically investigating document retrieval dynamics,
inverted index freshness, and temporal scoring priors under continuous streaming ingestion in a personal episodic memory system:
1. **Streaming Timeline:** 1-year episodic growth timeline from Day 1 to Day 365, scaling across $N \in [100, 250, 500, 813, 1200, 2000]$ documents.
2. **Corpus Composition ($N=2{,}000$):** 313 SROIE test receipts (with authentic Malaysian merchant headers, line items, and transaction dates) + 500 MS-COCO photo distractors + 1,187 synthetic and dev commercial document distractors.
3. **Index Maintenance Regimes Evaluated:**
   - **Full Re-Index Oracle:** Complete recomputation of global document frequencies and IDF from scratch at each checkpoint ($O(N)$ write cost).
   - **Dynamic Online Incremental BM25:** $O(1)$ posting list append with on-the-fly exact query-term IDF computation at search time.
   - **Periodic Batch Epoch Re-indexing:** Updates cached IDF table every $\Delta N = 100$ documents.
   - **Frozen Baseline IDF:** Freezes IDF statistics at cold start ($N_0 = 100$ documents; Day 18) and never updates global term weights.
   - **Dense Vector Flat Append:** Exact cosine similarity over appended 768-dim `BGE-base-en-v1.5` embeddings.
4. **Episodic Temporal Routing:** Adaptive Temporal Router vs. Static Recency Prior vs. Uniform Retrieval across Atemporal and Temporal queries.
5. **Statistical Rigor:** Paired permutation tests ($B=10{,}000$) with Holm-Bonferroni correction and $B=2{,}000$ bootstrap confidence intervals.

---

## Headline Findings

### 1. Zero Approximation Loss of Dynamic Online Incremental BM25 ($O(1)$ Write Cost):
- Across all checkpoints ($N=100$ to $N=2{,}000$) and all query slices (Verbatim, Paraphrase, Temporal), **Dynamic Online Incremental BM25 achieves 100% mathematical parity with the Full Re-Index Oracle** (exact identical MRR: $0.6595$ Word Verbatim, $0.4015$ Word Paraphrase, $0.6806$ Word Temporal; $0.8411$ Char Verbatim, $0.2933$ Char Paraphrase, $0.7852$ Char Temporal; $p = 1.000$).
- While the Full Re-Index Oracle incurs an $O(N)$ re-indexing latency penalty ($18.2$ ms/doc at $N=2{,}000$), Dynamic Online updates require only **$0.15$ ms/doc** write time (a **$121\times$ reduction in mobile ingestion compute**), computing exact IDF for the 3–8 query terms in $< 5\,\mu\text{s}$ at retrieval time.

### 2. Lexical IDF Drift and the Frozen Baseline Collapse:
- Over the 365-day streaming timeline, the vocabulary expands from $1{,}077$ terms ($N=100$) to **$7{,}226$ terms** ($N=2{,}000$), leaving $6{,}149$ terms ($85.1\%$ of the corpus vocabulary) completely unseen by the frozen baseline.
- The IDF Mean Absolute Error (MAE) between frozen and true global weights grows monotonically from $0.00$ to **$1.65$**.
- This statistical drift induces severe retrieval degradation:
  - On semantic paraphrases, Word BM25 collapses from $0.4015$ down to **$0.2554$ MRR** (a **$-36.4\%$ decay**, diff $-0.1461$, $p_{\text{perm}} < 0.0001$).
  - On verbatim lookups, Char 3–5g BM25 drops from $0.8411$ to **$0.8083$ MRR** (diff $-0.0328$, $p_{\text{perm}} = 0.0026$).
  - On temporal queries, Word BM25 drops from $0.6806$ to **$0.6011$ MRR** (diff $-0.0795$).

### 3. Adaptive Temporal Router vs. Static Recency Penalty:
- Personal memory queries frequently carry episodic time anchors ("payment in June 2018", "receipt from 15/05/2018").
- **Static Recency Prior Failure:** Applying a standard static exponential decay prior ($\exp(-\Delta t / \tau)$ measured from today) severely penalizes older historical documents, degrading atemporal verbatim queries from $0.8411$ to $0.8269$ MRR and dropping episodic temporal queries to **$0.7222$ MRR** (trailing uniform retrieval by $-0.0630$ MRR).
- **Adaptive Temporal Router Superiority:** An intent-gated classifier detecting temporal markers triggers a target-date Gaussian temporal kernel ($\tau = 14$ days), boosting episodic temporal MRR from $0.7852$ (Uniform) to **$0.9413$ MRR** (a massive **$+0.1561$ lift**, $p < 0.0001$), while preserving exact historical lookups on atemporal queries ($0.8433$ vs. $0.8411$ MRR, $p > 0.05$).

---

## Tabular Summary

### Streaming Retrieval Progression ($N=100 \to 2{,}000$)
| Checkpoint ($N$) | Day of Year | Word Dynamic (Verb) | Word Frozen (Verb) | Char Dynamic (Verb) | Char Frozen (Verb) | Dense BGE (Verb) | IDF MAE Drift |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | Day 18 | 0.6440 | 0.6440 | 0.7213 | 0.7213 | 0.5237 | 0.000 |
| 250 | Day 46 | 0.7656 | 0.7627 | 0.8450 | 0.8340 | 0.4945 | 0.591 |
| 500 | Day 90 | 0.7517 | 0.7482 | 0.8524 | 0.8315 | 0.4589 | 0.974 |
| 813 | Day 140 | 0.7428 | 0.7391 | 0.8492 | 0.8276 | 0.4421 | 1.218 |
| 1,200 | Day 202 | 0.6984 | 0.6892 | 0.8245 | 0.8012 | 0.4185 | 1.389 |
| 2,000 | Day 365 | **0.6595** | **0.6423** | **0.8411** | **0.8083** | **0.4028** | **1.647** |

### Temporal Intent & Recency Routing ($N=2{,}000$)
| Query Condition | Strategy | MRR | 95% Bootstrap CI | Hit@1 (%) | Hit@10 (%) |
|:---|:---|:---:|:---:|:---:|:---:|
| Atemporal (Verbatim) | Uniform Retrieval (No Prior) | 0.8411 | [0.7742, 0.8984] | 79.41% | 94.12% |
| Atemporal (Verbatim) | Static Recency Prior (Day 365) | 0.8269 | [0.7603, 0.8861] | 77.45% | 93.14% |
| Atemporal (Verbatim) | **Adaptive Temporal Router** | **0.8433** | [0.7770, 0.8997] | **79.41%** | **95.10%** |
| Episodic Temporal | Uniform Retrieval (No Prior) | 0.7852 | [0.7157, 0.8546] | 70.59% | 94.12% |
| Episodic Temporal | Static Recency Prior (Day 365) | 0.7222 | [0.6463, 0.7952] | 63.73% | 93.14% |
| Episodic Temporal | **Adaptive Temporal Router** | **0.9413** | [0.8969, 0.9779] | **92.16%** | **98.04%** |

