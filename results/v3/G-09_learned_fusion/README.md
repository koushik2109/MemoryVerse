# G-09: Cross-Modal Fusion Beyond Reciprocal Rank Fusion (RRF)

## Overview
This directory contains experimental results for **G-09**, systematically investigating cross-modal fusion architectures
between lexical text proxies (Character 3–5g BM25, Word BM25) and document-native visual late-interaction (ColQwen2):
1. **Evaluated Paradigms (7 Formulations):**
   - **Standalone Baselines:** Char 3–5g BM25, Word BM25, ColQwen2.
   - **Heuristic RRF ($k=60$):** Standard reciprocal rank fusion.
   - **Dev-Tuned RRF ($k^*$):** Optimal smoothing parameter $k^*=1$ tuned on Dev.
   - **Normalized Convex Fusion ($\alpha^*$):** Grid-searched linear combination $\alpha^*=0.05$ tuned on Dev.
   - **Supervised Logistic Reranker:** 6-dimensional feature vector trained on Dev.
   - **Pairwise Neural RankNet:** 2-layer MLP ranker with pairwise cross-entropy loss.
   - **Query-Gated Adaptive Fusion:** Dynamically predicts modality weights based on query intent.
   - **Tri-Modal Fusion:** Multi-view combination of Char BM25 + Word BM25 + ColQwen2.
2. **Evaluation Protocol:**
   - $N=918$ evaluated queries (306 Verbatim, 306 Light Paraphrase, 306 Answerable questions).
   - Strict 50/50 Stratified Split: $459$ Dev queries (tuning & training) and $459$ held-out Test queries (reporting).
   - Statistical Rigor: Paired permutation tests ($B=10{,}000$) with Holm-Bonferroni correction and $B=2{,}000$ bootstrap confidence intervals.

---

## Headline Findings

### 1. The RRF Regularization Advantage:
- Unconstrained supervised rankers (**Logistic Regression:** $0.5003$ MRR; **Pairwise RankNet:** $0.5158$ MRR) underperform **Heuristic RRF ($k=60$:** $0.5393$ MRR) across the held-out test suite.
- Supervised feature-based rankers overfit to score margins on the Dev set, whereas RRF's rank-inversion ($1/(k + r)$) acts as an outlier-resistant rank damper that prevents extreme mispredictions when one modality is completely blind.

### 2. Tri-Modal Fusion Breaks the Ceiling:
- Combining all three distinct representations (**Character 3–5g BM25 + Word BM25 + ColQwen2**) establishes a new empirical ceiling of **$0.5601$ MRR** across the entire 459-query test suite (Hit@1 = $51.20\%$, Hit@10 = $65.58\%$), beating:
  - Standalone ColQwen2 ($0.4749$ MRR) by **$+0.0852$ MRR** ($p_{\text{perm}} < 0.0001$).
  - Standalone Char BM25 ($0.4372$ MRR) by **$+0.1229$ MRR** ($p_{\text{perm}} < 0.0001$).
  - Heuristic 2-way RRF ($0.5393$ MRR) by **$+0.0208$ MRR** ($p_{\text{perm}} < 0.0001$).
- On **Light Paraphrases**, Tri-Modal Fusion reaches **$0.5722$ MRR** (Hit@1 = $49.02\%$, Hit@10 = $74.51\%$), delivering a **$+0.0535$ MRR lift** over 2-way RRF ($0.5187$, $p < 0.0001$) and **$+0.0778$ lift** over ColQwen2 alone ($0.4944$).

### 3. Modality Specialization across Regimes:
- **Verbatim Lookups:** Every fusion method achieves a perfect **$1.0000$ MRR** on the held-out test split, as lexical and visual top-1 predictions reinforce each other.
- **Answerable Questions:** Tri-Modal Fusion achieves the highest answerable MRR of **$0.1080$** (Hit@10 = $22.22\%$), outperforming standalone ColQwen2 ($0.0578$) and standalone Word BM25 ($0.0297$).

---

## Tabular Summary

### Held-Out Test Evaluation ($N=459$ Queries)
| Fusion Paradigm | Overall MRR | 95% Bootstrap CI | Hit@1 (%) | Hit@10 (%) | Light Paraphrase MRR |
|:---|:---:|:---:|:---:|:---:|:---:|
| Standalone Char 3–5g BM25 | 0.4372 | [0.3953, 0.4819] | 39.65% | 50.98% | 0.2622 |
| Standalone Word BM25 | 0.4636 | [0.4216, 0.5073] | 42.48% | 54.25% | 0.3997 |
| Standalone ColQwen2 | 0.4749 | [0.4340, 0.5172] | 41.39% | 57.73% | 0.4944 |
| Heuristic RRF ($k=60$) | 0.5393 | [0.4976, 0.5819] | 48.80% | 64.05% | 0.5187 |
| Dev-Tuned RRF ($k^*=1$) | 0.5393 | [0.4976, 0.5819] | 48.80% | 64.05% | 0.5187 |
| Normalized Convex ($\alpha^*=0.05$) | 0.5277 | [0.4860, 0.5721] | 48.80% | 60.57% | 0.5096 |
| Supervised Logistic Reranker | 0.5003 | [0.4567, 0.5460] | 48.80% | 52.29% | 0.4496 |
| Pairwise Neural RankNet | 0.5158 | [0.4734, 0.5613] | 48.80% | 55.34% | 0.4910 |
| Query-Gated Adaptive Fusion | 0.5296 | [0.4879, 0.5732] | 48.80% | 59.91% | 0.5073 |
| **Tri-Modal Fusion (Char+Word+ColQwen)** | **0.5601** | **[0.5189, 0.6037]** | **51.20%** | **65.58%** | **0.5722** |

