# G-05: Cross-Encoder Cascades, Learned-Sparse (SPLADE), and Late-Interaction (ColBERTv2) under OCR Noise

## Overview
This directory contains experimental results for **G-05**, systematically investigating:
1. **Candidate Depth Sensitivity ($K_1$-Sweep):** Sweeping first-stage candidate depth $K_1 \in [5, 10, 15, 20, 30, 50, 100]$ for two-stage cross-encoder cascades over Word BM25 and Char 3–5g BM25 first-stage candidate pools.
2. **Modern Retrieval Paradigm Benchmarking:** Evaluating whether learned-sparse term expansion (**SPLADE**) or token-level contextual late interaction (**ColBERTv2**) can overcome optical OCR noise without heavy cross-encoders, benchmarked head-to-head against subword **Char 3–5g BM25**, **Word BM25**, and dense **BGE-base**.

- **Corpus:** Primary 813-distractor pool (313 SROIE receipts + 500 COCO distractors)
- **Evaluation Probes:** $N=306$ held-out probes across Verbatim Lookups, Light Paraphrases, and Answerable Questions ($N=918$ queries per condition, $N=1,836$ total queries evaluated across 7 retrieval paradigms = $12,852$ ranking decisions)
- **Degradation Regimes:** Apple Clean Real-OCR vs. Apple Degraded Optical Blur (CER $\approx 20\%$)
- **Statistical Rigor:** Paired permutation tests ($B=10,000$) with Holm-Bonferroni FWER control and $B=2,000$ bootstrap confidence intervals.

---

## Benchmark Results

### 1. Paradigm Comparison (Verbatim Lookups)

| System Paradigm | Clean MRR [95% CI] | Blur MRR [95% CI] | Blur $\Delta$ vs. Char | Serving Latency (ms) | Index Memory (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Char 3–5g BM25** | **0.9437** [0.9188, 0.9645] | **0.8271** [0.7846, 0.8651] | Baseline | **0.96 ms** | **48.2 MB** |
| **Two-Stage Char + Cross-Enc ($K_1=20$)** | 0.9350 [0.9086, 0.9581] | 0.7844 [0.7403, 0.8243] | $-0.0427$ ($p < 0.001$) | 19.36 ms | 328.0 MB |
| **Word BM25** | 0.8386 [0.8003, 0.8738] | 0.6950 [0.6473, 0.7388] | $-0.1320$ ($p < 0.001$) | 0.10 ms | 12.4 MB |
| **Two-Stage Word + Cross-Enc ($K_1=20$)** | 0.7823 [0.7415, 0.8198] | 0.6561 [0.6062, 0.7023] | $-0.1710$ ($p < 0.001$) | 18.50 ms | 292.0 MB |
| **Late-Interaction (ColBERTv2)** | 0.3896 [0.3383, 0.4426] | 0.3145 [0.2656, 0.3634] | $-0.5126$ ($p < 0.0001$) | 28.50 ms | 185.0 MB |
| **Dense BGE-base** | 0.3300 [0.2827, 0.3789] | 0.2796 [0.2344, 0.3247] | $-0.5475$ ($p < 0.0001$) | 8.40 ms | 420.0 MB |
| **Learned-Sparse (SPLADE)** | 0.0774 [0.0535, 0.1031] | 0.0597 [0.0408, 0.0808] | $-0.7674$ ($p < 0.0001$) | 3.80 ms | 98.0 MB |

---

### 2. Candidate Depth Sensitivity ($K_1$-Sweep)

#### Verbatim Lookups (Apple Degraded Optical Blur)
| First-Stage $K_1$ | Word Hit@$K_1$ | Word Stage 1 MRR | Word Reranked MRR | Word Net Lift | Char Hit@$K_1$ | Char Stage 1 MRR | Char Reranked MRR | Char Net Lift | Rerank Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $K_1=5$ | 0.7745 | 0.6952 | 0.6695 | $-0.0257$ | 0.8529 | 0.8271 | 0.7984 | $-0.0288$ | 4.7 ms |
| $K_1=10$ | 0.8301 | 0.6952 | 0.6618 | $-0.0334$ | 0.8922 | 0.8271 | 0.7878 | $-0.0393$ | 9.3 ms |
| $K_1=15$ | 0.8464 | 0.6952 | 0.6572 | $-0.0380$ | 0.8922 | 0.8271 | 0.7832 | $-0.0439$ | 13.9 ms |
| $K_1=20$ | 0.8660 | 0.6952 | 0.6561 | $-0.0391$ | 0.8987 | 0.8271 | 0.7844 | $-0.0427$ | 18.5 ms |
| $K_1=30$ | 0.8889 | 0.6952 | 0.6528 | $-0.0424$ | 0.9052 | 0.8271 | 0.7853 | $-0.0419$ | 27.7 ms |
| $K_1=50$ | 0.8954 | 0.6952 | 0.6492 | $-0.0460$ | 0.9085 | 0.8271 | 0.7846 | $-0.0425$ | 46.1 ms |
| $K_1=100$ | 0.8987 | 0.6952 | 0.6472 | $-0.0480$ | 0.9281 | 0.8271 | 0.7844 | $-0.0427$ | 92.1 ms |

#### Light Paraphrases (Apple Degraded Optical Blur)
| First-Stage $K_1$ | Word Hit@$K_1$ | Word Reranked MRR | Word Net Lift | Char Hit@$K_1$ | Char Stage 1 MRR | Char Reranked MRR | Char Net Lift |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $K_1=5$ | 0.3987 | 0.3416 | $-0.0082$ | 0.2908 | 0.2146 | 0.2408 | $+0.0262$ |
| $K_1=10$ | 0.4967 | 0.3511 | $+0.0014$ | 0.3399 | 0.2146 | 0.2432 | $+0.0286$ |
| $K_1=20$ | 0.5882 | 0.3515 | $+0.0018$ | 0.4085 | 0.2146 | 0.2509 | **$+0.0363$** |
| $K_1=50$ | 0.7320 | 0.3514 | $+0.0017$ | 0.4935 | 0.2146 | 0.2522 | $+0.0376$ |
| $K_1=100$ | 0.8235 | 0.3528 | $+0.0030$ | 0.6569 | 0.2146 | 0.2564 | $+0.0417$ |

---

## Headline Architectural Takeaways

1. **Failure of Learned-Sparse Expansion (SPLADE) on Degraded Receipts:**
   SPLADE achieves only $0.0774$ MRR on clean receipts and drops to $0.0597$ MRR under optical blur (trailing Char 3–5g BM25 by $-0.7674$ MRR, $p < 0.0001$). Because SPLADE relies on BERT WordPiece subword tokens to expand related lexical concepts across a 30,522-dimensional vocabulary, optical noise fragments character strings into invalid subwords. The model cannot fire meaningful term expansions, causing extreme sparsity and retrieval collapse.

2. **Late-Interaction (ColBERTv2) Exceeds Dense Bi-Encoders but Trails Character Subwords:**
   ColBERTv2 achieves $0.3896$ clean and $0.3145$ degraded MRR on verbatim receipts, significantly outperforming dense BGE-base ($0.3300$ clean, $0.2796$ blur, $p < 0.01$). ColBERT's token-level MaxSim interaction retains alignment between intact character sequences that single-vector pooling blurs away. However, it still trails Character 3–5g BM25 by $-0.5126$ MRR ($p < 0.0001$) due to contextual token embedding drift under OCR corruption, while incurring 29.7$\times$ higher latency ($28.5$ ms vs. $0.96$ ms).

3. **The Cross-Encoder Cascade Penalty on Verbatim Entity Lookups:**
   Reranking top-$K_1$ candidates with a cross-encoder (MiniLM-L6) does *not* improve verbatim receipt retrieval at any depth $K_1 \in [5..100]$. On Char BM25, cross-encoding imposes a persistent penalty (Clean: $-0.0087$; Blur: $-0.0427$ at $K_1=20$, $p < 0.001$). Because standard cross-encoders are pre-trained on fluent web passages (MS MARCO), they systematically penalize short, semi-structured receipt text in favor of natural prose distractors. Cross-encoding only yields positive lift on semantic paraphrases ($+0.0363$ lift at $K_1=20$, $p = 0.02$).

4. **Strict Pareto Dominance of Subword Lexical Indexing:**
   Character 3–5g BM25 achieves strictly higher retrieval accuracy than all neural alternatives (Dense, SPLADE, ColBERTv2, and Two-Stage Cascades) on structured documents under OCR noise, while running entirely on CPU in $< 1.0$ ms with negligible memory ($48.2$ MB).
