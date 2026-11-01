# T2-04: Fusion and Cascades with Native Visual Model (ColQwen2)

## Overview
This study evaluates multimodal fusion and cascade architectures combining fast CPU sparse retrieval (Word BM25, Character BM25) with heavy GPU document-native visual retrieval (ColQwen2-v1.0) on $N=306$ held-out probes.

- **Git Commit:** `6b9f2aaa8a873049f7d95fd831e22f868cd5e636`
- **Candidate Pool:** 813 test items (receipts + images)
- **Primary Regimes Evaluated:** Light Paraphrase & Answerable Controls

## Headline Findings

| Architecture | MRR | Hit@1 | Hit@10 | Diff vs. ColQwen2 | Paired $p$ (vs ColQwen2) | Serving Latency (ms) | Speedup vs. Full ColQwen2 |
|---|---|---|---|---|---|---|---|
| **Word BM25 (Standalone CPU)** | 0.4003 | 0.3235 | 0.5817 | $-0.0960$ | $p = 0.0001$ | **0.87 ms** | **286.4x** |
| **ColQwen2-v1.0 (Full GPU)** | 0.4962 | 0.4052 | 0.6569 | — | — | 249.21 ms | 1.0x |
| **Cascade: Word BM25 ($K=20$) $\to$ ColQwen2** | 0.4513 | 0.3954 | 0.5621 | $-0.0449$ | $p = 0.0210$ | **7.00 ms** | **35.6x** |
| **Cascade: Word BM25 ($K=50$) $\to$ ColQwen2** | **0.4822** | **0.4020** | **0.6373** | **$-0.0140$** | **$p = 0.1620$ (n.s.)** | **16.19 ms** | **15.4x** |
| **Cascade: Word BM25 ($K=100$) $\to$ ColQwen2** | **0.4907** | **0.4052** | **0.6503** | **$-0.0055$** | **$p = 0.4812$ (n.s.)** | **31.52 ms** | **7.9x** |
| **Cascade: Char BM25 ($K=50$) $\to$ ColQwen2** | 0.3445 | 0.2810 | 0.4510 | $-0.1517$ | $p < 10^{-5}$ | 16.19 ms | 15.4x |
| **RRF Fusion (Word BM25 + ColQwen2)** | 0.5312 | 0.4346 | 0.7255 | **$+0.0350$** | **$p = 0.0042$** | 250.08 ms | 1.0x |

### 1. Optimal Pareto Trade-Off: Word BM25 ($K=50$) Cascade
- Evaluating full ColQwen2 over the entire 813-item collection takes **249.21 ms** on a Tesla T4 GPU.
- A two-stage cascade using **Word BM25** to select the top $K=50$ candidates and ColQwen2 to rerank them achieves **0.4822 MRR** (recovering **97.2%** of full ColQwen2 capacity, with no statistically significant difference, diff $-0.0140$, $p = 0.1620$).
- Serving latency drops from 249.21 ms to **16.19 ms**—a **15.4x latency reduction**.

### 2. High-Accuracy Frontier: RRF Fusion Beats Both Components
- Combining Word BM25 and ColQwen2 via Reciprocal Rank Fusion ($k=60$) achieves **0.5312 MRR** (+0.0350 lift over full ColQwen2 alone, $p = 0.0042$, and +0.1309 over Word BM25 alone, $p < 10^{-5}$).
- RRF proves that lexical and native visual features capture complementary relevance signals under paraphrasing.
