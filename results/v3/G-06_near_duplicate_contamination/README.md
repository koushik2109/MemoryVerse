# G-06: Real-World Retrieval Pool Contamination and Near-Duplicate Distractor Collisions

## Overview
This directory contains experimental results for **G-06**, systematically addressing the real-world challenge
of retrieval pool contamination by near-duplicate documents (same-merchant template duplicates and camera multi-shot photo bursts).

- **Base Corpus:** $N=813$ standard benchmark documents (313 SROIE receipts + 500 COCO distractors)
- **Near-Duplicate Pool:** $N=325$ authentic distractors (79 real-world same-vendor receipts + 246 burst multi-shot duplicates, reaching $N=1{,}138$ at $\rho=40\%$)
- **Contamination Rates Swept:** $\rho \in [0\%, 5\%, 10\%, 20\%, 40\%]$ (corpus size scaling from $813$ to $1{,}138$ documents)
- **Evaluation Probes:** $N=306$ held-out probes across Fine-Grained Transaction Queries, Coarse Vendor Queries, and Natural Questions ($N=918$ queries per condition, $N=9{,}180$ total query evaluations across 6 paradigms = $55{,}080$ evaluated ranking decisions)
- **Degradation Regimes:** Apple Clean Real-OCR vs. Apple Degraded Optical Blur (CER $\approx 20\%$)
- **Statistical Rigor:** Paired permutation tests ($B=10{,}000$) with Holm-Bonferroni correction and $B=2{,}000$ bootstrap confidence intervals.

---

## Benchmark Results

### 1. Specific Transaction Queries (Apple Degraded Optical Blur)

| System Paradigm | Base MRR ($\rho=0\%$) | Contaminated MRR ($\rho=40\%$) | MRR Retention | Hit@1 ($\rho=40\%$) | Near-Dup Collision Rate | Serving Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Char 3–5g BM25** | **0.8697** | **0.7003** [0.6630, 0.7375] | **80.5%** | **51.96%** | **4.25%** | **0.96 ms** |
| **Tuned Hybrid** | 0.8691 | 0.6835 [0.6441, 0.7206] | 78.6% | 48.37% | 4.25% | 9.36 ms |
| **Word BM25** | 0.8162 | 0.6296 [0.5796, 0.6775] | 77.1% | 42.81% | 4.90% | 0.10 ms |
| **Two-Stage Cascade ($K_1=20$)** | 0.8371 | 0.5153 [0.4746, 0.5557] | 61.6% | 17.32% | 76.47% | 19.36 ms |
| **Dense BGE-base** | 0.4391 | 0.3073 [0.2608, 0.3541] | 70.0% | 15.36% | 22.22% | 8.40 ms |
| **Late-Interaction (ColBERTv2)** | 0.4112 | 0.2906 [0.2452, 0.3370] | 70.7% | 16.01% | 25.16% | 28.50 ms |

---

### 2. Distractor Collision Rates across Query Specificity Regimes ($\rho=40\%$ Degraded)

| Query Specificity Regime | Char 3–5g BM25 Collision | Word BM25 Collision | Dense BGE-base Collision | ColBERTv2 Collision | Cascade ($K_1=20$) Collision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Specific Transaction** (*Vendor + Total + Date*) | **4.25%** | 4.90% | 22.22% | 25.16% | 76.47% |
| **Natural Question** (*Total at Vendor?*) | **15.69%** | 17.32% | 22.88% | 26.14% | 75.82% |
| **Coarse Vendor** (*Vendor Only*) | **22.55%** | 20.92% | 20.59% | 24.51% | 74.18% |

---

### 3. Mitigation Strategies at $\rho=40\%$ Contamination (Apple Degraded Optical Blur)

| Mitigation Strategy | Realized MRR [95% CI] | Hit@1 | Net $\Delta$MRR vs. Raw | Paired Permutation $p$-val | Serving Latency | Index Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Char BM25 (Unmitigated)** | 0.7003 [0.6630, 0.7375] | 51.96% | Baseline | — | 0.96 ms | 48.2 MB |
| **Ingestion MinHash LSH Dedup ($\tau=0.85$)** | **0.8610** [0.8250, 0.8932] | **81.70%** | **$+0.1607$** | **$p < 0.0001$** | **0.82 ms** | **38.6 MB** |
| **Transaction Anchor Weighting (TAW 2.5$\times$)** | 0.6997 [0.6642, 0.7348] | 49.35% | $-0.0006$ | $p = 0.9739$ (n.s.) | 1.02 ms | 48.2 MB |
| **Post-Retrieval MMR Diversity ($\lambda=0.7$)** | 0.6623 [0.6206, 0.7052] | 51.96% | $-0.0380$ | $p < 0.0001$ | 1.84 ms | 48.2 MB |

---

## Headline Architectural Takeaways

1. **The Distractor Collision Mechanism: Dense Bi-Encoders Collapse under Template Redundancy:**
   When the index contains multiple documents from the same vendor chain or recurring template, dense bi-encoders (BGE-base) suffer severe distractor collisions ($22.22\%$ on specific transaction queries). Single-vector pooling blurs fine-grained numeric differences (dates, prices, invoice IDs) into a general vendor semantic centroid. In contrast, Character 3–5g BM25 maintains fine-grained discrimination over unique numeric tokens and date subwords, keeping collision rate down to **$4.25\%$** ($5.2\times$ lower than Dense BGE).

2. **Cross-Encoder Cascades are Highly Vulnerable to Burst Duplicates:**
   Reranking top-20 candidates with a cross-encoder (`all-MiniLM-L6-v2`) causes a catastrophic **$76.47\%$ collision rate** under $40\%$ contamination. Cross-encoders trained on web passages mistake near-identical burst duplicate photos for the canonical target document, shuffling the correct target out of Rank 1.

3. **Ingestion-Time MinHash LSH Deduplication Delivers Massive Lift:**
   Collapsing burst multi-shot duplicates at ingestion time ($\tau=0.85$) eliminates 248 redundant distractor documents, boosting Char BM25 MRR from $0.7003$ to **$0.8610$** ($+0.1607$ lift, $p < 0.0001$) and restoring Hit@1 from $51.96\%$ to **$81.70\%$**, while reducing index memory by $19.9\%$ ($38.6$ MB vs. $48.2$ MB) and query serving latency to **$0.82$ ms**.
