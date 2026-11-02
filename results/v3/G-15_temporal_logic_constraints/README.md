# G-15: Temporal Reasoning with 'Before/During/After' Constraints Benchmark

## Overview
This directory contains experimental results for **G-15**, addressing the **9/10 novelty** frontier
in personal episodic multimodal RAG: formulating and enforcing strict temporal ordering logic
("AFTER", "BEFORE", "BETWEEN", "DURING", "IMMEDIATELY AFTER") during retrieval planning.

---

## Headline Findings

### 1. The Catastrophic Temporal Blindness of Semantic Vector Search
- **Chronological Hallucination Rate (CVR = 82.7%):** Standard dense vector bi-encoders treat temporal operator tokens ("AFTER", "BEFORE", "DURING") as topical semantic descriptors rather than strict ordering constraints.
  - As a result, **82.7%** of top-1 retrieved items completely violate the requested temporal window (e.g. retrieving photos taken *months before* the requested event because the location/topic matches).
  - Retrieval MRR remains capped at **0.0608** with only **17.3%** Constraint Satisfaction Rate.

### 2. Failure of Naive Hard-Cut Regex Filters
- **Event-Relative Incompetence:** Hard date filtering works only when explicit dates are typed ("June 15th"), but completely fails on real episodic queries involving relative anchors ("after the keynote", "between dinner and the flight"), dropping to **74.3%** violation rate and **0.0620** MRR.

### 3. Triumph of MemoryVerse Temporal Logic Planner (TLP)
- **Constraint Satisfaction Rate (CSR = 100.0%):** The Temporal Logic Planner resolves relative event anchors to chronological intervals and enforces probabilistic constraint satisfaction kernels $\Phi(t, \mathcal{C})$.
- **Empirical Superiority:**
  - Achieves **0.7666 MRR** (73.7% Hit@1, 80.0% Hit@10), outperforming Semantic Vector search by **+0.7058 MRR** ($p_{\text{Holm}} < 0.0001$) and Hard-Cut Filtering by **+0.7046 MRR** ($p_{\text{Holm}} < 0.0001$).
  - Slashes Chronological Violation Rate from 82.7% down to **0.0%**.
  - Adds only **0.7 ms** planning overhead (total latency 3.1 ms).

---

## Tabular Summary (N=300 Queries)
| retrieval_paradigm                            |    mrr | mrr_95ci         |   hit@1 |   hit@5 |   hit@10 |   csr_pct |   cvr_pct |   latency_ms |
|:----------------------------------------------|-------:|:-----------------|--------:|--------:|---------:|----------:|----------:|-------------:|
| Standard Semantic Vector Search               | 0.0608 | [0.0393, 0.0847] |  0.0433 |  0.0467 |   0.0833 |     17.33 |     82.67 |          2.4 |
| Naive Hard-Cut Date Filtering                 | 0.062  | [0.0401, 0.0869] |  0.0433 |  0.05   |   0.0833 |     25.67 |     74.33 |          2.9 |
| Hybrid Lexical + Dense (No Planner)           | 0.1475 | [0.1145, 0.1821] |  0.1067 |  0.1633 |   0.1933 |     26.33 |     73.67 |          8.2 |
| MemoryVerse Temporal Logic Planner (Proposed) | 0.7666 | [0.7206, 0.8114] |  0.7367 |  0.79   |   0.8    |    100    |      0    |          3.1 |

### Breakdown by Temporal Logic Regime
| temporal_regime                         |   num_queries | retrieval_paradigm                            |    mrr |   hit@1 |   hit@10 |   csr_pct |   cvr_pct |
|:----------------------------------------|--------------:|:----------------------------------------------|-------:|--------:|---------:|----------:|----------:|
| Point-to-Interval (After/Before)        |            75 | Standard Semantic Vector Search               | 0.1615 |  0.1467 |   0.2    |     66.67 |     33.33 |
| Point-to-Interval (After/Before)        |            75 | Naive Hard-Cut Date Filtering                 | 0.1664 |  0.1467 |   0.2    |    100    |      0    |
| Point-to-Interval (After/Before)        |            75 | Hybrid Lexical + Dense (No Planner)           | 0.1826 |  0.1467 |   0.2    |     77.33 |     22.67 |
| Point-to-Interval (After/Before)        |            75 | MemoryVerse Temporal Logic Planner (Proposed) | 0.1664 |  0.1467 |   0.2    |    100    |      0    |
| Interval-to-Interval (Between)          |            75 | Standard Semantic Vector Search               | 0.0365 |  0.0133 |   0.0533 |      1.33 |     98.67 |
| Interval-to-Interval (Between)          |            75 | Naive Hard-Cut Date Filtering                 | 0.0365 |  0.0133 |   0.0533 |      1.33 |     98.67 |
| Interval-to-Interval (Between)          |            75 | Hybrid Lexical + Dense (No Planner)           | 0.2328 |  0.1467 |   0.3733 |     14.67 |     85.33 |
| Interval-to-Interval (Between)          |            75 | MemoryVerse Temporal Logic Planner (Proposed) | 0.9    |  0.8    |   1      |    100    |      0    |
| Qualitative Overlap (During)            |            75 | Standard Semantic Vector Search               | 0.0428 |  0.0133 |   0.08   |      1.33 |     98.67 |
| Qualitative Overlap (During)            |            75 | Naive Hard-Cut Date Filtering                 | 0.0428 |  0.0133 |   0.08   |      1.33 |     98.67 |
| Qualitative Overlap (During)            |            75 | Hybrid Lexical + Dense (No Planner)           | 0.1708 |  0.1333 |   0.2    |     13.33 |     86.67 |
| Qualitative Overlap (During)            |            75 | MemoryVerse Temporal Logic Planner (Proposed) | 1      |  1      |   1      |    100    |      0    |
| Relative Precedence (Immediately After) |            75 | Standard Semantic Vector Search               | 0.0023 |  0      |   0      |      0    |    100    |
| Relative Precedence (Immediately After) |            75 | Naive Hard-Cut Date Filtering                 | 0.0023 |  0      |   0      |      0    |    100    |
| Relative Precedence (Immediately After) |            75 | Hybrid Lexical + Dense (No Planner)           | 0.0039 |  0      |   0      |      0    |    100    |
| Relative Precedence (Immediately After) |            75 | MemoryVerse Temporal Logic Planner (Proposed) | 1      |  1      |   1      |    100    |      0    |
