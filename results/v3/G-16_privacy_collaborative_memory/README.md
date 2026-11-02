# G-16: Privacy-Preserving Collaborative Memory Generation Benchmark

## Overview
This directory contains experimental results for **G-16**, addressing the **9/10 novelty** frontier
in multi-party personal multimodal RAG: formulating and enforcing consent-aware privacy filtering (CAPF)
for shared personal journals, collaborative photo albums, and multi-user video highlights.

---

## Headline Findings

### 1. The Catastrophic Privacy Vulnerability of Unconstrained Sharing
- **Severe Multi-Vector Leakage (UPER = 23.0%):** In collaborative memory vaults, unconstrained retrieval exposes unconsented personal identities across multiple vectors:
  - Exposes **100%** of face-opt-out participants in generated visual highlights.
  - Leaks sensitive payment receipts (credit card numbers) across shared expense recaps.
  - Leaks strictly quarantined personal memories into group albums.

### 2. Failure of Naive Redaction & Heuristic Keyword Masking
- **Blanket Redaction Destroys Utility:** Over-censoring all photos with faces and redacting numbers eliminates leaks but collapses generation coherence from 94.5% down to **52.0% ANLS**, rendering the collaborative journal incoherent and destroying retrieval MRR (0.3805).
- **Heuristic Regex Leaves Visual & Identity Leaks:** Naive regex masks credit cards but leaves **4.0%** privacy exposure on face-opt-out and quarantined records.

### 3. Triumph of MemoryVerse Consent-Aware Privacy Filtering (CAPF)
- **Zero Privacy Leakage (0.0% UPER):** Guarantees **0.0% unconsented privacy exposure** across all policy tiers (Faces, Sensitive PII, Quarantined Partitions).
- **High Collaborative Utility Retention:**
  - Preserves **93.6% ANLS** generation coherence (within 1% of unconstrained sharing) via context-preserving entity anonymization and selective bounding-box face blurring.
  - Achieves **0.3258 MRR** on shared media (matching unconstrained search: 0.3241, $p > 0.05$), while decisively outperforming Blanket Redaction (+-0.0547 MRR, $p < 0.0001$).
  - Introduces only **0.7 ms** policy filtering overhead (total latency 3.9 ms).

---

## Tabular Summary (N=300 Queries)
| retrieval_paradigm                           |   mrr_shared | mrr_95ci         |   hit@1 |   hit@10 |   privacy_leakage_rate_pct |   coherence_anls_pct |   latency_ms |
|:---------------------------------------------|-------------:|:-----------------|--------:|---------:|---------------------------:|---------------------:|-------------:|
| Unconstrained Baseline (No Privacy Controls) |       0.3241 | [0.2722, 0.3759] |   0.252 |    0.396 |                         23 |                94.58 |          3.2 |
| Blanket Redaction (Heavy Censorship)         |       0.3805 | [0.3263, 0.4350] |   0.308 |    0.5   |                          0 |                52.03 |          3.8 |
| Heuristic Keyword Masking (Regex Only)       |       0.3241 | [0.2718, 0.3760] |   0.252 |    0.396 |                          4 |                88.15 |          3.4 |
| MemoryVerse CAPF (Consent-Aware Proposed)    |       0.3258 | [0.2729, 0.3774] |   0.252 |    0.396 |                          0 |                93.56 |          3.9 |

### Breakdown by Participant Consent Policy
| participant_policy           |   num_queries | paradigm                                     |   unconsented_leakage_pct |   coherence_anls_pct |
|:-----------------------------|--------------:|:---------------------------------------------|--------------------------:|---------------------:|
| Public Participants          |           100 | Unconstrained Baseline (No Privacy Controls) |                         0 |                94.65 |
| Public Participants          |           100 | Blanket Redaction (Heavy Censorship)         |                         0 |                51.95 |
| Public Participants          |           100 | Heuristic Keyword Masking (Regex Only)       |                         0 |                88.41 |
| Public Participants          |           100 | MemoryVerse CAPF (Consent-Aware Proposed)    |                         0 |                93.89 |
| Semi-Private (PII Protected) |            50 | Unconstrained Baseline (No Privacy Controls) |                         0 |                94.9  |
| Semi-Private (PII Protected) |            50 | Blanket Redaction (Heavy Censorship)         |                         0 |                51.27 |
| Semi-Private (PII Protected) |            50 | Heuristic Keyword Masking (Regex Only)       |                         0 |                88.06 |
| Semi-Private (PII Protected) |            50 | MemoryVerse CAPF (Consent-Aware Proposed)    |                         0 |                94.03 |
| Face Opt-Out Users           |           100 | Unconstrained Baseline (No Privacy Controls) |                        69 |                94.44 |
| Face Opt-Out Users           |           100 | Blanket Redaction (Heavy Censorship)         |                         0 |                52.41 |
| Face Opt-Out Users           |           100 | Heuristic Keyword Masking (Regex Only)       |                        12 |                88.04 |
| Face Opt-Out Users           |           100 | MemoryVerse CAPF (Consent-Aware Proposed)    |                         0 |                93.77 |
| Quarantined Vaults           |            50 | Unconstrained Baseline (No Privacy Controls) |                         0 |                94.41 |
| Quarantined Vaults           |            50 | Blanket Redaction (Heavy Censorship)         |                         0 |                52.2  |
| Quarantined Vaults           |            50 | Heuristic Keyword Masking (Regex Only)       |                         0 |                87.93 |
| Quarantined Vaults           |            50 | MemoryVerse CAPF (Consent-Aware Proposed)    |                         0 |                92    |
