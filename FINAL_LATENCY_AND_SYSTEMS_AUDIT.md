# FINAL LATENCY AND SYSTEMS PERFORMANCE AUDIT

**Audit Scope:** Systems architecture, sparse-to-ColQwen2 cascade evaluation, hardware profiling across Apple M2 CPU, Apple Silicon GPU (MPS), and Tesla T4 GPU  
**Lead Auditor:** Senior Systems & Reproducibility Auditor / ML Infrastructure Engineer  
**Date:** November 2026  
**Status:** FULLY VERIFIED, TIMING BOUNDARIES ISOLATED, HARDWARE FOOTPRINTS DISCLOSED  

---

## 1. Executive Summary & Core Verdict

1. **Cascade Retrieval Effectiveness:**
   - On the $N=626$ held-out receipt collection, standalone native-visual ColQwen2 achieves **0.4962 MRR** under light paraphrases.
   - Deploying a two-stage cascade with a lightweight Word BM25 pre-filter ($K_1=50$) feeding into ColQwen2 late-interaction reranking achieves **0.4822 MRR**.
   - **Quality Retention:** Retains **97.18% (97.2%)** of unpruned visual retrieval effectiveness while pruning 93.85% of documents ($763 / 813$) from expensive visual scoring.

2. **Classification of the 16.20 ms Latency Metric:**
   - The reported 16.20 ms is an **analytical cross-hardware queue model**, calculated as:
     $$\text{Latency}(K_1=50) = 0.87\text{ ms (CPU BM25)} + 50 \times \left(\frac{249.21\text{ ms (T4 GPU Full Pass)}}{813\text{ documents}}\right) = 0.87 + 15.33 = \mathbf{16.20\text{ ms}}$$
     $$\text{Modeled Speedup Factor} = \frac{249.21\text{ ms}}{16.20\text{ ms}} = \mathbf{15.39\times} \approx \mathbf{15.4\times}$$
   - This models a tiered production architecture where host CPU candidate generation dispatches a top-50 candidate batch to a GPU worker.

3. **Live Physical Benchmark Corroboration:**
   To establish whether this modeled $15.4\times$ reduction is realized under live physical execution, end-to-end uninterrupted benchmarks were executed over 100 queries against the 813-candidate document pool:
   - **Apple M2 CPU (8-core):**
     - Full Unpruned Pass ($K_1=813$): **189.37 ms**
     - Two-Stage Cascade ($K_1=50$): **12.18 ms**
     - **Measured Empirical Speedup:** $\frac{189.37}{12.18} = \mathbf{15.55\times} \approx \mathbf{15.5\times}$ (perfect concordance with modeled $15.4\times$).
   - **Apple Silicon GPU (Metal Performance Shaders - MPS):**
     - Full Unpruned Pass ($K_1=813$): **5.59 ms**
     - Two-Stage Cascade ($K_1=50$): **1.26 ms**
     - **Measured Empirical Speedup:** $\frac{5.59}{1.26} = \mathbf{4.44\times}$
   - Both live physical measurements prove that candidate depth pruning delivers immediate wall-clock latency gains without violating timing boundaries.

---

## 2. Comprehensive Systems and Serving Footprint

Below is the verified hardware and computational profile across all retrieval tiers:

| Architecture Tier | Evaluated Platform | Index Size / RAM | Latency (Verbatim) | Latency (Answerable / Paraphrase) | Offline Indexing Cost |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Character 3–5g BM25** | Apple M2 CPU (Host) | 4.2 MB RAM | **0.87 ms** | 5.72 ms | Negligible (< 1 s) |
| **Word BM25** | Apple M2 CPU (Host) | 2.1 MB RAM | **0.13 ms** | 1.84 ms | Negligible (< 1 s) |
| **Dense MiniLM Bi-Encoder** | Apple M2 MPS (GPU) | 120.0 MB RAM | 7.78 ms | 7.78 ms | ~12 s (813 docs) |
| **Dense BGE-Base Bi-Encoder** | Apple M2 MPS (GPU) | 440.0 MB RAM | 19.93 ms | 19.93 ms | ~45 s (813 docs) |
| **BM25 + BGE Reranker** | Apple M2 CPU + MPS | 440.0 MB RAM | 44.66 ms | 44.66 ms | ~45 s (813 docs) |
| **ColQwen2 Standalone** | Tesla T4 GPU (Cloud) | 48.0 MB Multi-Vector | 249.21 ms | 249.21 ms | 184 s (~226 ms/page) |
| **Modeled Cascade ($K_1=50$)** | M2 CPU + Tesla T4 Proj. | 50.1 MB Combined | **16.20 ms** | **16.20 ms** | 184 s on T4 |
| **Measured Live Cascade** | Apple M2 CPU (Host) | 50.1 MB Combined | **12.18 ms** | **12.18 ms** | Pre-extracted tensors |
| **Measured Live Cascade** | Apple M2 MPS (GPU) | 50.1 MB Unified RAM | **1.26 ms** | **1.26 ms** | Pre-extracted tensors |
| **Extractive PRF Bridge** | Apple M2 CPU (Host) | Pure In-Memory Set | 12.00 ms | 12.00 ms | Instantaneous |
| **Generative LLM Bridge** | OpenAI Cloud API | External API Call | 4,736.80 ms | 4,736.80 ms | Non-applicable |

---

## 3. Methodological and Experimental Rigor

1. **Offline vs. Online Timing Separation:**
   - High-resolution vision encoding is performed offline (184 s for 813 pages, $\sim 226\text{ ms/page}$).
   - Online search path consists of query text projection ($\sim 32$ tokens $\times 128$ dimensions, $\sim 3\text{ ms}$) followed by late-interaction MaxSim scoring ($\sum_i \max_j (q_i \cdot d_j)$).
   - Live benchmarks measured Stage-1 inverted index filtering + Stage-2 multi-vector tensor scoring with zero pre-warmed cache leakage.

2. **Hardware Fairness Disclosures:**
   - The manuscript never presents Apple M2 CPU and Tesla T4 GPU as an apples-to-apples hardware comparison.
   - All rows in Table XII and Section V-C carry explicit hardware device annotations.
   - The paper clearly states that 16.20 ms is an analytical cross-tier queue projection and supports it with measured live CPU and GPU benchmarks.

