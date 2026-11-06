# Forensic Audit Report: Latency Evidence Classification, Hardware Profiling, and Cascade Model Verification

**Repository:** MemoryVerse — IJCDS Manuscript Evidence Package  
**Audit Scope:** Section V-C (Computational Resource and Latency Profile), Table XII (`tab:system_costs`), `scripts/gapfill_t2_04_fusion_cascade.py`, and `FINAL_BLOCKER_RESOLUTION/GPU_LIVE_EXECUTION_REPORT.md`  
**Deliverable File:** `LATENCY_EVIDENCE_CLASSIFICATION.md`  
**Audit Date:** November 2026  
**Status:** COMPLETED, CLASSIFIED, AND SCIENTIFICALLY RECONCILED  

---

## 1. Executive Summary and Taxonomy of Latency Evidence

A critical source of potential reviewer challenge in systems papers is ambiguity between **directly measured wall-clock runtimes** and **analytically modeled projections**, particularly when pipelines span heterogeneous hardware platforms (e.g., Apple Silicon ARM64 CPU/MPS vs. Cloud NVIDIA Tesla T4 GPUs).

This audit establishes a definitive, verified taxonomy across all latency figures cited in the manuscript:

```
+-----------------------------------------------------------------------------------------------------------------+
|                                        RETRIEVAL LATENCY EVIDENCE CLASSIFICATION                                |
+-------------------------------+-------------------+----------------------------+--------------------------------+
| Architecture Tier             | Latency / Metric  | Measurement Nature         | Hardware Profile               |
+-------------------------------+-------------------+----------------------------+--------------------------------+
| Standalone ColQwen2 (Full)    | 249.21 ms         | Empirically Measured       | NVIDIA Tesla T4 GPU (Cloud)    |
| Modeled Hybrid Cascade (K=50) | 16.20 ms (15.4x)  | Analytically Modeled       | M2 CPU + Tesla T4 Proj.        |
| Live CPU Cascade (K=50)       | 12.18 ms (15.5x)  | Empirically Measured       | Apple M2 Host CPU (8-core)     |
| Live GPU Cascade (K=50)       | 1.26 ms (4.4x)    | Empirically Measured       | Apple Silicon GPU (MPS)        |
| BM25 Verbatim Lookup          | 0.13 ms – 0.87 ms | Empirically Measured       | Apple M2 CPU / Xeon Host       |
| Dense MiniLM Bi-Encoder       | 7.78 ms           | Empirically Measured       | Apple M2 MPS                   |
| Dense BGE-Base Bi-Encoder     | 19.93 ms          | Empirically Measured       | Apple M2 MPS                   |
| BM25 + BGE Reranker           | 44.66 ms          | Empirically Measured       | Apple M2 CPU + MPS             |
| Extractive PRF (Multi-hop)    | 12.00 ms          | Empirically Measured       | Apple M2 CPU                   |
| Generative LLM Bridging       | 4,736.80 ms       | Empirically Measured       | OpenAI Cloud API (gpt-4o-mini) |
+-------------------------------+-------------------+----------------------------+--------------------------------+
```

---

## 2. Deconstruction of the 16.20 ms Cascade Figure

### A. Mathematical Provenance
The 16.20 ms figure cited in the manuscript text (Abstract line 26, Section V-C line 551) originates from `scripts/gapfill_t2_04_fusion_cascade.py` (lines 44–62):

$$\text{Latency}(K_1=50) = \text{Latency}_{\text{sparse}} + K_1 \times \left(\frac{\text{Latency}_{\text{ColQwen2}}}{N_{\text{corpus}}}\right)$$
$$\text{Latency}(50) = 0.87\text{ ms} + 50 \times \left(\frac{249.21\text{ ms}}{813}\right) = 0.87 + 15.3265 = \mathbf{16.20\text{ ms}}$$
$$\text{Speedup Factor} = \frac{249.21\text{ ms}}{16.1965\text{ ms}} = \mathbf{15.39\times} \approx \mathbf{15.4\times}$$

### B. Analytical Assumptions & Boundaries
1. **Model Nature:** This is an **analytical queue/cost projection**, not an uninterrupted physical timer log on a unified motherboard.
2. **Cross-Hardware Composition:** It models a production tier where a low-cost host CPU executes sub-millisecond sparse candidate pre-filtering ($0.87\text{ ms}$), dispatching the top 50 candidates to a GPU serving worker running ColQwen2 scoring ($15.33\text{ ms}$).
3. **Linearity Assumption:** It assumes late-interaction scoring scales linearly with candidate depth ($50 / 813 = 6.15\%$ of compute).

---

## 3. Live Physical Benchmark Verification (Apple M2 CPU & MPS GPU)

To confirm whether this modeled $15.4\times$ latency reduction holds under real uninterrupted physical execution, live end-to-end benchmarks were executed across 100 test queries over the full 813-candidate document pool (`FINAL_BLOCKER_RESOLUTION/GPU_LIVE_EXECUTION_REPORT.md`):

### A. Live Host CPU Execution (Apple M2)
- **Unpruned Full Pass ($K_1=813$):** 189.37 ms (BM25: 0.17 ms, MaxSim: 189.21 ms).
- **Two-Stage Cascade ($K_1=50$):** **12.18 ms** (BM25: 0.15 ms, MaxSim: 12.02 ms).
- **Measured Speedup:** $\frac{189.37}{12.18} = \mathbf{15.55\times}$.
- **Finding:** The live CPU benchmark directly validates the analytical model: reducing candidate depth from 813 to 50 delivers an empirical **$15.5\times$ wall-clock speedup** ($189.37\text{ ms} \to 12.18\text{ ms}$), in near-perfect agreement with the modeled $15.4\times$ factor.

### B. Live Hardware-Accelerated Execution (Apple Silicon Metal MPS)
- **Unpruned Full Pass ($K_1=813$):** 5.59 ms (BM25: 0.23 ms, MaxSim: 5.36 ms).
- **Two-Stage Cascade ($K_1=50$):** **1.26 ms** (BM25: 0.11 ms, MaxSim: 1.15 ms).
- **Measured Speedup:** $\frac{5.59}{1.26} = \mathbf{4.44\times}$.
- **Finding:** On integrated Apple Silicon GPU memory where tensor transfer overhead is zero (unified RAM architecture), sub-2 millisecond late interaction is physically achievable at $K_1=50$.

---

## 4. Full Model Inference vs. Tensor Operations Delineation

To maintain absolute scientific transparency, the paper must delineate the boundary between **offline document indexing**, **online query encoding**, and **late-interaction scoring**:

1. **Offline Document Indexing (Full Vision Transformer):**
   - ColQwen2-v1.0 processes high-resolution document images through its vision encoder offline.
   - For 813 pages, this required **184 seconds (3.1 minutes)** on Nvidia Tesla T4 GPU ($\sim 226\text{ ms/page}$), yielding 48.0 MB of multi-vector index storage ($\sim 59\text{ KB/page}$).
   - This cost is incurred once at index time, never on the search path.

2. **Online Query Time:**
   - **Query Text Encoding:** Query text is passed through the ColQwen2 text projection layer to produce multi-vector query representations ($\sim 32$ tokens $\times$ 128 dims, $\sim 3\text{ ms}$ on GPU).
   - **Late-Interaction Ranking (MaxSim):** Computes $\sum_{i} \max_j (q_i \cdot d_j)$ against candidate document tensors.
   - The live benchmarks on Apple M2 CPU (12.18 ms) and MPS GPU (1.26 ms) directly executed the uninterrupted **Stage-1 inverted index lookup + Stage-2 MaxSim tensor operator** over candidate document tensors.

---

## 5. Required Manuscript Clarifications

To prevent any reviewer misunderstanding:
1. In the **Abstract (Line 26)** and **Section I (Introduction)**:
   - Ensure the wording explicitly reads: *"deploying two-stage cascades ($K_1=50$ Word BM25 pre-filtering $\to$ ColQwen2 reranking) retains 97.2% of native visual MRR (0.4822 vs. 0.4962) with an analytically modeled $15.4\times$ latency reduction (16.20 ms modeled vs. 249.21 ms standalone GPU pass; verified via live CPU cascade at 12.18 ms, a $15.5\times$ measured reduction)."*
2. In **Table XII (`tab:system_costs`)**:
   - Maintain the explicit hardware profile annotations in every row header (`Apple M2 CPU`, `Apple M2 MPS`, `Tesla T4 GPU`).
   - Reiterate in the caption note that rows represent deployment-specific serving profiles rather than single-machine benchmark comparisons.

