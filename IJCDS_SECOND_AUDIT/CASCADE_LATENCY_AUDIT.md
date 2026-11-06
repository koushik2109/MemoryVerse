# REPRODUCIBILITY AUDIT: TWO-STAGE CASCADE LATENCY & TIMING PROFILES
**Manuscript:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Auditor:** Independent Systems & Reproducibility Auditor  
**Date:** November 2026  
**Audit Directory:** `IJCDS_SECOND_AUDIT/`  

---

## 1. Executive Summary & Verdict

### Final Classification:
### **VERDICT: UNVERIFIED (MODELED ESTIMATE PRESENTED AS MEASURED RUNTIME)**

```
+----------------------------------------------------------------------------------------------------+
|                                    CASCADE LATENCY CLASSIFICATION                                  |
|                                                                                                    |
|                       [ UNVERIFIED (MODELED ESTIMATE PRESENTED AS MEASURED) ]                       |
|                                                                                                    |
| 1. The reported cascade latency of 16.20 ms and the claimed 15.4x speedup were NEVER directly       |
|    measured in a live end-to-end pipeline.                                                         |
| 2. The value is derived from an analytical linear equation implemented in                          |
|    `scripts/gapfill_t2_04_fusion_cascade.py`:                                                      |
|         Latency = 0.87 ms (Apple M2 CPU) + 50 * (249.21 ms / 813) (NVIDIA Tesla T4 GPU) = 16.20 ms |
| 3. This formula mathematically combines execution times across two disjoint hardware architectures |
|    (Apple Silicon ARM64 CPU vs. NVIDIA Tesla T4 GPU) while omitting tensor transfer, IPC, and      |
|    batching overheads.                                                                             |
| 4. The manuscript must explicitly describe this figure as an analytical projection rather than an  |
|    empirically observed latency.                                                                   |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Source Code Deconstruction

Inspection of `scripts/gapfill_t2_04_fusion_cascade.py` (lines 105–125) reveals the exact implementation:

```python
# Serving latency profiles (from Table XII of manuscript):
# Word BM25: 0.87 ms CPU
# ColQwen2 full (813 docs): 249.21 ms GPU
# Per-candidate ColQwen2 scoring: 249.21 / 813 = 0.3065 ms / candidate
base_sparse_lat = 0.87
colqwen_per_doc_lat = 249.21 / 813.0

for K in k_values:
    lat_k = base_sparse_lat + K * colqwen_per_doc_lat
    speedup = 249.21 / lat_k
    
    # Word BM25 -> ColQwen2
    in_k_word = (rank_word <= K)
    hit_k_word = float(np.mean(in_k_word))
    casc_word_rr = np.where(in_k_word, rr_col, 0.0)
```

### Analysis of Implementation Defects:

#### Defect 1: Purely Analytical Arithmetic
For $K = 50$:
$$\text{Latency} = 0.87 + 50 \times \left(\frac{249.21}{813.0}\right) = 0.87 + 50 \times 0.30653 = 0.87 + 15.3265 = \mathbf{16.1965\text{ ms}} \approx \mathbf{16.20\text{ ms}}$$
$$\text{Speedup} = \frac{249.21}{16.1965} = \mathbf{15.386\times} \approx \mathbf{15.4\times}$$
No timer (`time.perf_counter()`), CUDA event (`torch.cuda.Event`), or end-to-end execution benchmark was invoked.

#### Defect 2: Cross-Hardware Heterogeneity
- The Stage 1 sparse baseline ($0.87\text{ ms}$) was timed on a local developer machine (**Apple Silicon M2, ARM64 CPU**).
- The Stage 2 visual baseline ($249.21\text{ ms}$) was timed on a cloud instance (**NVIDIA Tesla T4 GPU**).
- Summing an Apple M2 CPU latency with a scaled NVIDIA Tesla T4 GPU latency assumes a zero-latency hybrid interconnect between incompatible hardware platforms.

#### Defect 3: Omission of Practical Pipeline Overheads
The linear formula assumes that reranking 50 documents takes exactly $50/813$ of the time required to rank 813 documents. In practice:
1. **GPU Kernel Launch Overhead:** CUDA kernel launches incur non-zero fixed latency regardless of tensor length.
2. **Tensor / Image Transfer:** Transferring 50 high-resolution document images or precomputed multi-vector embeddings to GPU VRAM introduces PCIe bus latency.
3. **ColBERT / Late-Interaction Batching:** In ColQwen2, MaxSim operator performance depends on tensor padding and matrix multiplication tiling, which does not scale strictly linearly at small batch sizes ($K=50$).

#### Defect 4: Proxy Reranking Simulation
Line 120 reveals:
```python
casc_word_rr = np.where(in_k_word, rr_col, 0.0)
```
The experiment never actually executed ColQwen2 late interaction over the top-50 candidates. It simply checked whether the ground-truth document was present in the top-50 Word BM25 output; if present, it assigned the target's full-corpus ColQwen2 reciprocal rank. This proxy evaluation assumes that the relative ordering of the top-50 candidates under ColQwen2 is identical to their relative ordering in the full 813-item index.

---

## 3. Systematic Classification of Retrieval Latencies

| System / Configuration | Reported Latency | Measurement Nature | Hardware Platform | Audit Status |
|---|---|---|---|---|
| **Word BM25 (Full Pool)** | 0.87 ms | Directly Measured | Apple Silicon M2 CPU | **VERIFIED** |
| **Character 3–5g BM25 (Full Pool)** | 3.42 ms | Directly Measured | Apple Silicon M2 CPU | **VERIFIED** |
| **Dense BGE-base (CPU)** | 48.30 ms | Directly Measured | Apple Silicon M2 CPU | **VERIFIED** |
| **Native ColQwen2 (Full 813 Pool)** | 249.21 ms | Directly Measured | NVIDIA Tesla T4 GPU | **VERIFIED** |
| **Extractive PRF (Relational)** | 12.0 ms | Directly Measured | Apple Silicon M2 CPU | **VERIFIED** |
| **Generative LLM Bridging** | 4,736.8 ms | Directly Measured | OpenAI API (`gpt-4o-mini`) | **VERIFIED** |
| **Two-Stage Cascade (K=50)** | **16.20 ms** | **Analytical Model** | **M2 CPU + Tesla T4 GPU** | **UNVERIFIED (MODELED)** |
| **Cascade Speedup Factor** | **15.4×** | **Analytical Ratio** | **Cross-Hardware Model** | **UNVERIFIED (MODELED)** |

---

## 4. Verification of Accuracy & Retention Claims

While the latency is an analytical estimate, the retrieval accuracy numbers were verified from probe reciprocal ranks:
- Native ColQwen2 MRR: $0.4962$
- Two-Stage Cascade (K=50) MRR: $0.4822$
- MRR Retention:
  $$\frac{0.4822}{0.4962} = 0.971785 \implies \mathbf{97.18\%} \approx \mathbf{97.2\%}$$
- Word BM25 Top-50 Hit Rate: $0.9718$ (indicating that 297 out of 306 targets were captured in the top-50 filter).
- **Status:** **VERIFIED (MATHEMATICALLY EXACT ON PROXY ASSUMPTIONS)**.

---

## 5. Required Manuscript Replacement Text

### Current Text (`ijcds_body.tex`, lines 645–655):
```latex
As shown in Table~\ref{tab:colqwen_cascade}, the two-stage cascade achieves an MRR of 0.4822, retaining 97.2\% of native ColQwen2 retrieval performance (0.4962 MRR) while reducing mean query latency from 249.21\,ms to 16.20\,ms---a 15.4$\times$ speedup.
```

### Exact Replacement Text:
```latex
As shown in Table~\ref{tab:colqwen_cascade}, the two-stage cascade achieves an MRR of 0.4822, retaining 97.2\% of native ColQwen2 retrieval performance (0.4962 MRR) over the 813-item test pool. Based on component latency profiling (0.87\,ms CPU sparse pre-filtering plus an analytical 15.33\,ms GPU reranking allocation for 50 candidates), the modeled cascade latency is 16.20\,ms compared to 249.21\,ms for exhaustive GPU visual scoring---representing an analytical 15.4$\times$ latency reduction. End-to-end production latencies will vary depending on PCIe bus transfer overheads and batch scheduling.
```

---
**Auditor Signature:** Independent Systems & Reproducibility Auditor  
**Date:** November 4, 2026

