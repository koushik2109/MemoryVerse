# PHASE 7 – CORPUS SCALING, CASCADE ECONOMICS & LATENCY PROFILING
**Module:** Phase 7 Deliverable  
**Auditor:** Senior Systems Architect, IR Engineer & Performance Reviewer  
**Output Path:** `MEMORYVERSE_FINAL_AUDIT/PHASE7_SYSTEMS.md`  

---

## 1. EXECUTIVE SUMMARY & AUDIT SCOPE

Phase 7 audits the production engineering claims, computational scalability, and operational economics reported in the MemoryVerse manuscript:
1. **Corpus Scaling to 500,000 Distractors:** Evaluating sparse lexical index size, build time, memory footprint, and query latency across corpus expansions ($1\text{K} \to 500\text{K}$).
2. **Two-Stage Cascade (Word BM25 $\to$ ColQwen2):** Verifying the claimed 97.2% MRR retention (0.4822 vs. 0.4962) and investigating the 16.20 ms ($15.4\times$ speedup) latency figure.

### Key Audit Findings:
- **Cascade Retrieval Retention Verified:** The two-stage cascade ($K_1=50$ Word BM25 pre-filtering followed by ColQwen2 reranking) achieves $\text{MRR} = 0.4822$, retaining **97.18% of standalone ColQwen2 MRR (0.4962)**. The retrieval retention calculation is mathematically exact and verified.
- **CRITICAL DEFECT: Cascade Latency was Modeled, Not Measured:**  
  The claimed latency of **16.20 ms** was **never measured end-to-end in live execution**. In `scripts/gapfill_t2_04_fusion_cascade.py` (lines 105–125), the authors summed CPU latency measured on an Apple M2 laptop ($0.87\text{ ms}$) with scaled GPU time measured on a remote Tesla T4 GPU ($50 \times 249.21/813 = 15.33\text{ ms}$), omitting PCIe transfer and process overhead. The reranking was computed via array indexing (`np.where(rank_word <= 50, rr_col, 0.0)`), not live neural scoring.
- **500K Distractor Pool Verified with Caveats:** At 500,000 items, Word BM25 (0.9412 MRR) and Character BM25 (0.9385 MRR) remain accurate. However, distractor generation synthesized 500K records from 20 vendor name templates; the lack of diverse vocabulary artificially prevented severe lexical collision.

---

## 2. DECONSTRUCTION OF THE CASCADE LATENCY MODEL

### 2.1 The Code Implementation (`scripts/gapfill_t2_04_fusion_cascade.py`)
Lines 105–125 reveal the exact formula used:
```python
# [scripts/gapfill_t2_04_fusion_cascade.py, Lines 105-125]
    # Stage 1: CPU lexical pre-filtering on top-K
    t_cpu = 0.87  # ms measured on Apple M2
    
    # Stage 2: GPU ColQwen2 scoring on K1 candidates
    # Scaled linearly from 813-doc full corpus time
    t_gpu_per_doc = 249.21 / 813.0  # 0.3065 ms per document
    t_rerank = 50 * t_gpu_per_doc   # 15.33 ms
    
    total_cascade_latency = t_cpu + t_rerank  # 16.20 ms
```

### 2.2 Systems & Methodological Critique:
1. **Cross-Hardware Summation:** Summing execution time from an Apple M2 ARM CPU with a cloud Nvidia Tesla T4 GPU creates an artificial hybrid latency that does not represent any real server architecture.
2. **Missing PCIe & Serialization Overhead:** Transferring 50 candidate document images over the PCIe bus and allocating GPU tensor memory incurs non-trivial latency (typically 5–15 ms) not accounted for in linear per-document scaling.
3. **No Live Candidate Scoring:** Retrieval was evaluated by filtering pre-computed ColQwen2 rank arrays rather than executing a live pipeline.

### 2.3 Required Remediation:
- In the manuscript text (Contribution 3, Section 5.C, and Table VII), 16.20 ms must be explicitly labeled as an **"analytical model estimate based on component hardware profiling"**.
- A self-contained Kaggle notebook (`KAGGLE_COLQWEN2_CASCADE_BENCHMARK.ipynb`) has been developed to benchmark true end-to-end latency with live batched candidate scoring on an Nvidia Tesla T4 GPU.

---

## 3. TWO-STAGE CASCADE RETRIEVAL EFFECTIVENESS RECALCULATION

Evaluating the candidate cutoffs in `results/v4/R-02/cascade_tradeoff_summary.csv`:

| Pre-filtering Cutoff ($K_1$) | Cascade MRR [95% CI] | Standalone ColQwen2 MRR | Relative Retention | Modeled Latency | Modeled Speedup |
|---|---|---|---|---|---|
| $K_1 = 10$ | 0.4410 [0.395, 0.487] | 0.4962 | 88.88% | 3.94 ms | $63.3\times$ |
| $K_1 = 20$ | 0.4652 [0.419, 0.512] | 0.4962 | 93.75% | 7.00 ms | $35.6\times$ |
| **$K_1 = 50$ (Recommended)**| **0.4822 [0.436, 0.528]** | **0.4962** | **97.18%** | **16.20 ms** | **$15.4\times$** |
| $K_1 = 100$ | 0.4915 [0.444, 0.539] | 0.4962 | 99.05% | 31.52 ms | $7.9\times$ |
| Standalone ColQwen2 ($K_1=813$)| 0.4962 [0.448, 0.547] | 0.4962 | 100.00% | 249.21 ms | $1.0\times$ |

**Finding:** The trade-off curve is mathematically valid. Setting $K_1 = 50$ retains over 97% of ColQwen2's accuracy while reducing theoretical candidate scoring by $16.3\times$.

---

## 4. CORPUS SCALING TO 500,000 DOCUMENTS AUDIT

Evaluating the scale expansion benchmarks in `results/v6/scaling_500k_metrics.csv`:

| Corpus Size ($N$) | Word BM25 MRR | Char 3–5g BM25 MRR | Char Inverted Index RAM | Word Inverted Index RAM | Char Query Latency | Word Query Latency |
|---|---|---|---|---|---|---|
| **813 (Baseline)** | 0.8841 | 0.9471 | 12.4 MB | 1.8 MB | 0.87 ms | 0.22 ms |
| **10,000** | 0.9120 | 0.9450 | 114.2 MB | 16.5 MB | 1.45 ms | 0.38 ms |
| **50,000** | 0.9250 | 0.9431 | 540.8 MB | 78.2 MB | 2.68 ms | 0.62 ms |
| **100,000** | 0.9312 | 0.9418 | 1.05 GB | 154.0 MB | 4.12 ms | 0.81 ms |
| **250,000** | 0.9380 | 0.9402 | 2.58 GB | 378.5 MB | 6.25 ms | 0.98 ms |
| **500,000** | **0.9412** | **0.9385** | **4.82 GB** | **684.2 MB** | **8.42 ms** | **1.12 ms** |

### 4.1 Technical Verification:
- **Asymptotic Convergence:** As $N$ expands, Word BM25 MRR increases from 0.8841 to 0.9412 due to IDF sharpening on shared distractor vocabulary.
- **Index Footprint:** Character 3–5g BM25 requires $4.82\text{ GB}$ of RAM at 500K documents (vs. $684\text{ MB}$ for Word BM25), reflecting a $7.0\times$ memory multiplier due to n-gram token cardinality.
- **Vocabulary Diversity Caveat:** Because distractors were synthesized from 20 repeated vendor templates, this benchmark measures scaling against large document volume with low vocabulary entropy. The manuscript must explicitly state this boundary condition.

