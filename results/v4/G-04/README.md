# G-04: Lexical and Native-Visual Fusion & Two-Stage Cascades

## Executive Summary
This experiment addresses Reviewer Gap **G-04**: can text proxies and native visual representations be synergistically combined rather than treated as mutually exclusive alternatives?

### Key Findings
1. **RRF Hybrid Fusion Establishes New SOTA:**
   - Standalone Word BM25: **0.4003 MRR**
   - Standalone ColQwen2-v1.0: **0.4962 MRR**
   - **RRF Fusion (Word BM25 + ColQwen2):** **0.5745 MRR** [95% CI: 0.528, 0.621], Hit@10 = **0.7353**
   - RRF delivers **$+0.0782$ MRR lift over ColQwen2** ($p < 10^-5$) and **$+0.1742$ MRR lift over Word BM25** ($p < 10^-5$), proving that lexical and visual signals provide complementary information on receipts.
2. **Two-Stage Cascades Deliver 15.4x Latency Reduction:**
   - Standalone ColQwen2 requires **249.21 ms** GPU latency per query across the 813-item corpus.
   - A two-stage cascade using Word BM25 first stage ($K_1=50$, taking 0.87 ms) followed by ColQwen2 MaxSim reranking over only 50 candidate images achieves:
     - **0.4822 MRR** (retaining **97.2%** of full ColQwen2 capacity)
     - **16.20 ms** end-to-end latency (**15.4x speedup**!)
   - At $K_1=100$: reaches **0.4907 MRR** (98.9% capacity) at **31.52 ms** (7.9x speedup).
3. **Tri-Modal Fusion:**
   - Tri-modal fusion combining Char BM25 + Word BM25 + ColQwen2 achieves **0.5601 MRR** across the expanded 459-probe test suite.

### Manuscript Implications (Section VI / Table XII)
This provides an engineering recommendation: practitioners do not need to choose between fast CPU lexical search and accurate GPU visual search. A two-stage cascade achieves 97.2% of native visual accuracy at 16.2 ms serving latency.
