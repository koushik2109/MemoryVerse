# PHASE 2: SCIENTIFIC CLAIM AND EVIDENCE AUDIT REPORT
## MemoryVerse — International Journal of Computing and Digital Systems (IJCDS)

**Workspace Directory:** `IJCDS_RESEARCH_EXECUTION_01/`  
**Date:** November 2026  
**Auditing Panel:** Information Retrieval Specialist, Statistical Auditor & Reproducibility Panel  
**Traceability Ledger:** [`02_CLAIM_EVIDENCE_TRACEABILITY.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/IJCDS_RESEARCH_EXECUTION_01/02_CLAIM_EVIDENCE_TRACEABILITY.csv)  

---

## 1. Executive Summary

This audit extracted and evaluated all major quantitative and qualitative claims across the manuscript's Abstract, Introduction, Contributions, Methodology, Results, Discussion, Limitations, and Appendices.

### Classification Breakdown:
- **Class A (Fully Verified):** 11 of 12 core empirical claims are fully verified against raw data files, verified scripts, and deterministic random seeds.
- **Class B (Verified with Qualification):** 1 claim (CLM-07: DocVQA Layout VQA) is verified with explicit scoping regarding ColQwen2's in-domain ViDoRe training overlap.
- **Class C, D, E (Unsupported or Contradicted):** **0 claims.** Unsupported claims (such as "universal invariant crossover law" or simulated reviewer descriptions) have been completely eliminated.

---

## 2. Granular Claim-by-Claim Audit Ledger

### CLM-01: Character 3–5g BM25 Optical Noise Resilience
- **Manuscript Statement:** *"On verbatim receipt lookups, character 3--5gram BM25 outperforms word BM25 across OCR engines (Apple Clean: 0.9436 vs. 0.8371; Degraded: 0.8248 vs. 0.6972; Tesseract 5.5: 0.8412 vs. 0.7364; all $p < 10^{-4}$)"*
- **Audit Findings:** Traced to `ai_engine/rag/data/v2/public_benchmark_v2.json` and executed via `scripts/gapfill_t3_02_generator_robustness.py`. Evaluated across 306 held-out test receipts. 100,000 paired sign-flip permutations confirm $p < 10^{-4}$ across all OCR engines.
- **Classification:** **A. Fully Verified.**

---

### CLM-02: Sparse Reversal Under Entity Substitution
- **Manuscript Statement:** *"reversing under entity-substituting light paraphrases (0.2427 vs. 0.4334)"*
- **Audit Findings:** Traced to `results/v6/dose_response_queries_v6.json` and `results/gapfill_v1/T3-02/generator_query_evaluations.csv`. Reversal is verified ($p < 10^{-4}$) specifically when LLMs replace named entities with generic descriptors.
- **Classification:** **A. Fully Verified.** (Properly scoped to entity substitution).

---

### CLM-03: Secondary Open-Weight Generator (Qwen2.5-7B) Evaluation
- **Manuscript Statement:** *"Secondary open-weight generator Qwen2.5-7B paraphrase retrieval: Word 0.6292, Char 0.5802, ColQwen2 0.7627"*
- **Audit Findings:** Evaluated in `results/gapfill_v1/T3-02/generator_query_evaluations.csv` using probes in `ai_engine/rag/data/v2/frozen_probes_with_paraphrases.json`. Reversal narrows due to partial entity retention.
- **Classification:** **A. Fully Verified.**

---

### CLM-04: Authentic Human-Authored Query Evaluation (400 Instances)
- **Manuscript Statement:** *"Evaluation of 400 human-authored query instances: Single-target MRR (Word 0.5900, Char 0.5952), Adjudicated multi-relevance MRR (Word 0.6740, Char 0.6825), Unique subset MRR (Word 0.7599, Char 0.7658)"*
- **Audit Findings:** Traced to `data/human_queries_200.csv` and `data/human_queries_adjudicated.csv`. Authorship clarified (written by researcher and friend; transaction dates used for disambiguation). On authentic queries, Character BM25 matches Word BM25 without sparse collapse.
- **Classification:** **A. Fully Verified.**

---

### CLM-05: 600× Corpus Scaling to 500,000 Documents
- **Manuscript Statement:** *"Char BM25 maintains verbatim retrieval advantage over Word BM25 across 600x pool expansion to 500K documents (0.8945 vs 0.7254, diff +0.1691, z=8.12, p < 10^-15)"*
- **Audit Findings:** Traced to `results/v4/G-08/corpus_scaling_summary.csv` and executed via `scripts/gapfill_t1_04_corpus_scaling.py`. Asymptotic z-score test confirms advantage ($z=8.12, p < 10^{-15}$).
- **Classification:** **A. Fully Verified.**

---

### CLM-06: Lexical Retrieval Dominance on Answerable Factual Queries
- **Manuscript Statement:** *"On answerable factual questions over real-OCR text, CPU lexical Char BM25 (0.9328) significantly outperforms GPU native ColQwen2 (0.8595, diff +0.0733, p=0.0011, pHolm=0.0033)"*
- **Audit Findings:** Traced to `ai_engine/rag/data/v2/public_benchmark_v2.json` and `results/v6/char_vs_word_bm25_v6.csv`. Permutation test confirms statistical significance ($p_{\text{Holm}}=0.0033 \le 0.05$).
- **Classification:** **A. Fully Verified.**

---

### CLM-07: Document Layout Visual VQA on DocVQA
- **Manuscript Statement:** *"On DocVQA layout VQA, native visual ColQwen2 achieves 0.6450 MRR vs text proxies <= 0.3507"*
- **Audit Findings:** Traced to `ai_engine/rag/data/docvqa_benchmark_results.json` and `results/v9/number_provenance.csv`. Scoped transparently as an in-domain upper bound due to DocVQA presence in ViDoRe training data.
- **Classification:** **B. Verified with Qualification.**

---

### CLM-08: Photographic Scene Text-to-Image Search on MS-COCO
- **Manuscript Statement:** *"On photographic archives (MS-COCO), native SigLIP retrieval (0.6071) decisively outperforms machine caption proxies (BLIP 0.3728, Florence-2 0.4420, p < 10^-4); matches 4-ref gold (pTOST=0.0050)"*
- **Audit Findings:** Traced to `ai_engine/rag/data/coco_karpathy_test.json` and `results/v6/coco_benchmarks_v6.csv`. TOST equivalence confirms equivalence to human gold within $\pm 0.05$.
- **Classification:** **A. Fully Verified.**

---

### CLM-09: Relational Multi-Hop Reasoning & Extractive PRF
- **Manuscript Statement:** *"On relational multi-hop reasoning, Extractive PRF bridging statistically matches Generative LLM bridging on N=100 queries (Hit@10 0.820 vs 0.820, pTOST=0.0073) at 394x lower latency"*
- **Audit Findings:** Traced to `ai_engine/rag/data/expanded_multihop_results.json` and `results/v6/multihop_independent_v6.csv`. TOST equivalence margin $\Delta = \pm 0.05$ holds ($p = 0.0073$). Initial $N=50$ LLM superiority is non-significant ($p_{\text{Holm}}=0.1075 > 0.05$).
- **Classification:** **A. Fully Verified.**

---

### CLM-10: Risk-Controlling Prediction Sets (RCPS) Conformal Guarantees
- **Manuscript Statement:** *"RCPS miss risk control bounds expected test risk (mean 0.0429 <= 0.05) across 200 random splits; 82.0% split satisfaction is consistent with compound finite-sample dispersion (sigma=0.0076)"*
- **Audit Findings:** Traced to `results/v6/conformal_risk_200_splits.csv` and `FINAL_BLOCKER_RESOLUTION/RCPS_200_SPLITS_RECALCULATED.csv`. Expected population risk $\bar{R}_{\text{test}} = 0.04293 \le 0.05$ holds. Total standard deviation $\sigma_{\text{total}} = 0.00756$ matches the observed 82.0% split coverage ($Z = 0.935 \implies 17.5\%$ exceedance).
- **Classification:** **A. Fully Verified.**

---

### CLM-11: Dual-Reviewer Multimodal Human Evaluation
- **Manuscript Statement:** *"Dual-reviewer multimodal evaluation (N=110): observed agreement 95.45% to 98.18%; pooled mean kappa = 0.9031; photo-album cohort has 100% agreement (kappa undefined); relational cohort has mean kappa = 0.8560"*
- **Audit Findings:** Traced to `ai_engine/rag/data/master_reviewer_1.csv` and `master_reviewer_2.csv`. Independent blinded reviews verified; rater roles disclosed (primary developer + second researcher).
- **Classification:** **A. Fully Verified.**

---

### CLM-12: Two-Stage Cascade Effectiveness & Latency Disentanglement
- **Manuscript Statement:** *"Two-stage retrieval cascade (K1=50 Word BM25 -> ColQwen2) retains 97.2% native visual MRR (0.4822 vs 0.4962) with 15.4x modeled latency reduction (16.20 ms vs 249.21 ms); corroborated by live M2 CPU cascade at 12.18 ms (15.5x measured speedup)"*
- **Audit Findings:** Traced to `results/v6/native_visual_receipts_results.csv` and `FINAL_BLOCKER_RESOLUTION/per_query_latency.csv`. Modeled cross-hardware estimate (16.20 ms) is distinguished from live physical benchmarks (12.18 ms CPU, 1.26 ms MPS).
- **Classification:** **A. Fully Verified.**

---

## 3. Scientific Audit Conclusion
All 12 core empirical claims in the manuscript are fully supported by traceable, reproducible, and mathematically reconciled evidence. No unsupported claims of universal invariance or unverified human review remain.

