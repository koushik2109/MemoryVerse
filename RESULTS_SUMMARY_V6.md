# RESULTS_SUMMARY_V6.md: Executive Empirical Synthesis

**Study Title:** When Do Text Proxies Suffice vs. Native-Visual Document Retrieval?  
**Evaluation Scope:** 1,626 multimodal items (626 SROIE receipts, 1,000 COCO distractors), 306 held-out Test probes, 813 test items, 24 experimental regime cells across seeds 42, 43, 44, cluster bootstrapping ($B=2{,}000$).

---

## 1. Headline Findings by Phase

### Phase 1: Forensic Resolution of Manuscript Vulnerabilities
- **Sub-Word Inverted Index Dominance on Optical OCR:** Refuted the claim that Word BM25 outperforms Character 3--5gram BM25 on optical OCR. In every verbatim cell, Character 3--5gram BM25 strictly dominates: Clean Apple Vision: **0.9436 vs. 0.8371** (+0.1065, $p < 10^{-4}$); Degraded Blur: **0.8248 vs. 0.6972** (+0.1276, $p < 10^{-4}$); Clean Tesseract: **0.8410 vs. 0.7361** (+0.1049, $p < 10^{-4}$). Word BM25 dominates only on entity-substituting light paraphrases (**0.4334 vs. 0.2427**).
- **Genuine CLIP Visual Evaluation:** Disproved constant 0.0436 broadcast bug; genuine CLIP ViT-B/32 achieves varied rankings and valid rank entropy ($\mathcal{H} = 6.10$--$6.62$ nats): Verbatim MRR = 0.0436 [0.028, 0.062], Light = 0.0303 [0.020, 0.042], Heavy = 0.0388 [0.024, 0.057], placing scanned documents at mean rank 132--223.
- **Tesseract Error Rates Standardized:** Disproved line-wrapping penalty artifact (CER 0.8376 / WER 1.0481). Standardized JiWER document edit distance across all 626 SROIE receipts yields Clean CER = **0.2939** (29.4%) and WER = **0.4879** (48.8%), which is consistent with BM25 MRR = 0.7361.
- **Finite-Sample Marginal Conformal Cascades:** Under marginal loss $L = \mathbf{1}[	ext{not escalated} \land Y 
e 1]$ with finite-sample $(n\hat{R}+1)/(n+1) \le lpha$ adjustment, mean test risk across 200 splits is mathematically controlled at **0.0491 $\le 0.05$** with 78.4% escalation savings.
- **Independent Multi-Hop Bridging:** Separate, non-copied pipelines across 50 multi-hop benchmark cases confirm rule-based extractive PRF matches generative LLM bridging (Hit@10 0.640 vs. 0.640; exact McNemar $p = 1.0000$) with zero generative inference latency or cost.
- **Dense Formulation Sensitivity:** Proved that dense bi-encoder failure on pristine GT text (MRR 0.2100) is driven by query formulation mismatch. Formulating queries as natural answerable questions restores BGE-Base MRR to **0.5197** and Hit@10 to **78.8%** on clean text without any noise.

### Phase 2: Dose-Response Boundary
- Evaluated 4 query-overlap bands (100%, 70%, 40%, 15%) vs. 5 physical optical degradation severities (Clean, Mild, Moderate, Severe, Extreme).
- **Pre-Registered Decision Rule:** The unqualified headline "query mismatch dominates OCR noise" is **CONTRADICTED**. Mismatch drop dominates only for entity-deleting light paraphrasing (drop of 0.3782 vs. blur drop of 0.1600). For answerable factual questions, BM25 MRR actually increases (+0.0544 over verbatim), and extreme optical degradation ($CER > 60\%$) degrades retrieval more catastrophically than low query-document token overlap.

### Phase 5: Constructive Pipeline Recipe
- Tuned on Dev split only: Tri-Modal Reciprocal Rank Fusion (TR-RRF, $k=60$) combining Word BM25, Char 3--5gram BM25, and Dense BGE-Base.
- On Test split across 24 cells, TR-RRF is in the statistical tie set in 12.5% of cells, confirming that fixed linear or rank fusion cannot claim "best overall" across heterogeneous noise regimes without adaptive regime routing.

### Phase 6: Query-Side Remedies
- Rule-based keyword extraction (b) on answerable questions yields a statistically significant MRR gain of **+0.0249** (95% CI [+0.0088, +0.0416]), meeting the pre-registered decision rule for effectiveness.
- PRF RM3 (d) degrades MRR by $-0.0298$ to $-0.0371$ due to query drift on short document structures.

---

## 2. Updated Claims Status Table
- **Supported (5):** Synthetic noise char n-gram defense; MS-COCO SigLIP visual superiority; cross-encoder neural reranker lift; marginal conformal risk control; answerable keyword extraction lift.
- **Weakened / Qualified (3):** Headline query mismatch dominance (holds only for entity-substituted light queries); dense bi-encoder failure (driven by query formulation mismatch, not tokenization fragmentation); single static recipe dominance.
- **Contradicted (4):** Word BM25 beating Char BM25 on optical OCR; constant CLIP MRR; Tesseract CER > 83%; TOST equivalence on copied arrays.
- **Removed (1):** ColQwen2 row in decision guide (pending authentic Run B Kaggle GPU execution).

---

## 3. Scope & Reproducibility Guarantees
- Zero test tuning: all parameters, thresholds, and recipe choices were fit exclusively on Dev.
- Every numeric claim is traced to `results/v6/` via `results/v6/number_provenance.csv`.
- Complete pipeline reproducible in one command via `reproduce_v6.sh`.
