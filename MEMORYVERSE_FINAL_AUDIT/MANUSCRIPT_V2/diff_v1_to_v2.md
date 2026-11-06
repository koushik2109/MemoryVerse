# MANUSCRIPT V2 REVISION DIFF & EDITORIAL CHANGELOG
**Target Journal:** International Journal of Computing and Digital Systems (IJCDS)  
**Base Source:** `submission_package_ijcds/source/ijcds_body.tex`  
**Revised Source:** `MEMORYVERSE_FINAL_AUDIT/MANUSCRIPT_V2/ijcds_body.tex`  
**Date:** November 4, 2026  

---

## 1. ITEM-BY-ITEM REVISION LOG

### Item 1: Contribution 2 (Line 25) — Crossover Dynamics & Query Count Correction
- **Original Wording:**
  > `\item \textbf{Quantitative Regime Map and Invariant Crossover Threshold:} Establishing continuous logistic response surfaces ($R^2 > 0.95$) predicting retrieval effectiveness as a joint function of Document Character Error Rate (CER) and Query--Document Overlap. We establish the existence of an invariant critical crossover boundary ($o^* \approx 0.51$--$0.55$): whenever query--document overlap exceeds $52\%$, character $n$-grams consistently outperform whole words; below $52\%$ overlap (under entity-substituting paraphrases), whole-word indexing outperforms character $n$-grams (0.4334 vs.\ 0.2427). Through controlled query ablation ($N=306$) and multi-generator dose-response ($N=1{,}730$), we show this reversal is generator-invariant and driven by entity co-specification mass rather than syntactic phrasing ($p = 0.6617$). As a negative result, standard marginal conformal gating fails on semantic subgroups (entity-rewrite miss risk $0.0627 > 0.05$), justifying the simpler fixed-margin cascade.`
- **Revised Wording (V2):**
  > `\item \textbf{Quantitative Overlap Dynamics and Entity Formulation Regimes:} Analyzing query formulation dynamics across a multi-generator dose-response suite ($N=1{,}930$, comprising 1,224 GPT-4o-mini, 400 human-authored, and 306 Qwen2.5-7B queries). We show that the relative effectiveness of character $n$-grams versus whole words is governed by entity preservation: under aggressive entity-substituting paraphrases, word indexing outperforms character $n$-grams below ~65\% lexical overlap (0.4334 vs.\ 0.2427), whereas under authentic human formulation where entity anchors persist, character and word BM25 perform comparably across all overlap levels (MRRs 0.555--0.577). As a negative result, standard marginal conformal gating fails on semantic subgroups (entity-rewrite miss risk $0.0627 > 0.05$), justifying the simpler fixed-margin cascade.`
- **Scientific Rationale:** Resolves the $1{,}730 \to 1{,}930$ query reporting error, retracts the overfitted $[0.51, 0.55]$ universal threshold, and replaces the invalid ANOVA invariance claim ($p=0.6617$) with the empirical finding that human queries exhibit no steep crossover reversal.

---

### Item 2: Contribution 3 (Line 26) — Cascade Latency & Human Evaluation Disentanglement
- **Original Wording:**
  > `\item \textbf{Production Cascades and Operational Economics:} Demonstrating that while native visual retrieval excels when 2D geometry governs answerability (DocVQA MRR 0.6450 vs.\ proxies $\le 0.3507$, noting disclosed pretraining overlap) and photographic search (SigLIP MRR 0.6071 vs.\ caption proxies $\le 0.4420$), deploying two-stage cascades ($K_1=50$ Word BM25 pre-filtering $\to$ ColQwen2 reranking) retains 97.2\% of native visual MRR (0.4822 vs.\ 0.4962) while slashing latency by $15.4\times$ (16.20\,ms vs.\ 249.21\,ms). An end-to-end task audit ($N=150$) with dual-reviewer validation ($\kappa = 0.9031$) demonstrates that upstream retrieval MRR bounds downstream LLM answer accuracy.`
- **Revised Wording (V2):**
  > `\item \textbf{Production Cascades and Operational Economics:} Demonstrating that while native visual retrieval excels when 2D geometry governs answerability (DocVQA MRR 0.6450 vs.\ proxies $\le 0.3507$, noting disclosed in-domain ViDoRe pretraining overlap) and photographic search (SigLIP MRR 0.6071 vs.\ caption proxies $\le 0.4420$), deploying two-stage cascades ($K_1=50$ Word BM25 pre-filtering $\to$ ColQwen2 reranking) retains 97.2\% of native visual MRR (0.4822 vs.\ 0.4962) with an analytically modeled $15.4\times$ latency reduction (16.20\,ms vs.\ 249.21\,ms). An automated end-to-end question-answering audit ($N=150$) demonstrates that upstream retrieval MRR strictly bounds downstream generation accuracy (ANLS 0.3990 ColQwen2 vs.\ 0.3246 Char BM25), while independent dual-reviewer auditing across $N=110$ multimodal personal retrieval answers confirms high inter-annotator consensus (mean $\kappa = 0.9031$).`
- **Scientific Rationale:** Explicitly labels 16.20 ms as an analytical model estimate and cleanly separates the automated 150 document QA benchmark from the authentic 110-answer multimodal personal retrieval audit.

---

### Item 3: Section 6.2 (Lines 629–632) — Tri-Partite Benchmark Disentanglement
- **Original Wording:**
  > Merged relational retrieval ($N=100$), document QA ($N=150$), and dual-reviewer human audit into a single confusing paragraph implying 110 of the 150 document QA queries were reviewed by humans.
- **Revised Wording (V2):**
  > Dedicated paragraphs clearly presenting:
  > 1. Relational multi-hop benchmark ($N=100$, TOST equivalence $p=0.0073$).
  > 2. Automated end-to-end document QA benchmark ($N=150$, ANLS/EM metrics from GPT-4o-mini).
  > 3. Qualitative dual-reviewer human audit ($N=110$ answers from personal photo albums $N=60$ and multi-hop queries $N=50$, mean $\kappa = 0.9031$).
- **Scientific Rationale:** Completely eliminates dataset conflation, preserving valid automated metrics while attributing human review exclusively to its true empirical source.

---

### Item 4: Limitations (Line 648–649) — Finite-Sample RCPS & Latency Modeling
- **Original Wording:**
  > Reported 82.0% RCPS coverage without explanation and stated cascade latency as a flat measurement.
- **Revised Wording (V2):**
  > Added mathematical explanation: binomial sampling theory on test sets of size $N_{\text{test}} = 1{,}377$ predicts an 82.0% empirical satisfaction rate under a nominal $1-\delta = 0.90$ true population risk guarantee. Clarified that 16.20 ms represents an analytical model summing Apple M2 CPU pre-filtering with scaled Tesla T4 GPU reranking.

---

### Item 5: Appendix A (Lines 667 & 669) — Holm-Bonferroni Multiplicity Corrections
- **Family 4 Correction:**
  > Corrected MiniLM cross-encoder rank from Rank 2 to Rank 3. Threshold $\alpha_3 = 0.0100 \implies p_{\text{Holm}} = 0.0450 \le 0.05$ (Statistically Significant).
- **Family 6 Correction:**
  > Corrected multiplier on LLM bridging from $2$ to $m=5$. Threshold $\alpha_1 = 0.0100 \implies p_{\text{Holm}} = 0.1075 > 0.05$ (Non-Significant under family-wise control).

