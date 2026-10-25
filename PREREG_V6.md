# Pre-Registration Document (v6): Forensic Resolution of Empirical Contradictions

**Study Title:** When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise  
**Pre-Registration Date:** 2026-10-25T19:15:00+05:30  
**Repository Branch:** `research`  
**Execution Target:** Run A (Local CPU Environment) & Run B (Kaggle GPU Packaging)  

---

## 1. Objectives & Forensic Resolution Scope

This pre-registration governs Phase 0 and Phase 1 (Run A) forensic corrections addressing nine critical contradictions identified in `AUDIT_V6.md`:
1. **Contradiction (a):** Recomputing Word BM25 vs. Char 3–5gram BM25 across all query types and optical OCR sources to eliminate contradictory manuscript claims.
2. **Defect (b):** Executing genuine CLIP image-to-text retrieval across receipt images and COCO distractors with verified non-zero ranking entropy, replacing degenerate hardcoded constants.
3. **Artifact (c):** Re-tuning hybrid fusion weights ($\alpha$) strictly on the Dev split for redesigned answerable factual questions, eliminating the stale $\alpha^*=1.0$ collapse to Dense MiniLM.
4. **Alignment Defect (d):** Recomputing Tesseract OCR CER/WER using standard Levenshtein edit distance and JiWER under canonical text normalization, correcting line-length filtering inflation.
5. **Formulation Defect (e):** Reformulating conformal cascade risk control as *marginal loss* $L = 1[\text{not escalated} \land Y \ne 1]$ with finite-sample $(n+1)$ quantile calibration and Learn-then-Test bounds over 200 cluster-stratified splits, establishing valid distribution-free coverage.
6. **Provenance Defect (f):** Executing Grounded Extractive PRF bridging and Generative LLM bridging as independent code paths with separate caches, evaluating exact paired discordant outcomes without array copying.
7. **Provenance Mapping (g):** Emitting `results/v6/number_provenance.csv` mapping every published metric to its exact query type, text condition, probe subset, and distractor pool.
8. **Scope Delineation (h):** Explicitly marking GPU-dependent ColQwen2 evaluations as pending Phase 3, and testing headline query-mismatch vs. OCR-noise claims across probe types.
9. **Representation Deconstruction (i):** Evaluating dense bi-encoders on clean ground-truth text across short verbatim snippets, question-style queries, full sentences, and length-padded controls to isolate query-format mismatch from noise collapse.

---

## 2. Pre-Registered Hypotheses & Decision Rules

### Hypothesis 1.1: Sparse Inverted Index Dominance under Optical OCR (Confirmatory)
- **Statement:** Character 3–5gram BM25 matches or exceeds Word-level BM25 across verbatim queries under both proprietary Apple Vision and open-source Tesseract optical OCR.
- **Statistical Test:** Paired sign-flip permutation tests ($N_{\text{perm}}=100{,}000$) with cluster bootstrap 95% CIs ($B=2{,}000$) and Holm-Bonferroni correction ($m=4$).
- **Decision Rule:**
  - If $\text{MRR}_{\text{char}} - \text{MRR}_{\text{word}} > 0$ with $p_{\text{Holm}} < 0.05$, claim "Character $n$-gram BM25 significantly outperforms Word BM25 on optical OCR."
  - If the 95% CI contains zero, claim "Character $n$-gram and Word BM25 achieve statistically equivalent retrieval under optical OCR."
  - If $\text{MRR}_{\text{word}} - \text{MRR}_{\text{char}} > 0$ with $p_{\text{Holm}} < 0.05$, claim "Word BM25 is superior." The manuscript must strictly follow the empirical verdict.

### Hypothesis 1.2: Receipt Visual Retrieval with Genuine CLIP (Confirmatory)
- **Statement:** Zero-shot image retrieval via `openai/clip-vit-base-patch32` on receipt documents produces non-degenerate per-query ranking entropy ($H(R_q) > 0$), but achieves near-chance MRR due to lack of text recognition capability.
- **Quality Gate:** Assert that per-query rank entropy across the test set satisfies $H > 1.0$ nat and that rank variance $> 0$.
- **Decision Rule:** Report genuine empirical MRR and 95% CIs.

### Hypothesis 1.3: Dev-Tuned Hybrid Recovery on Answerable Questions (Confirmatory)
- **Statement:** Re-tuning hybrid weight $\alpha$ strictly on the Dev split ($N=306$) on answerable questions (`heavy_paraphrases_v3.json`) yields an optimal $\alpha^* \in (0, 1)$ that leverages lexical signals, eliminating the dense collapse ($\alpha^*=1.0$).
- **Tuning Constraint:** Grid search $\alpha \in [0.0, 1.0]$ in steps of 0.05 strictly on the 306 Dev probes against Dev documents. Zero test item IDs used.
- **Decision Rule:** Select $\alpha^*$ maximizing Dev MRR. Evaluate on Test with frozen $\alpha^*$.

### Hypothesis 1.4: Tesseract OCR Error Normalization (Confirmatory)
- **Statement:** Normalizing Tesseract OCR outputs (lowercase, collapse whitespace, strip punctuation) and computing standard Levenshtein distance on document text lowers clean CER below 0.35 and WER below 0.50, reconciling OCR quality with BM25 MRR (0.74).
- **Decision Rule:** Report both line-aligned and document-level / bag-of-words error rates computed with JiWER.

### Hypothesis 1.5: Marginal Conformal Cascade Risk Control (Confirmatory)
- **Statement:** Defining risk as the marginal expected loss:
  $$L(X, Y, \tau) = 1[\text{Stage 1 Margin} \ge \tau \land \text{Stage 1 Top-1} \ne Y]$$
  and calibrating $\tau$ via the finite-sample adjusted rule:
  $$\tau^* = \inf \left\{ \tau \in \mathcal{T} : \frac{n \hat{R}(\tau) + 1}{n + 1} \le \alpha \right\}$$
  ensures that across 200 random cluster-stratified 50/50 splits, mean test marginal risk is bounded by $\le \epsilon = 0.05$.
- **Decision Rule:** Claim "controls risk" if and only if mean test marginal risk $\le 0.05$ AND empirical coverage $\ge 50\%$.

### Hypothesis 1.6: Independent Multi-Hop Extractive vs. LLM Bridging (Confirmatory)
- **Statement:** Grounded extractive pseudo-relevance feedback bridging and generative LLM bridging executed via independent pipelines exhibit non-identical query-by-query outputs.
- **Quality Gate:** Assert that extractive bridge outputs and LLM bridge outputs have separate file caches and non-identical candidate rank vectors.
- **Decision Rule:** Report exact discordant counts (Win/Loss/Tie), exact two-tailed McNemar test $p$-value, and paired difference CIs.

### Hypothesis 1.9: Dense Bi-Encoder Formulation Sensitivity (Exploratory)
- **Statement:** Dense bi-encoders (`all-MiniLM-L6-v2`, `BAAI/bge-base-en-v1.5`) perform poorly on clean ground-truth text when queried with short verbatim alphanumeric snippets, but improve when queries are phrased as natural questions or full sentences with sentential context.
- **Evaluated Query Formats on Clean Ground Truth:**
  1. Short verbatim snippets (original verbatim probes)
  2. Answerable factual questions
  3. Full sentential queries with descriptive context
  4. Length-controlled padded snippets (spans padded with neutral stopwords to match question length)
- **Decision Rule:** Compare MRRs to determine the proportion of dense degradation attributable to query length / syntactic form versus representation noise.

---

## 3. Dataset Integrity & Zero Test-Tuning Contract

1. **Frozen Dataset Partitions:**
   - Corpus: 626 SROIE receipts + 1,000 COCO distractors ($N=1{,}626$ total items).
   - Split: Exact 50/50 split by Item ID ($N=813$ Dev items, $N=813$ Test items).
   - Probes: $N=306$ Dev probes, $N=306$ Test probes.
2. **Zero Test-Split Tuning Invariant:**
   - All parameters ($\alpha^*, k_1^*, b^*, \tau^*$) are tuned strictly on the 813 Dev items using the 306 Dev probes.
   - Code assertions verify that no Test item IDs appear in tuning dictionaries, parameter optimization loops, or conformal calibration splits.
3. **Statistical Protocols:**
   - Multi-seed evaluation: Seeds 42, 43, 44.
   - Cluster bootstrap by Item ID ($B=2{,}000$, 95% percentile intervals).
   - Paired permutation tests ($N_{\text{perm}}=100{,}000$ sign flips).
   - Step-down Holm-Bonferroni correction within each hypothesis family.
   - Analytically verified uniform random baseline ($H_N/N \approx 0.00895$).
