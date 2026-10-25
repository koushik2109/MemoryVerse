# AUDIT_V6.md: Forensic Audit of Experimental Contradictions and Code Paths

**Target Investigation:** Problems (a) through (i) identified in the manuscript and results tables.  
**Audit Date:** 2026-10-25  
**Auditor:** Antigravity IDE Autonomous Pair Programmer  

---

## 1. Problem (a): Word BM25 vs. Character n-gram BM25 Contradiction

### Manuscript Claim
The Abstract and Contributions state:
> *"Character n-grams defend strictly against synthetic noise (+0.1008 MRR at 20% noise), whereas word BM25 is superior under optical OCR (+0.0996 clean, +0.0777 blur; p < 0.001)."*

### Empirical Data in `results/v3/`
Inspection of `results/v3/headline_regime_grid_v3.csv` and `results/v3/char_vs_word_bm25.csv` demonstrates the exact opposite in every single verbatim cell:
- **Ground Truth**: Char 3–5gram BM25 MRR = **0.9853** vs. Tuned Word BM25 = **0.8572** ($\Delta = +0.1281$)
- **Clean Apple Vision OCR**: Char 3–5gram BM25 MRR = **0.9436** vs. Tuned Word BM25 = **0.8371** ($\Delta = +0.1065$)
- **Degraded Optical Blur**: Char 3–5gram BM25 MRR = **0.8248** vs. Tuned Word BM25 = **0.6972** ($\Delta = +0.1276$)
- **Synthetic 20% Noise**: Char 3–5gram BM25 MRR = **0.7252** vs. Tuned Word BM25 = **0.3961** ($\Delta = +0.3291$)

### Code Path & Root Cause
- **Code Path**: `ai_engine/rag/scripts/run_phase3_reviewer_evidence_v3.py` (lines 483–500) and `results/v3/headline_regime_grid_v3.csv`.
- **Root Cause**: The "+0.1008" and "+0.0996" figures were carried over from an early legacy v2 script (`run_journal_revision_experiments.py` Hypothesis B1) where Word BM25 was evaluated under a different tokenization/distractor setting. In the public benchmark evaluation, Character 3–5gram BM25 strictly dominates Word BM25 across all optical OCR conditions on verbatim lookups because OCR errors frequently split or corrupt word boundaries while leaving 3–5gram character shingles intact.

---

## 2. Problem (b): Degenerate CLIP Rows in 12-Cell Regime Grid

### Manuscript / Table Issue
In `results/v3/headline_regime_grid_v3.csv` and Table II (`tab:regime_grid_v3`), the CLIP Visual row reports:
`MRR = 0.0436`, `CI = [0.044, 0.044]`, `Hit@10 = 0.0817`, `CI = [0.082, 0.082]`, `Mean_Latency_ms = 12.5` across **all 12 cells** (every combination of verbatim/light/heavy and clean/blur/syn/gt).

### Code Path & Root Cause
- **Code Path**: `ai_engine/rag/scripts/run_phase3_reviewer_evidence_v3.py` lines 528–532:
  ```python
  # 6. CLIP Visual
  r_clip = 250  # fixed nominal visual ranking on text-only receipts
  sys_mrrs["CLIP Visual"].append(0.0436)
  sys_h10s["CLIP Visual"].append(0.0817)
  sys_lats["CLIP Visual"].append(12.5)
  ```
- **Root Cause**: CLIP Visual was never actually queried against receipt images in `run_phase3_reviewer_evidence_v3.py`. The script developer placed placeholder constants (`0.0436` MRR, `0.0817` Hit@10) with zero variance, producing degenerate zero-width confidence intervals. CLIP image embeddings and query text encoders must be executed genuinely across receipt images and COCO distractors, verified with an entropy check (per-query ranking entropy > 0).

---

## 3. Problem (c): Tuned Hybrid Collapsing to Dense MiniLM on Heavy Paraphrase

### Empirical Data in `results/v3/`
In `results/v3/headline_regime_grid_v3.csv`:
- **Heavy Paraphrase, Clean OCR**:
  - `Dense MiniLM (chunked)`: MRR = **0.2837**, Hit@10 = **0.4444**
  - `Tuned Hybrid`: MRR = **0.2837**, Hit@10 = **0.4444** ($\alpha^* = 1.0$)
  - `BM25 (Tuned)`: MRR = **0.9014**, Hit@10 = **0.9771**
- **Heavy Paraphrase, Synthetic 20% Noise**:
  - `Dense MiniLM (chunked)`: MRR = **0.1061**, Hit@10 = **0.2102**
  - `Tuned Hybrid`: MRR = **0.1061**, Hit@10 = **0.2102** ($\alpha^* = 1.0$)
  - `BM25 (Tuned)`: MRR = **0.4567**, Hit@10 = **0.6536**

### Code Path & Root Cause
- **Code Path**: `results/v3/dev_tuned_parameters_v3.json` lines 25–28 and 49–52:
  ```json
  "heavy:::clean_ocr": {
    "alpha": 1.0,
    "dev_mrr": 0.0253
  },
  "heavy:::syn_20": {
    "alpha": 1.0,
    "dev_mrr": 0.0113
  }
  ```
- **Root Cause**: `dev_tuned_parameters_v3.json` was generated on the original synthetic heavy paraphrase probes where BM25 collapsed to ~0.008 due to aggressive token removal, so dense retrieval had slightly higher (though near-zero) MRR (~0.025), leading the grid tuner to select $\alpha^* = 1.0$ (pure dense). When heavy paraphrases were redesigned into answerable factual questions (`results/v3/heavy_paraphrases_v3.json`), the tuning script was not re-run on Dev. Evaluating on Test with the stale $\alpha^* = 1.0$ forced Tuned Hybrid to completely discard BM25 (despite BM25 scoring 0.9014) and copy Dense MiniLM exactly.

---

## 4. Problem (d): Tesseract Clean CER 0.8376 / WER 1.0481 Discrepancy

### Empirical Inconsistency
In Section V-H and Table XI (`tab:engine_replication_v3`), Clean Tesseract OCR is reported with CER = **0.8376** and WER = **1.0481**. However, BM25 retrieval on Clean Tesseract achieves MRR = **0.7364** and Hit@10 = **0.9248**. If 84% of characters and 105% of words were transcription errors, BM25 could not possibly locate the correct receipts in over 92% of queries.

### Code Path & Root Cause
- **Code Path**: `ai_engine/rag/scripts/run_engine_and_dataset_replication_v3.py` lines 98–142 (`compute_cer_wer_line_aligned`):
  ```python
  if abs(len(gl) - len(hl)) > len(gl) * 0.70:
      continue
  ...
  if best_j != -1 and best_c_err < len(gl_chars) * 0.75:
      used_hyp.add(best_j)
  ...
  # Penalize all unmatched hypothesis lines as full insertions:
  for j, hl in enumerate(norm_hyp):
      if j not in used_hyp:
          total_c_err += len(hl)
          total_w_err += len(hl.split())
  ```
- **Root Cause**: Tesseract groups bounding boxes and wraps lines differently than SROIE ground truth lines (e.g. concatenating multi-column text). Because `abs(len(gl) - len(hl)) > len(gl) * 0.70` filtered out these lines, valid recognized text was rejected from alignment. Every unmatched ground truth line was counted as 100% deletions, and every unmatched Tesseract line was counted as 100% insertions, artificially inflating CER and WER beyond 100%. Recomputing with standard document-level Levenshtein edit distance and `jiwer` after canonical normalization will restore true transcription error rates.

---

## 5. Problem (e): Conformal Cascade Risk Formulation and Finite-Sample Coverage

### Empirical Issue
In Table tab:conformal_cascade_v3 and `results/v3/conformal_splits_summary_v3.csv`:
- `Conformal (Empirical Calibration)`: Mean test miss rate = **0.0579** [0.0555, 0.0603], empirical coverage = **36.5%**.
- `Conformal ((n+1) Quantile Adjusted)`: Mean test miss rate = **0.0574** [0.0550, 0.0598], empirical coverage = **37.0%**.
- `Conformal (Learn-then-Test 95% UCB)`: Mean test miss rate = **0.0529** [0.0513, 0.0545], empirical coverage = **37.0%**.
All 95% confidence intervals for test miss rate strictly exclude the nominal target $\epsilon = 0.05$.

### Code Path & Root Cause
- **Code Path**: `ai_engine/rag/scripts/run_conformal_cascade_v2.py` lines 384–385 and 404–405:
  ```python
  misses = int(np.sum(1 - cal_s1_h1[non_esc_mask]))
  risk_emp = misses / n_non_esc  # CONDITIONAL RATIO!
  ```
- **Root Cause**: The script formulated risk as a *conditional ratio*:
  $$R_{\text{cond}}(\tau) = \mathbb{P}(\text{Stage 1 Miss} \mid \text{Not Escalated}) = \frac{\sum_{i=1}^n 1[M_i \ge \tau \land Y_i \ne 1]}{\sum_{i=1}^n 1[M_i \ge \tau]}$$
  Standard split conformal prediction bounds the *marginal expected loss*:
  $$L(X, Y, \tau) = 1[M \ge \tau \land Y \ne 1], \quad R_{\text{marg}}(\tau) = \mathbb{E}[L(X, Y, \tau)] \le \alpha$$
  Because the denominator $n_{\text{non\_esc}}$ is itself a random variable that shrinks as $\tau$ increases, standard quantile correction on calibration splits does not control the conditional ratio. Defining loss as the marginal loss and calibrating $\lambda$ with the exact finite-sample rule $((n \hat{R} + 1)/(n+1) \le \alpha)$ or Learn-then-Test guarantees valid marginal risk control.

---

## 6. Problem (f): Extractive vs. LLM Multi-Hop Bridging Identical Outputs

### Empirical Issue
`results/v3/multihop_paired_results.csv` and Section V-G reported Grounded Extractive Bridging and Generative LLM Bridging both scoring exactly $0.640$ on multi-hop and $0.820$ overall.

### Code Path & Root Cause
- **Code Path**: `ai_engine/rag/scripts/run_multihop_equivalence_v2.py` lines 149–150:
  ```python
  extractive_bridge_all = np.copy(llm_bridge_all)
  extractive_bridge_mh = np.copy(llm_bridge_mh)
  ```
  and `run_journal_revision_experiments.py` line 1952.
- **Root Cause**: The extractive bridging outputs were created by copying the LLM bridge array using `np.copy()`. The extractive pseudo-relevance feedback bridging algorithm was never executed independently end-to-end on the 100 queries. An independent extractive pipeline must be evaluated and compared per-query against LLM bridging.

---

## 7. Problem (g): Number Discrepancies Across Tables

### Empirical Issue
The same system (e.g. Word BM25 on clean text) appears with differing numbers across tables:
- `0.8572`: Table tab:regime_grid_v3 (Verbatim, Ground Truth text, $N=306$, 813 distractors).
- `0.8371`: Table tab:regime_grid_v3 (Verbatim, Clean Apple Vision OCR, $N=306$, 813 distractors).
- `0.8324`: Table tab:practical_stack & tab:engine_replication_v3 (Verbatim, Clean OCR, seed-averaged).
- `0.8317`: Early v2 pooled split.
- `0.9455`: Table tab:tier2_physical (Scene-text Set 2, $N=55$ on 191 album items).

### Code Path & Root Cause
- **Code Path**: Different scripts pooled different subsets of probes and distractors without maintaining a unified provenance mapping.
- **Resolution**: Generate `results/v6/number_provenance.csv` mapping every single number in the manuscript to its exact source file, probe count, query type, condition, and distractor pool.

---

## 8. Problem (h): ColQwen2 Decision Guide Row & Headline Definition

### Empirical Issue
1. Table tab:decision_guide included a row: `Document Page | Clean Real OCR | Light Paraphrase | ColQwen2 Document-Native | 0.4580 [0.408, 0.509] | 38.40`, but no experimental table in Section V presented the underlying ColQwen2 run.
2. The headline claim "semantic query mismatch is 18x worse than OCR noise" compared verbatim lookups to light paraphrasing, but when evaluated on answerable question queries, BM25 achieved MRR = **0.9014** (higher than clean OCR verbatim 0.8371).

### Code Path & Root Cause
- **Code Path**: `IEEE_Paper_MemoryVerse.tex` Table tab:decision_guide and `results/v5/native_visual_receipts_results.csv`.
- **Resolution**: ColQwen2 requires GPU execution (Phase 3 on Kaggle). In Phase 1, ColQwen2 is marked as pending Phase 3 or deleted from CPU-only fragments. The headline claim must be formulated rigorously across matched effect-size ranges and query styles.

---

## 9. Problem (i): Dense Bi-Encoder Baseline on Ground-Truth Verbatim Spans

### Empirical Inconsistency
In `headline_regime_grid_v3.csv`, `Dense MiniLM (chunked)` scores only **0.2100** MRR on **clean ground-truth text** for verbatim queries. Since ground-truth text has 0% character corruption and 0% OCR error, the collapse cannot be attributed solely to "tokenization fragmentation and noise drift."

### Code Path & Root Cause
- **Code Path**: `ai_engine/rag/scripts/run_phase3_reviewer_evidence_v3.py` lines 441 & 503.
- **Root Cause**: Dense bi-encoders (e.g. MiniLM, BGE) are trained on semantic search (MS-MARCO, natural questions, passage retrieval). Verbatim receipt probes are short entity snippets (e.g., store names, tax amounts, dates) without sentential context. Dense encoders map short lexical tokens poorly compared to full questions or natural sentences. To decouple "dense is poor at snippet lookup" from "OCR noise breaks dense representations", we must evaluate dense models on clean ground-truth text with:
  (1) Verbatim short spans
  (2) Question-style queries
  (3) Full sentential queries with context
  (4) Length-controlled padded spans.
