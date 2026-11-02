# N-04 Diagnostic Report: FUNSD EasyOCR Blur Formula Discrepancy and Tie-Handling Analysis

## 1. Executive Summary
- **Issue:** The footnote of Table VII(b) claims that full-collection MRR is mathematically capped by $0.322 \times \text{MRR}_{\text{non-blank}} \approx 0.02\text{--}0.05$, citing non-blank MRR of $0.1412$ (Word) and $0.1784$ (Char). However:
  - $0.322 \times 0.1412 = 0.0455$, but the table displays **0.0227** (Word BM25).
  - $0.322 \times 0.1784 = 0.0574$, but the table displays **0.0462** (Char BM25).
- **Diagnosis & Root Cause:**
  1. **Optical Detection Cliff / Blank Rate:**
     - Across the entire FUNSD dataset under optical blur, CRAFT text detection in EasyOCR failed completely on **67.8%** of forms (135 / 199 forms yielded 0 characters).
     - On the held-out test evaluation probes ($N=99$), **70.7%** of target forms (70 / 99) were 100% blank (0 characters detected). Only **29.3%** (29 / 99) had non-empty OCR transcriptions.
  2. **Evaluation & Tie-Breaking Artifact in `run_funsd_easyocr_replication.py`:**
     - In `run_funsd_easyocr_replication.py`, ranking was computed via `ranks = np.argsort(-scores)`.
     - When a target form was blank, its score was `0.0`. Under numpy's stable `argsort`, documents with score `0.0` were ordered by their array index in the corpus (0..598), rather than receiving an unretrieved zero-rank or true average tied rank.
     - For Word BM25, this stable index ordering produced the exact number **0.0227** printed in Table VII(b).
     - For Char BM25, average tie-breaking over the 600-item pool produced **0.0463** (reported as 0.0462).
  3. **Strict Information Retrieval Metrics:**
     - Under the standard IR rule where an unretrieved document (score $\le 0.0$) receives $\text{RR} = 0.0$:
       - **Word BM25:**
         - Non-blank subset ($N=29$): $\text{MRR} = 0.0345$
         - Full collection ($N=99$): $\text{MRR} = 0.0101$
         - Exact formula: $\frac{29}{99} \times 0.0345 = 0.0101$.
       - **Char BM25 (3-5gram):**
         - Non-blank subset ($N=29$): $\text{MRR} = 0.1492$
         - Full collection ($N=99$): $\text{MRR} = 0.0437$
         - Exact formula: $\frac{29}{99} \times 0.1492 = 0.0437$.
     - Under average tie-breaking over tied zero-scoring documents:
       - Word BM25 full collection: $\text{MRR} = 0.0132$
       - Char BM25 full collection: $\text{MRR} = 0.0463$.

## 2. Recommendation and Proposed Fix (Phase 2/4/5)
- **Do not present EasyOCR Blur as a retrieval robustness finding.** It is purely an upstream OCR pipeline detection failure (CRAFT text detector threshold collapse on blurred administrative fonts).
- **Options for Table VII(b):**
  - **Option A (Recommended per N-04):** Recompute the column using strict IR unretrieved evaluation ($\text{RR}=0$ for zero scores: Word BM25 = 0.0101, Char BM25 = 0.0437) or report the non-blank subset clearly, and explicitly label it as an OCR pipeline catastrophic failure cliff, not an index robustness comparison.
  - **Option B:** Drop the `EasyOCR Blur` column entirely from Table VII(b), keeping `EasyOCR Clean` (which proves cross-engine replication) and focusing blur degradation analysis on Tesseract and Apple Vision where OCR engines did not suffer 70% blank detection failure.
- **Decision Needed from User:** Choose Option A (recomputed numbers + disclaimer) or Option B (drop EasyOCR Blur column). Marked for Phase 4 decision.

## 3. Impact on Headline Claims
- Does this weaken or reverse a headline claim? **No.** The paper's primary claims rely on SROIE receipts (Table II) and Tesseract/Apple Vision on FUNSD. The failure of CRAFT/EasyOCR under optical blur was already treated as an optical detection cliff in the footnote.

