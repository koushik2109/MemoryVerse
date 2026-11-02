# N-02 Diagnosis Report: Table IV Truncation and Stream Naming Bug

## Executive Summary
The reviewer's suspicion is **100% confirmed**. There is a dual semantic-relabeling and configuration bug in Table IV:
1. **The Truncation Claim is Spurious:** Standalone Word BM25 does *not* drop from 0.8371 to 0.8069 due to $K_1=10$ truncation. In `results/v2/ablation_results.csv`, `BM25 Alone` achieves **0.8107** on Real-OCR (not 0.8069). Furthermore, with Hit@10 = 0.9444, truncation mathematically cannot cost 0.030 MRR.
2. **The "Clean Stream" Relabeling Bug:**
   - In the underlying code (`ai_engine/rag/scripts/run_ablation_v2.py`, lines 10, 538–540), "Clean Stream" was defined as the **CLIP visual pixel feature stream** (`s_c = np.dot(test_clip_img_embs, test_clip_q)`), which was hypothesized to provide an uncorrupted visual proxy when text OCR is noisy.
   - In LaTeX drafting, `BM25 + Clean Stream` was mistakenly renamed to `BM25 First-Stage Stream Only ($K_1=10$)`.
   - `+ Clean Stream` (which was `Tuned Hybrid + CLIP Clean Stream`) was mistakenly renamed to `Oracle Ground-Truth Text Stream ($K_1=10$)`.
   - Because both evaluated the corrupted text stream (Apple Clean OCR or Optical Blur OCR) blended with CLIP visual scores (`gamma_star * s_c`), they varied across noise columns and were identical in 3 of 4 columns (0.8317, 0.8069, 0.6702) because CLIP weights on text queries are near zero.
   - An actual Oracle Ground-Truth Text stream would evaluate pristine text and would be constant ($0.8572$ unconstrained or $0.8550$ truncated) across all noise columns.

## Source Code Evidence
- File: `ai_engine/rag/scripts/run_ablation_v2.py`
  - Line 10: `4. Clean Stream (CLIP visual pixel stream)`
  - Line 538: `s_hyb_clean = min_max_normalize(s_hybrid) + gamma_hyb_star * s_c`
  - Line 540: `s_bm25_clean = s_b + gamma_bm25_star * s_c`
  - Line 549: `s_bm25_alone = s_b` (achieved 0.8107 on real_ocr)
- File: `results/v2/ablation_results.csv`:
  - `clean, BM25 + Clean Stream`: 0.8317
  - `clean, + Clean Stream`: 0.8299
  - `syn_20, BM25 + Clean Stream`: 0.3694
  - `syn_20, + Clean Stream`: 0.3694
  - `real_ocr, BM25 + Clean Stream`: 0.8069
  - `real_ocr, + Clean Stream`: 0.8121
  - `real_ocr_deg, BM25 + Clean Stream`: 0.6702
  - `real_ocr_deg, + Clean Stream`: 0.6684

## Proposed Fix (for Phase 2 / R-01)
1. **Rerun / Re-tabulate Table IV:**
   - Clearly delineate true standalone streams:
     - `Word BM25 (Standalone)`: Clean 0.8317, Syn 20% 0.3694, Real-OCR 0.8107, Real-Deg 0.6684.
     - `Tuned Hybrid (Standalone)`: Clean 0.8299, Syn 20% 0.3694, Real-OCR 0.8121, Real-Deg 0.6684.
     - `True Oracle Ground-Truth Text`: constant across all columns (0.8317 at K=10, 0.8572 unconstrained).
     - `Visual Pixel Stream Addition (+ CLIP Visual)`: correctly labelled as multimodal fusion with pixel stream, not "Oracle Ground-Truth Text".
2. **Revise Manuscript Prose:**
   - Remove the false claim that truncation alone drops BM25 from 0.8371 to 0.8069.
   - Explain that $K_1=10$ candidate truncation causes negligible loss (0.8371 $\to$ 0.8320, $\le 0.005$ MRR).

