# R-03: FUNSD EasyOCR Blur Retrieval Recomputation and Audit

## 1. Overview and Problem Diagnosis
In item **N-04**, an empirical discrepancy was identified in the footnote of Table VII(b):
The footnote claimed that full-collection MRR is capped by $0.322 \times \text{MRR}_{\text{non-blank}} \approx 0.02\text{--}0.05$, citing non-blank MRR of $0.1412$ (Word) and $0.1784$ (Char). However:
- $0.322 \times 0.1412 = 0.0455 \ne 0.0227$ (reported Word BM25).
- $0.322 \times 0.1784 = 0.0574 \ne 0.0462$ (reported Char BM25).

Investigation traced the root cause to two factors:
1. **Optical Detection Cliff:** EasyOCR 1.7.2's CRAFT detector collapsed on blurred administrative fonts, returning 0 characters on 70.7% of the test targets (70 of 99 targets had zero characters).
2. **Evaluation Tie-Breaking Bug:** In `scripts/run_funsd_easyocr_replication.py`, ranking was computed using `np.argsort(-scores)`. Because target documents occupied corpus indices 0..99 while 500 distractors occupied indices 100..599, blank documents (with score 0.0) were placed ahead of the 500 distractors due to numpy's stable sorting, receiving artificial ranks $\le 100$ and inflated MRRs.

## 2. Experimental Execution (R-03)
Script `scripts/rerun_funsd_easyocr_r03.py` was executed to recompute all metrics across:
- 5 Systems: BM25 (Tuned Word), BM25 (Char 3-5gram), Dense MiniLM, Dense BGE-base, Tuned Hybrid.
- 2 Query Regimes: Verbatim and Light Paraphrase.
- 3 Tie-Handling Rules:
  1. `original_stable`: Reproduces the historical artifact (Word BM25 = 0.0227, Char BM25 = 0.0462).
  2. `strict_ir`: Standard IR convention where unretrieved documents (score $\le 0.0$) receive $\text{RR} = 0.0$ and $\text{Hit@10} = 0.0$.
  3. `average_ties`: Average fractional rank over all tied documents in the 600-item pool.
- Subset Breakdown: Full collection ($N=99$), Non-blank targets ($N=29$, 29.3%), and Blank targets ($N=70$, 70.7%).

## 3. Results Summary

### Table 1: EasyOCR Blur under Verbatim Queries ($N=99$)
| System | Original Stable (Historical) | Strict IR Unretrieved (Standard) [95% CI] | Average Tie-Breaking | Non-Blank Targets ($N=29$) | Blank Targets ($N=70$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Tuned Word)** | 0.0227 | **0.0101** [0.000, 0.030] | 0.0132 | 0.0345 | 0.0000 |
| **BM25 (Char 3-5g)** | 0.0462 | **0.0437** [0.010, 0.086] | 0.0463 | 0.1492 | 0.0000 |
| **Dense MiniLM** | 0.0261 | **0.0247** [0.012, 0.047] | 0.0247 | 0.0569 | 0.0114 |
| **Dense BGE-base** | 0.0540 | **0.0435** [0.024, 0.069] | 0.0435 | 0.0950 | 0.0222 |
| **Tuned Hybrid** | 0.0408 | **0.0359** [0.019, 0.061] | 0.0359 | 0.0768 | 0.0189 |

### Table 2: EasyOCR Blur under Light Paraphrases ($N=99$)
| System | Original Stable (Historical) | Strict IR Unretrieved (Standard) [95% CI] | Average Tie-Breaking | Non-Blank Targets ($N=29$) | Blank Targets ($N=70$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Tuned Word)** | 0.0226 | **0.0202** [0.000, 0.051] | 0.0226 | 0.0690 | 0.0000 |
| **BM25 (Char 3-5g)** | 0.0251 | **0.0230** [0.000, 0.053] | 0.0250 | 0.0786 | 0.0000 |
| **Dense MiniLM** | 0.0316 | **0.0292** [0.015, 0.052] | 0.0292 | 0.0674 | 0.0135 |
| **Dense BGE-base** | 0.0562 | **0.0432** [0.020, 0.076] | 0.0432 | 0.0892 | 0.0242 |
| **Tuned Hybrid** | 0.0253 | **0.0252** [0.005, 0.056] | 0.0252 | 0.0723 | 0.0000 |

## 4. Key Takeaways
1. **Mathematical Consistency Restored:**
   Under Strict IR, the full-collection MRR exactly equals the weighted non-blank performance:
   $$\text{MRR}_{\text{all}} = \frac{29}{99} \times \text{MRR}_{\text{non-blank}} + \frac{70}{99} \times 0.0 = 0.2929 \times \text{MRR}_{\text{non-blank}}$$
   - Word BM25: $0.2929 \times 0.0345 = 0.0101$
   - Char BM25: $0.2929 \times 0.1492 = 0.0437$
2. **Clear Recommendation:**
   In Table VII(b), either:
   - Report the recomputed strict IR numbers: Word BM25 = 0.0101 [0.000, 0.030], Char BM25 = 0.0437 [0.010, 0.086], with the updated non-blank math ($0.293 \times 0.1492 = 0.0437$).
   - Or drop the `EasyOCR Blur` column as a pipeline detection cliff, focusing degradation on Tesseract and Apple Vision where OCR did not collapse.

