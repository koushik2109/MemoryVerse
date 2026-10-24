# Execution Runlog (RUNLOG.md)

## System & Environment
- **Platform:** macOS-27.2-arm64-arm-64bit-Mach-O (Apple Silicon)
- **Acceleration:** Apple Silicon MPS (`torch.backends.mps.is_available() == True`)
- **Python Version:** 3.14.7 (v3.14.7:823f0323ee6)
- **PyTorch Version:** 2.13.0
- **Transformers Version:** 5.15.1
- **Sentence-Transformers Version:** 6.0.0
- **Random Seeds:** 42, 43, 44 everywhere for multi-seed evaluations.

## Execution Log

### Phase 0: Setup and Audit
- **Timestamp:** 2026-10-25T00:36:52+05:30
- **Action:** Executed 3-number reproduction test on 97-item held-out test split (55 probes):
  - BM25 Clean MRR: `0.9818` (Observed: `0.9818`, Diff: `0.0000`, Match: PASS)
  - Dense MiniLM Clean MRR: `0.8527` (Observed: `0.8527`, Diff: `0.0000`, Match: PASS)
  - Hybrid Fixed Clean MRR: `0.9909` (Observed: `0.9909`, Diff: `0.0000`, Match: PASS)
- **Audit Documentation:** Created `AUDIT.md` detailing datasets, 50/50 split logic by item ID, parameter tuning locus ($\alpha, \beta, \gamma, \tau$), and probe provenance (85 private album probes vs. public COCO/ESC-50).
- **Statistical Library:** Implemented `ai_engine/rag/eval_lib.py` with Hit@K, MRR, NDCG@K, Clopper-Pearson binomial CIs, cluster bootstrap CIs ($B=2,000$), paired permutation tests ($N_{\text{perm}}=10,000$), and step-down Holm-Bonferroni correction.
- **Unit Testing:** Created `ai_engine/rag/tests/test_eval_lib.py`; all 8 unit tests passed in 0.50s.
- **Pre-Registration:** Authored and committed `PREREG_ADDENDUM.md` prior to any Phase 1/2 experimental data generation or results.
