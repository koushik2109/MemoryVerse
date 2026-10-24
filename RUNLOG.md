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

### Phase 1: Public Scene-Text and Document Benchmark Construction
- **Timestamp:** 2026-10-25T00:45:00+05:30
- **Action:** Acquired public ICDAR 2019 SROIE dataset (626 receipt images, bounding box text files, and key entity JSONs).
- **Tooling:** Built and compiled native Swift Apple Vision OCR binary (`run_ocr_apple_vision`) using `swiftc -O` targeting Vision Framework `VNRecognizeTextRequest` (.accurate mode).
- **Execution:** Executed Apple Vision OCR across all 626 images in 80.23s (0.128s/image).
  - Measured Clean OCR Mean CER: `0.4519`
  - Measured Clean OCR Mean WER: `0.6503`
- **Probe Extraction:** Extracted 612 unique single-target factual probes (verified that each probe appears in exactly one target item across the corpus).
- **Corpus Pooling:** Combined 626 SROIE receipt records with 1,000 MS-COCO image distractors (`coco_1k_images`) to form a 1,626-item distractor-rich multimodal corpus.
- **Split Execution:** Strictly partitioned 50/50 by item ID (Seed 42): Dev (813 docs, 306 probes), Test (813 docs, 306 probes).
- **Dataset Frozen:** Saved to `ai_engine/rag/data/v2/public_benchmark_v2.json`.

### Phase 2.1: Paraphrase Generation & Freezing
- **Timestamp:** 2026-10-25T01:03:30+05:30
- **Model:** `gpt-4o-mini` (temperature = 0.0) via HTTP API.
- **Generation:** 612 probes across Dev and Test splits with automated token overlap constraints:
  - Verbatim: original query text
  - Light Paraphrase: target 30%-50% token overlap, preserving key numeric/entity identifiers. Observed Mean: `41.2%`.
  - Heavy Paraphrase: target <15% token overlap, conceptual rephrasing with zero rare entity copying. Observed Mean: `0.1%`.
- **Freezing:** Saved `ai_engine/rag/data/v2/frozen_probes_with_paraphrases.json` (SHA-256 in MANIFEST.json).
- **Audit Sample:** Generated `ai_engine/rag/data/v2/paraphrase_audit_sample.csv` (61 rows = 10% sample) for human inspection.

### Phase 2.2: Degraded Real-OCR Variant
- **Timestamp:** 2026-10-25T01:04:40+05:30
- **Degradation Pipeline:** Applied physical corruptions to all 626 SROIE images before OCR:
  - Gaussian blur (radius = 1.5)
  - 50% spatial downscale (bilinear)
  - Deterministic rotation +/- 4 degrees with white fill
  - JPEG compression artifact (quality = 20)
- **OCR Execution:** Apple Vision OCR on degraded images completed in 47.91s (0.077s/img).
  - Measured Degraded OCR Mean CER: `0.5218`
  - Measured Degraded OCR Mean WER: `0.7561`
- **Output:** Saved to `ai_engine/rag/data/v2/sroie_degraded_ocr.json` (SHA-256 in MANIFEST.json).

