# Configuration Log: Forking-Paths Accounting (v3)

**Tracking File for Evaluation Configurations, Hyperparameter Sweeps, and Engine Variants**

| Run ID | Timestamp | Script | Component / Parameter | Values Evaluated | Split Used | Selection Criterion | Frozen Output Path | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CFG-001** | 2026-10-25T13:20:00 | Phase 0 | Random Baseline | Uniform Random Ranking ($N=813, N=1,000$) | Analytical | Expected MRR $H_N/N$ | `AUDIT.md` | Expected MRR 0.00895 ($N=813$), 0.00749 ($N=1,000$) |
| **CFG-002** | 2026-10-25T13:36:00 | Phase 1.1 | OCR Normalization & Alignment | (1) Legacy Unnormalized, (2) Normalized Full, (3) Normalized Line-Aligned | SROIE Corpus ($N=626$) | Methodological Sanity | `results/v3/ocr_error_rates.csv` | Corrected clean CER is 0.1387 (vs 0.3938 legacy); degraded CER is 0.3086 (vs 0.4760 legacy) |
| **CFG-003** | 2026-10-25T13:37:00 | Phase 1.2 | Dense Truncation & Chunking | Standard 512-limit vs Chunked MaxP ($\le 150$ words window) | Held-out Test ($N=306$ probes, $813$ docs) | Decision Rule D1.2 | `results/v3/dense_truncation_impact.csv` | 0.64% receipts truncated; MaxP MRR identical to unchunked (MiniLM: 0.2030 vs 0.2056, BGE: 0.3028 vs 0.3022) |
| **CFG-004** | 2026-10-25T13:43:00 | Phase 1.3 | Apple Vision OCR Settings | Level: accurate vs fast; LanguageCorrection: True vs False | SROIE Corpus ($N=626$) & Test Split | Decision Rule D1.3 | `results/v3/ocr_engine_settings.csv` | Fast level severely degrades (CER 0.5137); language correction flag does not alter AUROC inversion (~0.23) |
| **CFG-005** | 2026-10-25T13:45:00 | Phase 1.4 | Heavy Paraphrase Redesign | Answerable question queries via `openai/gpt-oss-120b` (T=0.0) | Held-out Test ($N=306$ probes, $813$ docs) | Decision Rule D1.4 | `results/v3/heavy_redesign_results.csv` | Replaced chance queries (0.0089) with answerable queries; BM25 MRR 0.8878, MiniLM 0.2873, BGE 0.5191 |
