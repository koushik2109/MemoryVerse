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
### Phase 2.3 - 2.5: Multi-Stream Regime Analysis Grid Evaluation
- **Timestamp:** 2026-10-25T01:37:30+05:30
- **Script:** `ai_engine/rag/scripts/run_regime_grid_v2.py`
- **Total Wall-Clock Time:** 780.48s (13.0 min)
- **Evaluated Suite:**
  - 18 Cells: 3 query types (verbatim, light paraphrase, heavy paraphrase) x 6 noise regimes (clean, syn_10, syn_20, syn_40, real_ocr, real_ocr_deg).
  - 12 Retrieval Systems: BM25 (word), BM25 (char 3-5gram), Fuzzy Lexical, Dense MiniLM, Dense BGE-base, Dense Multilingual, Hybrid (tuned on Dev), All-Stream RRF, CLIP Visual, BM25 + CLIP, Hybrid + CLIP, RAMP-RAG.
  - Multi-Seed Evaluation: Seeds 42, 43, 44 on synthetic noise; 306 held-out test queries, 813 test items (pooled with COCO distractors).
  - Statistical Validation: Clustered bootstrap over target items ($B=2,000$), paired sign-flip permutation tests ($N_{\text{perm}}=10,000$) against Tuned Hybrid, and family-wise step-down Holm-Bonferroni correction.
- **Key Empirical Outputs:**
  - `results/v2/regime_grid_results.csv`
  - `results/v2/regime_grid_raw_test_scores.csv`
  - `results/v2/crossover_summary.csv`
  - `results/v2/regime_grid_heatmap.png`
  - `results/v2/mrr_vs_noise_by_query_type.png`
  - `latex_fragments/table_regime_grid.tex`

### Phase 4: Component Ablation & RAMP-RAG Architecture Re-Evaluation
- **Timestamp:** 2026-10-25T02:04:30+05:30
- **Script:** `ai_engine/rag/scripts/run_ablation_v2.py`
- **Total Wall-Clock Time:** 313.66s (5.2 min)
- **Evaluated Models:** `ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-base`.
- **Key Empirical Results:**
  - `+ Cascade (MiniLM)` gains $+0.0410$ MRR on Clean ($p_{\text{Holm}} = 0.0014$) and $+0.0339$ MRR on 20% Noise ($p_{\text{Holm}} = 0.0011$).
  - `+ Cascade (BGE-reranker)` achieves the highest overall retrieval accuracy across all conditions: Clean MRR `0.8821` ($p_{\text{Holm}} = 0.0011$), 20% Noise MRR `0.4154` ($p_{\text{Holm}} = 0.0011$), Real OCR MRR `0.8555` ($p_{\text{Holm}} = 0.0011$), Real Degraded OCR MRR `0.6969`.
  - `+ Reliability` degrades performance: 20% Noise MRR drops to `0.3574` vs. Baseline `0.3694` (diff CI `[-0.0222, -0.0025]`). H1 Decision Rule: **FAIL (REJECTED)**.
  - `+ Router` achieves `0.3765` MRR vs. `All-Stream RRF` `0.1827` MRR. H2 Decision Rule: **PASS**.
  - Observable Reliability AUROC: High on uniform synthetic character corruption (10% noise AUROC `0.9772`, 20% noise `0.9948`), but completely inverts on real optical blur/compression (`0.2674`), proving why heuristic reliability fails in realistic conditions.
- **Output Artifacts:**
  - `results/v2/ablation_results.csv`
  - `results/v2/reliability_auroc.csv`
  - `latex_fragments/table_ablation_v2.tex`

### Phase 3: Mechanism Analysis (Why Dense Text Proxies Collapse)
- **Timestamp:** 2026-10-25T02:10:15+05:30
- **Script:** `ai_engine/rag/scripts/run_mechanism_analysis_v2.py`
- **Total Wall-Clock Time:** 124.83s (2.1 min)
- **Tokenization Fragmentation:** WordPiece subwords/word escalates from `1.347` (clean) to `2.246` (20% noise) and `2.994` (40% noise, $+122\%$ fragmentation).
- **Embedding Drift:** Mean cosine similarity to clean embedding drops from `1.0000` to `0.4655` (MiniLM) and `0.7007` (BGE-base) at 20% noise, and to `0.2604` (MiniLM) and `0.5595` (BGE-base) at 40% noise.
- **Error Attribution:** Dense models collapse specifically on `Numeric-Heavy` (MiniLM MRR `0.033`) and `Short` probes (MRR `0.068`), while `BM25 (Char 3-5gram)` achieves the highest noise resilience across all property buckets (Numeric-Heavy MRR `0.415`, Short MRR `0.473`, Rare-Token MRR `0.643`, Named Entity MRR `0.504`).
- **Output Artifacts:**
  - `results/v2/mechanism_tokenization_fragmentation.csv`
  - `results/v2/mechanism_embedding_drift.csv`
  - `results/v2/mechanism_error_attribution.csv`
  - `results/v2/fig_tokenization_fragmentation.png`
  - `results/v2/fig_embedding_drift.png`
  - `results/v2/fig_error_attribution.png`
  - `MECHANISM.md`

### Phase 5: Conformal Cascade Calibration & Cost-Quality Pareto Frontier
- **Timestamp:** 2026-10-25T02:25:22+05:30
- **Script:** `ai_engine/rag/scripts/run_conformal_cascade_v2.py`
- **Total Wall-Clock Time:** 244.35s (4.1 min)
- **Workload:** N=2,754 pooled query instances (306 probe clusters x 3 query variants [verbatim, light paraphrase, heavy paraphrase] x 3 representation conditions [clean OCR, syn_20, real_ocr]).
- **Protocol:** 200 random cluster-stratified 50/50 splits (seeds 1000..1199).
- **Target Risk Bound:** Non-escalated miss rate $\epsilon \le 0.05$.
- **Key Empirical Results:**
  - Empirical Calibration: Calibrated $\tau^* = 0.384$. Mean held-out test miss rate = `0.0579` [5th: `0.0325`, 50th: `0.0538`, 95th: `0.0873`]. Split coverage $\le 0.05$: `36.5%`. Mean escalation rate: `89.5%` [5th: `86.9%`, 95th: `91.2%`]. Mean latency = `74.15 ms`. Test MRR = `0.3527`.
  - Conservative 95% UCB Calibration: Calibrated $\tau^* = 0.400$. Mean held-out test miss rate = `0.0529` [5th: `0.0325`, 50th: `0.0541`, 95th: `0.0703`]. Split coverage $\le 0.05$: `37.0%`. Mean escalation rate: `90.3%`. Mean latency = `74.81 ms`. Test MRR = `0.3522`.
  - Pareto Comparison: Never Escalate achieves MRR `0.3266` at `0.45 ms` (miss rate `0.7262`); Always Escalate achieves MRR `0.3485` at `82.82 ms` (0 miss rate). Conformal cascade saves ~10% latency vs Always Escalate while achieving higher MRR (`0.3527` vs `0.3485`) by avoiding cross-encoder false positives on obvious high-margin matches.
- **Output Artifacts:**
  - `results/v2/conformal_cascade_results.csv`
  - `results/v2/conformal_splits_summary.csv`
  - `results/v2/conformal_pareto_curve.csv`
  - `results/v2/fig_conformal_pareto.png`
  - `latex_fragments/table_conformal_v2.tex`

### Phase 7: COCO Karpathy-Test Native-Visual vs. Proxy Retrieval Benchmark
- **Timestamp:** 2026-10-25T02:32:37+05:30
- **Script:** `ai_engine/rag/scripts/run_coco_native_visual_v2.py`
- **Total Wall-Clock Time:** 33.58s
- **Corpus:** N=1,000 MS-COCO Karpathy-test images.
- **Queries:** N=1,000 held-out human descriptions (caption 3/4) from Karpathy test.
- **Key Empirical Results:**
  - `BM25 (BLIP Proxy)`: Hit@1 = `0.1220`, Hit@10 = `0.3840`, MRR = `0.2048`
  - `Dense MiniLM (BLIP Proxy)`: Hit@1 = `0.2420`, Hit@10 = `0.6530`, MRR = `0.3728`
  - `Dense BGE-base (BLIP Proxy)`: Hit@1 = `0.2430`, Hit@10 = `0.6460`, MRR = `0.3695`
  - `CLIP ViT-B/32 (Native Pixels)`: Hit@1 = `0.3390`, Hit@10 = `0.7570`, MRR = `0.4709`
  - `Hybrid (CLIP + BGE-base BLIP)`: Hit@1 = `0.3880`, Hit@10 = `0.7870`, MRR = `0.5186`
  - `Dense BGE-base (Gold Proxy Upper Bound)`: Hit@1 = `0.4580`, Hit@10 = `0.8320`, MRR = `0.5835`
- **Pre-Registered Hypothesis Testing (Native Pixels vs. Best Caption Proxy):**
  - Hit@1: Diff $+0.0960$ [$+0.0620, +0.1300$], $p_{\text{Holm}} = 0.00040$ -> Sig = **True**
  - Hit@5: Diff $+0.1020$ [$+0.0650, +0.1400$], $p_{\text{Holm}} = 0.00040$ -> Sig = **True**
  - Hit@10: Diff $+0.1040$ [$+0.0720, +0.1390$], $p_{\text{Holm}} = 0.00040$ -> Sig = **True**
  - MRR: Diff $+0.0981$ [$+0.0690, +0.1263$], $p_{\text{Holm}} = 0.00040$ -> Sig = **True**
- **Verdict:** **CONFIRMED (HOLDS ACROSS ALL METRICS)**. Native pixel retrieval decisively outperforms machine caption proxies across all metrics on the 1,000-image benchmark.
- **Output Artifacts:**
  - `results/v2/coco_native_visual_results.csv`
  - `results/v2/coco_hypothesis_test.csv`
  - `latex_fragments/table_coco_v2.tex`

### Phase 6: Multi-Hop Equivalence Testing via Two One-Sided Tests (TOST)
- **Timestamp:** 2026-10-25T02:23:31+05:30
- **Script:** `ai_engine/rag/scripts/run_multihop_equivalence_v2.py`
- **Total Wall-Clock Time:** 0.01s
- **Dataset:** 100 diagnostic queries (50 Single-Hop, 50 Multi-Hop).
- **Equivalence Margin:** $\Delta = \pm 0.05$ TargetHit@10 at $\alpha = 0.05$ (90% Confidence Interval).
- **Key Empirical Results:**
  - Grounded Extractive Bridge vs. Generative LLM Bridge (Multi-Hop Only, N=50): Extractive Hit@10 = `0.640`, LLM Hit@10 = `0.640`. Paired difference $\bar{D} = +0.0000$ (SE = 0.0000). $90\%$ Equivalence CI = `[+0.0000, +0.0000]`. $p_{\text{TOST}} = 0.00000 < 0.05$. Verdict: **EQUIVALENT**.
  - Grounded Extractive Bridge vs. Generative LLM Bridge (All Queries, N=100): Extractive Hit@10 = `0.820`, LLM Hit@10 = `0.820`. Paired difference $\bar{D} = +0.0000$ (SE = 0.0000). $90\%$ Equivalence CI = `[+0.0000, +0.0000]`. $p_{\text{TOST}} = 0.00000 < 0.05$. Verdict: **EQUIVALENT**.
  - Baseline Hybrid (No Bridge) vs. LLM Bridge (Multi-Hop, N=50): Paired difference $\bar{D} = -0.0200$, $90\%$ CI `[-0.0785, +0.0385]`, $p_{\text{TOST}} = 0.19694$. Verdict: **INCONCLUSIVE** (extends beyond $-0.05$).
- **Output Artifacts:**
  - `results/v2/multihop_equivalence_results.csv`
  - `latex_fragments/table_multihop_v2.tex`

### Phase 8: Deliverables & Reproduction Harness
- **Timestamp:** 2026-10-25T02:35:00+05:30
- **Master Reproduction Script:** Created `reproduce.sh` with automated verification of frozen datasets and sequential execution of all Phase 2--7 scripts.
- **Verification:** Audited SHA-256 hashes of all 26 outputs across `results/v2/` and `latex_fragments/` into `MANIFEST.json`.

---

## Phase 0 & Phase 1 Debug Runlog (v3)

### Phase 0: Pipeline Audit & Reproduction Check
- **Timestamp:** 2026-10-25T13:18:00+05:30
- **Actions:**
  - Audited pipeline end-to-end; wrote `AUDIT.md` documenting root causes of CER/WER mismatch, TOST array copying artifact, conformal calibration risk definition, and heavy paraphrase chance performance.
  - Successfully reproduced 5/5 target numbers from `results/v2/`:
    1. Verbatim BM25 clean MRR: `0.8317` (Matches 0.832)
    2. Verbatim Dense MiniLM clean MRR: `0.2100` (Matches 0.210)
    3. Light Paraphrase BM25 clean MRR: `0.4010` (Matches 0.401)
    4. COCO Native CLIP MRR: `0.4709` (Matches 0.4709)
    5. Conformal Test Miss Rate: `0.0579` (Matches 0.0579)
  - Compiled `CLAIMS_LEDGER.csv` auditing 17 claims across Abstract, Intro, Sec IV/V, and Conclusion.
  - Pre-registered hypotheses and decision rules in `PREREG_V3.md`.
  - Computed analytical random-ranking baselines: $\mathbb{E}[\text{MRR}] = 0.00895$ ($N=813$), $0.00749$ ($N=1,000$).

### Phase 1: Debugging and Empirical Verification
- **Timestamp:** 2026-10-25T13:45:00+05:30
- **Script:** `ai_engine/rag/scripts/run_phase1_debug_v3.py`
- **Total Wall-Clock Time:** 6m 12s
- **Sub-Experiment 1.1 (CER/WER Sanity):**
  - Evaluated on $N=626$ SROIE receipts:
    - Clean Apple Vision: Legacy Unnorm CER `0.3938` / WER `0.5796` -> Normalized Line-Aligned CER `0.1387` / WER `0.2165`.
    - Degraded Optical: Legacy Unnorm CER `0.4760` / WER `0.7052` -> Normalized Line-Aligned CER `0.3086` / WER `0.4595`.
  - Saved 20 side-by-side inspection pairs to `results/v3/ocr_samples.md` and error rates to `results/v3/ocr_error_rates.csv`.
- **Sub-Experiment 1.2 (Dense Truncation & Chunked Retrieval):**
  - Token lengths measured under MiniLM and BGE tokenizers: only 4/626 receipts (0.64%) exceed 512 tokens; mean tokens lost is 0.032%.
  - Chunked MaxP retrieval evaluated on held-out Test split ($N=306$ probes, $N=813$ docs):
    - MiniLM: Unchunked MRR `0.2056` vs Chunked MaxP `0.2030`.
    - BGE-base: Unchunked MRR `0.3022` vs Chunked MaxP `0.3028`.
    - Numeric-heavy probes: MiniLM Unchunked `0.0790` vs Chunked `0.0770` (BM25: `0.7668`).
  - Saved to `results/v3/dense_truncation_impact.csv`.
- **Sub-Experiment 1.3 (Apple Vision OCR Configurations):**
  - Compiled native Swift worker `run_ocr_apple_vision_v3`.
  - Evaluated 4 configurations (`accurate`/`fast` $\times$ `languageCorrection` True/False):
    - Fast level severely degrades accuracy (CER jumps from 0.1386 to 0.5137; BM25 MRR drops from 0.8107 to 0.5063).
    - Language correction flag does NOT resolve the AUROC inversion (AUROC remains ~0.23 under both True and False).
  - Saved to `results/v3/ocr_engine_settings.csv`.
- **Sub-Experiment 1.4 (Heavy Paraphrase Redesign):**
  - Generated answerable question queries using `openai/gpt-oss-120b` ($T=0.0$) via Groq API.
  - Filtered with uniqueness checking across all 626 SROIE receipts (302 retained, 4 non-unique dropped).
  - Saved 50 audit samples to `results/v3/heavy_audit_sample.csv` and frozen queries to `results/v3/heavy_paraphrases_v3.json`.
  - Evaluated retrieval MRR: BM25 `0.8878`, MiniLM `0.2873`, BGE-base `0.5191` (vs legacy chance `0.0089`, random baseline `0.0089`).
  - Saved to `results/v3/heavy_redesign_results.csv`.

### Phase 2: Empirical Fixes and Retractions (v3)
- **Timestamp:** 2026-10-25T14:18:00+05:30
- **Script:** `ai_engine/rag/scripts/run_phase2_fixes_v3.py`
- **Total Wall-Clock Time:** 6.90s (cached pooled queries)
- **Sub-Experiment 2.1 (Conformal Cascade Finite-Sample Re-evaluation):**
  - Evaluated on $N=2{,}754$ pooled queries across 200 random cluster-stratified 50/50 splits (seeds 1000..1199):
    - Empirical Calibration: Test Miss Rate `0.0579` [0.0560, 0.0598], Coverage `36.5%` [29.8%, 43.6%], Escalation `89.5%`, MRR `0.3527`.
    - $(n+1)$ Quantile Adjusted: Test Miss Rate `0.0574`, Coverage `37.0%`, Escalation `89.6%`.
    - Learn-then-Test (95% Clopper-Pearson UCB): Test Miss Rate `0.0529`, Coverage `37.0%`, Escalation `90.3%`.
    - Fixed Threshold Equal Escalation ($\tau = 0.380$): Test Miss Rate `0.0559`, Coverage `30.0%`, Escalation `89.5%`, MRR `0.3526`.
  - Refutes strict guarantee claim: test miss rate exceeds 0.05 in ~63% of splits; simple fixed threshold matches MRR and latency at equal escalation.
  - Outputs: `results/v3/conformal_splits_summary_v3.csv`, `results/v3/conformal_cascade_v3.csv`.
- **Sub-Experiment 2.2 (Multi-Hop TOST Artifact Removal):**
  - Evaluated on 50 diagnostic multi-hop queries:
    - Replaced TOST on copied arrays with exact discordant analysis and McNemar tests:
      - Proposed MM-RAG vs Hybrid (No Multi-Hop): Proposed hits on 2 queries (`mh_35`, `mh_50`) and misses on 1 (`mh_25`); net diff +1/50 (+0.0200), exact McNemar $p=1.0000$.
      - Proposed MM-RAG vs Single-Stage Dense: Identical 32/50 hits (net diff 0.0000), exact McNemar $p=1.0000$.
  - Output: `results/v3/multihop_paired_results.csv`.
- **Sub-Experiment 2.3 (H2 Query Router vs. Fair Baselines):**
  - Evaluated Router against Tuned Hybrid and All-Stream RRF across 4 noise regimes with 100,000 sign flips:
    - 20% noise: Router (0.3765) adds only $+0.0071$ over Tuned Hybrid (0.3694, $p_{\text{Holm}}=0.042$).
    - Clean OCR: Diff $+0.0010$ ($p_{\text{Holm}}=1.0$). Real OCR: Diff $0.0000$ ($p_{\text{Holm}}=1.0$). Real Degraded OCR: Diff $-0.0017$ ($p_{\text{Holm}}=1.0$).
  - Reconciles Section IV-I and V-D: Section V-D's claim of +0.1938 was an artifact of comparing to unnormalized strawman All-Stream RRF (0.1827). H2 is KILLED.
  - Output: `results/v3/router_evaluation_v3.csv`.
- **Sub-Experiment 2.4 (Light-Paraphrase Paired Tests):**
  - Evaluated Tuned Hybrid, BM25+CLIP, and Hybrid+CLIP vs BM25 across 6 noise regimes with 100,000 sign flips:
    - Traced unsourced "0.125" figure to Hybrid+CLIP (`0.1249`).
    - On syn_20, BM25 collapses to `0.0875` [0.0713, 0.1055], while Tuned Hybrid reaches `0.1110` (diff $+0.0235$, $p_{\text{Holm}}=0.00176$), Hybrid+CLIP reaches `0.1249` (diff $+0.0374$, $p_{\text{Holm}}=0.00024$), and BM25+CLIP reaches `0.1324` (diff $+0.0449$, $p_{\text{Holm}}=0.00024$).
  - Output: `results/v3/light_paraphrase_paired_tests.csv`.
- **Sub-Experiment 2.5 (Char n-gram vs. Word BM25 Boundary):**
  - Evaluated Word BM25 vs Char (3-5)gram BM25 across 6 noise regimes $\times$ 2 query types with 100,000 sign flips:
    - Verbatim 20% synthetic noise: Char BM25 (0.4703) beats Word BM25 (0.3694) by $+0.1008$ ($p_{\text{Holm}}=0.00012$).
    - Verbatim clean OCR: Word BM25 (0.8317) beats Char BM25 (0.7306) by $+0.1011$ ($p_{\text{Holm}}=0.00012$).
    - Verbatim real degraded OCR: Word BM25 (0.6684) beats Char BM25 (0.5907) by $+0.0777$ ($p_{\text{Holm}}=0.00012$).
  - Documents exact regime boundary: character n-grams protect against synthetic character noise; word BM25 is superior under optical OCR.
  - Output: `results/v3/char_vs_word_bm25.csv`.



