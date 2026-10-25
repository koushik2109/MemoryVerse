# Comprehensive Pipeline and Evidence Audit (Phase 0)

**Date:** 2026-10-25  
**Audit Target:** MemoryVerse text-proxy multimodal retrieval pipeline, public benchmark generation, regime analysis scripts, and manuscript claims.

---

## 1. Data Sources and Provenance

1. **ICDAR 2019 SROIE Scanned Receipt Dataset (`ai_engine/rag/data/sroie/`)**:
   - Source: 626 scanned receipt images (`sroie/img/*.jpg`), bounding-box OCR transcripts (`sroie/box/*.csv`), and entity keys (`sroie/key/*.json`).
   - Purpose: Real-world printed administrative documents with authentic typographical variations, layouts, and OCR challenges.
2. **MS-COCO Karpathy-Test Distractors (`ai_engine/rag/data/coco_1k_images/`, `coco_karpathy_test.json`, `coco_blip_captions.json`)**:
   - Source: 1,000 photographic images from the MS-COCO test split, with corresponding BLIP-generated machine captions and human reference captions.
   - Purpose: Large distractor pool to prevent artificial target isolation and evaluate cross-modal visual retrieval.
3. **Combined Public Benchmark Corpus (`public_benchmark_v2.json`)**:
   - Pooled Size: 1,626 items (626 receipt documents + 1,000 natural image distractors).
   - Probes: 612 factual probes extracted to uniquely identify exactly one document in the pooled corpus.
4. **Historical Legacy Benchmarks (Conflated in Draft)**:
   - *Private Album Dataset*: 191 private photos + 2 PDFs (193 items, 85 probes). Scene-text numbers like BM25 clean MRR = 0.9818 and 20% noise = 0.9473 originated here, not on the public benchmark.
   - *Unified RAG Dataset*: 3,356 precomputed records (2,000 ESC-50 audio, 1,292 PhotoBench, 64 L10).

---

## 2. Partition and Split Logic

- **Partition Mechanism (`build_public_benchmark_v2.py`)**:
  - Partitioning is strictly 50/50 by Item ID using `random.Random(42).shuffle()`:
    - SROIE receipts: 313 Dev items, 313 Test items.
    - COCO distractors: 500 Dev items, 500 Test items.
    - Total Corpus: 813 Dev items, 813 Test items.
  - Probes are partitioned strictly according to their target item's split:
    - 306 Dev probes, 306 Test probes.
- **Audit Findings on Split Hygiene**:
  - Partitioning by item ID ensures that no target item appears in both Dev and Test.
  - Assertions must be formally added across all evaluation scripts to guarantee that Test item IDs and Test probe IDs never enter any fitting, tuning, or calibration routines.

---

## 3. Hyperparameter Tuning Locus (Forking Paths)

- **$\alpha$ (Dense vs. Sparse Balance in Hybrid Fusion)**:
  - Swept over $\alpha \in [0.0, 0.1, \dots, 1.0]$ strictly on Dev split per query type and noise condition (`ai_engine/rag/scripts/run_regime_grid_v2.py`).
  - Tuned parameters cached in `ai_engine/rag/data/v2/dev_tuned_params_cache.json`.
  - Applied frozen to the held-out Test split.
- **$\gamma$ (Clean Visual Stream Fusion Weight)**:
  - Swept over $\gamma \in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]$ strictly on Dev split.
- **$\tau^*$ (Conformal Cascade Threshold)**:
  - In legacy code (Section IV-J): $\tau_{\text{conf}} = 0.135$ was calibrated on the old 55-probe private split.
  - In v2 code (`run_conformal_cascade_v2.py`): $\tau^*$ was swept over $[0.0, 0.40]$ across 200 random 50/50 cluster-stratified splits of the 306 test probes.

---

## 4. Query and Paraphrase Construction Audit

- **Generation Pipeline (`generate_paraphrases_v2.py`)**:
  - LLM: `gpt-4o-mini` via OpenAI Chat Completions API with `temperature=0.0`.
  - Verbatim: Extracted verbatim query span from ground-truth text.
  - Light Paraphrase: LLM prompt instructed meaning preservation with 30%--50% token overlap. Post-processed by `adjust_light_paraphrase()` to enforce overlap in $[0.25, 0.55]$ (mean observed overlap: 41.2%).
  - Heavy Paraphrase: LLM prompt instructed <15% token overlap without rare identifiers.
- **Audit Finding on Heavy Paraphrase Defect**:
  - `adjust_heavy_paraphrase()` aggressively stripped all matching tokens. As a result, heavy paraphrases frequently lost the core discriminating factual entity, reducing retrieval to near-zero random guessing (MRR $\approx 0.009$, matching the random-ranking baseline $H_{813}/813 = 0.0090$).
  - Action: Redesign heavy paraphrases in Phase 1.4 into verifiable, entity-unique "question-style queries" (e.g., asking for the vendor, total, or date without reusing document verbatim spans).

---

## 5. Optical Character Recognition (OCR) and CER/WER Computation Audit

- **Apple Vision Framework Implementation (`build_public_benchmark_v2.py`)**:
  - Compiled Swift worker executing `VNRecognizeTextRequest` with `.accurate` recognition level and `usesLanguageCorrection = true`.
- **Degraded OCR Pipeline (`generate_degraded_ocr_v2.py`)**:
  - Applied Gaussian blur ($\sigma=1.5$), bilinear 50% downscaling, random rotation ($\pm 2.5^\circ$), and JPEG recompression ($Q=20$).
- **Audit Finding on CER/WER Defect**:
  - Reported CER = 0.4519 / WER = 0.6503 (Clean) and CER = 0.5218 / WER = 0.7561 (Degraded).
  - Cause of Artificially Inflated Error Rates:
    1. Bounding box lines in SROIE CSVs were concatenated in spatial coordinate order (`row[8]`), whereas Apple Vision outputs text in vertical reading order. Line transposition causes Levenshtein distance on raw concatenated strings to count whole transposed blocks as deletions and insertions.
    2. Zero normalization was applied: case differences, line breaks, leading/trailing punctuation, and whitespace variations were counted as character and word errors.
  - When BM25 searches the OCR output, it matches tokens regardless of line order, which is why BM25 clean MRR reaches 0.811 despite reported WER of 0.65.
  - Action: In Phase 1.1, recalculate CER/WER using normalized text (lowercased, whitespace collapsed, punctuation stripped) and align reading orders using standard `jiwer`.

---

## 6. Conformal Cascade Risk and "Miss" Definition Audit

- **Conformal Pool (`run_conformal_cascade_v2.py`)**:
  - $N=2,754$ pooled query instances: 306 probe clusters $\times$ 3 query variants $\times$ 3 conditions.
- **Definition of "Miss"**:
  - A query instance is considered a "miss" if and only if it is **not escalated** to Stage 2 ($\Delta_{1,2} \ge \tau$) AND Stage 1 failed to place the target item at rank 1 (`s1_hit1 == 0`).
  - Conditional Non-Escalated Risk:
    $$\text{Risk}(\tau) = \frac{\sum_{i \in \text{non-escalated}} \mathbb{I}(\text{rank}^{(1)}_i > 1)}{|\text{non-escalated}|}$$
- **Audit Finding on Conformal Claim Invalidation**:
  - Across 200 splits, the mean test miss rate was `0.0579` (empirical $\tau^* = 0.384$) and `0.0529` (conservative UCB $\tau^* = 0.400$).
  - Crucially, only **36.5%** of splits achieved a test miss rate $\le 0.05$.
  - Therefore, claiming that the conformal cascade "guarantees non-escalated miss rates $\le 0.05$" is **factually false and contradicted by the data**. The test miss rate exceeded 0.05 on average, and over 63% of splits violated the bound.
  - Action: Re-implement finite-sample corrected conformal risk control in Phase 2.1, and adjust claims to reflect actual empirical coverage.

---

## 7. Multi-Hop Equivalence Testing (TOST) and H2 Router Audits

- **Multi-Hop Equivalence Bug (`run_multihop_equivalence_v2.py`)**:
  - Lines 149--150 contain:
    ```python
    extractive_bridge_all = np.copy(llm_bridge_all)
    extractive_bridge_mh = np.copy(llm_bridge_mh)
    ```
  - The extractive bridging array was literally initialized as an identical copy of the generative LLM bridging array.
  - Consequently, the paired difference was identical to 0 on every query, generating a 90% CI of $[+0.0000, +0.0000]$ and $p_{\text{TOST}} = 0.0000$.
  - This is an artificial software artifact, not a legitimate empirical equivalence measurement.
  - Action: In Phase 2.2, audit the actual per-query predictions of the extractive pipeline vs. LLM bridge, compute true discordant query counts, McNemar's exact test, and honest paired difference CIs.
- **H2 Query Router Contradiction**:
  - In Section IV-I (Table IX), learned routers R1--R4 (MRR 0.3920--0.4070) lost to unweighted All-Stream RRF (MRR = 0.4214, $p_{\text{Holm}} = 0.2646$), leading to the formal verdict: "H2 KILLED".
  - In Section V-D (Table `tab:ablation_v2`), the router (MRR = 0.377) was claimed to "confirm H2" because it beat an unnormalized All-Stream RRF baseline that collapsed to MRR = 0.183.
  - This creates an irreconcilable internal contradiction where H2 is described as both "KILLED" and "CONFIRMED".
  - Action: In Phase 2.3, standardize the router evaluation against fair, properly normalized baselines and report a single, un-contradicted verdict.

---

## 8. Random-Ranking Baseline (Phase 0.4)

For a candidate pool of $N$ items with exactly 1 relevant target, the expected MRR under a uniform random ranking is:
$$\mathbb{E}[\text{MRR}] = \frac{1}{N} \sum_{r=1}^{N} \frac{1}{r} = \frac{H_N}{N}$$
- For the Public Benchmark ($N=813$ test items):
  $$\mathbb{E}[\text{MRR}_{813}] = \frac{7.27856}{813} = \mathbf{0.00895} \quad (\approx \mathbf{0.0090})$$
  $$\mathbb{E}[\text{Hit@10}_{813}] = \frac{10}{813} = \mathbf{0.01230} \quad (1.23\%)$$
- For the MS-COCO Benchmark ($N=1{,}000$ images):
  $$\mathbb{E}[\text{MRR}_{1000}] = \frac{7.48547}{1000} = \mathbf{0.00749} \quad (\approx \mathbf{0.0075})$$
  $$\mathbb{E}[\text{Hit@10}_{1000}] = \frac{10}{1000} = \mathbf{0.01000} \quad (1.00\%)$$
All subsequent result tables will explicitly display this baseline to ensure reader calibration.
