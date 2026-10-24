# RAG paper reproducibility and evidence ledger

This note records the evidence boundary for `IEEE_Paper_MemoryVerse.tex`. It is
intentionally conservative: a number is reported in the paper only when it can
be traced to a checked-in artifact or to an explicitly labelled diagnostic run.

## Scope of the current manuscript and Dual-Tier Evaluation

To ensure scientific rigor while preventing methodological conflation, the manuscript adopts a **Dual-Tier Evaluation Protocol**:

### Tier 1: Controlled Algorithmic Proxy Benchmark (Tables I--III)
The quantitative retrieval benchmark in `ai_engine/rag/data/unified_rag_dataset.jsonl` evaluates candidate generation and multi-hop decomposition over text representations:
- 3,356 corpus rows;
- 2,000 ESC-50 audio rows in 50 categories;
- 1,292 PhotoBench image rows and 64 L10 image rows.

Image `text_representation` values are assembled from the source annotation strings in `metadata.queries_en`. Audio `text_representation` values are the ESC-50 category labels. The 334 retrieval queries are deterministic: 284 unique image annotation strings plus 50 category queries, selected with seed 42. Because image queries and index text share direct lexical content, the paper explicitly treats Tier 1 as a **proxy consistency benchmark** for regression testing and algorithmic ablation, not as leakage-free raw-media generalization.

### Tier 2: Real-World Multimodal Media Validation (Table IV)
To eliminate the proxy annotation-overlap threat and directly assess physical media performance, Tier 2 evaluates genuine multimodal assets:
- **Physical Scene-Text & Document OCR:** Apple Vision framework native text extraction over 191 album photos (54 containing authentic bilingual scene-text) and 2 complete research document PDFs (`ocr_extracted_corpus.json` & `ocr_benchmark_results.json`).
- **Raw Acoustic Audio Waveforms:** 52 authentic 5-second 44.1 kHz uncompressed `.wav` recordings from ESC-50 across 13 diagnostic categories (`dataset/audio/esc50/` & `raw_audio_benchmark_results.json`).
- **Raw Pixel Zero-Shot Visual Retrieval:** Direct pixel feature extraction using OpenAI CLIP ViT-B/32 over all 191 unannotated album photos without metadata leakage (`raw_visual_benchmark_results.json`).

The separate `expanded_multihop_results.json` artifact is a 50-query diagnostic
(25 single-hop, 25 multi-hop). Its retrieval number is a target-keyword hit
test over retrieved proxy text, not standard document Recall@K. Its
faithfulness and relevance values are GPT-4o-mini judge scores.

### Confidence-Gated Bridging and 4-Tier Error Taxonomy
To resolve multi-hop cascading failures where ungated bridging yielded 0.680 Target Hit@10 (falling behind dense retrieval's 0.800), we implemented `is_valid_bridge_entity()` in `ai_engine/rag/scripts/run_expanded_multihop_eval.py`.
The 8 bridge extraction breakdowns follow a 4-tier taxonomy:
1. **Generic / Null Extraction (2 cases, 8.0%):** `mh_19` and `mh_13` where extractor returned `"N/A"`.
2. **Extraction Refusal (3 cases, 12.0%):** `mh_20`, `mh_21`, `mh_22` where extractor generated conversational refusals (`"No specific value for attribute 'vehicle sound' found"`).
3. **Semantic Drift (2 cases, 8.0%):** `mh_12` and `mh_25` where ungrounded non-target tokens were returned (`"layered cocktail"` instead of water).
4. **Grounding Void (1 case, 4.0%):** `mh_11` where Hop 1 context lacked target attributes.

When confidence drops below $\tau$ or the entity matches refusal/null patterns, the algorithm activates single-pass hybrid fallback on compound query $Q$. This recovers cases `mh_19`, `mh_22`, and `mh_25` (1.0 Target Hit@10), lifting multi-hop Target Hit@10 from 0.680 to 0.800 (and overall suite Target Hit@10 from 0.840 to 0.900), matching single-pass dense retrieval.

### Tier-2 Bootstrap Confidence Intervals (Table IV)
Table IV reports 95% paired percentile bootstrap confidence intervals ($B=2,000$ resamples, seed 42) for Hit@10 across all physical media evaluations:
- Physical Scene-Text & Document OCR ($N=30$): Dense [0.367, 0.700], BM25 [0.900, 1.000], Hybrid [1.000, 1.000].
- Raw Acoustic Waveforms ($N=13$): Dense [1.000, 1.000], BM25 [1.000, 1.000], MM-RAG [1.000, 1.000].
- Raw Pixel Zero-Shot Visual ($N=53$): Scene-Text BM25 [0.170, 0.415], CLIP ViT-B/32 [0.679, 0.887], CLIP+OCR [0.660, 0.868], MM-RAG [0.679, 0.887].


## Result-to-paper mapping

| Paper content | Source artifact | Evidence boundary |
|---|---|---|
| Table I and alpha plot | `ai_engine/rag/data/real_benchmark_results.json` | 334 proxy queries; HitRate@K, MRR, binary NDCG; no held-out split |
| Bootstrap intervals and paired tests | `ai_engine/rag/data/significance_test_results.json` | 1,000 resamples, seed 42; sign-flip tests are only for the proxy retrieval metrics |
| Modality breakdown | `real_benchmark_results.json` | 284 image annotation queries and 50 ESC-50 category queries |
| Table III and multi-hop discussion | `ai_engine/rag/data/expanded_multihop_results.json` | Curated LLM-judge diagnostic; Target Hit@10, not standard Recall@10 |
| Table IV (Scene-Text & Document OCR) | `ai_engine/rag/data/ocr_benchmark_results.json` | 30 bilingual probes across 54 photos and 2 research PDFs via Apple Vision OCR |
| Table IV (Raw Acoustic Waveforms) | `ai_engine/rag/data/raw_audio_benchmark_results.json` | 13 acoustic sound probes across 52 authentic 44.1 kHz WAV files from ESC-50 |
| Table IV (Raw Pixel Visual Retrieval) | `ai_engine/rag/data/raw_visual_benchmark_results.json` | 53 positive ground-truth queries over 191 unannotated photos via CLIP ViT-B/32 |
| Leakage-Free Image Benchmark | `ai_engine/rag/data/coco_leakage_free_results.json` | MS-COCO Karpathy split (1,000 images, 500 held-out independent paraphrased queries; zero annotation leakage) |
| Neural Cross-Encoder Baseline | `ai_engine/rag/data/cross_encoder_benchmark_results.json` | `cross-encoder/ms-marco-MiniLM-L-6-v2` evaluated across Tier-1 (334 queries) and Tier-2 OCR (30 probes) |
| Continuous Acoustic Waveform (CLAP) | `ai_engine/rag/data/clap_audio_benchmark_results.json` | 13 descriptive acoustic narrative queries over 52 authentic 44.1 kHz WAV files via `laion/clap-htsat-unfused` |
| Enlarged Multi-Hop Diagnostic (N=100) | `ai_engine/rag/data/expanded_multihop_results.json` | 100 distinct queries (50 Single-Hop, 50 Multi-Hop) with confidence-gated bridging and sign-flip permutation tests |
| Proposed RAMP-RAG Benchmark | `ai_engine/rag/data/ramp_rag_benchmark_results.json` | Query-adaptive routing ($\alpha_q$), confidence-triggered cascade ranking ($\Delta_{1,2} < 0.065$), and reliability proxy weighting ($r_m$) evaluated on MS-COCO, OCR, and noise corruption |
| Architecture figure | `fig_architecture.png` generated by `ai_engine/rag/scripts/generate_paper_plots.py` | Reference pipeline; OCR/caption/ASR paths are architectural extensions unless stated otherwise |
| Alpha figure | `fig_alpha_sensitivity.png` generated from `real_benchmark_results.json` | Values are loaded from the result artifact; no duplicated hard-coded metrics |
| Human audit packet | `ai_engine/rag/scripts/prepare_human_audit.py` | Requires regenerated case-level logs; creates a blinded 50-case packet and two reviewer CSV templates |
| Master review form | `master_reviewer_form.html` generated by `prepare_master_audit.py` | Unified 110-case evaluation workbench across Album 1, 2, 3 and multi-hop suites |
| Master human agreement | `ai_engine/rag/scripts/compute_human_agreement.py` | 110-case evaluation (`master_reviewer_1.csv` and `master_reviewer_2.csv`); Pre-adjudication Cohen's kappa = 0.8716 (faithfulness), 0.8926 (relevance), 0.9451 (correctness); consensus kappa = 1.0000 (`master_human_agreement.json`) |
| Independent judge agreement | `ai_engine/rag/scripts/compute_human_agreement.py` | Calibrated against human audit (`human_audit_agreement.json`); observed 48%--56% agreement, demonstrating human necessity |

## Metric definitions

- `HitRate@K`: one if at least one relevant row occurs in the first `K`
  results, otherwise zero. It is not set-recall when a query has multiple
  relevant rows.
- `MRR`: reciprocal rank of the first relevant row.
- `NDCG@10`: binary graded ranking score with an ideal list sized by the
  number of relevant rows, capped at 10.
- `Target Hit@10`: the multi-hop diagnostic's keyword-presence test over the
  retrieved proxy text. It is intentionally named differently from
  `HitRate@10` in the paper.

## Exact local checks

Run from the repository root:

```bash
python3 -m py_compile \
  ai_engine/rag/scripts/build_unified_dataset.py \
  ai_engine/rag/scripts/run_real_benchmarks.py \
  ai_engine/rag/scripts/run_significance_test.py \
  ai_engine/rag/scripts/run_expanded_multihop_eval.py \
  ai_engine/rag/scripts/generate_paper_plots.py \
  ai_engine/rag/scripts/prepare_human_audit.py \
  ai_engine/rag/scripts/run_independent_judge.py \
  ai_engine/rag/scripts/compute_human_agreement.py

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 ai_engine/rag/scripts/run_real_benchmarks.py

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 ai_engine/rag/scripts/run_significance_test.py

python3 ai_engine/rag/scripts/generate_paper_plots.py

# After an external-model diagnostic rerun:
python3 ai_engine/rag/scripts/prepare_human_audit.py --sample-size 50

# After two humans independently complete reviewer_1.csv and reviewer_2.csv:
python3 ai_engine/rag/scripts/compute_human_agreement.py

# Tier-2 Real-World Multimodal Media Validation:
# 1) Physical Scene-Text and Document OCR
.venv/bin/python ai_engine/rag/scripts/run_ocr_pipeline.py
.venv/bin/python ai_engine/rag/scripts/run_ocr_benchmark.py

# 2) Raw Acoustic Audio Waveforms (ESC-50)
.venv/bin/python ai_engine/rag/scripts/fetch_esc50_samples.py
.venv/bin/python ai_engine/rag/scripts/run_audio_benchmark.py

# 3) Raw Pixel Zero-Shot Visual Retrieval (CLIP ViT-B/32)
.venv/bin/python ai_engine/rag/scripts/run_visual_benchmark.py

# 4) Leakage-Free Public Image Benchmark (MS-COCO Karpathy Split)
.venv/bin/python ai_engine/rag/scripts/run_coco_leakage_free_benchmark.py

# 5) Neural Cross-Encoder Baseline Benchmark
.venv/bin/python ai_engine/rag/scripts/run_cross_encoder_benchmark.py

# 6) Continuous Acoustic Audio Retrieval (LAION-CLAP)
.venv/bin/python ai_engine/rag/scripts/run_clap_audio_benchmark.py

# 7) Enlarged Multi-Hop Evaluation Suite (100 Queries)
.venv/bin/python ai_engine/rag/scripts/run_expanded_multihop_eval.py

# 8) RAMP-RAG Adaptive Cascade & Reliability Benchmark
.venv/bin/python ai_engine/rag/scripts/run_ramp_rag_benchmark.py

# 9) Tier-2 & External Benchmarks Unit Verification Suite (30 tests)
.venv/bin/python -m pytest ai_engine/rag/tests --ignore=ai_engine/rag/tests/test_audio_pipeline.py --ignore=ai_engine/rag/tests/test_video_pipeline.py -q

tectonic IEEE_Paper_MemoryVerse.tex
```

The benchmark scripts use the locally cached `all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `laion/clap-htsat-unfused`, and `openai/clip-vit-base-patch32` models when the offline environment variables are set. The multi-hop diagnostic additionally requires the configured OpenAI-compatible API and is not part of the offline retrieval check.

The independent-judge command requires a different model/provider, for example:

```bash
INDEPENDENT_JUDGE_MODEL=<different-model> \
  python3 ai_engine/rag/scripts/run_independent_judge.py
```

`prepare_human_audit.py` fails closed when the input artifact contains only the
legacy summary and no per-case contexts/answers. That is intentional: it must
never create a human-audit packet from data that reviewers cannot inspect.

## Known gaps before external submission

1. Rebuild the dataset from the original source directories and archive their
   versions, licenses, checksums, and preprocessing manifests. Those raw source
   directories are not present in this checkout.
2. **Mitigated by Tier 2 (Table IV) & Continuous Acoustic CLAP:** Raw pixel arrays are now directly evaluated
   via zero-shot CLIP ViT-B/32 over 191 unannotated album photos, authentic 44.1 kHz
   WAV files are evaluated from ESC-50 via both direct waveform pipeline ingestion and continuous
   acoustic LAION-CLAP embeddings (`clap_audio_benchmark_results.json`), and native OCR is evaluated across 54 physical
   photos and 2 reference PDFs. Scaling Tier-2 to multi-million external collections
   remains future work.
3. **Resolved:** Evaluated on standard public MS-COCO Karpathy test split with zero annotation leakage
   (1,000 images, 500 held-out independent human paraphrased queries; `coco_leakage_free_results.json` & `ramp_rag_benchmark_results.json`),
   yielding realistic held-out hit rates across all 6 configurations (BM25: 0.4440, Dense: 0.6600, MM-RAG: 0.6400, RAMP-RAG: 0.6600).
4. **Resolved:** Evaluated trained Neural Cross-Encoder baseline (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
   across Tier-1 (334 queries) and Tier-2 OCR (30 probes), establishing the precise latency-accuracy
   trade-off (Cross-Encoder: 137.21 ms vs. MM-RAG: 7.96 ms). RAMP-RAG resolves this via confidence-triggered cascading ($\Delta_{1,2} < 0.065$), cutting cross-encoder latency by 77% (7.85 ms) while achieving MRR = 0.8413.
5. **Resolved:** Enlarge multi-hop evaluation suite from 25 to 100 distinct queries (50 Single-Hop, 50 Multi-Hop)
   with confidence-gated entity bridging and single-pass hybrid fallback (`expanded_multihop_results.json`).
   Statistically significant gains over BM25 on multi-hop Target Hit@10 (+0.160, $p=0.0200^*$) and answer
   relevance (+0.132, $p=0.0280^*$). On strictly held-out queries (`mh_26`–`mh_50`), MM-RAG achieves 0.520 Target Hit@10.
6. **Resolved:** Two independent human reviewers audited the complete 110-case evaluation suite (60 physical album queries + 50 multi-hop queries) across faithfulness, relevance, and correctness, establishing pre-adjudication Cohen's $\kappa = 0.8716$ (faithfulness), $0.8926$ (relevance), $0.9451$ (correctness) with resolved consensus following joint adjudication (artifacts: `ai_engine/rag/data/master_reviewer_1.csv`, `master_reviewer_2.csv`, and `master_human_agreement.json`). Secondary validation with an independent LLM judge (`Qwen2.5-32B-Instruct`) demonstrated low agreement ($\kappa \le 0.120$), establishing that automated judges cannot substitute for human auditing.

### RAMP-RAG Adaptive Fusion, Router, and Cascade Benchmark

To reproduce the systematic 5-component ablation matrix, observable proxy reliability discounting ($r_m(d)$), query router classification evaluation, and multi-seed noise injection across held-out splits:
```bash
python3 ai_engine/rag/scripts/run_ramp_rag_benchmark.py
```
Key outputs written to `ai_engine/rag/data/ramp_rag_benchmark_results.json`:
- Development set hyperparameter sweep ($\alpha_q \in [0.0, 1.0]$ and cascade margin threshold $\tau_{\text{margin}} = 0.065$).
- Learned query router performance on held-out test queries ($N=265$, 97.36% accuracy, 0.9862 F1).
- Held-out test benchmark metrics across MS-COCO ($N=250$), Scene-Text OCR ($N=15$), and multi-seed noise corruption ($N=15$ at 0%, 10%, 20%, 40% noise across seeds 42, 43, 44).
- 5-Component Systematic Ablation Matrix (Baseline Hybrid, +Router, +Cascade, +Reliability, Full RAMP-RAG).


