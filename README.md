# MemoryVerse: RAMP-RAG Research Reproduction Guide

This repository contains the official implementation, evaluation suite, and benchmark datasets for:
**"Multi-Stage Multimodal RAG with Text-Proxy Hybrid Retrieval" (RAMP-RAG)**.

---

## 1. Experimental Settings & Reproducibility Matrix

| Parameter / Protocol | Exact Setting | Specification Details |
|---|---|---|
| **Random Seeds** | `42, 43, 44` | Applied to every stochastic step: noise injection, permutation tests, bootstrap resampling, and dataset partitions. |
| **Dev / Test Split** | Strict 50 / 50 by Image ID | Partitioned with zero target leakage. Dev: 96 items (27 text photos, 68 non-text photos, 1 PDF doc); Test: 97 items (27 text photos, 69 non-text photos, 1 PDF doc). |
| **Dev Probes** | 30 Probes | 15 photo scene-text probes + 15 document probes, strictly targeting Dev items. Used exclusively for hyperparameter tuning. |
| **Test Probes** | 55 Probes | 30 photo scene-text probes + 25 document probes, strictly targeting Test items. Zero tuning on Test. |
| **MS-COCO Benchmark** | 1,000 images, 500 queries | Partitioned into 250 Dev queries and 250 Test queries. Evaluated using gold human captions and realistic BLIP-generated proxies. |
| **Dense Models** | `all-MiniLM-L6-v2` (384-dim)<br>`BAAI/bge-base-en-v1.5` (768-dim) | Pre-trained SentenceTransformers, normalized embeddings, cosine similarity. |
| **Sparse Model** | `PurePythonBM25` ($k_1=1.5, b=0.75$) | Bilingual CJK + English alphanumeric tokenization. |
| **Cross-Encoder Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Applied on top-5 retrieved candidates during cascade ranking. |
| **Clean Stream Model** | `openai/clip-vit-base-patch32` | Raw image pixel embeddings for photos; uncorrupted text embeddings for documents and queries. |
| **Proxy Generator** | `Salesforce/blip-image-captioning-base` | Generated 1,000 realistic VLM captions for COCO images (`max_new_tokens=25`). |
| **Noise Injection** | 0%, 10%, 20%, 40% | Character-level random substitution, deletion, and insertion across seeds 42, 43, 44. |
| **Observable Reliability** | Zero Oracle | $r_{\text{ocr}}(d) = \bar{c}_{\text{word}}(d) \cdot \rho_{\text{vocab}}(d)$, computed strictly from observable dictionary validity and length regularity of corrupted text. |
| **Confidence Intervals** | 2,000 Bootstrap Resamples | Exact Clopper-Pearson binomial intervals used for saturated proportions (1.0 or 0.0). |
| **Significance Tests** | Paired Permutation Tests | 10,000 sign-flip permutations with step-down Holm-Bonferroni correction vs. Convex Hybrid baseline. |
| **Latency Protocol** | Warm Cache, 3 Runs | Scope includes query encoding, retrieval, score fusion, and reranking/cascade. Mean and P95 reported in milliseconds. |

---

## 2. Repository Structure

```
├── IEEE_Paper_MemoryVerse.tex          # LaTeX manuscript (strictly 10 pages)
├── RESULTS.md                          # Empirical decisions, outcomes, and claim verifications
├── README.md                           # This reproducibility specification
├── notebooks/
│   └── ramp_rag_clean_stream_eval.ipynb # Executable evaluation notebook (runs top-to-bottom)
├── ai_engine/rag/
│   ├── data/
│   │   ├── ocr_extracted_corpus.json   # 191 photos (54 with scene text) + 2 PDFs
│   │   ├── ramp_rag_probes_split.json  # Strict 50/50 Dev/Test split (30 Dev, 55 Test probes)
│   │   ├── coco_karpathy_test.json     # 5,000 MS-COCO test records
│   │   ├── coco_blip_captions.json     # 1,000 BLIP-generated realistic caption proxies
│   │   ├── journal_revision_results.json # Full benchmark result payload (Parts A & B)
│   │   ├── mrr_hit10_vs_noise.png      # Multi-seed noise sweep figure
│   │   ├── pareto_cascade.png          # Conformal cascade cost-quality Pareto figure
│   │   ├── query_type_heatmap.png      # 5-category query taxonomy heatmap
│   │   ├── csv_tables/                 # Exported CSV tables (A1, A6, A7, etc.)
│   │   └── latex_tables/               # Publication-ready LaTeX tables
│   └── scripts/
│       ├── prepare_ramp_rag_probes.py  # Generates strict Dev/Test split & probes
│       ├── run_journal_revision_experiments.py # Master runner for Part A & Part B
│       ├── generate_notebook.py        # Executable notebook builder
│       └── generate_noise_plots.py     # High-resolution matplotlib plot generator
```

---

## 3. Frozen Pre-Registered Analysis Plan & Decision Protocol

## 3. Pre-Specified Analysis Plan (Frozen Before Test Evaluation)

*Pre-specified and frozen prior to running Test split evaluations for Part 0, Experiment 1 (H1), and Experiment 2 (H2). Zero test tuning.*
*Recorded commit hash: edf5c694a4180891c70dd271ca1a7d8a83e8a98e (October 24, 2026).*

### 3.1 Global Decision Rules
A claim of improvement is admitted into the manuscript **if and only if**:
1. The paired permutation test difference against the corresponding baseline achieves step-down **Holm-corrected $p < 0.05$**;
2. The **95% cluster bootstrap confidence interval** of the paired difference strictly excludes zero;
3. The empirical sign of the difference holds in **at least 2 of the 3 random seeds** (Seeds 42, 43, 44).
If any condition fails, the paper must report **"no reliable difference"** and eliminate any claim of superiority.

### 3.2 Statistical Protocol & Cluster Bootstrap
- **Cluster Bootstrap Over Probes:** When aggregating multi-seed sweeps (Seeds 42, 43, 44), seeds within each probe are averaged first, and resampling is performed strictly over unique probe clusters ($B = 2,000$ resamples of probe IDs).
- **Clopper-Pearson Exact Intervals:** Used for saturated proportions (Hit@K = 1.0 or 0.0).
- **Holm Correction:** Step-down Holm-Bonferroni correction applied across all primary hypothesis comparisons per family.
- **Observable Signals Only:** All reliability estimates ($r_{\text{ocr}}$) and routing features must be strictly observable from query text and retrieved candidate representations. Oracle variants are allowed only as explicitly labeled theoretical upper bounds.
- **Standardized Latency Protocol:** Local warm cache, query encoding included, scope = retrieval + reranking/fusion, mean and P95 reported over 3 runs.

### 3.3 Hypotheses & Pre-Specified Kill Criteria

#### Part 0: Prerequisites & Auditing
- **0.1 Ablation Rows:** Add 'BM25 alone' and 'BM25 + clean stream' across 0%, 10%, 20%, 40% noise to test whether clean-stream gain is simply dropping corrupted dense vectors.
- **0.2 Clean Baseline Reconciliation:** Explain distractor pool differences between 55-probe test-split index (105 items) and full corpus index (210 items).
- **0.3 COCO MRR Audit:** Re-run and audit COCO test evaluations where duplicate MRRs occurred due to default $r_{\text{ocr}}=1.0$ and verify with genuine CLIP image-vector representations.
- **0.4 AUROC of $r_{\text{ocr}}$:** Report AUROC for observable $r_{\text{ocr}}$ predicting corrupted vs. clean documents across noise levels.

#### Part A: Experiment 1 - Heterogeneous-Quality Corpus (Hypothesis H1)
- **Design:** Index of scene-text/document items (targets + distractors). Corrupt document-side OCR text only (queries stay clean). Fraction of bad documents $f_{\text{bad}} \in \{0, 0.1, 0.3, 0.5, 0.7\}$ with 40% character noise, plus continuous $\text{Beta}(2,5)$ scaled to $[0, 0.6]$. Seed-controlled independent assignment.
- **Methods:** BM25, dense MiniLM, dense bge-base, hybrid fixed $\alpha=0.7$, hybrid + clean stream (CLIP), BM25 + clean stream, reliability-weighted fusion ($r_{\text{ocr}}$ weighting text and $\gamma(1-r_{\text{ocr}})$ weighting CLIP; $\gamma$ tuned on Dev only), oracle-reliability (upper bound), all-stream RRF (BM25 + dense + CLIP).
- **Hypothesis H1:** Reliability-weighted fusion with clean stream beats the best single stream and fixed hybrid for $f_{\text{bad}} \in \{0.3, 0.5, 0.7\}$.
- **Kill Criterion:** No Holm-significant gain over the best single stream and over all-stream RRF for any $f_{\text{bad}} \ge 0.3$. If killed, report as negative result.

#### Part B: Experiment 2 - Mixed-Intent Router (Hypothesis H2)
- **Design:** Unified index containing COCO images (gold and BLIP proxies + CLIP vectors), photos (OCR + CLIP vectors), and document sentences. Balanced mixed query workload ($\ge 100$ queries per intent where data permits): (a) COCO paraphrases (captions 3 & 4), (b) exact scene-text lookups, (c) identifiers/filenames, (d) CJK bilingual queries, (e) numeric queries. Dev/Test split 50/50 by item ID. Router ground-truth labels = winning stream on Dev (ties broken by cheaper compute cost).
- **Routers:** R1 (surface-features), R2 (MiniLM query embedding + logistic regression), R3 (Query Performance Prediction features: max/mean IDF, BM25 top-k entropy, dense gap, CLIP gap, length, CJK/digit ratios + GradientBoosting), R4 (R2 + R3 combined).
- **Baselines:** Single streams (BM25, dense MiniLM, CLIP visual), fixed hybrid, all-stream RRF, oracle selector (upper bound).
- **Hypothesis H2:** A learned router (R2, R3, or R4) beats the best single stream AND all-stream RRF on the mixed workload and closes at least 50% of the oracle gap.
- **Kill Criterion:** No Holm-significant gain over the better of {best single stream, all-stream RRF}. If killed, report as negative result.

#### Part C: Optional Extensions (C1 – C4)
- **C1:** Noise-robust dense encoder (MiniLM fine-tuned contrastively with character corruptions on Dev).
- **C2:** Multi-encoder sweep on uniform noise (MiniLM, bge-base, e5-base, multilingual-e5).
- **C3:** Paraphrased scene-text probes.
- **C4:** Expanded multi-hop evaluation.

---

## 4. Running the Complete Suite

To reproduce all benchmarks from scratch:

```bash
# 1. Run the heterogeneous corpus and mixed router experimental harness
python3 ai_engine/rag/scripts/run_heterogeneous_experiments.py
```

