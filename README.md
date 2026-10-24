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
│   │   ├── clean_stream_experiments_results.json # Full benchmark result payload
│   │   ├── mrr_hit10_vs_noise.png      # High-resolution noise sweep figure
│   │   ├── csv_tables/                 # Exported CSV tables (Tasks 2, 3, 4, 5)
│   │   └── latex_tables/               # Publication-ready LaTeX tables (Tasks 2, 3, 4, 5)
│   └── scripts/
│       ├── prepare_ramp_rag_probes.py  # Generates strict Dev/Test split & probes
│       ├── download_coco_1k.py         # Concurrent downloader for 1,000 COCO images
│       ├── generate_coco_blip_captions.py # BLIP caption generation on Apple Silicon MPS
│       ├── generate_noise_plots.py     # High-resolution matplotlib plot generator
│       └── run_clean_stream_experiments.py # Master experimental harness (Tasks 1-5)
```

---

## 3. Running the Complete Suite

To reproduce all benchmarks from scratch:

```bash
# 1. Prepare strict Dev/Test split and expanded probes
.venv/bin/python ai_engine/rag/scripts/prepare_ramp_rag_probes.py

# 2. Run the complete clean-stream fusion, noise sweep, ablation, and latency benchmarks
.venv/bin/python ai_engine/rag/scripts/run_clean_stream_experiments.py

# 3. Generate high-resolution figures
python3 ai_engine/rag/scripts/generate_noise_plots.py
```
