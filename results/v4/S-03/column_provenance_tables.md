# Column-Provenance Audit for Tables II, III, IV, V, VII, VIII, IX, X, XI, XII, XIII (S-03)

This document establishes the precise operational specification for every column in all tables of the manuscript.

---

## 1. Table II: Regime Analysis Grid (`tab:regime_grid_v3`)
- **Document Pool:** Full Evaluation Pool ($N=813$ documents: 313 held-out test SROIE receipts + 500 MS-COCO distractors).
  - *Distractor Treatment:* The 500 MS-COCO distractor images have their ground-truth caption text indexed as fallback for text models.
- **Split:** Held-Out Public Test Split ($N=306$ probes, seed 42).
- **Evaluation Mode:** Unconstrained ranking over all 813 candidates.
- **Dense Embedding Mode:** Unchunked full-text representations (dot-product ranking over precomputed document embeddings; query latency measured at 0.01 ms for dot product, full latency in Table XII).

| Column Header | Text / Input Source | Query Sets Evaluated | Chunked? | Pool Specification |
| :--- | :--- | :--- | :--- | :--- |
| **Clean GT MRR** | Ground Truth Text | Verbatim ($N=306$), Light Paraphrase ($N=306$), Answerable ($N=306$) | Unchunked | 813 (313 SROIE + 500 COCO caption text) |
| **Syn 20% MRR** | Synthetic 20% Character Sub/Del/Ins Noise | Verbatim ($N=306$), Light Paraphrase ($N=306$), Answerable ($N=306$) | Unchunked | 813 (313 SROIE perturbed + 500 COCO perturbed) |
| **Apple Clean MRR** | Apple Vision Neural OCR (Clean Scans) | Verbatim ($N=306$), Light Paraphrase ($N=306$), Answerable ($N=306$) | Unchunked | 813 (313 SROIE OCR + 500 COCO caption text) |
| **Apple Blur MRR** | Apple Vision Neural OCR (Gaussian blur $\sigma=1.5$, 50% downscale, deterministic rotation $\pm4^\circ$, JPEG Q20) | Verbatim ($N=306$), Light Paraphrase ($N=306$), Answerable ($N=306$) | Unchunked | 813 (313 SROIE Blur OCR + 500 COCO caption text) |
| **Latency (ms)** | Profiling Hardware (Apple M2 CPU for sparse; M2 MPS for dense/CLIP; Tesla T4 for ColQwen2) | Single-probe execution latency | N/A | Batch Size = 1, Warm Cache |

*Note on Visual Models in Table II:*
- ColQwen2-v1.0 Native and CLIP Visual evaluate raw image pixels directly.
- Under "Clean GT" and "Apple Clean", both evaluate identical unperturbed document images.
- Under "Syn 20% MRR", character noise is not applicable to image pixels; values reflect unperturbed native images ($^\dagger$).
- Under "Apple Blur MRR", visual models evaluate optically degraded blurry receipt images (`sroie/img_degraded`).

---

## 2. Table III: Dense Bi-Encoder Sensitivity to Query Formulation (`tab:dense_formulation_v6`)
- **Document Pool:** Full Evaluation Pool ($N=813$ documents).
- **Text Source:** Clean Ground-Truth Receipt Text ($0\%$ noise).
- **Split:** Held-Out Public Test Split ($N=306$ probes).

| Column Header | Metric | Query Formulations Tested | Chunked / Embedding Mode |
| :--- | :--- | :--- | :--- |
| **MiniLM Unchk** | MRR@10 | Rows 1–5: Short Verbatim Snippets, Length-Padded Spans, Sentential Context, Keyword-Matched, Natural Answerable | Unchunked single vector |
| **MiniLM Chunk** | MRR@10 | Rows 1–5 | Chunked MaxP (150-word sliding windows $\le 256$ tokens) |
| **BGE Unchk** | MRR@10 | Rows 1–5 | Unchunked single vector |
| **BGE Chunk** | MRR@10 | Rows 1–5 | Chunked MaxP (150-word sliding windows $\le 256$ tokens) |

---

## 3. Table IV: Architectural Component Ablation (`tab:ablation_v3`)
- **Document Pool:** Full Evaluation Pool ($N=813$ documents).
- **Split:** Held-Out Public Test Split ($N=306$ probes, seed 42).
- **Candidate Truncation:** First-stage candidate depth strictly capped at $K_1=10$.
  - Standalone Word BM25 drops from unconstrained 0.8371 to 0.8107 under $K_1=10$ candidate pool truncation.

| Column Header | Text / Input Source | Evaluated Systems | Candidate Depth ($K_1$) | Pool Specification |
| :--- | :--- | :--- | :--- | :--- |
| **Clean MRR** | Ground Truth Text ($0\%$ noise) | Fixed-weight Hybrid ($\alpha=0.5$), BM25 Only, MiniLM Only, Oracle GT Stream, Learned Router, Reliability Weighting | $K_1=10$ | 813 Pool |
| **Syn 20% MRR** | Synthetic 20% Character Corruption | All 6 Ablation Rows | $K_1=10$ | 813 Pool |
| **Real-OCR MRR** | Apple Vision Neural OCR (Clean) | All 6 Ablation Rows | $K_1=10$ | 813 Pool |
| **Real-Deg MRR** | Apple Vision Neural OCR (Optical Blur Degraded) | All 6 Ablation Rows | $K_1=10$ | 813 Pool |

---

## 4. Table V: Production Cross-Encoder Reranker Cascade Ablation (`tab:practical_stack`)
- **Document Pool:** Full Evaluation Pool ($N=813$ documents).
- **Text Source:** Clean Apple Vision Real-OCR.
- **Split:** Held-Out Public Test Split ($N=306$ probes).
- **Candidate Depth:** $K_1=20$ for cross-encoder reranking.

| Column Header | Description | Metric | Hardware |
| :--- | :--- | :--- | :--- |
| **Clean MRR** | Retrieval effectiveness on clean Real-OCR | MRR@10 | M2 CPU / T4 GPU |
| **Hit@10** | Fraction of queries with target in top 10 | Hit@10 | M2 CPU / T4 GPU |
| **Lat. (ms)** | End-to-end execution latency per query | Milliseconds | Batch Size = 1 |
| **MRR Lift ($p$-value)** | Absolute MRR delta over baseline and paired permutation $p$-value | Delta ($p$) | Monte Carlo Permutation ($B=100,000$) |

---

## 5. Table VII: OCR Error Rates Across Engines (`tab:ocr_error_rates_v6`)
- **Document Pool:** SROIE Test Split ($N=306$ receipt transcripts) and FUNSD ($N=100$ forms).
- **Ground Truth:** Pristine human transcriptions.

| Column Header | Metric | Definition | Engines Evaluated |
| :--- | :--- | :--- | :--- |
| **CER (%)** | Character Error Rate | Levenshtein edit distance / reference characters | Apple Clean, Apple Blur, Tesseract Clean, Tesseract Blur, EasyOCR Clean, EasyOCR Blur |
| **WER (%)** | Word Error Rate | Word-level Levenshtein edit distance / reference words | All 6 OCR conditions |

---

## 6. Table VIII: Cross-Modal Text-to-Image Retrieval on MS-COCO (`tab:coco_benchmarks_v3`)
- **Document Pool:** MS-COCO 2014 Karpathy Test Split ($N=1{,}000$ candidate images).
- **Query Set:** Exactly 1 held-out caption per image (Caption 3, strictly excluded from indexed representations).

| Column Header | Metric | Query Source | Representations Evaluated |
| :--- | :--- | :--- | :--- |
| **Hit@1** | Fraction target ranked at rank 1 | Caption 3 | Native Vision (CLIP, SigLIP) vs Machine Captions (BLIP, BLIP-2, Florence-2) vs 4-Ref Gold |
| **Hit@5** | Fraction target in top 5 | Caption 3 | All 7 representation streams |
| **Hit@10** | Fraction target in top 10 | Caption 3 | All 7 representation streams |
| **MRR** | Mean Reciprocal Rank | Caption 3 | All 7 representation streams |

---

## 7. Table IX: Document Visual Question Answering on DocVQA (`tab:docvqa_v9`)
- **Document Pool:** 500-page held-out test split of DocVQA ($N=500$ document pages).
- **Query Set:** $N=500$ human-authored factual questions.

| Column Header | Retrieval Stream | Input Modality | Metrics Evaluated |
| :--- | :--- | :--- | :--- |
| **MRR** | Tesseract BM25, Florence-2 BM25, ColQwen2-v1.0 | Text OCR vs Native Image | MRR@10 |
| **Hit@1** | Top-1 retrieval rate | Text OCR vs Native Image | Hit@1 |
| **Hit@5** | Top-5 retrieval rate | Text OCR vs Native Image | Hit@5 |
| **Hit@10** | Top-10 retrieval rate | Text OCR vs Native Image | Hit@10 |
| **nDCG@5** | Normalized Discounted Cumulative Gain at rank 5 | Text OCR vs Native Image | nDCG@5 (measured locally vs ViDoRe published) |

---

## 8. Table X: Cross-Engine Replication on SROIE and FUNSD (`tab:engine_replication_v3`)
- **Part (a):** SROIE Pure Receipt Pool ($N=626$ receipts, 0 COCO distractors).
- **Part (b):** Scanned Administrative Form Corpus (FUNSD, $N=100$ forms, 700 distractors).

| Column Header | Engine / Condition | Query Types Evaluated | Evaluation Pool |
| :--- | :--- | :--- | :--- |
| **Ground Truth** | Pristine manual transcription | Verbatim ($N=306$), Light Paraphrase ($N=306$) | 626 Receipts / 100 Forms |
| **Apple Clean** | Apple Vision Neural OCR | Verbatim, Paraphrase | 626 Receipts / 100 Forms |
| **Tesseract Clean** | Tesseract 5.5 LSTM OCR | Verbatim, Paraphrase | 626 Receipts / 100 Forms |
| **Tesseract Blur** | Tesseract 5.5 under blur | Verbatim, Paraphrase | 626 Receipts / 100 Forms |
| **EasyOCR Clean** | EasyOCR 1.7.2 CRAFT + CRNN | Verbatim, Paraphrase | 626 Receipts / 100 Forms |
| **EasyOCR Blur** | EasyOCR 1.7.2 under blur | Verbatim, Paraphrase | Strict IR unretrieved metric ($\text{RR}=0$) |

---

## 9. Table XI: Secondary Open-Weight and Human Query Reformulation (`tab:secondary_queries`)
- **Document Pool:** Full Evaluation Pool ($N=813$ documents).
- **Input Text:** Apple Vision Clean Real-OCR.

| Column Header | Retrieval Model | Query Split Evaluated | Hardware |
| :--- | :--- | :--- | :--- |
| **Word BM25** | Robertson BM25 ($k_1=0.9, b=0.5$) | Primary Synthetic ($N=306$), Qwen2.5-7B ($N=306$), Human $N=100$, Human $N=200$ | Apple M2 CPU |
| **Char BM25** | Character 3--5gram BM25 | Primary Synthetic, Qwen2.5-7B, Human $N=100$, Human $N=200$ | Apple M2 CPU |
| **Dense BGE** | BGE-Base-en-v1.5 (unchunked) | Primary Synthetic, Qwen2.5-7B, Human $N=100$, Human $N=200$ | Apple M2 MPS |
| **Dev-Hybrid** | Convex linear fusion | Primary Synthetic, Qwen2.5-7B, Human $N=100$, Human $N=200$ | Apple M2 MPS |
| **ColQwen2** | ColQwen2-v1.0 Native Visual | Primary Synthetic, Qwen2.5-7B, Human $N=100$, Human $N=200$ | Nvidia Tesla T4 GPU |

---

## 10. Table XII: Hardware Profiling and Serving Economics (`tab:system_costs`)
- **Hardware Ensembles:** Apple M2 (16 GB unified RAM) vs Nvidia Tesla T4 GPU (14.56 GB VRAM).
- **Profiling Protocol:** Batch Size = 1, Warm Cache, $N=306$ probe queries.

| Column Header | Metric | Definition |
| :--- | :--- | :--- |
| **Indexing Latency** | Throughput | Pages or transcripts processed per second |
| **Query Latency (ms)** | End-to-end execution | Total milliseconds from raw query string to ranked Top-10 candidates |
| **Index Footprint** | Memory / Disk size | Megabytes required to store index in RAM or disk |
| **Hardware Required** | Serving infrastructure | Commodity CPU vs Dedicated Accelerator GPU |

---

## 11. Table XIII: Empirical Decision Guide (`tab:decision_guide_v6`)
- **Synthesis:** Maps all 8 tested regimes into optimal architectures, verified scores, and statistically tied sets ($p > 0.05$).

| Column Header | Content | Verification Basis |
| :--- | :--- | :--- |
| **Target Modality** | Document Page, Photographic Image, Multi-Hop Query | Validated across SROIE, FUNSD, DocVQA, COCO |
| **Optical Condition** | Clean Neural OCR, Real-Degraded Blur, Synthetic Noise, Precomputed Proxies | Defined by evaluated OCR pipeline |
| **Query Regime** | Verbatim Lookup, Light Paraphrase, Answerable Question, Relational Multi-Hop | Query formulation taxonomy |
| **Optimal Architecture** | Empirically highest-performing model | Table II, Table VIII, Table IX, Table X |
| **Score [95% CI]** | Primary effectiveness metric (MRR or Target Hit@10) | Cluster bootstrap 95% CI |
| **Statistically Tied Set ($p > 0.05$)** | Models with no statistically significant difference from optimal | Two-tailed paired permutation test or TOST |
| **Architectural Recommendation** | Actionable design guidance for system builders | Serving latency and compute trade-offs |
