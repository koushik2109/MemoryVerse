# Column Provenance for Tables II, III, IV, and X

This document maps the exact pool, text source, query formulation, probe count $N$, and candidate parameters for every column and row in Tables II, III, IV, and X.

---

## 1. Table II (`tab:regime_grid_v3`): Multimodal Text-Proxy Retrieval Regime Analysis
- **Corpus Pool:** 813 items (313 SROIE receipt evaluation subset + 500 MS-COCO distractors with ground-truth caption fallback).
- **Evaluation Split:** Held-out test split, $N=306$ probes.
- **Metric:** MRR@10 with 95% cluster-bootstrap confidence intervals.

| Column Header | Text Source / Input Stream | Corpus Pool | Query Set Formulation | Candidate Depth |
| :--- | :--- | :--- | :--- | :--- |
| **Clean GT MRR** | Ground Truth transcription (SROIE GT + COCO GT captions) | 813 pool | Block 1: Verbatim ($N=306$)<br>Block 2: Light Paraphrase ($N=306$)<br>Block 3: Answerable Questions ($N=306$) | Full pool |
| **Syn 20% MRR** | 20% synthetic uniform character noise on GT text | 813 pool | Block 1: Verbatim ($N=306$)<br>Block 2: Light Paraphrase ($N=306$)<br>Block 3: Answerable Questions ($N=306$) | Full pool |
| **Apple Clean MRR** | Apple Vision Neural OCR (pristine scans) | 813 pool | Block 1: Verbatim ($N=306$)<br>Block 2: Light Paraphrase ($N=306$)<br>Block 3: Answerable Questions ($N=306$) | Full pool |
| **Optical Blur MRR** | Apple Vision Neural OCR on Gaussian-blurred scans (Text models); Direct degraded image pixels (ColQwen2, CLIP) | 813 pool | Block 1: Verbatim ($N=306$)<br>Block 2: Light Paraphrase ($N=306$)<br>Block 3: Answerable Questions ($N=306$) | Full pool |
| **Latency (ms)** | Execution time (warm cache, batch size=1, hardware specified) | 813 pool | Verbatim & Answerable queries | Isolated dot product (dense) / Inverted index (sparse) |

---

## 2. Table III (`tab:dense_formulation_v6`): Dense Bi-Encoder Sensitivity to Query Formulation
- **Corpus Pool Audit:** Discloses pool divergence:
  - Row 1 MiniLM (0.1919) was evaluated on the **pure receipt pool ($N=626$)**.
  - Row 1 BGE (0.3256) and Table II MiniLM (0.2100) were evaluated on the **813-distractor pool** (with COCO caption fallback).
- **Text Source:** Clean ground-truth receipt text ($0\%$ noise).
- **Evaluation Split:** Held-out test split, $N=306$ probes.

| Row Formulation | Text Source | Corpus Pool | Query Description |
| :--- | :--- | :--- | :--- |
| **1. Short Verbatim Snippets** | Pristine SROIE Ground Truth | Mixed (MiniLM 626 vs BGE 813) | Unmodified short text fragments from receipts |
| **2. Length-Padded Spans** | Pristine SROIE Ground Truth | Pure receipt pool / 813 pool | Padded with neutral tokens to match normal passage length |
| **3. Full Sentential Context** | Pristine SROIE Ground Truth | Pure receipt pool / 813 pool | Wrapped into complete synthetic declarative sentences |
| **4. Keyword-Matched Multi-Entity** | Pristine SROIE Ground Truth | Pure receipt pool / 813 pool | Multi-entity co-specification (vendor + total + date) |
| **5. Natural Answerable Questions** | Pristine SROIE Ground Truth | Pure receipt pool / 813 pool | Natural language question targeting key receipt fact |

---

## 3. Table IV (`tab:ablation_v3`): Architectural Component Ablation
- **Corpus Pool:** 813-item pool (with COCO caption fallback).
- **Evaluation Split:** Held-out test split, $N=306$ verbatim probes.
- **Candidate Depth:** First-stage candidate truncation $K_1=10$.
  - *Note on divergence:* Table IV BM25 stream achieves 0.8069 because scoring is truncated to top-$K_1=10$ candidate set, whereas Table II standalone BM25 achieves 0.8371 without $K_1=10$ truncation.
  - *Note on Oracle row:* Matches BM25 stream in 3 of 4 columns because ground-truth text stream relies on the same $K_1=10$ candidate generation.

| Column Header | Text Source / Pipeline Stream | Corpus Pool | Query Formulation | Candidate Depth |
| :--- | :--- | :--- | :--- | :--- |
| **Clean MRR** | Ground Truth text stream | 813 pool | Verbatim ($N=306$) | $K_1 = 10$ |
| **Syn 20% MRR** | 20% synthetic character noise stream | 813 pool | Verbatim ($N=306$) | $K_1 = 10$ |
| **Real-OCR MRR** | Apple Vision Clean OCR stream | 813 pool | Verbatim ($N=306$) | $K_1 = 10$ |
| **Real-Deg MRR** | Apple Vision Degraded Blur OCR stream | 813 pool | Verbatim ($N=306$) | $K_1 = 10$ |

---

## 4. Table X (`tab:engine_replication_v3`, `tab:funsd_v3`, `tab:easyocr_v9`): Cross-Engine Replication

### (a) SROIE Retail Receipts
- **Corpus Pool:** Pure SROIE receipt corpus $N=626$ (unannotated COCO distractors treated as empty text).
- **Evaluation Split:** $N=306$ held-out test probes.
- **Query Types:** Verbatim ($N=306$) and Light Paraphrase ($N=306$).

| Column Header | OCR Engine / Degradation | Corpus Pool | Query Types |
| :--- | :--- | :--- | :--- |
| **Ground Truth** | Pristine SROIE ground truth transcriptions | 626 pool | Verbatim & Light Paraphrase |
| **Apple Clean** | Apple Vision Neural OCR (clean scans) | 626 pool | Verbatim & Light Paraphrase |
| **Tesseract Clean**| Tesseract 5.5 LSTM OCR (clean scans) | 626 pool | Verbatim & Light Paraphrase |
| **Tesseract Blur** | Tesseract 5.5 LSTM OCR (optical blur scans) | 626 pool | Verbatim & Light Paraphrase |
| **EasyOCR Clean** | EasyOCR 1.7.2 Deep Neural OCR (clean scans) | 626 pool | Verbatim & Light Paraphrase |
| **EasyOCR Blur** | EasyOCR 1.7.2 Deep Neural OCR (optical blur scans)| 626 pool | Verbatim & Light Paraphrase |

### (b) FUNSD Administrative Forms
- **Corpus Pool:** Scanned administrative forms ($N=199$ / $N=700$ distractor pool).
- **Evaluation Split:** $N=100$ held-out form probes.
- **Query Types:** Verbatim ($N=100$) and Light Paraphrase ($N=100$).

| Column Header | OCR Engine / Degradation | Corpus Pool | Query Types |
| :--- | :--- | :--- | :--- |
| **Ground Truth** | Pristine FUNSD form transcriptions | FUNSD pool | Verbatim & Light Paraphrase |
| **Apple Clean** | Apple Vision Neural OCR (clean forms) | FUNSD pool | Verbatim & Light Paraphrase |
| **Tesseract Clean**| Tesseract 5.5 LSTM OCR (clean forms) | FUNSD pool | Verbatim & Light Paraphrase |
| **Tesseract Blur** | Tesseract 5.5 LSTM OCR (optical blur forms) | FUNSD pool | Verbatim & Light Paraphrase |
| **EasyOCR Clean** | EasyOCR 1.7.2 Deep Neural OCR (clean forms) | FUNSD pool | Verbatim & Light Paraphrase |
| **EasyOCR Blur** | EasyOCR 1.7.2 Deep Neural OCR (optical blur forms)| FUNSD pool | Verbatim & Light Paraphrase |

