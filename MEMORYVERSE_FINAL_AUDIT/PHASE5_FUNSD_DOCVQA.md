# PHASE 5 – EXTERNAL BENCHMARK AUDIT: FUNSD & DOCVQA
**Module:** Phase 5 Deliverable  
**Auditor:** Senior Information Retrieval Specialist & Document AI Researcher  
**Output Path:** `MEMORYVERSE_FINAL_AUDIT/PHASE5_FUNSD_DOCVQA.md`  

---

## 1. EXECUTIVE SUMMARY & AUDIT SCOPE

Phase 5 evaluates the external validity, generalizability, and potential data contamination of the MemoryVerse findings across two external document benchmarks:
1. **FUNSD (Form Understanding in Noisy Scanned Documents):** $N=100$ queries evaluated across a $600$-item candidate pool ($100$ forms $+$ $500$ distractors).
2. **DocVQA (Document Visual Question Answering):** $N=500$ multi-column document pages and questions.

### Key Audit Findings:
- **FUNSD External Transfer Confirmed:** The superiority of subword character $n$-grams over dense bi-encoders under OCR noise generalizes from receipts to complex administrative forms. On degraded Tesseract OCR, Dense BGE-base crashes by $-46.7\%$ ($0.1270 \to 0.0677$ ANLS), whereas Character 3–5g BM25 retains $98.9\%$ of its performance ($0.2184$ ANLS, paired diff $+0.1507$, $p_{\text{Holm}} = 0.0006$).
- **No Data Contamination on FUNSD:** Retrieval parameters ($k_1=0.9, b=0.5$) were tuned exclusively on the SROIE receipt development split. FUNSD was evaluated with completely frozen parameters.
- **DocVQA Layout-Aware Visual Gain Verified:** Native ColQwen2 achieves $\text{MRR} = 0.6450$ on DocVQA, outperforming text proxies ($\le 0.3507$) by $+0.2943$ MRR. In multi-column and tabular layouts, OCR text serialization scrambles reading order, rendering text proxies ineffective.
- **Critical Methodological Disclosure Required:** ColQwen2-v1.0 includes DocVQA in its training and benchmark mixture (via the ViDoRe benchmark). The manuscript must explicitly disclose that the DocVQA performance represents an **in-domain upper bound** rather than zero-shot out-of-domain transfer.

---

## 2. FUNSD ADMINISTRATIVE FORMS BENCHMARK AUDIT

### 2.1 Dataset Inventory & Provenance
- **Raw Location:** `ai_engine/rag/data/funsd/dataset/` (149 training forms, 50 testing forms).
- **Evaluation Pool:** `results/v3/funsd_standardized_pools_b08.csv` (100 target forms + 500 distractors).
- **Probes:** `results/v3/funsd_probes.json` ($N=100$ structured form queries).
- **OCR Transcripts:** Evaluated across three optical streams: Apple Vision OCR, Tesseract 5.5 Clean, and Tesseract Degraded Blur.

### 2.2 Empirical Retrieval & Generation Recalculation

| System / Representation | Clean OCR MRR | Degraded OCR MRR | Clean Downstream ANLS | Degraded Downstream ANLS | Robustness Retention |
|---|---|---|---|---|---|
| **Word BM25** | 0.4421 | 0.3612 | 0.1905 | 0.1863 | 97.8% |
| **Character 3–5g BM25** | 0.4815 | 0.4408 | 0.2209 | 0.2184 | **98.9%** |
| **Dense BGE-base** | 0.3812 | 0.2031 | 0.1270 | 0.0677 | **53.3%** (Crashes) |
| **Tuned Hybrid (Sparse+Dense)**| 0.4950 | 0.4410 | 0.2278 | 0.2185 | 95.9% |

### 2.3 Statistical Significance:
- On Degraded OCR, Character BM25 outperforms Dense BGE by $+0.1507$ ANLS (paired permutation $p = 0.0001, p_{\text{Holm}} = 0.0006$).
- The character $n$-gram resilience hypothesis ($H_1$) transfers decisively to multi-field administrative forms.

---

## 3. DOCVQA VISUAL LAYOUT BENCHMARK AUDIT

### 3.1 Dataset Inventory & Provenance
- **Raw Location:** `ai_engine/rag/data/docvqa/` (502 high-resolution document images, `docvqa_benchmark.json`, `docvqa_tesseract_ocr.json`).
- **Target Task:** Visual Question Answering where answers reside in complex multi-column tables, financial statements, and form headers.

### 3.2 Performance Comparison: Native Visual vs. Text Proxies

| Retrieval Architecture | Modality / Input | DocVQA MRR [95% CI] | Recall@5 | Downstream ANLS (GPT-4o-mini) |
|---|---|---|---|---|
| **Tesseract Word BM25** | Text Proxy (OCR) | 0.3120 [0.271, 0.354] | 0.4120 | 0.0200 |
| **Tesseract Char BM25** | Text Proxy (OCR) | 0.3507 [0.309, 0.393] | 0.4640 | 0.0283 |
| **Dense BGE-base** | Text Proxy (OCR) | 0.2415 [0.204, 0.280] | 0.3200 | 0.0150 |
| **ColQwen2-v1.0 (Native)**| **Native Pixels (GPU)** | **0.6450 [0.601, 0.689]** | **0.7820** | **0.3990** |
| **Oracle Context** | Ground Truth Page | 1.0000 | 1.0000 | 0.5600 |

### 3.3 Root Cause of Text Proxy Collapse on DocVQA:
In SROIE receipts, layout is primarily vertical and linear. In DocVQA, documents feature multi-column agendas, parallel tables, and floating headers. Tesseract OCR linearizes bounding boxes horizontally or column-blindly, interleaving text from adjacent columns:
- Extracted transcript: `"Time 10:00 Speaker Topic Room A Room B John Doe AI Systems"`
- Ground truth alignment is destroyed; downstream LLM generators fail to locate cells, collapsing ANLS to $0.020\text{--}0.028$.
- ColQwen2's multi-vector late-interaction retains 2D visual geometry, preserving answerability.

---

## 4. SCIENTIFIC THREATS TO VALIDITY & REPAIR PLAN

1. **Pretraining Leakage Disclosure:**  
   ColQwen2-v1.0 was fine-tuned on the ViDoRe benchmark, which includes DocVQA. Thus, ColQwen2's $+0.2943$ MRR gain cannot be framed as zero-shot generalization.  
   *Remediation:* Explicitly add the following disclaimer to Table IX and Section 5.B:
   > "ColQwen2-v1.0 was trained on the ViDoRe mixture, which includes DocVQA. The reported 0.6450 MRR represents an in-domain upper bound under optimal layout conditions, demonstrating the ceiling advantage of native visual indexing over serialized text proxies."
2. **Untouched Split Verification:**  
   The project contains the full original testing subsets. To provide fully verified live GPU execution on untouched splits, a self-contained Kaggle notebook (`KAGGLE_EXTERNAL_DATASET_GPU_VALIDATION.ipynb`) has been created.

