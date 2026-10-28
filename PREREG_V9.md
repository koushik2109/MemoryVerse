# PREREG_V9.md: Pre-Registration for Phase 3.1 Third OCR Engine Generalization

**Date:** 2026-10-27  
**Study Milestone:** Phase 3.1 Cross-Engine Generalization & Replication  
**Target Engine:** EasyOCR 1.7.2 (CRAFT text detector + PyTorch ResNet-CRNN recognizer, English `['en']` language pack)  
**Baseline Engines:** Apple Vision Neural OCR, Tesseract 5.5.3 LSTM OCR  
**Corpus & Partition:** SROIE 2019 Scanned Receipts ($N=626$) + COCO Distractors ($N=500$), Total $N=813$ items  
**Evaluation Partition:** Held-out Test split ($N=306$ probes: Verbatim, Light Paraphrase, Answerable Questions)  

---

## 1. Context and Objective

Our headline findings establish that textual proxy retrieval quality is governed by an interaction between query lexical form and OCR representation granularity:
1. Under verbatim lookups, Character 3–5gram BM25 decisively outperforms Word BM25 on Apple Vision (+0.1065 clean, +0.1276 degraded blur) and Tesseract 5.5 (+0.1049 clean, +0.0560 blur).
2. Under entity-substituting light paraphrases, Word BM25 outperforms Character BM25.
3. Dense bi-encoders collapse on receipt text primarily due to single-entity query formulation mismatch, with subword tokenization fragmentation (+122.3% subwords/word) explaining additional degradation under noise.

To ensure these conclusions are not artifacts of the specific OCR systems evaluated (proprietary Apple Vision and open-source Tesseract 5.5), Phase 3.1 executes an authentic replication using a **third distinct OCR architecture: EasyOCR 1.7.2**. EasyOCR couples a character-region-awareness (CRAFT) deep neural text detector with a deep convolutional recurrent neural network (CRNN) recognizer with CTC decoding.

---

## 2. Pre-Registered Hypotheses

All hypotheses are marked as **CONFIRMATORY** because they test directional predictions established by the primary benchmark:

- **H3.1a [CONFIRMATORY - Subword Lexical Invariance]:** Under verbatim entity queries on EasyOCR-transcribed text (both clean and degraded blur), Character 3–5gram BM25 achieves strictly higher retrieval MRR than Word BM25 with a paired difference whose 95% cluster-bootstrap confidence interval strictly excludes zero ($p_{\text{Holm}} < 0.05$).
- **H3.1b [CONFIRMATORY - Semantic Paraphrase Reversal]:** Under entity-substituting light paraphrases, Word BM25 achieves higher MRR than Character 3–5gram BM25 ($p_{\text{Holm}} < 0.05$), replicating the finding that character n-grams generate spurious collisions when exact entity tokens are absent.
- **H3.1c [CONFIRMATORY - Dense bi-encoder deficit]:** Dense bi-encoders (MiniLM) score substantially lower than Character 3–5gram BM25 on clean EasyOCR text (MRR $< 0.30$ vs $> 0.80$), confirming representation mismatch across deep OCR transcriptions.
- **H3.1d [CONFIRMATORY - Optical Blur Sensitivity]:** Optical Gaussian blur significantly degrades transcription accuracy (increasing CER and WER) on EasyOCR, but Character 3–5gram BM25 exhibits higher degradation resilience ($\Delta \text{MRR}_{\text{char}} < \Delta \text{MRR}_{\text{word}}$).

---

## 3. Specifications of the Third Engine (EasyOCR)

- **Library & Version:** `easyocr==1.7.2`
- **Underlying Models:**
  - Detector: CRAFT (Character Region Awareness for Text Detection)
  - Recognizer: ResNet backbone + 2-layer Bidirectional LSTM + CTC loss
  - Model Source: Official PyTorch weights maintained by JaidedAI
- **Language Pack:** English (`['en']`)
- **Execution Hardware:** CPU / Apple Silicon PyTorch backend
- **Input Images:**
  - Clean: 626 SROIE receipt images (`ai_engine/rag/data/sroie/img/`)
  - Degraded: 626 SROIE receipt images degraded under identical optical blur parameters (Gaussian blur $\sigma=1.5$, 50% downscale, rotation $\pm 4^\circ$, JPEG Q20; `ai_engine/rag/data/sroie/img_degraded/`)

---

## 4. Transcription Error Metrics & Normalization Protocol

Document edit distance will be evaluated against human-annotated Ground Truth text using standardized **JiWER 4.0.0**:
- **Document Text Normalization:**
  - Lowercase conversion
  - Collapsing consecutive whitespace characters to single spaces
  - Stripping leading and trailing whitespace
  - Standardized punctuation stripping for character edit distance alignment
- **Evaluated Metrics:**
  - **CER (Character Error Rate):** $\text{CER} = \frac{S_c + D_c + I_c}{N_c}$ via full-document Levenshtein distance.
  - **WER (Sequence Word Error Rate):** $\text{WER}_{\text{seq}} = \frac{S_w + D_w + I_w}{N_w}$ aligning token sequences.
  - **MER (Multiset Error Rate, Bag-of-Words):** $\text{MER}_{\text{BoW}} = \frac{\sum_w |c_{\text{ref}}(w) - c_{\text{hyp}}(w)|}{\sum_w c_{\text{ref}}(w)}$.
- **Qualitative Audit:**
  - 10 side-by-side qualitative comparisons (Ground Truth vs Apple Vision vs Tesseract 5.5 vs EasyOCR) saved to `results/v9/easyocr_samples_v9.md`.

---

## 5. Retrieval Systems & Evaluation Design

1. **Evaluated Retrieval Architectures:**
   - Word BM25 ($k_1=0.9, b=0.5$)
   - Character 3–5gram BM25 ($k_1=0.9, b=0.5$)
   - Dense MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)
   - Dense BGE-Base (`BAAI/bge-base-en-v1.5`)
   - Tuned Hybrid (convex sparse-dense combination)
2. **Evaluation Workload:**
   - Held-out Test split: $N=306$ probes (102 Verbatim, 102 Light Paraphrase, 102 Answerable Questions).
   - Distractor pool: 500 COCO image captions (total corpus $N=813$ items).
3. **Statistical Invariants:**
   - Cluster Bootstrapping: $B=2{,}000$ resamples clustered by receipt item ID.
   - Random seeds: 42, 43, 44.
   - Significance tests: Paired permutation test with $N_{\text{perm}}=100{,}000$ resamples.
   - Family-wise error control: Holm-Bonferroni correction within each query type family.
   - Permutation floor: Report $p < 10^{-4}$ or $p < 1e-5$ at the floor.
   - Random ranking baseline reported next to results ($MRR = 0.00895$).

---

## 6. Pre-Registered Decision Rules

1. **Replication of Verbatim Character Dominance:**
   - Supported if and only if: $\text{MRR}_{\text{char}} - \text{MRR}_{\text{word}} > 0$, the 95% bootstrap CI strictly excludes zero, and $p_{\text{Holm}} < 0.05$ on both clean and degraded EasyOCR.
2. **Replication of Paraphrase Reversal:**
   - Supported if and only if: $\text{MRR}_{\text{word}} - \text{MRR}_{\text{char}} > 0$, the 95% bootstrap CI strictly excludes zero, and $p_{\text{Holm}} < 0.05$ on light paraphrases.
3. **Synthesis in Replication Table:**
   - If both rules hold, mark claim as **REPLICATED (Across All 3 OCR Engines)** in `results/v9/replication.csv`.
   - If either fails, mark as **ENGINE-SPECIFIC (Apple Vision / Tesseract Only)**.

---

## 7. Frozen Deliverables

- `results/v9/sroie_easyocr_clean_ocr.json`
- `results/v9/sroie_easyocr_degraded_ocr.json`
- `results/v9/easyocr_error_rates_v9.csv`
- `results/v9/easyocr_samples_v9.md`
- `results/v9/easyocr_replication_v9.csv`
