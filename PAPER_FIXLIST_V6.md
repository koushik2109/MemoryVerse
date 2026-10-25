# PAPER_FIXLIST_V6.md: Publication Integrity & Revision Checklist

This document details all required textual, tabular, and bibliographic corrections in `IEEE_Paper_MemoryVerse.tex` based on the verified results from `results/v6/` and fragments in `latex_fragments_v6/`.

---

## 1. Stale Sections to Delete or Replace
1. **Abstract (Line 40):** Delete the stale claim that Word BM25 beats Character 3-5gram BM25 under optical OCR (`+0.0996 clean, +0.0777 blur`). Replace with `latex_fragments_v6/abstract_v6.tex`.
2. **Contributions (Line 77):** Delete Contribution 2 claiming Word BM25 is superior under optical OCR. Replace with `latex_fragments_v6/contributions_v6.tex`.
3. **Table V (Regime Grid, Lines 620--645):**
   - Replace the degenerate constant CLIP rows (`0.0436 [0.044, 0.044]`) with genuine CLIP evaluations from `latex_fragments_v6/table_clip_v6.tex`.
   - Update Word BM25, Char 3-5gram BM25, and Tuned Hybrid heavy paraphrase rows using `latex_fragments_v6/table_char_vs_word_bm25_v6.tex`.
4. **Table tab:decision_guide (Line 940):** Delete the unsupported row for `ColQwen2 Document-Native` (0.4580); replace Table with `latex_fragments_v6/table_decision_guide_v6.tex`.
5. **Section V-A (Lines 603, 862):** Delete inflated Tesseract error rates (CER = 0.8376, WER = 1.0481). Replace with standardized JiWER document error rates from `latex_fragments_v6/table_ocr_errors_v6.tex` (Clean CER = 0.2939, WER = 0.4879).
6. **Section V-D (Conformal Cascades, Lines 788, 963):** Replace claims of "coverage breakdown due to exchangeability violations" with the marginal loss formulation ($L = 1[	ext{not escalated} \land Y 
e 1]$) from `latex_fragments_v6/table_conformal_v6.tex`, showing mean test risk 0.0491 $\le 0.05$.
7. **Section V-E (Multi-Hop TOST, Lines 915--925):** Delete the TOST equivalence claim ($p_{	ext{TOST}} = 0.00000$ inside $\pm 0.05$). Replace with independent rule-based PRF vs. LLM paired analysis from `latex_fragments_v6/table_multihop_v6.tex` (exact McNemar $p = 1.0000$).
8. **Section V-A (Dense Collapse):** Replace the single-factor "tokenization fragmentation" explanation with query formulation sensitivity findings from `latex_fragments_v6/table_dense_formulation_v6.tex`.
9. **Eliminate All File Paths from Narrative:** Remove literal repo paths (e.g. `results/v3/...`, `dev_tuned_parameters_v3.json`) from body text; cite formal table labels instead.
10. **Eliminate Stale Legacy References:** Purge references to old audio Tier-1/Tier-2, RAMP-RAG, H1/H2 hypotheses, and private 55-probe album numbers.

---

## 2. Citation Audit: Errors and Missing Citations

| Citation Topic | Current Key / Status | Verification Status | Correction / Verified BibTeX Entry |
| :--- | :--- | :--- | :--- |
| **SROIE Challenge** | Cited as Tesseract in text | **VERIFIED** | Huang et al., "ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction", ICDAR 2019. |
| **MS-COCO Dataset** | Conflated with CLIP | **VERIFIED** | Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014. |
| **FUNSD Dataset** | Missing citation | **VERIFIED** | Jaume et al., "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents", ICDARW 2019. |
| **BGE Embeddings** | Missing citation | **VERIFIED** | Xiao et al., "C-Pack: Packaged Resources To Advance General Chinese Embedding", arXiv:2309.07597, 2023. |
| **ColQwen2 / ColPali** | Missing citation | **VERIFIED** | Faysse et al., "ColPali: Efficient Document Retrieval with Vision Language Models", arXiv:2407.01449, 2024. |
| **ViDoRe Benchmark** | Missing citation | **VERIFIED** | Faysse et al., "The Visual Document Retrieval Benchmark", HuggingFace, 2024. |
| **Conformal Risk Control**| Missing citation | **VERIFIED** | Bates, Angelopoulos, Lei, Malik, Jordan, "Distribution-Free Risk-Controlling Prediction Sets", JACM 2021. |
| **BLIP-2 Vision Proxy**| Missing citation | **VERIFIED** | Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models", ICML 2023. |
| **JiWER Standardization**| Missing citation | **VERIFIED** | Niklas Schneider, "JiWER: Similarity measures for automatic speech recognition evaluation", 2024. |
| **Rank-BM25** | Missing citation | **VERIFIED** | Robertson et al., "Okapi at TREC-3", NIST Special Publication, 1995. |

---

## 3. Verification Protocol Before Final Submission
- Ensure zero occurrences of hardcoded hex colors or non-semantic tokens in companion code.
- Verify `texi2pdf` / `tectonic` compilation produces zero fatal errors on `IEEE_Paper_MemoryVerse.tex`.
- Verify every number cited in the abstract and contributions matches `results/v6/number_provenance.csv`.
