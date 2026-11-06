# FINAL SUBMISSION CHECKLIST (IJCDS)
**Manuscript:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Journal:** International Journal of Computing and Digital Systems (IJCDS)  
**Authors:** Anurag Krishna Vennela and G. Sai Balaji Koushik (Department of Computer Science and Engineering, National Institute of Technology Warangal, India)  
**Date:** November 2026  

---

## 1. Manuscript Metadata & Structural Requirements

- [x] **Journal Name:** International Journal of Computing and Digital Systems (IJCDS)
- [x] **Target Submission System:** EDAS Conference and Journal Management System
- [x] **Manuscript Title:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*
- [x] **Authors & Affiliations:**
  - Author 1: Anurag Krishna Vennela (National Institute of Technology Warangal, India; Email: `av832145@student.nitw.ac.in`)
  - Author 2: G. Sai Balaji Koushik (National Institute of Technology Warangal, India; Email: `sg832128@student.nitw.ac.in`)
- [x] **Abstract Word Count:** 182 words (Complies with IJCDS 150–250 word limit).
- [x] **Keywords (Index Terms):** Document Retrieval, Vision-Language Models, Visual RAG, Optical Character Recognition, Dense Retrieval Failure, Conformal Prediction.
- [x] **Total Page Count:** **Exactly 15 pages** (Verified via `pdfinfo submission_package_ijcds/source/main.pdf`). Complies with IJCDS standard research article length (12–15 pages).
- [x] **Document Class & Style:** IEEEtran / IJCDS custom two-column layout (`IEEEtran.cls`, `ijcds_header.tex`, `report.cls`, `report.sty`).
- [x] **Fonts & Geometry:** 10pt Times New Roman equivalent, standard column gutters, fully justified.

---

## 2. Figures & Tables Inventory & Quality Check

| Item | Caption / Subject | Size / Float | Visual Quality | Data Verified | Status |
|---|---|---|---|---|---|
| **Fig. 1** | Retrieval Regime Decision Boundary Map | Single Column (`[t]`) | High-res PDF vector | Verified against 12 grid regimes | **PASSED** |
| **Fig. 2** | Subword Token Fragmentation vs Cosine Drift | Single Column (`[t]`) | Vector plot (dual axis) | Verified against WordPiece counts | **PASSED** |
| **Fig. 3** | Dose-Response Crossover Curve | Single Column (`[t]`) | Vector plot (3 generators) | Update N=1730 -> N=1930 in caption | **READY AFTER EDIT** |
| **Fig. 4** | Risk-Controlling Prediction Sets (RCPS) Pareto Front | Single Column (`[t]`) | Vector scatter / line | Verified over 200 calibration splits | **PASSED** |
| **Tab. I** | Headline Regime Grid (Clean vs Real OCR vs Degraded) | Two Column (`table*`) | Formatted cleanly | 18/18 cells match `headline_regime_grid_v3.csv` | **PASSED** |
| **Tab. II** | Recall@K Accuracy Comparisons | Single Column | Professional booktabs | All cells verified against test probes | **PASSED** |
| **Tab. III** | Chunked vs Unchunked Dense Sensitivity | Single Column | Professional booktabs | Paired delta verified ($\Delta < 0.007$) | **PASSED** |
| **Tab. IV** | Query Formulation Sensitivity (Verbatim vs Questions) | Single Column | Professional booktabs | All cells match `table_iv_corrected.csv` | **PASSED** |
| **Tab. V** | Holm-Bonferroni Hypothesis Families Summary | Two Column (`table*`) | Formatted cleanly | Update Family 4 & Family 6 p-values | **READY AFTER EDIT** |
| **Tab. VI** | MS-COCO Native Visual vs 4-Ref Human Gold | Single Column | Professional booktabs | TOST equivalence verified ($p=0.0050$) | **PASSED** |
| **Tab. VII** | External FUNSD Layout Transfer Benchmark | Single Column | Professional booktabs | Verified on Tesseract & EasyOCR | **PASSED** |
| **Tab. VIII** | DocVQA Native Visual vs OCR Text Proxy | Single Column | Professional booktabs | ColQwen2 gain verified ($+0.2472$) | **PASSED** |
| **Tab. IX** | Two-Stage ColQwen2 Cascade Efficiency | Single Column | Professional booktabs | MRR verified; clarify modeled latency | **READY AFTER EDIT** |
| **Tab. X** | Relational Multi-Hop Retrieval & Query Bridging | Single Column | Professional booktabs | Extractive PRF vs LLM verified | **PASSED** |
| **Tab. XI** | Query Overlap Dose-Response MRR Breakdown | Single Column | Professional booktabs | Update sample size to 1,930 | **READY AFTER EDIT** |
| **Tab. XII** | Corpus Scaling Behavior to 500,000 Documents | Single Column | Professional booktabs | MRR values verified; disclose distractor model | **READY AFTER EDIT** |
| **Tab. XIII** | End-to-End RAG Answer Accuracy & Human Agreement | Single Column | Professional booktabs | Disentangle 100 RAG vs 110 album audit | **READY AFTER EDIT** |

---

## 3. References & Citation Integrity

- [x] **Total References in `.tex`:** 48 bibliography items.
- [x] **Total Cited References:** 47 references currently cited in text.
- [x] **Orphaned Reference Identified:** `\bibitem{zheng2023judging}` (L. Zheng et al., "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena", NeurIPS 2023).
  - *Action:* Insert `\cite{zheng2023judging}` into Section 6.2 as specified in `MANUSCRIPT_CORRECTIONS.md`.
  - *Result after fix:* Exactly 48 / 48 references cited ($100\%$ citation integrity).
- [x] **Reference Formatting:** Complete author lists, full publication venues, page numbers, and publication years present for all 48 entries.

---

## 4. Reproducibility & Open Science Readiness

- [x] **Public Benchmark Dataset:** Complete 1,626-item benchmark with disjoint 813/813 splits present in `ai_engine/rag/data/v2/public_benchmark_v2.json`.
- [x] **Evaluation Scripts:** All evaluation and table generation scripts present in `scripts/`.
- [x] **Deterministic Seeds:** Random seeds (`seed=42`, splits `42--241`) documented and verified.
- [x] **Credential Safety:** Checked `.env`, code, and logs: zero API keys or secrets present.
- [x] **Hardware Transparency:** CPU (Apple M2) vs GPU (Tesla T4) execution environments properly differentiated.

---

## 5. EDAS Submission Protocol

When submitting via EDAS:
1. **Track / Conference:** International Journal of Computing and Digital Systems (Regular Issue).
2. **Manuscript PDF:** Use compiled `submission_package_ijcds/source/main.pdf` (ensure all text corrections from `MANUSCRIPT_CORRECTIONS.md` are compiled into the PDF prior to upload).
3. **PDF Compliance Check:** Ensure PDF passes EDAS PDF eXpress / font embedding checks (Type 1 or TrueType fonts only; no Type 3 bitmap fonts). All fonts in `main.pdf` are embedded Type 1 fonts.
4. **Author Bios & Photos:** Ensure short biographical sketches of Anurag Krishna Vennela and G. Sai Balaji Koushik are ready for camera-ready submission.

---
**Auditor Signature:** Independent Statistical & Reproducibility Auditor  
**Date:** November 4, 2026

