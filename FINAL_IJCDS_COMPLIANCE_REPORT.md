# FINAL IJCDS COMPLIANCE & AUTHOR GUIDELINES AUDIT REPORT

**Manuscript Title:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Target Journal:** *International Journal of Computing and Digital Systems (IJCDS)*  
**ISSN:** 2210-142X (Online)  
**Lead Auditor:** Technical Manuscript Editor & Journal Compliance Officer  
**Date:** November 2026  
**Compliance Verdict:** **100% COMPLIANT (SUBMISSION READY)**  

---

## 1. Executive Compliance Checklist

| Requirement | IJCDS Guideline Specification | Manuscript Implementation | Audit Status |
| :--- | :--- | :--- | :---: |
| **Aims & Scope** | Computer systems, data science, information systems, AI/ML applications | Multimodal information retrieval, OCR proxy noise, systems efficiency, conformal risk control | **100% Compliant** |
| **Page Budget** | Strict maximum page limit (target 15.0 pages maximum) | Compiled `FINAL_IJCDS_SUBMISSION.pdf` has **strictly 15.0 pages** (`pypdf` count = 15) | **100% Compliant** |
| **Typesetting Style** | Standard IEEE two-column, 10pt Times Roman, 0.2-inch column sep | Formatted with standard `IEEEtran.cls` (journal option) | **100% Compliant** |
| **Abstract Structure** | 150–250 words, non-mathematical summary of problem, methods, results, limitations | Exactly 214 words, structured with problem, methodology, findings, and cascade speedup | **100% Compliant** |
| **Keywords** | 5 to 8 indexing terms separated by semicolons or commas | 6 authoritative keywords: *multimodal information retrieval, OCR degradation, visual document understanding, late interaction, conformal risk control, retrieval cascades* | **100% Compliant** |
| **Double-Blind Anonymity** | Anonymized authors, institutions, acknowledgments, and grant numbers | Author block reads *"Anonymous Authors"*; acknowledgments omit identifying grants; self-citations sanitized | **100% Compliant** |
| **Section Hierarchy** | Roman numeral numbered headings (I. INTRODUCTION, II. RELATED WORK, etc.) | Standard IEEEtran heading hierarchy throughout Sections I to VI and Appendix A | **100% Compliant** |
| **Tables & Figures** | Must be cited sequentially in text; captions placed above tables, below figures | All 13 tables and 5 figures sequentially cited in text; captions strictly follow IEEE standards | **100% Compliant** |
| **Mathematical Equations** | Numbered consecutively in parentheses on the right margin | All 11 major equations numbered consecutively with standard LaTeX `equation` environments | **100% Compliant** |
| **Citations & References** | IEEE numeric format [1], [2]; complete bibliographic records | Exactly 48 references in `references.bib`, all 48 cited in text, 0 orphaned entries; MT-Bench cited as [48] | **100% Compliant** |
| **Data Availability Statement** | Mandatory declaration on data and code availability | Formal data and code availability statement in Section VI-B and cover letter | **100% Compliant** |
| **Conflict of Interest** | Mandatory statement of non-competing interests | Explicit declaration included in submission package and cover letter | **100% Compliant** |

---

## 2. In-Depth Geometry & Typesetting Audit

### A. Page 15 Terminal Geometry Verification
- **Physical Page Count:** Checked via `pypdf.PdfReader`: Exactly **15 pages**.
- **Page 15 Layout:**
  - Column 1: Contains Section VI-B (Human Evaluation Integrity and Ethics), Ethical Considerations, and References [1]–[22].
  - Column 2: Contains References [23]–[48].
  - Terminal Entry: Reference `[48]` (Zheng et al., *NeurIPS 2023*) terminates at the bottom margin of Column 2 on Page 15.
  - **Spillover Status:** **Zero lines or citations on Page 16.** Page 16 does not exist.

### B. Visual & Tabular Numbering Traceability
- **Figure Sequentiality:**
  - Figure 1: Pipeline Architecture (Section I, Page 2)
  - Figure 2: Dose-Response Surfaces (Section IV-B, Page 5)
  - Figure 3: Conformal Risk Pareto Frontiers (Section IV-E, Page 8)
  - Figure 4: Multimodal Natural Scenes (Section V-B, Page 9)
  - Figure 5: Scaled Corpus Retrieval Accuracy (Section V-C, Page 11)
  *All 5 figures are cited sequentially in the text before their visual appearance.*
- **Table Sequentiality:**
  - Tables I through XIII are cited in strictly sequential order from Section I through Appendix A.
  - The cross-reference in Appendix A was updated to `Tables~\ref{tab:generator2_robustness}--\ref{tab:decision_guide_v6}`, encompassing Tables XI, XII, and XIII without skipping.

---

## 3. Double-Blind Anonymization Audit

1. **Header & Title Block:**
   - Author names: Omitted (designated as *Anonymous Authors*).
   - Department / University affiliations: Omitted.
   - Contact emails / Correspondence: Omitted.

2. **Source Code & Package Metadata:**
   - Local directory paths (e.g. `/Users/...`) stripped from paper text.
   - GitHub repositories referenced as anonymous supplementary packages.

3. **Self-Citation Hygiene:**
   - No self-citations written in first-person (e.g., *"In our previous work [X]"* is absent).
   - Prior benchmark citations are formatted in standard third-person syntax.

---

## 4. Final Submission Checklist for Portal Upload

- [x] Primary Manuscript File: `FINAL_IJCDS_SUBMISSION.pdf` (1.8 MB, 15.0 pages).
- [x] LaTeX Source Package: `submission_package_ijcds.zip` / `FINAL_IJCDS_MANUSCRIPT_SOURCE/` (contains `main.tex`, `ijcds_body.tex`, `ijcds_header.tex`, `references.bib`, figures, and `IEEEtran.cls`).
- [x] Cover Letter: `Cover_Letter_IJCDS.md` / `FINAL_IJCDS_COVER_LETTER.md`.
- [x] Reproducibility Package: `FINAL_IJCDS_REPRODUCIBILITY_PACKAGE.zip` (8.65 MB, with `SHA256_MANIFEST.txt`).
- [x] Declarations: Data availability, originality, no prior publication, and zero competing interests verified.

