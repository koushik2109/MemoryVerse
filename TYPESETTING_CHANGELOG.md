# Typesetting Change Log (IJCDS Camera-Ready)

**Manuscript Title:** When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise  
**Target Venue:** International Journal of Computing and Digital Systems (IJCDS)  
**Synchronized Source Files:**
- `submission_package_ijcds/source/ijcds_body.tex`
- `submission_package_ijcds/source/ijcds_header.tex`
- `submission_package_ijcds/Manuscript_MemoryVerse_IJCDS.pdf`
- `ijcds_body.tex` (repository root)
- `ijcds_header.tex` (repository root)
- `ijcds/ijcds_body.tex`
- `ijcds/ijcds_header.tex`
- `paper_v4/ijcds_body.tex`
- `paper_v4/ijcds_header.tex`
- `paper_v4/ijcds/ijcds_body.tex`
- `paper_v4/ijcds/ijcds_header.tex`

---

## [X] ITEM 1 — DONE: Fix Minimum Font Size Violation in Table III

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `./ijcds_body.tex`, `./ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Change:
- **Location:** Table III (`tab:dense_formulation_v6`) footer note.
- **Before:**
  ```tex
  \raggedright\tiny Note: Chunked MaxP evaluates 150-word sliding windows ($\le 256$ tokens); difference between unchunked and chunked is $<0.007$ MRR across all formulations (e.g., Answerable chunked: MiniLM 0.3669, BGE 0.5197), confirming sequence truncation does not explain dense collapse.
  ```
- **After:**
  ```tex
  \raggedright\scriptsize Note: Chunked MaxP evaluates 150-word sliding windows ($\le 256$ tokens); difference between unchunked and chunked is $<0.007$ MRR across all formulations (e.g., Answerable chunked: MiniLM 0.3669, BGE 0.5197), confirming sequence truncation does not explain dense collapse.
  ```

---

## [X] ITEM 2 — DONE: Fix Equation Numbering Sequence Break

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `./ijcds_body.tex`, `./ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Verification of Mathematical Sequence:
- **Sequence Mapping:**
  1. Equation (1): Single-stage dense cosine similarity $S_{\text{dense}}(q, d)$ (Section 2.A).
  2. Equation (2): Sparse BM25 retrieval score $S_{\text{sparse}}(q, d)$ (Section 2.A).
  3. Equation (3): Reciprocal Rank Fusion $\text{RRF}(q, d)$ (Section 2.A, Eq.~\ref{eq:rrf}).
  4. Equation (4): Cross-encoder neural reranking score $S_{\text{rerank}}(q, d)$ (Section 2.B).
  5. Equation (5): Bag-of-Words symmetric-difference rate $R_{\text{BoW}}$ (Section 5.E).
- **Status:** The sequence is unbroken (1, 2, 3, 4, 5). The $R_{\text{BoW}}$ formula is verified as Equation (5) in the compiled PDF without skips or manual counter resets.

---

## [X] ITEM 3 — DONE: Fix Math-Mode Kerning Glitches in Equation (2)

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `./ijcds_body.tex`, `./ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Change:
- **Location:** Equation (2) (Section 2.A, sparse BM25 scoring formulation).
- **Before:**
  ```tex
  \begin{equation}
      S_{\text{sparse}}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
  \end{equation}
  ```
- **After:**
  ```tex
  \begin{equation}
      S_{\text{sparse}}(q, d) = \sum_{t \in q} \mathrm{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\mathrm{avgdl}}\right)}
  \end{equation}
  ```
- **Typographic Impact:** Multi-letter terms $\mathrm{IDF}$ and $\mathrm{avgdl}$ are rendered in upright roman mathematical font ($\mathrm{IDF}$ and $\mathrm{avgdl}$), preventing italic math kerning from formatting them as multiplied scalar variables ($I \times D \times F$ and $a \times v \times g \times d \times l$).

---

## [X] ITEM 4 — DONE: Define Acronym in the Abstract

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `./ijcds_body.tex`, `./ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Change:
- **Location:** Abstract (line 1).
- **Before:**
  ```tex
  On natural images, native SigLIP (MRR 0.607) significantly surpasses machine captions, yet statistically matches four-reference human gold descriptions within $\pm 0.05$ TOST bounds ($p_{\text{TOST}}=0.005$).
  ```
- **After:**
  ```tex
  On natural images, native SigLIP (MRR 0.607) significantly surpasses machine captions, yet statistically matches four-reference human gold descriptions within $\pm 0.05$ Two One-Sided Tests (TOST) bounds ($p_{\text{TOST}}=0.005$).
  ```
- **Compliance:** Abstract length is 182 words (compliant with IJCDS $\le 200$ word strict limit).

---

## [X] ITEM 5 — DONE: Enforce Table Roman Numeral Consistency

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `./ijcds_body.tex`, `./ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Change:
- **Location:** Note directly beneath Table I (`tab:sota_comparison`, line 59).
- **Before:**
  ```tex
  Table~\ref{tab:sota_comparison} is a qualitative paradigm map, not a cross-system numerical comparison.
  ```
  *(Rendered as "Table 1" in compiled output).*
- **After:**
  ```tex
  Table~I is a qualitative paradigm map, not a cross-system numerical comparison.
  ```
- **Compliance:** Harmonized with `TABLE I.` Roman numeral heading convention used across the manuscript.

---

## [X] ITEM 6 — DONE: Clean Up Orphaned Transition Word in Section 7

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `./ijcds_body.tex`, `./ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Change:
- **Location:** Section 7 (*Limitations*), Limitation 4 (line 645).
- **Before:**
  ```tex
  \item \textbf{Hardware Serving Non-Equivalence and Latency Economics:} Latency and resource profiles reflect production deployment tiers (Apple M2 CPU for lexical indexes; Nvidia Tesla T4 GPU for late interaction). Conversely, late interaction demands substantial multi-vector storage ($\sim 59$\,KB/page) that scales linearly with corpus size.
  ```
- **After:**
  ```tex
  \item \textbf{Hardware Serving Non-Equivalence and Latency Economics:} Latency and resource profiles reflect production deployment tiers (Apple M2 CPU for lexical indexes; Nvidia Tesla T4 GPU for late interaction). Late interaction demands substantial multi-vector storage ($\sim 59$\,KB/page) that scales linearly with corpus size.
  ```

---

## [X] ITEM 7 — DONE: Fix Bibliography Layout Hack Placement

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `submission_package_ijcds/source/ijcds_header.tex`, and all synchronized project files.

### Exact Changes:
1. **Removed floating macro from body:**
   - **Location:** `ijcds_body.tex` right before `\begin{thebibliography}` (line 667).
   - **Removed snippet:**
     ```tex
     {\makeatletter\gdef\@openbib@code{\setlength{\itemsep}{0.5pt}\setlength{\parsep}{0pt}}\makeatother}
     ```
2. **Relocated cleanly to preamble:**
   - **Location:** `ijcds_header.tex` (preamble before `\begin{document}`).
   - **Added snippet:**
     ```tex
     \makeatletter
     \gdef\@openbib@code{\setlength{\itemsep}{0.5pt}\setlength{\parsep}{0pt}}
     \makeatother
     ```
- **Typographic Impact:** Eliminates mid-document macro redefinitions and prevents vertical spacing glitches before the reference list.

---

## Document Compilation & Verification Status

| Target File | Page Count | Status |
| :--- | :---: | :---: |
| `submission_package_ijcds/source/main.pdf` | **15 pages** | Pass (0 errors, 0 margin overflows) |
| `submission_package_ijcds/Manuscript_MemoryVerse_IJCDS.pdf` | **15 pages** | Pass (0 errors, 0 margin overflows) |
| `main.pdf` (root) | **15 pages** | Pass (0 errors, 0 margin overflows) |
| `ijcds/ijcds_master_doc.pdf` | **15 pages** | Pass (0 errors, 0 margin overflows) |
| `paper_v4/main.pdf` | **15 pages** | Pass (0 errors, 0 margin overflows) |
| `paper_v4/ijcds/ijcds_master_doc.pdf` | **15 pages** | Pass (0 errors, 0 margin overflows) |
| `ijcds_submission_package.zip` | Rebuilt (5.38 MB) | Clean submission bundle verified |
