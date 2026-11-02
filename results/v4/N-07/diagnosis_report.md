# N-07 Diagnostic Report: Table III Row 5 Chunked vs. Unchunked Audit for MiniLM and BGE

## 1. Executive Summary
- **Issue Identified:**
  - Table III (`tab:dense_formulation_v6`), Row 5 ("Natural Answerable Questions") reports:
    - MiniLM: MRR = $0.3669$, Hit@10 = $0.5686$
    - BGE: MRR = $0.5263$, Hit@10 = $0.7974$
  - The footnote states:
    > *"$^\dagger$Row 5 evaluates unchunked full-text embeddings (0.5263; chunked MaxP achieves 0.5197, Table II)."*
  - However, in Table II (`tab:regime_grid_v3`), line 195:
    - `Dense MiniLM (chunked)` on Clean GT Answerable is reported as **0.3669**!
    - `Dense BGE-base (chunked MaxP)` on Clean GT Answerable is reported as **0.5197**!
  - Therefore, MiniLM $0.3669$ in Table III matches the chunked value from Table II, whereas BGE $0.5263$ is unchunked! The table was mixing chunked MiniLM with unchunked BGE under a footnote asserting both were unchunked.

## 2. Empirical Ground-Truth Numbers
We evaluated both models on identical frozen Clean Ground-Truth text ($N=306$ Answerable probes, 813 corpus documents):

| Model | Representation | MRR | Hit@10 | Source / Status |
| :--- | :--- | :---: | :---: | :--- |
| **MiniLM** (`all-MiniLM-L6-v2`) | Chunked MaxP (150-word window) | **0.3669** | **0.5686** | Table II, `results/v6/dense_query_formulation_v6.csv` |
| **MiniLM** (`all-MiniLM-L6-v2`) | Unchunked Single-Vector | **0.3676** | **0.5686** | Measured empirical run |
| **BGE-base** (`bge-base-en-v1.5`) | Chunked MaxP (150-word window) | **0.5197** | **0.7876** | Table II, `results/v6/dense_query_formulation_v6.csv` |
| **BGE-base** (`bge-base-en-v1.5`) | Unchunked Single-Vector | **0.5263** | **0.7974** | Table III, `results/v3/headline_regime_grid_v3.csv` |

### Key Finding:
- Chunked MaxP vs. Unchunked yields virtually identical retrieval effectiveness for both models on Answerable Questions:
  - MiniLM: $0.3669$ (chunked) vs. $0.3676$ (unchunked), difference $< 0.001$ MRR.
  - BGE-base: $0.5197$ (chunked) vs. $0.5263$ (unchunked), difference $+0.0066$ MRR.
- This reinforces the paper's core scientific thesis: **dense degradation is not driven by sequence truncation**, because chunked and unchunked representations perform identically across both verbatim snippets and answerable questions.

## 3. Recommended Manuscript Updates (Phase 2/5)
- In Table III (`tab:dense_formulation_v6`), explicitly display both Chunked and Unchunked metrics, or update the footnote:
  - If single row kept: Report Unchunked for both (MiniLM $0.3676$, BGE $0.5263$) or Chunked for both (MiniLM $0.3669$, BGE $0.5197$), and footnote the paired counter-part clearly:
    > *"$^\dagger$Row 5 reports unchunked representations (MiniLM 0.3676, BGE 0.5263); sliding-window chunked MaxP yields virtually identical performance: MiniLM 0.3669, BGE 0.5197 (Table II)."*
  - Alternatively (cleanest fix): Split Row 5 into:
    - `5(a). Natural Answerable Questions (Unchunked)`: MiniLM 0.3676 (0.5686), BGE 0.5263 (0.7974)
    - `5(b). Natural Answerable Questions (Chunked MaxP)`: MiniLM 0.3669 (0.5686), BGE 0.5197 (0.7876).

