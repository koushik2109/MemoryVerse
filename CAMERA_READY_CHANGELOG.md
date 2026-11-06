# Camera-Ready Change Log

**Manuscript Title:** When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise  
**Target Venue:** International Journal of Computing and Digital Systems (IJCDS)  
**Synchronized Source Files:**
- `submission_package_ijcds/source/ijcds_body.tex`
- `submission_package_ijcds/Manuscript_MemoryVerse_IJCDS.pdf`
- `ijcds_body.tex` (repository root)
- `ijcds/ijcds_body.tex`
- `paper_v4/ijcds_body.tex`
- `paper_v4/ijcds/ijcds_body.tex`

---

## [X] Item 1 — DONE: RAG Scoping & Keyword Alignment

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `ijcds_body.tex`, `ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Changes:

1. **Keywords Block:**
   - **Before:**
     ```tex
     \textbf{Keywords:} Multimodal RAG, Text Proxies, Proxy Noise, Dense-Sparse Regime Analysis, Cascade Ranking, Conformal Risk Control, Document Retrieval. \vspace{1mm} \hrule}
     ```
   - **After:**
     ```tex
     \textbf{Keywords:} Text Proxies, Proxy Noise, Dense-Sparse Regime Analysis, Cascade Ranking, Conformal Risk Control, Document Retrieval. \vspace{1mm} \hrule}
     ```

2. **Abstract Opening:**
   - **Before:**
     ```tex
     \textbf{Abstract:} Multimodal retrieval systems often index textual proxies, such as OCR output or generated captions, instead of raw visual media.
     ```
   - **After:**
     ```tex
     \textbf{Abstract:} Multimodal document retrieval systems often index textual proxies, such as OCR output or generated captions, instead of raw visual media.
     ```

3. **Introduction Paragraph:**
   - **Before:**
     ```tex
     Recently, multimodal extensions of RAG have emerged to handle non-textual data, as surveyed by Abootorabi et~al.~\cite{abootorabi2025multimodal}. For example, Patel et~al.~\cite{patel2025multimodal} and Joshi et~al.~\cite{joshi2024multimodel} presented enterprise multimodal RAG pipelines combining OCR ingestion, multimodal embeddings, and LLM reasoning.
     ```
   - **After:**
     ```tex
     Recently, multimodal document retrieval systems have emerged to handle non-textual data, as surveyed by Abootorabi et~al.~\cite{abootorabi2025multimodal}. For example, Patel et~al.~\cite{patel2025multimodal} and Joshi et~al.~\cite{joshi2024multimodel} presented enterprise document-retrieval pipelines combining OCR ingestion, multimodal embeddings, and LLM reasoning.
     ```
   *(Note: Motivational introduction to RAG in paragraph 2 is preserved).*

4. **Section 2.B Heading and Opening:**
   - **Before:**
     ```tex
     \subsection{Multimodal RAG and Document Analysis}
     Integrating multimodal inputs into RAG workflows has advanced rapidly with Vision-Language Models (VLMs) and document foundation models.
     ```
   - **After:**
     ```tex
     \subsection{Multimodal Document Retrieval and Analysis}
     Multimodal document retrieval has advanced rapidly with Vision-Language Models (VLMs) and document foundation models.
     ```

5. **Figure 1 Caption & Reference Architecture:**
   - **Before:**
     ```tex
     \caption{Conceptual reference architecture of the text-proxy RAG framework.
     The reference Multi-Stage Multimodal RAG architecture is structured into four functional phases (Fig.~\ref{fig:architecture}):
     ```
   - **After:**
     ```tex
     \caption{Conceptual reference architecture of the text-proxy retrieval framework.
     The reference Multi-Stage Multimodal Document Retrieval architecture is structured into four functional phases (Fig.~\ref{fig:architecture}):
     ```

6. **Section 5 (Empirical Analysis) Prose:**
   - **Before:**
     ```tex
     To evaluate whether proxy corruption or semantic query mismatch is the dominant bottleneck in text-proxy RAG, we systematically decouple these effects:
     ```
   - **After:**
     ```tex
     To evaluate whether proxy corruption or semantic query mismatch is the dominant bottleneck in text-proxy retrieval, we systematically decouple these effects:
     ```

7. **Table XIII (Decision Guide) Caption:**
   - **Before:**
     ```tex
     \caption{Empirical Decision Guide for MM-RAG System Builders: Optimal Retrieval Architecture and Statistically Indistinguishable Set ($p > 0.05$) by Regime.}\label{tab:decision_guide_v6}
     ```
   - **After:**
     ```tex
     \caption{Empirical Decision Guide for Multimodal Retrieval System Builders: Optimal Retrieval Architecture and Statistically Indistinguishable Set ($p > 0.05$) by Regime.}\label{tab:decision_guide_v6}
     ```

8. **Section 8 (Conclusion) Opening:**
   - **Before:**
     ```tex
     In this work, we have established that intermediate textual proxies in multimodal RAG exhibit well-defined, regime-dependent efficacy boundaries.
     ```
   - **After:**
     ```tex
     In this work, we have established that intermediate textual proxies in multimodal document retrieval exhibit well-defined, regime-dependent efficacy boundaries.
     ```

---

## [X] Item 2 — DONE: DocVQA "In-Domain" Caveat Enforcement

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `ijcds_body.tex`, `ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Changes:

1. **Abstract:**
   - **Before:**
     ```tex
     On a 500-query DocVQA subset, ColQwen2 reaches MRR 0.6450, while tested text-proxy baselines reach at most 0.3507.
     ```
   - **After:**
     ```tex
     On a 500-query DocVQA subset, ColQwen2 reaches MRR 0.6450, while tested text-proxy baselines reach at most 0.3507 (though ColQwen2's advantage represents an in-domain upper bound due to pretraining overlap).
     ```

2. **Conclusion (Section 8):**
   - **Before:**
     ```tex
     on visual pages (DocVQA), native ColQwen2 substantially outperforms generic proxies (0.6450 vs.\ $\le 0.3507$); yet on receipts
     ```
   - **After:**
     ```tex
     on visual pages (DocVQA), native ColQwen2 substantially outperforms generic proxies (0.6450 vs.\ $\le 0.3507$), though ColQwen2's advantage represents an in-domain upper bound due to pretraining overlap; yet on receipts
     ```

---

## [X] Item 3 — DONE: Word BM25 Stemming Justification

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `ijcds_body.tex`, `ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Changes:

**Section 4.B (OCR Engines and Optical Degradation Conditions):**
- **Before:**
  ```tex
  Word BM25 uses standard whitespace and punctuation regex tokenization (\texttt{\textbackslash w+}), lowercase normalization, and the standard NLTK 179-word English stoplist without stemming (preserving alphanumeric serials, prices, and part numbers).
  ```
- **After:**
  ```tex
  Word BM25 uses standard whitespace and punctuation regex tokenization (\texttt{\textbackslash w+}), lowercase normalization, without stemming or stopword removal (to strictly preserve exact matches on alphanumeric serials, prices, and vendor codes, which stemming algorithms frequently destroy).
  ```

---

## [X] Item 4 — DONE: Prune Exploratory Audio/Photo Pilots (Section 6.C)

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `ijcds_body.tex`, `ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Changes:

1. **Section 6.C Deletion:**
   - **Before:**
     ```tex
     \subsection{Dual-Tier Feasibility Pilots and Systemic Extensions}
     Prior to establishing the frozen benchmark, exploratory pilots evaluated multimodal proxies across text ($N=334$) and physical media ($N=151$ probes: $N=85$ scene text, $N=13$ ESC-50 audio recordings \cite{piczak2015esc50} via LAION-CLAP \cite{wu2023clap}, and $N=53$ album photos via CLIP ViT-B/32 \cite{radford2021clip}). On physical media, sparse BM25 achieved MRR = 0.7916 on scene text, CLAP achieved MRR = 1.0000 on audio, and CLIP achieved Hit@10 = 0.792 (MRR = 0.6182) on raw photos, confirming modality-specific proxy bounds.
     ```
   - **After:** Completely deleted. Section 7 follows Section 6.B directly.

2. **Section 3.A Ingestion Item Deletion:**
   - **Before:**
     ```tex
     \begin{itemize}
         \setlength{\itemsep}{0.2pt plus 0.1pt}
         \setlength{\parskip}{0pt}
         \item \textbf{Documents and Forms:} Administrative forms and receipts are processed through document decomposition, optional Otsu binarization \cite{otsu1979} and Hough skew correction \cite{hough1962}, extracting text $T_d$ and bounding boxes via OCR engines (Apple Vision \cite{apple2020vision}, Tesseract 5.5 \cite{smith2007tesseract}, or EasyOCR \cite{jaidedai2020easyocr}).
         \item \textbf{Photographs:} Natural photographs are indexed via machine captions alongside native visual embeddings (CLIP \cite{radford2021clip}, SigLIP \cite{zhai2023siglip}).
         \item \textbf{Acoustic Media:} Continuous audio is transcribed using Whisper \cite{radford2023whisper} or embedded using contrastive audio representations (CLAP \cite{wu2023clap}).
     \end{itemize}
     ```
   - **After:**
     ```tex
     \begin{itemize}
         \setlength{\itemsep}{0.2pt plus 0.1pt}
         \setlength{\parskip}{0pt}
         \item \textbf{Documents and Forms:} Administrative forms and receipts are processed through document decomposition, optional Otsu binarization \cite{otsu1979} and Hough skew correction \cite{hough1962}, extracting text $T_d$ and bounding boxes via OCR engines (Apple Vision \cite{apple2020vision}, Tesseract 5.5 \cite{smith2007tesseract}, or EasyOCR \cite{jaidedai2020easyocr}).
         \item \textbf{Photographs:} Natural photographs are indexed via machine captions alongside native visual embeddings (CLIP \cite{radford2021clip}, SigLIP \cite{zhai2023siglip}).
     \end{itemize}
     ```

3. **Bibliography Pruning:**
   - Uncited entries `\bibitem{wu2023clap}` (LAION-CLAP) and `\bibitem{piczak2015esc50}` (ESC-50) were completely removed from `\begin{thebibliography}`.

---

## [X] Item 5 — DONE: Frame Conformal Risk as a Strict Negative Result

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `ijcds_body.tex`, `ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Changes:

1. **Introduction (Key Contributions — Contribution 2):**
   - **Before:**
     ```tex
     Through controlled query ablation ($N=306$) and multi-generator dose-response ($N=1{,}730$), we show this reversal is generator-invariant and driven by entity co-specification mass under controlled ablation rather than syntactic phrasing ($p = 0.6617$).
     ```
   - **After:**
     ```tex
     Through controlled query ablation ($N=306$) and multi-generator dose-response ($N=1{,}730$), we show this reversal is generator-invariant and driven by entity co-specification mass under controlled ablation rather than syntactic phrasing ($p = 0.6617$). As a negative result, standard marginal conformal gating fails on semantic subgroups (entity-rewrite miss risk $0.0627 > 0.05$), justifying the simpler fixed-margin cascade.
     ```

2. **Section 8 (Conclusion):**
   - **Before:**
     ```tex
     In cascade risk calibration, marginal conformal prediction achieved 58.5\% empirical split coverage against the 0.05 target, performing comparably to a simple fixed-margin baseline (MRR 0.3579 vs.\ 0.3577).
     ```
   - **After:**
     ```tex
     As a negative result, standard marginal conformal gating fails on semantic subgroups despite 58.5\% empirical split coverage against the 0.05 target: entity rewrites experience miss risk $0.0627 > 0.05$, while an equal-escalation fixed-margin cascade performs comparably (MRR 0.3579 vs.\ 0.3577), justifying the simpler fixed-margin cascade.
     ```

---

## [X] Item 6 — DONE: Tone Down Multi-Hop "Equivalence"

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `ijcds_body.tex`, `ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Changes:

1. **Abstract:**
   - **Before:**
     ```tex
     Query formulation strongly affects dense retrieval: BGE-Base MRR rises from 0.3256 on short snippets to 0.5263 on natural answerable questions. Results suggest selecting retrieval methods by query type and modality.
     ```
   - **After:**
     ```tex
     Query formulation strongly affects dense retrieval: BGE-Base MRR rises from 0.3256 on short snippets to 0.5263 on natural answerable questions. On a 100-query relational benchmark, extractive PRF statistically matches generative LLM bridging while eliminating a $394\times$ latency multiplier. Results suggest selecting retrieval methods by query type and modality.
     ```

2. **Section 6.B (Exploratory Multi-Hop Query Decomposition):**
   - **Before:** Contained detailed discordant pair counts for the underpowered 50-query hard subset and standalone `table:multihop_paired_v3`.
   - **After:**
     ```tex
     \subsection{Exploratory Multi-Hop Query Decomposition}
     \label{sec:multihop_reasoning}
     On the pooled $N=100$ relational multi-hop benchmark, extractive PRF and generative LLM achieve Hit@10 = $0.8200$ (82/100; paired difference $0.0000$ [95\% CI: $-0.0392, +0.0392$], formal TOST equivalence at $\Delta = \pm 0.05$, $p_{\text{TOST}} = 0.0073 < 0.01$). The initial 50-query subset remains underpowered ($p_{\text{TOST}}=0.1109$), so no equivalence claim is made for it; the extractive bridge eliminates the measured 394$\times$ latency multiplier. In a separate end-to-end question-answering audit ($N=150$ queries), Oracle context achieves ANLS = 0.5598 (Exact Match 0.5600), native ColQwen2 achieves ANLS = 0.3990 (EM 0.2933), and Char 3--5g BM25 achieves ANLS = 0.3246 (EM 0.2400). Dual-reviewer human auditing between two author-reviewers on 110 RAG answers generated by \texttt{gpt-4o-mini} confirmed high inter-rater agreement: observed agreement of 95.5\% for faithfulness ($\kappa = 0.8716$), 96.4\% for relevance ($\kappa = 0.8926$), and 98.2\% for correctness ($\kappa = 0.9451$; mean $\kappa = 0.9031$ \cite{zheng2023judging}).
     ```
   - Removed standalone `table:multihop_paired_v3` (underpowered 50-query diagnostic table).

3. **Section 8 (Conclusion):**
   - **Before:**
     ```tex
     On expanded relational multi-hop queries ($N=100$), extractive PRF is formally equivalent to generative LLM bridging under TOST (Hit@10 0.820 vs.\ 0.820, $p_{\text{TOST}}=0.0062 < 0.01$ at $\Delta=\pm 0.05$) while eliminating 394$\times$ LLM latency.
     ```
   - **After:**
     ```tex
     On a 100-query relational benchmark, extractive PRF statistically matches generative LLM bridging (Hit@10 0.820 vs.\ 0.820, $p_{\text{TOST}}=0.0073 < 0.01$ at $\Delta=\pm 0.05$) while eliminating a $394\times$ latency multiplier.
     ```

---

## [X] Item 7 — DONE: Fix Figure 1 Heading Numerals

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `ijcds_body.tex`, `ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Changes:

**Figure 1 Caption:**
- **Before:**
  ```tex
  Modular production components (e.g., dual-signal router, conformal cascade gate) are empirically evaluated and ablated in Section~V-C.
  ```
- **After:**
  ```tex
  Modular production components (e.g., dual-signal router, conformal cascade gate) are empirically evaluated and ablated in Section~5.C.
  ```

---

## [X] Item 8 — DONE: Scrub Unproven Causal Language

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `ijcds_body.tex`, `ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Changes:

1. **Section 5.D (Natural Photo Benchmark on MS-COCO):**
   - **Before:**
     ```tex
     the empirical deficit on photographic scenes is attributable to machine captioner quality rather than text as an information modality (Fig.~\ref{fig:coco_native_vs_proxy}).
     ```
   - **After:**
     ```tex
     the empirical deficit on photographic scenes is strongly associated with machine captioner quality rather than text as an information modality (Fig.~\ref{fig:coco_native_vs_proxy}).
     ```

2. **Section 5.E (Boundary Conditions and Robustness):**
   - **Before:**
     ```tex
     This indicates that the Character-to-Word BM25 reversal is specifically driven by entity-substitution paraphrasing rather than generalized paraphrasing.
     ```
   - **After:**
     ```tex
     This indicates that the Character-to-Word BM25 reversal is strongly correlated with entity-substitution paraphrasing rather than generalized paraphrasing.
     ```

---

## [X] Item 9 — DONE: Remove Cross-Hardware Millisecond Comparison

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `ijcds_body.tex`, `ijcds/ijcds_body.tex`, `paper_v4/ijcds_body.tex`, `paper_v4/ijcds/ijcds_body.tex`

### Exact Text Changes:

**Section 7 (Limitations and Threats to Validity — Limitation 4):**
- **Before:**
  ```tex
  \item \textbf{Hardware Serving Non-Equivalence and Latency Economics:} Latency and resource profiles reflect production deployment tiers (Apple M2 CPU for lexical indexes; Nvidia Tesla T4 GPU for late interaction). On the matched GPU host CPU, BM25 requires 0.96\,ms vs.\ ColQwen2 $>8{,}000$\,ms; conversely, late interaction demands substantial multi-vector storage ($\sim 59$\,KB/page) that scales linearly with corpus size.
  ```
- **After:**
  ```tex
  \item \textbf{Hardware Serving Non-Equivalence and Latency Economics:} Latency and resource profiles reflect production deployment tiers (Apple M2 CPU for lexical indexes; Nvidia Tesla T4 GPU for late interaction). Conversely, late interaction demands substantial multi-vector storage ($\sim 59$\,KB/page) that scales linearly with corpus size.
  ```

---

## Build Verification Status

- `submission_package_ijcds/source/main.pdf`: **15 pages**, 0 errors, 0 margin overflows.
- `submission_package_ijcds/Manuscript_MemoryVerse_IJCDS.pdf`: **15 pages**, 0 errors, 0 margin overflows.
- `main.pdf` (root): **15 pages**, 0 errors, 0 margin overflows.
- `ijcds/ijcds_master_doc.pdf`: **15 pages**, 0 errors, 0 margin overflows.
- `paper_v4/main.pdf`: **15 pages**, 0 errors, 0 margin overflows.
- `ijcds_submission_package.zip`: **Rebuilt and verified** (clean structure with `Manuscript_MemoryVerse_IJCDS.pdf`, `Cover_Letter_IJCDS.docx`, `Cover_Letter_IJCDS.md`, `README_SUBMISSION.md`, and clean `source/` folder).

---

## [X] Item 10 — DONE: Peer Review Consistency & Rigor Revisions (6.5/10 to 8/10+ Audit)

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `submission_package_ijcds/source/references.bib`, `scripts/make_coco_figure.py`, and all synchronized project files.

### 1. Internal Contradictions & Figure Alignments Resolved:
- **Figure 5 vs Table VI & Text:** Synchronized with `conformal_cascade_v6.csv`: $\tau^*=0.2369$, 78.4\% escalation, 58.5\% empirical split coverage, and 82.0\% RCPS coverage.
- **Figure 3 Replacement (`fig_generator_dose_response.png`):** Replaced U-shaped legacy figure with multi-generator dose-response curve ($N=1{,}730$ queries from GPT-4o-mini, Qwen2.5-7B, and authentic human annotators) demonstrating the clean monotonic crossover at $o^* \approx 0.52$ and generator invariance.
- **Figure 4 (COCO) Alignment & Bounding Margin:** Adjusted `x_max = 0.80` with uniform 0.2 tick intervals in `scripts/make_coco_figure.py`, providing a clean 40px margin around `0.621` (resolving bounding line collision). Explicitly defined human gold standard in figure caption (`single independent description vs. 4-reference concatenated descriptions`).
- **Table IV Caption Arithmetic:** Corrected from `drops by only -0.0024 to 0.8379` to `drops by only -0.0024 to 0.8347 under $K_1=10$ candidate truncation`.
- **Table II Synthetic Noise Column for Vision Models:** Replaced copied clean values (`0.0590$^\dagger$`, `0.8417$^\dagger$`, `0.0381$^\dagger$`, `0.4962$^\dagger$`, `0.0338$^\dagger$`, `0.8595$^\dagger$`) with `---$^\dagger$` to prevent improper cross-modality comparison.
- **Table V / Section 5.C p-values:** Explicitly distinguished raw $p$-values from Holm-adjusted $p_{\text{Holm}}$ in Table V and prose: MiniLM $+0.0279$ ($p=0.0090, p_{\text{Holm}}=0.0540$, borderline); BGE $+0.0475$ ($p=0.0011, p_{\text{Holm}}=0.0077$).
- **Table VII / Section 5.D p-values:** Harmonized CLIP vs. Florence-2 to raw $p=0.0380, p_{\text{Holm}}=0.0760$ (non-significant after correction).
- **Section 6.A (H1 Reliability Weighting):** Clarified raw $p=0.0189$ vs. Holm-adjusted $p_{\text{Holm}}=0.0945$.
- **500K Scaling Benchmark Test:** Identified test as paired asymptotic $z$-test ($z=8.12, p < 10^{-15}$; permutation test resolution floor $p < 10^{-5}$) and softened claim to consistently maintaining $+0.100$ to $+0.169$ MRR advantage across 4 orders of magnitude.
- **ColQwen2 Memory at Scale:** Disambiguated ~3 GB compressed disk index (50,000 pages at ~59 KB/page) from >100 GB uncompressed in-memory VRAM allocation during unpruned PyTorch inference without PLAID/quantization.
- **ANLS vs Exact Match:** Corrected Oracle performance to `ANLS = 0.5600 (Exact Match 0.5600)` in Section 6.B.
- **Retrieval MRR vs QA:** Clarified that the $N=150$ E2E QA suite includes 50 multi-column DocVQA pages where layout geometry governs answerability, explaining why ColQwen2's pooled MRR (0.6978 vs 0.5934) drives superior downstream ANLS (0.3990 vs 0.3246).
- **Conclusion Latency Ratio:** Specified exact comparison: `44x lower latency on answerable queries (5.72 ms vs 249.21 ms; 286x on verbatim lookups)`.
- **Limitation 3 (RCPS & Conformal Risk):** Corrected conformal prediction framing: marginal calibration guarantees expected risk $E[L] \le \alpha$, so 41.5% splits exceeding $\alpha$ is expected finite-sample variance; RCPS provides PAC-style high-probability control ($1 - \delta = 0.90$).

### 2. IJCDS Compliance & Structural Polish:
- **Abstract Condensation:** Compressed abstract from 249 words to 178 words ($\le 200$ word IJCDS strict limit), grounding in authentic human queries, rounding numbers to 3 decimals, and retaining all core findings.
- **Figure 2 Caption:** Changed `MRR@10` to `MRR` to match tables.
- **Human Annotation & Multi-Hop Procedure:** Documented annotator procedure in Section 4.B/4.C for the 400 human queries and 100 relational multi-hop queries.
- **Broken Bibliography Quotes:** Fixed missing closing quotes in Fateh 2024 (`fateh2024ocr`), Cormack 2009 (`cormack2009rrf`), and OpenAI 2024 (`openai2024gpt4o`).
- **Citation of Cohen (1960):** Added `\bibitem{cohen1960coefficient}` for inter-annotator $\kappa$ agreement.
- **Appendix A Family Enumeration:** Completed Family 4 ($m=7$) and corrected Family 2 verbatim tied test ($p=0.7420, p_{\text{Holm}}=1.0000$).

---

## [X] Item 11 — DONE: Table VII Justification Stretch & Multi-Line Overflow Fix

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `submission_package_ijcds/Manuscript_MemoryVerse_IJCDS.pdf`, and all synchronized project files.

### Identified Defect:
- In Table VII (`tab:coco_benchmarks_v3`), the first column was specified as `p{2.3cm}`. Because standard LaTeX `p{...}` columns use fully-justified text, strings that wrapped (such as `Baseline BLIP (MiniLM)`) suffered severe inter-word stretching on line 1 (`Baseline          BLIP 0`), while `(MiniLM)` overflowed onto line 2.
- Similarly, `Baseline BLIP (BGE)`, `Single Gold (BGE)`, and `4-Ref Multi-Caption` wrapped across multiple lines, causing awkward vertical misalignment with the adjacent single-line metric columns (`Hit@1` and `Hit@5`).

### Exact Changes:
1. **Column Specification:**
   - Changed from `@{}p{2.3cm}cccc@{}` to `@{}lcccc@{}` with `\setlength{\tabcolsep}{1.8pt}`. This enforces natural left alignment for method names, completely eliminating justification whitespace stretching and right-edge collision with numeric columns.
2. **Method Nomenclature Alignment:**
   - Updated method labels to concise, canonical names matching Figure 4 and Table V:
     - `Baseline BLIP (MiniLM)` $\rightarrow$ `BLIP + MiniLM`
     - `Baseline BLIP (BGE)` $\rightarrow$ `BLIP + BGE`
     - `4-Ref Multi-Caption` $\rightarrow$ `4-Ref Gold`
   - Every method name now renders on a single line with zero vertical overflow.
3. **Verification:**
   - Table VII spans $x=[71.2, 274.1]$\,pt, perfectly centered within the printable column width ($[56.7, 288.7]$\,pt).
   - Document compiled cleanly to **strictly 15 pages** across all 5 targets with **0 margin overflows**.
   - Rebuilt `submission_package_ijcds/Manuscript_MemoryVerse_IJCDS.pdf` and `ijcds_submission_package.zip`.

---

## [X] Item 12 — DONE: Editorial Polish (Orphaned "Conversely" & Table Roman Numerals)

**Target Files:** `submission_package_ijcds/source/ijcds_body.tex`, `submission_package_ijcds/Manuscript_MemoryVerse_IJCDS.pdf`, and all synchronized project files.

### 1. Limitation 4 Transition Word Cleanup:
- **Before:**
  ```tex
  \item \textbf{Hardware Serving Non-Equivalence and Latency Economics:} Latency and resource profiles reflect production deployment tiers (Apple M2 CPU for lexical indexes; Nvidia Tesla T4 GPU for late interaction). Conversely, late interaction demands substantial multi-vector storage ($\sim 59$\,KB/page) that scales linearly with corpus size.
  ```
- **After:**
  ```tex
  \item \textbf{Hardware Serving Non-Equivalence and Latency Economics:} Latency and resource profiles reflect production deployment tiers (Apple M2 CPU for lexical indexes; Nvidia Tesla T4 GPU for late interaction). Late interaction demands substantial multi-vector storage ($\sim 59$\,KB/page) that scales linearly with corpus size.
  ```

### 2. Table I Note Roman Numeral Alignment:
- **Before:**
  ```tex
  Table~\ref{tab:sota_comparison} is a qualitative paradigm map, not a cross-system numerical comparison.
  ```
  *(Compiled as "Table 1 is a qualitative paradigm map..." due to `\arabic` resolution).*
- **After:**
  ```tex
  Table~I is a qualitative paradigm map, not a cross-system numerical comparison.
  ```
  *(Matches the Roman numeral heading style `TABLE I.` used throughout the manuscript).*

### 3. Build & Package Verification:
- All 5 document targets compiled to **strictly 15 pages** with **0 errors and 0 margin overflows**.
- Updated `submission_package_ijcds/Manuscript_MemoryVerse_IJCDS.pdf` and rebuilt `ijcds_submission_package.zip`.


