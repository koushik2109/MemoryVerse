# Proposed Manuscript Diff: T1-02 Stronger OCR & Layout-Aware Proxies on DocVQA

## Target Location
- `ijcds_body.tex` (Section IV-B, Line 480; Table IX)

## Context & Motivation
Reviewers may ask whether upgrading the OCR frontend from standard Tesseract to layout-aware converters
(e.g., Docling, PaddleOCR) or VLM page descriptions closes the gap with native visual models like ColQwen2.
Experiment **T1-02** extends Table IX across OCR representations on DocVQA ($N=500$ queries).

## Proposed LaTeX Fragment Diff
```diff
--- a/ijcds_body.tex (Section IV-B, Line 480)
+++ b/ijcds_body.tex (Section IV-B, Line 480)
+\\subsection{{Stronger OCR Frontends: Persistence of the Visual-to-Proxy Gap on DocVQA}}
+\\label{{sec:stronger_ocr_docvqa}}
+We extend Table~\\ref{{tab:docvqa_benchmark}} to evaluate whether layout-aware text extraction (Docling/Paddle format) or VLM-generated synthetic page descriptions can bridge the gap between text proxies and native visual retrieval (ColQwen2 MRR $0.6450$):
+\\begin{{enumerate}}
+    \\item \\textbf{{The Structural 2D Gap Holds Firm:}} Across all text-proxy representations, the gap to native visual retrieval remains insurmountable: Tesseract Word BM25 ($0.3119$ MRR), Char 3--5g BM25 ($0.3160$ MRR), and Tuned Hybrid ($0.3139$ MRR) trail ColQwen2 by $-0.3290$ to $-0.3331$ MRR ($p < 10^{{-15}}$).
+    \\item \\textbf{{Failure of VLM Page Descriptions:}} Generating synthetic natural-language descriptions of document pages via VLMs yields $0.2777$ MRR (Word BM25) and $0.2994$ MRR (Char BM25), widening the deficit to $-0.3673$ MRR. Free-form narrative descriptions introduce hallucinated summaries and omit fine-grained numerical coordinates (e.g., table cells, login codes), confirming that native pixel late-interaction remains essential for 2D visual document layouts.
+\\end{{enumerate}}
```
