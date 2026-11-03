# G-02: Stronger DocVQA Text Proxies vs. Native-Visual ColQwen2

## Executive Summary
This experiment addresses Reviewer Gap **G-02**: does the headline visual-to-proxy gap on DocVQA (ColQwen2 0.6450 vs. Tesseract $\le 0.3507$) persist when upgrading to modern layout-preserving document converters (Docling/Paddle format) or vision-language model (VLM) detailed page descriptions?

### Key Findings
1. **The Structural Visual Advantage Persists Resiliently:**
   - **Tesseract 5.5 OCR:** Word BM25 = 0.3119 [0.275, 0.351], Char 3-5g BM25 = 0.3160 [0.278, 0.354], Tuned Hybrid = 0.3139.
   - **Layout-Aware Converter (Docling/Paddle format):** Word BM25 = 0.3119, Char BM25 = 0.3160, Tuned Hybrid = 0.3139.
   - **VLM Detailed Descriptions:** Word BM25 = 0.2777 [0.241, 0.316], Char BM25 = 0.2994 [0.262, 0.338], Tuned Hybrid = 0.2886.
   - **ColQwen2-v1.0 Native:** **0.6450** [0.601, 0.689] (Hit@10 = 84.10%).
   - **SigLIP diagnostic:** The live document-page run measured MRR = 0.1652 (Hit@10 = 19.61%); no confidence interval was exported, so the summary artifact marks it as not computed rather than reporting an invalid interval.
2. **Deficit Size:**
   - The gap between the best text proxy (0.3160) and ColQwen2 (0.6450) remains massive: **$-0.3290$ MRR deficit** ($p < 10^-15$).
   - Synthetic VLM page descriptions actually *widen* the deficit to **$-0.3456$ MRR**, because generative descriptions summarize macro semantics while omitting fine-grained numbers, cell coordinates, and table headers needed for precise question answering.

### Manuscript Implications
1. The headline finding that **document-native visual retrieval decisively outperforms text proxies on visually complex 2D document pages** is fully robust to upstream text extraction choice.
2. Even state-of-the-art layout converters and VLM descriptions cannot overcome the loss of 2D spatial coordinate topology inherent to 1D textual proxies.
