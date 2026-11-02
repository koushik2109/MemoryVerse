# G-01: Stronger and Modern Text Retrieval Baselines under Proxy Noise

## Executive Summary
This experiment addresses Reviewer Gap **G-01** and Limitation 10: does the dense bi-encoder deficit survive when moving to modern retrieval paradigms, including learned-sparse term expansion (SPLADE), token-level late interaction (ColBERTv2), multi-engine dense architectures (BGE-Small, BGE-Base), and production two-stage cross-encoder cascades?

### Key Findings
1. **The Dense Deficit Persists Across Modern Embeddings:**
   - On verbatim entity lookup, BGE-Base (0.3300 MRR) and BGE-Small (0.2997 MRR) trail Word BM25 (0.8386 MRR, diff $-0.5086$, $p < 10^-15$) and Char 3-5g BM25 (0.9437 MRR, diff $-0.6137$, $p < 10^-15$) by massive statistical margins.
   - Under physical optical blur, BGE-Base drops to 0.2796 MRR and BGE-Small to 0.2561 MRR, while Char BM25 retains 0.8271 MRR ($+0.5475$ lift, $p < 10^-15$).
   - On Tesseract Clean, the pattern holds identically: Char BM25 (0.9854) > Word BM25 (0.8571) > BGE-Small (0.3191).
2. **Learned-Sparse (SPLADE) Catastrophically Collapses Under OCR Noise:**
   - SPLADE relies on BERT masked-language-model logits to expand vocabulary weights. On fragmented and blurred OCR strings, MLM predictions become chaotic, causing SPLADE to collapse to **0.0774 MRR** (Clean) and **0.0597 MRR** (Optical Blur).
   - SPLADE is fundamentally ill-suited for uncurated optical text.
3. **Late Interaction (ColBERTv2) Fails to Bridge the Entity Anchor Gap:**
   - ColBERTv2 achieves **0.3896 MRR** (Clean Verbatim) and **0.3145 MRR** (Blur Verbatim), outperforming single-vector dense embeddings (+0.0596 over BGE-Base) but trailing sub-word Char BM25 by **-0.5541 MRR** ($p < 10^-15$).
   - Token-level MaxSim operations on corrupted sub-word pieces cannot substitute for exact lexical character matching.
4. **Two-Stage Cascades Recover Dense Paraphrase Weakness:**
   - A two-stage pipeline with Char BM25 first-stage ($K_1=20$) fed into a Cross-Encoder reranker recovers **0.9350 MRR** on Clean Verbatim and **0.7844 MRR** on Optical Blur, while boosting Paraphrase MRR from 0.2448 to 0.2735.

## Conclusion for Manuscript (Section VII Limitation 10)
Rather than merely conceding the dense baseline as a limitation, the paper can now report empirical verification: **the dense deficit is not an artifact of small or outdated models (MiniLM), but an intrinsic property of dense vector geometry on fragmented lexical anchors**, persisting across BGE-Small, BGE-Base, ColBERTv2, and SPLADE.
