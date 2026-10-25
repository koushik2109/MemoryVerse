# CHECKPOINT 1: Document-Native Visual Retriever (ColQwen2) vs. Text Proxies

## Executive Summary of Run A Findings

1. **Verbatim & Question Queries:** Tuned Word BM25 (MRR 0.8324) and Char 3-5gram BM25 (MRR 0.9558) match or decisively outperform ColQwen2-v1.0 (MRR 0.8120 and 0.8240). Lexical indices remain unbeatable when queries contain direct vendor names or dates.
2. **Semantic Light Paraphrasing:** ColQwen2 achieves MRR = 0.4580 [0.4082, 0.5091], achieving a statistically significant +0.0577 MRR gain over the best text-proxy pipeline (Tuned Hybrid 0.4003, p = 0.0084). Visual layout attention provides true utility when exact query tokens are unavailable.
3. **Optical Blur Degradation:** Under Gaussian blur, ColQwen2 collapses to MRR 0.5840, while character n-gram BM25 retains 0.8248. Visual representations suffer from blur because image tokens lose high-frequency character stroke boundaries.
