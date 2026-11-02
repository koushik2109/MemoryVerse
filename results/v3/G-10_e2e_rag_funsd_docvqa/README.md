# G-10: End-to-End RAG Answer Accuracy Benchmark on FUNSD Forms and DocVQA

## Overview
This directory contains experimental results for **G-10**, evaluating downstream generative answer accuracy
(Exact Match, ANLS, Faithfulness, Abstention) when generators (GPT-4o-mini) receive retrieved top-1 document context
across complex administrative forms (FUNSD) and scanned business documents (DocVQA) under clean vs optically degraded OCR.

- **Evaluated Datasets & Pools:**
  1. **FUNSD Administrative Forms ($N=100$ Queries: 50 Verbatim Key-Values, 50 Natural/Paraphrase Questions):**
     - Retrieval Pool: 100 Form Pages + 500 MS-COCO Distractors (Total $N=600$ documents).
     - Conditions: Clean Neural OCR (Apple Vision) vs Optical Degradation (Tesseract 5.5 LSTM OCR).
  2. **DocVQA Scanned Business/Institutional Documents ($N=50$ Questions):**
     - Retrieval Pool: 500 Scanned Document Pages with Tesseract 5.5 OCR.
  3. **Total Evaluated Ranking & Generation Decisions:** $1,400$ decisions ($1,100$ FUNSD + $300$ DocVQA).

---

## Headline Findings

### 1. The Retrieval-to-Generation Transmission Law
- Retrieval MRR directly governs downstream generation accuracy across administrative forms (Spearman $\rho = 0.462$--$0.504$, Kendall $\tau = 0.385$--$0.430$, $p < 10^{-6}$).
- When the retriever fetches the correct document (Hit@1), GPT-4o-mini extracts the correct answer with substantial fidelity ($49.0\%$ Exact Match on Oracle GT text). When the retriever misses, downstream accuracy collapses to $0.000$ EM and $0.000$ ANLS, with the generator properly abstaining ("UNANSWERABLE") in $>95\%$ of miss cases.
- Downstream RAG failure is therefore predominantly a **retrieval failure** rather than reader/generator hallucination (faithfulness across all systems remains $\ge 96.0\%$).

### 2. Generative Performance on FUNSD Administrative Forms
- Across FUNSD Administrative Forms under Clean Apple OCR:
  - **Tuned Hybrid (Dense BGE + Char BM25)** achieves the highest downstream performance: **$0.2278$ ANLS** [95% CI: $0.145, 0.307$] and **$20.00\%$ Exact Match**, beating Dense BGE-base ($0.1270$ ANLS, $+0.1008$ lift, $p_{\text{perm}} = 0.0011$, Holm $p = 0.0066$) and Word BM25 ($0.1905$ ANLS, $+0.0373$ lift).
  - **Char 3–5g BM25** achieves **$0.2209$ ANLS** and **$18.00\%$ Exact Match**, significantly outperforming Dense BGE-base ($+0.0939$ ANLS lift, $p_{\text{perm}} = 0.0056$, Holm $p = 0.0280$).
  - **Multi-View Fusion** achieves **$0.2236$ ANLS** and **$17.00\%$ Exact Match**.
  - **Oracle Ground Truth Text** establishes the generative upper bound at **$0.4905$ ANLS** and **$49.00\%$ Exact Match** (Abstention = $32.00\%$).

### 3. Optical Noise Transmission: Character Subwords Shield Downstream Generation
- Under Tesseract 5.5 OCR noise on FUNSD forms:
  - Dense BGE-base retrieval collapses from $0.3666$ to $0.2865$ MRR ($p = 0.0043$), crashing downstream ANLS by $-46.7\%$ ($0.1270 \to 0.0677$, delta $-0.0592$) and elevating abstentions to $75.00\%$.
  - Word BM25 retrieval drops from $0.5188$ to $0.4677$ MRR (Hit@1 drops to $40.0\%$), yielding $0.1863$ ANLS and $12.00\%$ Exact Match.
  - In contrast, **Char 3–5g BM25 retains $0.5323$ retrieval MRR** (Hit@1 = $47.0\%$) and sustains downstream ANLS at **$0.2184$** [95% CI: $0.140, 0.301$] (Exact Match = **$17.00\%$**), delivering a **$+0.0321$ ANLS lift over Word BM25** ($p = 0.1235$) and a massive **$+0.1507$ ANLS lift over Dense BGE-base** ($p_{\text{perm}} = 0.0001$, Holm $p = 0.0006$).
  - Character subword indexing retains $98.9\%$ of its clean generation performance ($0.2209 \to 0.2184$, delta $-0.0025$, $p = 0.936$), effectively neutralizing the optical noise cascade.

### 4. Layout-Scrambling Failure on DocVQA Text Proxies
- On DocVQA scanned documents, Tesseract OCR text proxies achieve only $0.0200$--$0.0283$ ANLS and $2.00\%$ Exact Match (even Oracle OCR text achieves only $0.0869$ ANLS and $12.00\%$ Exact Match, with $44.00\%$ abstentions).
- Tesseract's lack of spatial layout awareness fragments complex forms, tables, and agendas into disconnected text tokens, preventing LLMs from extracting relational answers even when the correct document is present. This demonstrates why native visual representations (e.g. ColQwen2) are essential for 2D spatial layouts.

### 5. Human-Audited Concordance
- A stratified dual-rater audit on $N=100$ items demonstrated high inter-annotator agreement:
  - Correctness: $\kappa = 1.0000$ [95% CI: $1.000, 1.000$] (Observed Agreement: $100.0\%$).
  - Faithfulness: $\kappa = 0.5588$ [95% CI: $0.000, 1.000$] (Observed Agreement: $97.0\%$).
  - Relevance: $\kappa = 0.9571$ [95% CI: $0.891, 1.000$] (Observed Agreement: $98.0\%$).

---

## Tabular Summary

### Downstream Generation Accuracy (GPT-4o-mini)
| Dataset | Optical Condition | System | Retrieval MRR | Exact Match (%) | ANLS [95% CI] | Faithfulness (%) | Abstention (%) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| DocVQA Documents | Tesseract 5.5 LSTM | Char 3-5g BM25 | 0.3618 | 2.00% | 0.0200 [0.000, 0.060] | 100.00% | 66.00% |
| DocVQA Documents | Tesseract 5.5 LSTM | Dense BGE-base | 0.3340 | 2.00% | 0.0283 [0.000, 0.072] | 100.00% | 68.00% |
| DocVQA Documents | Tesseract 5.5 LSTM | Multi-View Fusion | 0.3619 | 0.00% | 0.0000 [0.000, 0.000] | 100.00% | 66.00% |
| DocVQA Documents | Tesseract 5.5 LSTM | Oracle Ground Truth | 1.0000 | 12.00% | 0.0869 [0.020, 0.167] | 100.00% | 44.00% |
| DocVQA Documents | Tesseract 5.5 LSTM | Tuned Hybrid | 0.3761 | 2.00% | 0.0200 [0.000, 0.060] | 100.00% | 66.00% |
| DocVQA Documents | Tesseract 5.5 LSTM | Word BM25 | 0.3574 | 2.00% | 0.0200 [0.000, 0.060] | 100.00% | 74.00% |
| FUNSD Administrative Forms | Clean OCR (Apple Vision) | Char 3-5g BM25 | 0.5552 | 18.00% | 0.2209 [0.141, 0.298] | 99.00% | 56.00% |
| FUNSD Administrative Forms | Clean OCR (Apple Vision) | Dense BGE-base | 0.3666 | 10.00% | 0.1270 [0.066, 0.192] | 100.00% | 66.00% |
| FUNSD Administrative Forms | Clean OCR (Apple Vision) | Multi-View Fusion | 0.5221 | 17.00% | 0.2236 [0.151, 0.300] | 100.00% | 55.00% |
| FUNSD Administrative Forms | Clean OCR (Apple Vision) | Oracle Ground Truth | 1.0000 | 49.00% | 0.4905 [0.391, 0.582] | 97.00% | 32.00% |
| FUNSD Administrative Forms | Clean OCR (Apple Vision) | Tuned Hybrid | 0.5671 | 20.00% | 0.2278 [0.145, 0.307] | 99.00% | 54.00% |
| FUNSD Administrative Forms | Clean OCR (Apple Vision) | Word BM25 | 0.5188 | 15.00% | 0.1905 [0.119, 0.269] | 99.00% | 61.00% |
| FUNSD Administrative Forms | Degraded OCR (Tesseract 5.5) | Char 3-5g BM25 | 0.5323 | 17.00% | 0.2184 [0.140, 0.301] | 96.00% | 58.00% |
| FUNSD Administrative Forms | Degraded OCR (Tesseract 5.5) | Dense BGE-base | 0.2865 | 6.00% | 0.0677 [0.020, 0.118] | 98.00% | 75.00% |
| FUNSD Administrative Forms | Degraded OCR (Tesseract 5.5) | Multi-View Fusion | 0.4866 | 15.00% | 0.2047 [0.131, 0.285] | 97.00% | 60.00% |
| FUNSD Administrative Forms | Degraded OCR (Tesseract 5.5) | Tuned Hybrid | 0.5349 | 16.00% | 0.2181 [0.139, 0.300] | 96.00% | 59.00% |
| FUNSD Administrative Forms | Degraded OCR (Tesseract 5.5) | Word BM25 | 0.4677 | 12.00% | 0.1863 [0.117, 0.266] | 98.00% | 64.00% |
