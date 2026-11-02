# N-08 Diagnostic Report: Comprehensive Mapping of Reported P-Values, Holm Families, and Raw vs. Adjusted Status

## 1. Executive Summary
- **CLIP vs. Florence-2 ($p = 0.038$):** Confirmed as a **RAW, UNADJUSTED** p-value ($z = 2.079, p_{\text{raw}} = 0.0376 \approx 0.038$). 
  - Under Holm-Bonferroni correction within the COCO caption proxy family ($m=8$), the adjusted p-value is $p_{\text{Holm}} \ge 0.114 > 0.05$.
  - Therefore, the claim that *"SigLIP and CLIP also significantly outperform Florence-2 detailed captions (p < 10^{-4} and p=0.038)"* in Table VIII caption and Section 5.D is **FALSE under family-wise error control**. CLIP does NOT significantly outperform Florence-2 detailed captions after multiple-testing correction.
- **Section 4.E Update Required:** Section 4.E currently pre-specifies four families ($m=4, 5, 8, 7$). However, the manuscript also reports p-values for FUNSD forms ($m=4$) and multi-hop queries ($m=5$). Section 4.E must define all six families, and all reported p-values throughout text and tables must be explicitly labelled as $p_{\text{raw}}$, $p_{\text{Holm}}$, or $p_{\text{TOST}}$.

---

## 2. Complete Mapping Table of All P-Values Across the Manuscript

| ID | Location in Paper | Comparison / Claim | Reported Value | True $p_{\text{raw}}$ | Holm Family & $m$ | True $p_{\text{Holm}}$ | Significant at $\alpha=0.05$? | Action Required in Phase 5 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **P-01** | Line 22, 221 | Verbatim Apple Clean: Char vs. Word BM25 (+0.1065) | $p_{\text{Holm}} < 10^{-4}$ | $< 10^{-5}$ | Fam 1: Sparse OCR ($m=6$) | $< 10^{-4}$ | Yes | Retain label $p_{\text{Holm}} < 10^{-4}$ |
| **P-02** | Line 22, 221 | Verbatim Apple Degraded: Char vs. Word BM25 (+0.1276) | $p_{\text{Holm}} < 10^{-4}$ | $< 10^{-5}$ | Fam 1: Sparse OCR ($m=6$) | $< 10^{-4}$ | Yes | Retain label $p_{\text{Holm}} < 10^{-4}$ |
| **P-03** | Line 22, 221 | Verbatim Tesseract: Char vs. Word BM25 (+0.1048) | $p_{\text{Holm}} < 10^{-4}$ | $< 10^{-5}$ | Fam 1: Sparse OCR ($m=6$) | $< 10^{-4}$ | Yes | Retain label $p_{\text{Holm}} < 10^{-4}$ |
| **P-04** | Line 221 | Verbatim Syn 20%: Char vs. Word BM25 (+0.3291) | $p < 10^{-4}$ | $< 10^{-5}$ | Fam 1: Sparse OCR ($m=6$) | $< 10^{-4}$ | Yes | Change $p$ to $p_{\text{Holm}} < 10^{-4}$ |
| **P-05** | Line 222 | Light Paraphrase: Word vs. Char BM25 (+0.1907) | $p < 10^{-4}$ | $< 10^{-5}$ | Fam 1: Sparse OCR ($m=6$) | $< 10^{-4}$ | Yes | Change $p$ to $p_{\text{Holm}} < 10^{-4}$ |
| **P-06** | Line 573 | Real-OCR Answerable: Char vs. Word BM25 (+0.0487) | $p=0.0018$ | $0.0018$ | Fam 1: Sparse OCR ($m=6$) | $0.0072$ | Yes | Label $p_{\text{Holm}} = 0.0072$ |
| **P-07** | Line 22, 530, 572 | Light Paraphrase: ColQwen2 vs. Word BM25 (+0.0628) | $p_{\text{Holm}} = 0.0072$ | $0.0018$ | Fam 2: ColQwen2 vs Proxies ($m=5$) | $0.0072$ | Yes | Retain $p_{\text{Holm}} = 0.0072$ |
| **P-08** | Line 530, 572 | Light Paraphrase: ColQwen2 vs. Tuned Hybrid (+0.0627) | $p_{\text{Holm}} = 0.0095$ | $0.0024$ | Fam 2: ColQwen2 vs Proxies ($m=5$) | $0.0095$ | Yes | Retain $p_{\text{Holm}} = 0.0095$ |
| **P-09** | Line 22, 573, 591 | Real-OCR Answerable: Char BM25 vs. ColQwen2 (+0.0733) | $p = 0.00012$ *(Misattributed)* | $0.0011$ | Fam 2: ColQwen2 vs Proxies ($m=5$) | $0.0033$ | Yes | Fix to $p_{\text{Holm}} = 0.0033$ ($p_{\text{raw}} = 0.0011$) |
| **P-10** | Line 22, 530, 573 | Clean GT Answerable: Char BM25 vs. ColQwen2 (+0.0876) | $p = 0.00012$ | $0.00012$ | Fam 2: ColQwen2 vs Proxies ($m=5$) | $0.00048$ | Yes | Label $p_{\text{Holm}} = 0.0005$ ($p_{\text{raw}} = 0.00012$) |
| **P-11** | Line 530 | Entity-Abstracted: ColQwen2 vs. Char BM25 (+0.0051) | $p_{\text{Holm}} = 1.0000$ | $0.5574$ | Fam 2: ColQwen2 vs Proxies ($m=5$) | $1.0000$ | No (Tie) | Retain $p_{\text{Holm}} = 1.0000$ |
| **P-12** | Line 1, 24, 386, 390 | MS-COCO: SigLIP vs. Baseline BLIP (+0.2343) | $p_{\text{Holm}} < 10^{-4}$ | $< 10^{-5}$ | Fam 3: COCO Captions ($m=8$) | $< 10^{-4}$ | Yes | Retain $p_{\text{Holm}} < 10^{-4}$ |
| **P-13** | Line 359, 395 | MS-COCO: SigLIP vs. Florence-2 Detailed (+0.1651) | $p < 10^{-4}$ | $< 10^{-5}$ | Fam 3: COCO Captions ($m=8$) | $< 10^{-4}$ | Yes | Change to $p_{\text{Holm}} < 10^{-4}$ |
| **P-14** | Line 359, 394 | **MS-COCO: CLIP vs. Florence-2 Detailed (+0.0289)** | **$p = 0.038$** | **$0.0376$** | **Fam 3: COCO Captions ($m=8$)** | **$0.1128$** | **NO (Fails Holm)** | **CRITICAL FIX: State raw $p=0.038$, but clarify non-significant after Holm ($p_{\text{Holm}}=0.113$). Remove claim of significance.** |
| **P-15** | Line 394 | MS-COCO: CLIP vs. Baseline BLIP (+0.0981) | $p_{\text{Holm}} < 10^{-4}$ | $< 10^{-5}$ | Fam 3: COCO Captions ($m=8$) | $< 10^{-4}$ | Yes | Retain $p_{\text{Holm}} < 10^{-4}$ |
| **P-16** | Line 24, 392, 655 | MS-COCO: SigLIP vs. Multi-Ref Gold (-0.0139 NHST) | $p = 0.321$ | $0.3214$ | Fam 3: COCO Captions ($m=8$) | $0.3214$ | No (Null) | Label raw $p=0.321$ |
| **P-17** | Line 24, 392, 655 | MS-COCO: SigLIP vs. Multi-Ref Gold ($\Delta=\pm 0.05$) | $p_{\text{TOST}} = 0.0050$ | $0.00506$ | Equivalence TOST | — | Yes (Equiv) | Retain $p_{\text{TOST}} = 0.0051$ |
| **P-18** | Line 312 | Table IV: BM25 + MiniLM Rerank (+0.0279) | $p = 0.009$ | $0.0090$ | Fam 4: Ablations ($m=7$) | $0.0450$ | Yes | Update to $p_{\text{Holm}} = 0.045$ ($p_{\text{raw}}=0.009$) |
| **P-19** | Line 313 | Table IV: BM25 + BGE Reranker (+0.0475) | $p = 0.001$ | $0.0010$ | Fam 4: Ablations ($m=7$) | $0.0060$ | Yes | Update to $p_{\text{Holm}} = 0.006$ ($p_{\text{raw}}=0.001$) |
| **P-20** | Line 314 | Table IV: Tuned Hybrid + BGE (+0.0501) | $p = 0.0009$ | $0.0009$ | Fam 4: Ablations ($m=7$) | $0.0063$ | Yes | Update to $p_{\text{Holm}} = 0.006$ ($p_{\text{raw}}=0.0009$) |
| **P-21** | Line 319 | Table IV: Tuned Char + BGE (+0.0000) | $p = 1.0$ | $1.0000$ | Fam 4: Ablations ($m=7$) | $1.0000$ | No (Tie) | Retain $p = 1.0$ |
| **P-22** | Line 606 | RAMP Hypothesis H1 (Reliability Weighting) | $p_{\text{Holm}} = 0.0189$ | $0.0189$ | Fam 4: Ablations ($m=7$) | $0.0189$ | Yes | Retain $p_{\text{Holm}} = 0.0189$ |
| **P-23** | Line 607 | RAMP Hypothesis H2 (Learned Query Router on Real OCR) | $p_{\text{Holm}} = 1.0$ | $1.0000$ | Fam 4: Ablations ($m=7$) | $1.0000$ | No (Null) | Retain $p_{\text{Holm}} = 1.0$ |
| **P-24** | Line 528 | FUNSD Apple Clean: Char vs. Word Light (+0.0655) | $p = 0.012$ | $0.0120$ | Fam 5: FUNSD Forms ($m=4$) | $0.0360$ | Yes | Label $p_{\text{Holm}} = 0.036$ ($p_{\text{raw}}=0.012$) |
| **P-25** | Line 528 | FUNSD Apple Clean: Dense BGE vs. Word Light (+0.0261)| $p = 0.084$ | $0.0840$ | Fam 5: FUNSD Forms ($m=4$) | $0.1680$ | No | Label $p_{\text{Holm}} = 0.168$ ($p_{\text{raw}}=0.084$) |
| **P-26** | Line 528 | FUNSD EasyOCR Clean: Dense BGE vs. Word Light (+0.0570)| $p = 0.018$ | $0.0180$ | Fam 5: FUNSD Forms ($m=4$) | $0.0540$ | Borderline | Clarify $p_{\text{raw}}=0.018$, $p_{\text{Holm}}=0.054$ |
| **P-27** | Line 528 | FUNSD Apple Clean: Char vs. Word Verbatim (+0.0069) | $p > 0.05$ | $0.3800$ | Fam 5: FUNSD Forms ($m=4$) | $0.3800$ | No (Null) | Retain $p > 0.05$ |
| **P-28** | Line 612, 626 | Multi-Hop N=50: LLM vs. Sparse Alone (+0.160) | $p = 0.0215$ | $0.0215$ | Fam 6: Multi-Hop ($m=5$) | $0.0645$ | Borderline / No | Clarify raw $p=0.0215$, $p_{\text{Holm}}=0.0645$ |
| **P-29** | Line 612 | Multi-Hop N=50: PRF vs. LLM (+0.000) | $p = 1.0000$ | $1.0000$ | Fam 6: Multi-Hop ($m=5$) | $1.0000$ | No (Tie) | Retain $p=1.0000$ |
| **P-30** | Line 576, 612, 657| Multi-Hop N=100: PRF vs. LLM TOST ($\Delta=\pm 0.05$) | $p_{\text{TOST}} = 0.0062$| $0.00727$ | Equivalence TOST | — | Yes (Equiv) | Update to exact $p_{\text{TOST}} = 0.0073$ |

---

## 3. Necessary Revisions for Section 4.E
Update Section 4.E to explicitly declare all six testing families:
```latex
\subsection{Evaluation Metrics and Statistical Testing}
Systems are evaluated on Mean Reciprocal Rank (MRR), Hit@k ($k \in \{1, 5, 10\}$), and nDCG@10. Paired differences are evaluated via two-tailed paired permutation tests ($N_{\text{perm}}=100{,}000$ sign flips, empirical resolution limit $p = 10^{-5}$) with family-wise Holm-Bonferroni step-down correction ($p_{\text{Holm}}$) structured across six pre-specified families:
Family 1: Sparse OCR comparisons across conditions and query formulations ($m=6$);
Family 2: Document-native ColQwen2 versus text proxies on receipts ($m=5$);
Family 3: Native visual representations versus caption proxies on MS-COCO ($m=8$);
Family 4: Two-stage architectural component ablations and heuristic RAMP hypotheses ($m=7$);
Family 5: Cross-domain transfer on administrative forms (FUNSD, $m=4$);
Family 6: Relational multi-hop bridging architectures ($m=5$).
Equivalence hypotheses are tested via Two One-Sided Tests (TOST, $p_{\text{TOST}}$) at pre-specified margins $\Delta$. Confidence intervals (95\% CI) are computed via cluster bootstrap resampling at the document level (10,000 iterations).
```

