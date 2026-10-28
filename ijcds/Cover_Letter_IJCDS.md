# Cover Letter for Manuscript Submission

**To:**  
Prof. Dr. Wael M. El-Medany  
Editor-in-Chief  
*International Journal of Computing and Digital Systems (IJCDS)*  
University of Bahrain, Kingdom of Bahrain  

**Date:** October 28, 2026  
**Subject:** Submission of Original Regular Paper Manuscript for Peer Review  

---

### Manuscript Submission Metadata

| Attribute | Details |
| :--- | :--- |
| **Intended Publication Format:** | **Regular Paper** |
| **Manuscript Title:** | **When Do Text Proxies Suffice? A Multi-Regime Analysis of Intermediate Representations in Multimodal RAG** |
| **Corresponding Author:** | **Anurag Krishna Vennela** (Email: `anuragkrishnavennela@gmail.com`) |
| **Co-Author:** | **G. Sai Balaji Koushik** (Email: `gummadavellikoushik2109@gmail.com`) |
| **Preferred Author Address:** | Department of Computer Science and Engineering, National Institute of Technology Warangal, Warangal, Telangana 506004, India |
| **Telephone & Fax Numbers:** | Tel: `+91 870 2459191` \| Fax: `N/A` |

---

Dear Prof. Dr. El-Medany and Esteemed Editorial Board,

We are pleased to submit our original research manuscript titled **"When Do Text Proxies Suffice? A Multi-Regime Analysis of Intermediate Representations in Multimodal RAG"** for consideration as a Regular Paper in the *International Journal of Computing and Digital Systems (IJCDS)*. The manuscript is authored by Anurag Krishna Vennela and G. Sai Balaji Koushik from the Department of Computer Science and Engineering at the National Institute of Technology Warangal, India.

This work presents a comprehensive empirical regime analysis addressing a central architectural problem in applied digital systems and Multimodal Retrieval-Augmented Generation (MM-RAG): determining precisely when intermediate text proxies (such as optical character recognition transcripts and visual captions) suffice, and where native multi-vector or visual representations become essential. Through pre-registered, held-out benchmarking across 1,626 multimodal items (retail receipts from SROIE, scanned administrative forms from FUNSD, natural imagery from MS-COCO, and continuous acoustic audio from ESC-50), our study establishes five principal findings:

1. **Optical Degradation and Sub-Word Robustness:** Under verbatim optical OCR noise, character 3–5gram BM25 outperforms standard whole-word BM25 across all OCR regimes (Apple Clean: 0.9436 vs.\ 0.8371; Degraded Blur: 0.8248 vs.\ 0.6972; Tesseract 5.5: 0.8412 vs.\ 0.7364; $p_{\text{Holm}} < 10^{-4}$), whereas whole-word BM25 regains superiority under entity-substituting paraphrases (0.4334 vs.\ 0.2427);
2. **Document-Native Multi-Vector Generalization:** On scanned document images, native multi-vector models (ColQwen2-v1.0) achieve 0.4962 MRR under light paraphrasing (+0.0628 lift over Word BM25, $p_{\text{Holm}} = 0.0072$), while showing no significant difference from CPU lexical search on answerable factual questions ($p > 0.05$) where it incurs 249.21\,ms GPU latency vs.\ 0.87\,ms CPU latency;
3. **Micro-Mechanisms of Dense Bi-Encoder Deficit:** Dense bi-encoder collapse is associated with query formulation mismatch rather than sequence truncation constraints; structuring queries as natural answerable questions improves BGE-Base MRR from 0.3256 to 0.5263 on identical text;
4. **Risk-Controlled Cascade Ranking:** Split conformal cascade calibration under marginal loss bounds expected test miss risk to 0.0494 at a 78.4\% escalation rate across resampled splits of the pooled finite instances (statistically matched by equal-escalation fixed margin);
5. **Visual and Multi-Hop Modality Trade-Offs:** Native visual representations (SigLIP MRR = 0.6071) significantly outperform machine caption proxies (0.3728, $p < 10^{-4}$) on natural photographic imagery with no significant difference detected from human gold captions (0.5835, $p = 0.142$; equivalence not tested), while extractive pseudo-relevance feedback shows no detectable difference from generative LLM bridging in exploratory multi-hop retrieval ($p = 1.000$) with zero API cost.

We believe this investigation aligns directly with the core scope of *IJCDS* in applied computing systems, digital information retrieval, and reliable machine learning architectures. Our empirical findings provide actionable, hardware-profiled deployment guidelines for enterprise document digitization systems and edge computing environments.

We confirm that this manuscript represents original work that has not been published previously and is not under consideration for publication elsewhere. All authors have reviewed and approved the manuscript for submission, and there are no conflicts of interest to declare. All frozen datasets, reproducible harness scripts, and audit ledgers are publicly available.

Thank you for your time, consideration, and coordination of the peer-review process.

Sincerely yours,

**Anurag Krishna Vennela** (Corresponding Author)  
Department of Computer Science and Engineering  
National Institute of Technology Warangal, Telangana 506004, India  
Email: `anuragkrishnavennela@gmail.com`  

**G. Sai Balaji Koushik** (Co-Author)  
Department of Computer Science and Engineering  
National Institute of Technology Warangal, Telangana 506004, India  
Email: `gummadavellikoushik2109@gmail.com`  

---

### Informative Abstract (199 words; Limit: 200 words)

> *Multimodal Retrieval-Augmented Generation (MM-RAG) frequently indexes textual proxies over raw pixels. Evaluating a 1,626-item benchmark under frozen 50/50 Dev/Test splits ($N=306$ test probes, 813 items), we delineate when proxies suffice and when native representations are indispensable. On verbatim lookups, character 3--5gram BM25 outperforms word BM25 across OCR regimes (Apple Clean: 0.9436 vs.\ 0.8371; Degraded Blur: 0.8248 vs.\ 0.6972; Tesseract: 0.8412 vs.\ 0.7364; $p_{\text{Holm}} < 10^{-4}$), reversing under entity rewrites (0.2427 vs.\ 0.4334). On documents, native ColQwen2 achieves 0.4962 MRR under light paraphrases (+0.0628 lift over Word BM25, $p_{\text{Holm}} = 0.0072$), profiling at 249.21\,ms on Tesla T4 GPU vs.\ 0.87\,ms for CPU lexical lookups. Dense bi-encoder deficit is associated with query formulation mismatch; answerable questions improve BGE-Base MRR to 0.5263 (Hit@10 79.74\%). Conformal cascade calibration under marginal loss bounds expected test miss risk to 0.0494 at a 78.4\% escalation rate. On MS-COCO ($N=1{,}000$), native visual retrieval (SigLIP MRR = 0.6071) beats machine captions (0.3728, $p < 10^{-4}$) with no significant difference detected from human gold captions (0.5835, $p=0.142$; equivalence not tested). In exploratory multi-hop reasoning ($N=50$), extractive pseudo-relevance feedback exhibits no detectable difference from generative LLM bridging (Hit@10 0.640 vs.\ 0.640, exact McNemar $p = 1.0000$) with zero API cost.*

**Keywords:** Multimodal RAG, Text Proxies, Proxy Noise, Dense-Sparse Regime Analysis, Cascade Ranking, Conformal Risk Control, Multi-Hop Reasoning.
