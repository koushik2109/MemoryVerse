# COVER LETTER FOR SUBMISSION TO IJCDS
**International Journal of Computing and Digital Systems (IJCDS)**  
*University of Bahrain*  

**Date:** November 2026  

**To:**  
The Editor-in-Chief / Editorial Board  
International Journal of Computing and Digital Systems (IJCDS)  

**Subject:** Submission of Original Research Article (Regular Paper)  

Dear Editor-in-Chief and Editorial Board Members,

We are pleased to submit our original research manuscript entitled:  
**"When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise"**  
for consideration for publication as a Regular Paper in the *International Journal of Computing and Digital Systems (IJCDS)*.

---

### Article Classification
- **Article Type:** Regular Paper (Full Research Article)
- **Primary Subject Area:** Multimodal Information Retrieval, Document Intelligence, Applied Artificial Intelligence, Systems Architecture
- **Target Venue Fit:** IJCDS publishes original research on computational systems, data science, and applied machine learning. Our work directly addresses the fundamental efficiency and architectural trade-offs in deploying Multimodal Retrieval-Augmented Generation (MM-RAG) systems across cloud and edge computing environments.

---

### Author Information (For Portal Registration)

**Corresponding Author:**  
Anurag Krishna Vennela  
Department of Computer Science and Engineering  
National Institute of Technology Warangal  
Warangal, Telangana 506004, India  
Email: `anuragkrishnavennela@gmail.com`  

**Co-Author:**  
G. Sai Balaji Koushik  
Department of Computer Science and Engineering  
National Institute of Technology Warangal  
Warangal, Telangana 506004, India  
Email: `gummadavellikoushik2109@gmail.com`  

---

### Informative Abstract (214 words)
Multimodal Retrieval-Augmented Generation (MM-RAG) frequently indexes intermediate textual proxies (OCR transcriptions, machine captions) rather than raw visual media. Evaluating a 1,626-item multimodal benchmark across frozen 50/50 Dev/Test splits (813 corpus items, 306 test probes), we establish an empirical regime map delineating when text proxies suffice and when native visual models become indispensable. On verbatim receipt lookups, character 3–5gram BM25 outperforms word BM25 across OCR engines (Apple Clean: 0.9436 vs. 0.8371; Degraded Optical Blur: 0.8248 vs. 0.6972; Tesseract 5.5: 0.8412 vs. 0.7364; all $p < 10^{-4}$), reversing under entity-substituting light paraphrases (0.2427 vs. 0.4334). On document images, document-native ColQwen2 achieves 0.4962 MRR under paraphrasing (+0.0628 over Word BM25, $p = 0.0072$) but ties lexical search on answerable factual questions ($p = 1.0000$). On DocVQA ($N=500$), ColQwen2 (0.5497) substantially outperforms text proxies ($\le 0.3507$). Dense bi-encoder deficit is governed by query formulation mismatch rather than sequence truncation; natural answerable questions improve BGE-Base MRR from 0.3256 to 0.5263. On MS-COCO ($N=1,000$), native SigLIP retrieval (0.6071) decisively outperforms machine captions (BLIP 0.3728, Florence-2 0.4420, $p < 10^{-4}$). Conformal cascades and extractive multi-hop bridging reduce compute overhead while bounding marginal risk.

---

### Core Scientific Contributions
1. **Empirical Multi-Regime Map:** We establish the first comprehensive regime map across 1,626 multimodal items (SROIE, FUNSD, DocVQA, MS-COCO) delineating exact operational boundaries where text proxies match or exceed GPU native-visual models.
2. **Subword Robustness & Crossover Mechanics:** We prove that Character 3–5g BM25 provides robust optical noise tolerance across commercial and open-source OCR engines, demonstrating that the crossover effect is driven by entity substitution rather than natural phrasing.
3. **Systems & Computational Footprint:** We provide an empirical decision guide for multimodal RAG architectures, proving that two-stage retrieval cascades ($K_1=50$ BM25 $\to$ ColQwen2) retain 97.2% of native visual MRR while slashing latency by $15.5\times$ on live CPU execution ($12.18\text{ ms}$).

---

### Double-Blind Review Compliance
In strict compliance with IJCDS double-blind review policies:
- All author names, affiliations, contact emails, and personal repository links have been removed from the manuscript text, figures, headers, and metadata.
- Prior self-citations are referenced objectively in standard third-person syntax.
- The compiled manuscript PDF contains strictly anonymous placeholder headers.

---

### Mandatory Declarations
We formally confirm and declare that:
1. **Originality:** This manuscript is an original work that has not been published previously in any journal, conference proceedings, or book chapter.
2. **Concurrent Submission:** It is not currently under consideration, review, or submission for publication elsewhere.
3. **Approval:** All authors have read and approved the final submitted manuscript.
4. **No Competing Interests:** The authors declare that they have no competing financial or non-financial interests that could have influenced this work.
5. **Data and Code Availability:** Complete raw datasets, evaluation scripts, benchmark configurations, and reproduction instructions are bundled in the accompanying reproducibility package.

Thank you for your consideration of our work.

Sincerely,

**Anurag Krishna Vennela** (Corresponding Author)  
**G. Sai Balaji Koushik** (Co-Author)  
Department of Computer Science and Engineering  
National Institute of Technology Warangal, India  

