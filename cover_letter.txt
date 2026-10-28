COVER LETTER FOR SUBMISSION TO IJCDS
International Journal of Computing and Digital Systems
University of Bahrain

Date: October 28, 2026

To:
The Editor-in-Chief / Editorial Board
International Journal of Computing and Digital Systems (IJCDS)

Subject: Submission of Original Research Article

Dear Editor and Editorial Board Members,

We are pleased to submit our original research manuscript entitled:
"When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise"
for consideration for publication as a Regular Paper in the International Journal of Computing and Digital Systems (IJCDS).

---
ARTICLE TYPE
Regular Paper

---
AUTHOR INFORMATION (For EDAS Submission System)
Corresponding Author:
Anurag Krishna Vennela
Department of Computer Science and Engineering
National Institute of Technology Warangal
Warangal, Telangana 506004, India
Email: anuragkrishnavennela@gmail.com
Phone: +91-9876543210
Fax: N/A

Co-Author:
G. Sai Balaji Koushik
Department of Computer Science and Engineering
National Institute of Technology Warangal
Warangal, Telangana 506004, India
Email: gummadavellikoushik2109@gmail.com

---
INFORMATIVE ABSTRACT (Strictly <= 200 words)
Multimodal Retrieval-Augmented Generation (MM-RAG) frequently indexes intermediate textual proxies over raw visual media. Evaluating a 1,626-item multimodal benchmark under frozen 50/50 Dev/Test splits (813 corpus items, 306 test probes), we establish an empirical regime map delineating when text proxies suffice and when native representations are required. On verbatim receipt lookups, character 3--5gram BM25 outperforms word BM25 across OCR engines (Apple Clean: 0.9436 vs. 0.8371; Degraded: 0.8248 vs. 0.6972; Tesseract 5.5: 0.8412 vs. 0.7364; all p < 10^-4), reversing under entity-substituting light paraphrases (0.2427 vs. 0.4334). On document images, document-native ColQwen2 achieves 0.4962 MRR under paraphrasing (+0.0628 over Word BM25, p = 0.0072) but ties lexical search on answerable factual questions (p = 1.0000). On DocVQA (N=500), ColQwen2 (0.6450) substantially outperforms text proxies (<= 0.3507). Dense bi-encoder deficit is associated with query formulation mismatch rather than sequence truncation; natural answerable questions improve BGE-Base MRR from 0.3256 to 0.5263. On MS-COCO (N=1,000), native SigLIP retrieval (0.6071) significantly outperforms machine caption proxies (0.3728, p < 10^-4). Conformal cascades and extractive multi-hop bridging reduce compute overhead while bounding marginal risk.

---
THREE-SENTENCE CONTRIBUTION SUMMARY
1. We present an empirical multi-regime analysis across 1,626 multimodal items (SROIE, FUNSD, MS-COCO, DocVQA) establishing rigorous boundaries for when intermediate text proxies suffice and when native multimodal representations are indispensable.
2. We show that character 3--5gram BM25 provides robust optical noise tolerance across commercial and open-source OCR engines but reverses under entity-substituting paraphrases, while dense bi-encoder collapse is governed by query formulation mismatch rather than sequence truncation.
3. We identify the specific operational regimes where GPU-intensive document-native multi-vector models (ColQwen2) and vision embeddings (SigLIP) justify their compute overhead, consolidating our findings into an empirical decision guide for multimodal RAG system designers.

---
DOUBLE-BLIND REVIEW STATEMENT
In accordance with IJCDS double-blind review requirements, all identifying author details, institutional affiliations, contact emails, and personal repository URLs have been completely removed from the manuscript text, figures, headers, and PDF metadata. The blinded manuscript refers strictly to "two authors" in audit protocols and cites code availability through an anonymized repository placeholder.

---
ORIGINALITY AND CONCURRENT SUBMISSION STATEMENT
We confirm that:
1. This manuscript is an original work that has not been published previously in any journal, conference, or book.
2. It is not currently under consideration, review, or submission for publication elsewhere.
3. All authors have read and approved the manuscript, and there are no conflicts of interest associated with this submission.
4. All research data, scripts, and benchmark configurations are authentic, verified against frozen results files, and fully reproducible.

We thank you and the reviewers for your time and consideration of our manuscript.

Sincerely,

Anurag Krishna Vennela
G. Sai Balaji Koushik
Department of Computer Science and Engineering
National Institute of Technology Warangal
Warangal, India
