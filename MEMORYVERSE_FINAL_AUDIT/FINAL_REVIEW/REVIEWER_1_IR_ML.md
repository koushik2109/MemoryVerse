# ADVERSARIAL PEER REVIEW REPORT 1: INFORMATION RETRIEVAL & MACHINE LEARNING
**Target Journal:** International Journal of Computing and Digital Systems (IJCDS)  
**Reviewer Profile:** Senior Information Retrieval Researcher, Multi-Vector Dense Representation Specialist  
**Review Completed:** November 2026  
**Final Recommendation:** **MINOR REVISION** (Upgraded from Major Revision following Manuscript V2 Repairs)  

---

## 1. OVERALL EVALUATION & SUMMARY

This paper provides an exceptionally detailed, extensive empirical comparison between intermediate text proxies (Word BM25, Character 3–5gram BM25, dense bi-encoders) and document-native multimodal models (ColQwen2-v1.0, SigLIP, CLIP). The investigation addresses a fundamental, practical dilemma faced by enterprise retrieval architectures: *when is it safe and computationally economical to rely on OCR transcripts, and when does native visual pixel retrieval become essential?*

The core experimental findings—particularly the resilience of character subword $n$-grams over whole words under optical noise (+0.10 to +0.13 MRR) and the subword token fragmentation mechanism (+122% token expansion) that destroys dense bi-encoder embeddings—are methodologically robust, practically actionable, and supported by extensive empirical evidence.

In the initial submission draft, several significant technical over-claims hindered full acceptance (specifically, claiming an invariant crossover threshold at 0.51–0.55 overlap, conflating document QA with personal photo-album human review, and presenting modeled cascade latency as measured runtime). The corrections introduced in Manuscript V2 substantially resolve these concerns.

---

## 2. MAJOR STRENGTHS

1. **Practical Realism of the Evaluation Suite:** The inclusion of real optical capture devices (Apple Vision neural OCR, open-source Tesseract 5.5, EasyOCR), physical blur sweeps, and real OCR-corrupted receipts ($N=626$) gives the paper immense practical value beyond idealized synthetic benchmarks.
2. **Identification of the Dense Collapse Root Cause:** The tokenization analysis demonstrating that dense bi-encoder failure under OCR is driven by subword token fragmentation (+122.3% WordPiece expansion) rather than sequence truncation ($<0.7\%$) is a genuine, high-quality scientific contribution.
3. **Disclose of In-Domain ViDoRe Training Overlap:** In Table IX and Section 5.B, the authors transparently acknowledge that ColQwen2 was fine-tuned on ViDoRe (which includes DocVQA), framing ColQwen2's 0.6450 MRR as an in-domain upper bound rather than an exaggerated zero-shot claim.
4. **Viable Engineering Cascades:** The two-stage cascade ($K_1=50$ Word BM25 pre-filtering $\to$ ColQwen2 reranking) retains 97.2% of native visual MRR (0.4822 vs. 0.4962), demonstrating a sound balance between latency and accuracy.

---

## 3. REMAINING WEAKNESSES & PEER-REVIEW CONCERNS

1. **Modeled vs. Measured Latency in Cascades:** While Manuscript V2 correctly labels the 16.20 ms latency as an analytical model estimate, a live end-to-end GPU benchmark that includes PCIe transfer and batched multi-vector MaxSim scoring on candidate images would provide even stronger empirical footing. (The authors have appropriately addressed this by providing a companion Kaggle GPU benchmark notebook).
2. **Synthetic Distractor Vocabulary Entropy:** The 500,000 distractor corpus was constructed by replicating 20 vendor templates with randomized dates and numerical totals. While this stresses inverted index posting list traversal and RAM consumption, it does not fully replicate the vocabulary entropy of 500,000 distinct real-world retail chains.

---

## 4. DETAILED ASSESSMENT OF CLAIMS & CORRECTIONS

- **Crossover Dynamics:** The revision in Manuscript V2 retracting the universal crossover threshold and acknowledging that human queries exhibit no steep reversal (Word and Char BM25 are tied within 0.027 MRR) is scientifically honest and accurately reflects the data.
- **Document RAG Evaluation:** Delineating automated string ANLS/EM evaluation on the 150 document queries from the authentic 110-answer multimodal qualitative audit completely resolves the prior attribution ambiguity.

---

## 5. FINAL RECOMMENDATION
**Recommendation: MINOR REVISION.**  
The revised manuscript and experimental evidence meet high IR research standards and offer substantial empirical utility to the computational and digital systems community.

