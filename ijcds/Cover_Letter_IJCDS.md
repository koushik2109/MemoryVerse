# Cover Letter for Manuscript Submission

**To:**  
Prof. Dr. Wael M. El-Medany  
Editor-in-Chief  
*International Journal of Computing and Digital Systems (IJCDS)*  
University of Bahrain, Kingdom of Bahrain  

**Date:** October 27, 2026  
**Subject:** Submission of Original Regular Paper Manuscript for Peer Review  

---

Dear Prof. Dr. El-Medany and Esteemed Editorial Board,

We are pleased to submit our original research manuscript titled **"When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise"** for consideration as a Regular Paper in the *International Journal of Computing and Digital Systems (IJCDS)*. The manuscript is authored by Anurag Krishna Vennela and G. Sai Balaji Koushik from the Department of Computer Science and Engineering at the National Institute of Technology Warangal, India.

This work presents a comprehensive empirical regime analysis addressing a central architectural question in multimodal information retrieval and Multimodal Retrieval-Augmented Generation (MM-RAG): when intermediate text proxies (such as optical character recognition transcripts and visual captions) suffice, and where native multi-vector representations become essential. Through pre-registered, held-out benchmarking across 1,626 items (encompassing administrative receipts, scanned forms, natural images, and continuous acoustic audio), our study reveals four principal discoveries:
1. Character 3–5gram inverted indexes establish decisive superiority under verbatim optical OCR noise (+0.1065 clean, +0.1276 degraded blur, $p < 10^{-4}$), refuting the conventional reliance on whole-word BM25;
2. Dense bi-encoder collapse on document text stems from query-span formulation discrepancies rather than sequence truncation constraints, where answerable questions restore discriminability (BGE-Base MRR = 0.5263);
3. Distribution-free conformal cascade ranking rigorously bounds test miss risk below 0.05 while yielding 21.6% computational savings; and
4. Native visual encoders (SigLIP MRR = 0.6071) decisively surpass machine-generated caption proxies on natural imagery ($p < 10^{-4}$), while rule-based pseudo-relevance feedback matches multi-hop generative LLM bridging with zero inference latency.

We believe this investigation aligns directly with the core scope of *IJCDS* in applied computing systems, digital information retrieval, and reliable machine learning architectures. In particular, our empirical findings provide actionable, hardware-profiled deployment guidelines for enterprise document digitization systems and low-resource edge architectures, directly serving computing practitioners and researchers in digital systems.

We confirm that this manuscript represents original work that has not been published previously and is not under consideration for publication elsewhere. All authors have reviewed and approved the manuscript for submission, and there are no conflicts of interest to declare. All benchmarks, reproducible harness scripts, and pre-registered audit artifacts have been made publicly available.

Thank you for your time, consideration, and coordination of the peer-review process.

Sincerely yours,

**Anurag Krishna Vennela**  
Corresponding Author  
Department of Computer Science and Engineering  
National Institute of Technology Warangal, Telangana 506004, India  
Email: anuragkrishnavennela@gmail.com  

**G. Sai Balaji Koushik**  
Co-Author  
Department of Computer Science and Engineering  
National Institute of Technology Warangal, Telangana 506004, India  
Email: gummadavellikoushik2109@gmail.com  
