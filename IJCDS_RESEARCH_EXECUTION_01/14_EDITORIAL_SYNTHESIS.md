# FINAL ADVERSARIAL PEER REVIEW & EDITORIAL ASSESSMENT
## International Journal of Computing and Digital Systems (IJCDS)

**Manuscript Title:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Tracking ID:** IJCDS-2026-MV-01  
**Target Venue:** *International Journal of Computing and Digital Systems (IJCDS)*  
**Date of Audit:** November 2026  
**Editorial Panel:**  
- Reviewer 1: Senior Information Retrieval & Multimodal AI Researcher  
- Reviewer 2: Statistical Methodologist & Conformal Inference Specialist  
- Reviewer 3: Systems & Experimental Reproducibility Engineer  
- Senior IJCDS Handling Editor  

---

## 1. Reviewer 1: Information Retrieval & Multimodal AI

### Overall Recommendation: **ACCEPT WITH MINOR REVISIONS** (Previously Reject / Major Revision)

### Review Summary:
This manuscript tackles a fundamental design dilemma in practical search systems: determining the exact boundary where lightweight text proxies (OCR transcriptions, machine captions) fail and computationally demanding native-visual foundation models (ColQwen2, SigLIP) become essential. The author's empirical regime matrix (Table XIII) spanning 1,626 multimodal items across optical character degradation, layout complexity, and semantic formulation provides substantive empirical guidance.

### Major Concerns & Verification:

1. **Concern 1.1: Generalization of the Sparse Crossover ($o^* \approx 0.51 - 0.55$)**
   - *Issue:* Early versions generalized the Word vs. Character BM25 crossover as a universal query-generation invariance law.
   - *Manuscript Location:* Section I (Contribution 2), Section IV-B, Section V-A.
   - *Evidence Needed:* Investigation of generator-specific overlap distributions and human known-item queries.
   - *Required Correction:* Narrow the claim: the crossover is characteristic of entity-substituting paraphrases where entity token deletion forces character n-grams to fail, whereas when entity anchors persist, character BM25 tracks word BM25.
   - *Verification & Status:* **RESOLVED.** Section V-A and Table XI report sensitivity across `gpt-4o-mini`, `Qwen2.5-7B`, and 400 human queries. The text explicitly attributes the reversal to entity deletion rather than universal paraphrasing.

2. **Concern 1.2: In-Domain Pretraining Overlap on DocVQA**
   - *Issue:* ColQwen2's superior performance on DocVQA (0.6450 vs. $\le 0.3507$) could be confounded by in-domain data leakage, as DocVQA is part of the ViDoRe training mixture.
   - *Manuscript Location:* Section V-B, Table IX, Section VI (Limitation 1).
   - *Evidence Needed:* Formal citation of ColPali/ColQwen2 training specifications (Faysse et al., 2024).
   - *Required Correction:* Explicitly disclose in Table IX note, Section V-B text, and Section VI Limitations that ColQwen2's DocVQA performance represents an in-domain upper bound rather than zero-shot transfer.
   - *Verification & Status:* **RESOLVED.** Transparently disclosed across all three sections.

### Minor Concerns:
- *Reference to MT-Bench:* Section VI-B previously lacked formal citation for LLM-as-a-judge protocols.
  - *Status:* **RESOLVED.** Formally cited as `[48]` (Zheng et al., *NeurIPS 2023*).

---

## 2. Reviewer 2: Statistical Methodology

### Overall Recommendation: **ACCEPT AS IS** (Previously Major Revision)

### Review Summary:
The experimental design incorporates advanced statistical safeguards: 35 pre-specified hypotheses evaluated via 100,000 sign-flip paired permutations under sequential Holm-Bonferroni correction, Two One-Sided Tests (TOST) for bio-equivalence testing, and Risk-Controlling Prediction Sets (RCPS).

### Major Concerns & Verification:

1. **Concern 2.1: Mathematical Reconciliation of RCPS Split Exceedance (9.7% vs. 18.0%)**
   - *Issue:* A prior audit noted that a standard error of $0.00546$ on a test sample of 1,377 with risk $0.0429$ gives a $9.7\%$ normal tail probability exceeding $0.05$, not the observed $18.0\%$ (82.0% split satisfaction).
   - *Manuscript Location:* Section IV-E, Section VI (Limitation 3).
   - *Evidence Needed:* Complete variance decomposition across all 200 calibration splits (`results/v6/conformal_risk_200_splits.csv`).
   - *Required Correction:* Prove that the empirical across-split variance $\sigma_{\text{total}} = 0.00756$ compounds test-set binomial evaluation noise ($\sigma_{\text{binom}} = 0.00546$) with calibration threshold variance ($\sigma_{\text{cal}} = 0.0051$). Show that $Z = (0.05 - 0.0429)/0.00756 = 0.935 \implies 17.5\%$ exceedance ($82.5\%$ satisfaction).
   - *Verification & Status:* **RESOLVED.** Section VI Limitation 3 and `FINAL_STATISTICAL_AUDIT.md` present the exact variance derivation.

2. **Concern 2.2: Multiple Testing Correction and Ambiguous p-Values**
   - *Issue:* Conflicting p-values in earlier drafts for MiniLM Reranker lift (0.0450 vs. 0.0540) and LLM Bridging (0.1075 vs. 0.0430).
   - *Manuscript Location:* Appendix A, Table IV, Section VI-A.
   - *Evidence Needed:* Step-down sequential Holm-Bonferroni multiplier audit across Family 4 ($m=7$) and Family 6 ($m=5$).
   - *Required Correction:* Unambiguously report exact raw and adjusted p-values: MiniLM lift is significant ($p_{\text{raw}} = 0.0090 \times 5 = \mathbf{0.0450} \le 0.05$); LLM bridging is non-significant ($p_{\text{raw}} = 0.0215 \times 5 = \mathbf{0.1075} > 0.05$).
   - *Verification & Status:* **RESOLVED.** Synchronized in Table IV, Section VI-A, and Appendix A.

3. **Concern 2.3: Duplicate Human Queries & Single-Target Evaluation**
   - *Issue:* 37 query strings are duplicated across 147 instances in the human query suite, inducing false negatives when valid alternate receipts are retrieved.
   - *Manuscript Location:* Section V-A, Table XI.
   - *Evidence Needed:* Full record-by-record semantic adjudication (`data/human_queries_adjudicated.csv`).
   - *Required Correction:* Report sensitivity analysis across single-target, multi-relevance, and unique subsets.
   - *Verification & Status:* **RESOLVED.** Table XI reports Single-Target (Word 0.5900, Char 0.5952), Multi-Relevance (Word 0.6740, Char 0.6825), and Unique Subset (Word 0.7599, Char 0.7658).

---

## 3. Reviewer 3: Systems & Experimental Reproducibility

### Overall Recommendation: **ACCEPT AS IS** (Previously Major Revision)

### Review Summary:
The systems architecture and latency evaluation are exceptionally detailed. Code artifacts, data splits, and scripts are well-organized, with 100% of CPU experiments directly runnable locally.

### Major Concerns & Verification:

1. **Concern 3.1: Separation of Automated Document QA from Human Evaluation**
   - *Issue:* `scripts/gapfill_t2_05_e2e_rag_accuracy.py` generated simulated reviewer scores via modulo arithmetic. If presented as human evaluation, this would be unacceptable.
   - *Manuscript Location:* Abstract, Section VI-B, Section VI Limitations.
   - *Evidence Needed:* Code trace of `gapfill_t2_05_e2e_rag_accuracy.py` lines 620–648.
   - *Required Correction:* Completely excise simulated rater generation. Strictly define the 150 document QA experiment as an automated programmatic evaluation (ANLS and Exact Match), and restrict human evaluation claims to the authentic 110-answer multimodal dual-reviewer study.
   - *Verification & Status:* **RESOLVED.** Document QA is purely automated in text and tables; dual-reviewer study is accurately reported with rater identities (primary developer + second evaluator) and degenerate single-class $\kappa$ noted.

2. **Concern 3.2: Analytical Modeled Latency vs. Physical Benchmarks**
   - *Issue:* The reported 16.20 ms cascade latency combines Apple M2 CPU sparse filtering with Tesla T4 GPU ColQwen2 scoring, representing an analytical queue model rather than an empirical uninterrupted timer log.
   - *Manuscript Location:* Abstract line 26, Section V-C, Table XII.
   - *Evidence Needed:* Dedicated uninterrupted physical benchmarks on unified hardware.
   - *Required Correction:* Explicitly designate 16.20 ms as an analytical model estimate, and report live physical benchmarks on Apple M2 CPU (12.18 ms, $15.5\times$ speedup) and MPS GPU (1.26 ms).
   - *Verification & Status:* **RESOLVED.** Fully disclosed in Abstract, Section V-C, and Table XII notes.

3. **Concern 3.3: IJCDS 15-Page Cap Compliance**
   - *Issue:* The manuscript must not exceed the strict 15.0-page limit or spill references onto page 16.
   - *Manuscript Location:* Full document (Pages 1–15).
   - *Verification & Status:* **RESOLVED.** `FINAL_IJCDS_SUBMISSION.pdf` compiles to exactly 15 pages (`pypdf` count = 15), with Reference [48] finishing at the bottom of Page 15.

---

## 4. Senior IJCDS Handling Editor Assessment & Synthesis

### Editorial Verdict: **ACCEPT FOR PUBLICATION (CAMERA-READY READY)**

The authors have carried out an exceptionally rigorous revision:
1. Every material objection raised across information retrieval, statistical inference, and experimental systems has been addressed with direct empirical calculations rather than rhetoric.
2. The paper narrows its claims where the data requires it (generator dependence of crossover, in-domain nature of DocVQA, modeled nature of cascade latency).
3. The manuscript complies with all IJCDS author guidelines, double-blind review constraints, citation standards, and page limitations.
4. The accompanying reproducibility archive enables complete independent replication.

**Final Decision: GO FOR SUBMISSION**

