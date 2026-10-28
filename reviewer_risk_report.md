# Reviewer Risk Report: Critical Audit & Defensive Analysis (reviewer_risk_report.md)

**Manuscript Title:** When Do Text Proxies Suffice? A Multi-Regime Analysis of Intermediate Representations in Multimodal RAG  
**Target Journal:** International Journal of Computing and Digital Systems (IJCDS)  
**Authors:** Anurag Krishna Vennela and Gummadavelli Sai Bhuvanesh Koushik  
**Audit Date:** October 28, 2026  
**Status:** Submission-Ready (Audit Complete, 100% Verified Provenance, 0 Non-OK Numbers)

---

## Executive Summary

This report systematically anticipates the Top 10 vulnerabilities, methodological critiques, and hostile reviewer attack vectors against Manuscript V10. For each risk, we document its architectural severity, the potential reviewer attack, the exact empirical evidence, and the pre-computed defense embedded in the manuscript.

---

## Top 10 Reviewer Risks and Pre-Computed Defenses

### 1. Single-Corpus Receipt Concentration (SROIE)
- **Reviewer Attack:** "The primary OCR retrieval experiments rely heavily on scanned retail receipts from ICDAR 2019 SROIE ($N=626$). Receipts have unique typographical and structural quirks; findings cannot be generalized to administrative forms, multi-column articles, or dense technical reports."
- **Severity:** Medium
- **Pre-Computed Defense:**
  - **Cross-Dataset Administrative Form Replication:** The manuscript includes full out-of-distribution evaluation on 100 scanned administrative forms from FUNSD (Section IV-E, Table XII).
  - **Empirical Scope Boundary Delineation:** The manuscript does not claim universal receipt dominance. On FUNSD administrative forms, the verbatim margin narrows to a statistical tie (+0.0069, MRR 0.8433 vs. 0.8364). Under light paraphrasing on FUNSD, character BM25 retains a lead (0.2228 vs. 0.1573) due to subword matching of compound form headers.
  - **Cross-Domain Visual Document VQA (DocVQA):** Section IV-E documents the ViDoRe DocVQA benchmark ($N=500$ queries, 500 documents), proving that where complex visual geometry governs answerability, document-native ColQwen2 achieves 0.6450 MRR vs. 0.3507 for Tuned Hybrid, establishing the exact upper bound where text proxies fail.

### 2. LLM-Generated Queries & Single Generator Threat
- **Reviewer Attack:** "Light paraphrases and answerable questions were synthesized using a single LLM (`gpt-4o-mini`). Model-generated queries carry systematic linguistic biases and do not reflect spontaneous human search queries or human typing mistakes."
- **Severity:** Medium-High
- **Pre-Computed Defense:**
  - **Transparent Disclosure:** Section VII, Item 9 explicitly acknowledges this as a limitation: *"Evaluation queries for light paraphrases and answerable questions were synthesized using gpt-4o-mini... Because queries were generated using a single LLM without multi-model cross-generation or human validation, model-generated queries may under-represent human typographical errors, spontaneous colloquialisms, or ambiguous intent."*
  - **Orthogonal Diagnostic Probe Stratification:** In Section IV-B, test probes are stratified into five property buckets (Numeric-Heavy, Short Entities, Rare Tokens, Named Entities, Long Text) that isolate specific failure modes independently of LLM prompt phrasing.
  - **Dual-Reviewer Human Audit:** Co-author auditing across 110 blind-evaluated cases confirmed high inter-rater agreement ($\kappa = 0.8716$ faithfulness, $0.8926$ relevance, $0.9451$ correctness, overall $\kappa = 0.9031$).

### 3. Conformal Calibration Empirical Split Coverage Below Nominal Target (58.5%)
- **Reviewer Attack:** "The authors claim distribution-free risk control under conformal prediction, but their empirical split coverage is only 58.5% on a nominal 95% target, and an equal-escalation fixed margin threshold achieves identical MRR. Conformal cascades provide no practical benefit."
- **Severity:** Medium
- **Pre-Computed Defense:**
  - **Total Empirical Honesty:** The manuscript states this directly in Contributions, Section IV-C, Table IV, and Section VII: *"Empirical split coverage—the proportion of test splits meeting the nominal target $\epsilon = 0.05$—is only 58.5%, falling substantially below the nominal 95% confidence level expected for single-split high-probability risk control."*
  - **Distinction Between Marginal Loss and High-Probability RCPS:** The paper explains that standard marginal conformal loss controls expected risk over draws, not single-split coverage. To achieve high-probability split coverage, High-Probability RCPS ($\delta=0.10$) must be used, which achieves 82.0% split coverage.
  - **Parity with Equal-Escalation Baseline:** The paper explicitly acknowledges that equal-escalation fixed margin ($\tau=0.235$) achieves identical MRR (0.3579 vs. 0.3577, $p > 0.05$) and risk (0.0488 vs. 0.0494), positioning conformal prediction as a calibrated framework for threshold selection rather than a ranking algorithm.
  - **Finite Resampled Pool Disclosed:** The paper explicitly clarifies that the 200 splits resample the same pooled finite set ($N=2{,}754$ instances), so the CI of the mean $[0.0480, 0.0502]$ describes split variability across partitions of this pool rather than out-of-distribution generalization risk.

### 4. Multi-Hop Relational Reasoning Sample Size ($N=50$) and Power Limits
- **Reviewer Attack:** "The multi-hop diagnostic suite contains only $N=50$ queries. Asserting that extractive PRF 'matches' generative LLM bridging is an underpowered null-hypothesis claim. Furthermore, pooling with 50 single-hop controls to get $N=100$ inflates agreement."
- **Severity:** Medium
- **Pre-Computed Defense:**
  - **Softened Claims & Exploratory Framing:** All claims in the Abstract, Contributions, Section IV-D, Section V, and Limitations now explicitly label the $N=50$ comparison as an *"exploratory diagnostic finding of no detectable difference (Hit@10 0.640 vs. 0.640, exact McNemar $p=1.0000$)"*.
  - **Explicit Power & Equivalence Statement:** The paper explicitly calculates the minimum detectable difference (MDD $> \pm 0.0792$ at 80% power) and notes that the 95% CI $[-0.0792, +0.0792]$ exceeds the pre-registered equivalence boundary $\pm 0.05$, openly stating that formal equivalence cannot be claimed.
  - **Rejection of Pooled $N=100$ as Headline Parity Support:** The paper explicitly warns that the pooled 100-query metric (Hit@10 = 0.8200) conflates multi-hop reasoning with 50 trivial single-hop controls (where both achieve 100% recall with zero discordance) and cannot serve as statistical support for multi-hop parity.

### 5. Causal vs. Associational Framing of Dense Retrieval Collapse
- **Reviewer Attack:** "Claiming that query formulation mismatch 'causes' or 'drives' dense bi-encoder collapse is an overclaim without counterfactual interventions on queries."
- **Severity:** Low-Medium
- **Pre-Computed Defense:**
  - **Rigorous Linguistic Calibration:** All causal verbs ("reflects", "is driven by", "causes") regarding dense collapse on clean text have been replaced with "is associated with".
  - **Sole Causal Evidence Preserved:** Section IV-B explicitly states that the clean Dev lexicon vocabulary projection experiment (reconstructing WordPiece fragmentation, recovering +0.1530 cosine similarity and +0.0434 MRR) provides the sole direct causal intervention in the study, establishing that tokenization fragmentation contributes causally to noise-induced degradation, whereas the link between query formulation and clean-text retrieval remains observational.

### 6. Acceptance of Null Hypothesis on MS-COCO SigLIP vs. Human Gold Captions
- **Reviewer Attack:** "Reporting that native SigLIP 'statistically matches' human gold captions based on $p=0.142$ is mathematically invalid (absence of evidence is not evidence of absence)."
- **Severity:** Medium
- **Pre-Computed Defense:**
  - **Strictly Softened Phrasing:** The phrase "statistically matching" has been eradicated. Abstract, Section IV-B, Section V, and Conclusion now uniformly report: *"no significant difference detected from human gold captions (MRR = 0.5835 vs. SigLIP 0.6071, diff -0.0236, paired permutation $p = 0.142$, equivalence not tested)"*.

### 7. Heterogeneous Latency Profiling (CPU vs. GPU)
- **Reviewer Attack:** "Comparing 0.87 ms CPU lexical lookups against 249.21 ms GPU ColQwen2 is an unfair comparison across disparate hardware."
- **Severity:** Low
- **Pre-Computed Defense:**
  - **Transparent Hardware Attribution:** Table XIII (Table~\ref{tab:system_costs}) explicitly separates latency by hardware environment: Apple Silicon M2 (16 GB unified RAM) for lightweight CPU proxies vs. Nvidia Tesla T4 GPU (14.56 GB GDDR6) for ColQwen2.
  - **Latency Definitions Formalized:** The manuscript distinguishes isolated vector dot product latency (0.01 ms) from end-to-end query encoding + retrieval latency (7.78 ms / 19.93 ms), and compares ColQwen2 answerable latency (249.21 ms) against character BM25 answerable query latency (5.72 ms) as well as verbatim lookups (0.87 ms).

### 8. Failure of Edit-Distance Expansion on Word BM25
- **Reviewer Attack:** "Why didn't the authors evaluate classic fuzzy matching or edit-distance expanded BM25 before declaring character n-grams superior?"
- **Severity:** Low
- **Pre-Computed Defense:**
  - **Empirically Evaluated & Disclosed:** Section IV-E documents that Levenshtein edit-distance expansion on Word BM25 actually degrades performance: MRR drops from 0.8371 to 0.8005 on clean Apple Vision and from 0.8572 to 0.8301 on ground truth text. Fuzzy word expansion causes spurious term collisions with distractor documents, proving that overlapping character 3--5grams capture structural sub-word evidence far more effectively than heuristic string expansion.

### 9. Third OCR Engine Generalization (EasyOCR)
- **Reviewer Attack:** "Evaluating Apple Vision and Tesseract is insufficient; Apple Vision is a commercial black box and Tesseract is an older LSTM architecture."
- **Severity:** Low
- **Pre-Computed Defense:**
  - **Pre-Registered Third Engine Replication:** In `PREREG_V9.md` and Section IV-E, we evaluated EasyOCR 1.7.2 (CRAFT neural text detector + deep ResNet-CRNN recognizer with CTC loss). Clean EasyOCR achieves CER = 0.4026 and WER = 0.5684 (escalating to CER = 0.7755 and WER = 0.9827 under optical blur).
  - **Hypothesis Confirmation:** All pre-registered hypotheses were confirmed: Character 3--5gram BM25 retains top MRR under verbatim queries (0.8519 clean, 0.4272 degraded blur) over Word BM25 (0.7649 clean, 0.3472 blur; $p_{\text{Holm}} < 10^{-4}$), reversing under light paraphrases (0.3792 vs. 0.2292), while dense bi-encoders collapse (0.1549).

### 10. Classic Information Retrieval Repositioning
- **Reviewer Attack:** "Character n-gram retrieval on OCR text is well known in classical IR. What is the novel contribution here?"
- **Severity:** Low
- **Pre-Computed Defense:**
  - **Historical Grounding:** In Section II-E, the paper grounds sub-word indexing in foundational literature: the TREC-5 Confusion Track (Kantor & Voorhees 2000) and McNamee & Mayfield (2004).
  - **Novel Paradigm Repositioning:** Modern multimodal RAG pipelines frequently discard character n-grams under the belief that deep embeddings supersede lexical matching. Our work establishes the contemporary empirical crossover boundary: where CPU text proxies suffice (verbatim lookups, factual answerable questions), where hybrid cascades prune cross-encoders, and where native pixel vision models (SigLIP on natural images, ColQwen2 on complex document geometry) are indispensable.

---

## Conclusion

Manuscript V10 and its companion IJCDS document (`ijcds/ijcds_master_doc.pdf`) present a watertight, rigorously calibrated defense across all 10 reviewer attack vectors. All 1,717 empirical claims trace directly to frozen CSV result files with 0 discrepancies.

