# CHANGELOG: Manuscript V9 to Submission-Ready V10 (CHANGELOG.md)

**Project:** MemoryVerse (Multimodal Retrieval-Augmented Generation Regime Analysis)  
**Target Journal:** International Journal of Computing and Digital Systems (IJCDS)  
**Date:** October 28, 2026  
**Auditor / Experiment Engineer:** Antigravity (Google DeepMind pair programmer)  
**Status:** Complete & Submission-Ready  

---

## 1. Executive Summary & Audit Statistics

| Audit Category | Metric | Status |
| :--- | :--- | :--- |
| **Total Numerical Claims Extracted** | 1,717 claims across Abstract, Contributions, Tables, Body, Limitations, Conclusion | Complete |
| **Verified OK Claims** | 1,717 / 1,717 (100.0%) | Verified |
| **Mismatched / Untraceable Claims** | 0 (0.0%) | Resolved |
| **Automated LaTeX Tables Generated** | 13 publication tables via `scripts/make_tables.py` into `latex_fragments/` | Verified |
| **Abstract Word Count** | Exactly 199 words (IEEEtran / IJCDS limit: 200 words) | Verified |
| **LaTeX Compilation** | Tectonic build clean (0 errors, 0 undefined citations, 0 undefined references) | Verified |
| **IJCDS Template Build** | `ijcds/ijcds_master_doc.pdf` compiled cleanly (2.08 MB) | Verified |

---

## 2. Phase-by-Phase Changelog

### Phase 1: Number Audit & Single Source of Truth Tables
- **Generated `claims_ledger.csv`:** Full audit tracking 1,717 empirical claims against `results/v10/number_provenance.csv` and frozen result files. 100% verified with 0 non-OK rows.
- **Created `scripts/make_tables.py`:** Generates all 13 LaTeX tables directly from raw CSV results, removing manual copy-paste errors.
- **Unified ColQwen2 Evaluation:** Re-evaluated ColQwen2-v1.0 on Nvidia Tesla T4 GPU under identical seeds (42, 43, 44, $N_{\text{perm}}=100{,}000$). Resolved the discrepancy between $+0.2500$ (cross-engine subset without distractor text) and $+0.0628$ (headline regime grid). Point estimate: MRR = 0.4962 [0.4484, 0.5472]; lift over Word BM25 (0.4334) is $+0.0628$ [$+0.0200, +0.1113$] ($p_{\text{Holm}}=0.0072$); lift over Tuned Hybrid (0.4335) is $+0.0647$ [$+0.0228, +0.1125$] ($p_{\text{Holm}}=0.0095$).
- **Resolved Word BM25 Divergence (0.8371/0.4334 vs. 0.8324/0.4003):** Documented that headline grid indexes ground-truth distractor captions, whereas engine replication evaluated distractors as empty strings. Removed false footnote claiming "$\le 0.005$ MRR seed difference".
- **Fixed Chunked vs. Unchunked MiniLM Labels:** Corrected Table II label to `Dense MiniLM (unchunked)` (0.2056) to match Section IV-B text (unchunked 0.2056 vs. chunked MaxP 0.2030).
- **Ablation Table Footnotes:** Added explicit notes to Table V and Table VI explaining candidate depth $K_1=10$ (v2 multi-stream, 0.8121 baseline, 0.8555 cascade) vs. $K_1=20$ (production cascade over first-stage Word BM25 0.8371 reaching 0.8846).
- **Latency Discrepancies Resolved:** Clarified isolated vector dot-product latency (0.01 ms) vs. end-to-end query encoding latency (7.78 ms / 19.93 ms). Compared ColQwen2 answerable latency (249.21 ms) against CPU lexical answerable latency (5.72 ms) as well as verbatim lookups (0.87 ms).

### Phase 2: Claim Calibration & Overclaim Elimination
- **Softened Abstract:** Replaced "statistically matching human gold captions" with "no significant difference detected (p=0.142; equivalence not tested)".
- **Calibrated Causal Wording:** Replaced causal claims ("reflects", "is driven by", "causes") about dense retrieval collapse with "is associated with". Explicitly declared the clean Dev lexicon vocabulary projection experiment (+0.1530 cosine similarity, +0.0434 MRR recovery) as the sole direct causal intervention.
- **Reframed Conformal Calibration:**
  - Removed misleading generalization risk CI framing; clarified that the 200 splits resample the same pooled finite set ($N=2{,}754$ instances), so the CI $[0.0480, 0.0502]$ describes only split variability on this fixed pool.
  - Stated plainly that empirical split coverage is only 58.5% (falling below nominal 95% single-split coverage).
  - Acknowledged that equal-escalation fixed margin ($\tau=0.235$) achieves matching accuracy (MRR 0.3577 vs. 0.3579) and risk (0.0494 vs. 0.0488).
  - Reduced "21.6% savings" marketing language to factual "78.4% escalation rate".
- **Multi-Hop Statistical Power Statement:**
  - Labeled the $N=50$ comparison as an exploratory diagnostic finding of no detectable difference ($p=1.0000$).
  - Added formal power statement: minimum detectable difference (MDD) at $\alpha=0.05$ and 80% power exceeds $\pm 0.0792$, spanning beyond the $\pm 0.05$ equivalence margin.
  - Excluded the pooled 100-case suite (Hit@10 = 0.8200) from headline parity claims because it includes 50 single-hop controls with 100% recall and zero discordance.
- **Eliminated Overclaiming Language:**
  - Replaced "strictly dominates" with "consistently outperforms".
  - Replaced "decisively surpass" and "decisively beat" with "significantly outperform".
  - Replaced "refuted" with "rejected" and "not supported by empirical evidence".
- **Limitations Audited Against Raw Data:** Documented domain sensitivity between SROIE receipts (Word BM25 > Char BM25 under light paraphrases) and FUNSD administrative forms (Char BM25 > Word BM25 under light paraphrases: 0.2228 vs. 0.1573 Apple, 0.2442 vs. 0.2019 Tesseract) due to compound form headers.

### Phase 3: Strengthened Evidence & Baselines
- **Third OCR Engine Replication (EasyOCR 1.7.2):** Pre-registered in `PREREG_V9.md` and evaluated across clean and degraded blur OCR ($N=306$ held-out probes). Clean EasyOCR achieves CER = 0.4026 and WER = 0.5684; degraded blur escalates to CER = 0.7755 and WER = 0.9827. Verbatim Char 3--5gram BM25 retains top MRR (0.8519 clean, 0.4272 blur) over Word BM25 (0.7649 clean, 0.3472 blur), reversing under light paraphrases (0.3792 vs. 0.2292), with dense bi-encoders collapsing (0.1549).
- **Visual Document Retrieval (DocVQA):** Evaluated ViDoRe DocVQA subsample (500 queries, 500 documents). Text proxies achieve MRR $\le 0.3507$ (Word BM25 0.3107, Char BM25 0.3141, BGE 0.2737, Hybrid 0.3507), whereas document-native ColQwen2 achieves MRR = 0.6450 (Hit@10 = 0.8410), defining the regime boundary where visual document geometry governs answerability.
- **Edit-Distance BM25 Evaluation:** Implemented and tested Levenshtein edit-distance expansion on Word BM25. MRR drops from 0.8371 to 0.8005 on clean Apple Vision and from 0.8572 to 0.8301 on ground truth, proving that fuzzy word expansion causes distractor false positives and cannot substitute for overlapping character 3--5grams.

### Phase 4: Literature & Baseline Repositioning
- **Classic Information Retrieval Grounding:** Integrated foundational citations in Section II-E:
  - Kantor & Voorhees (2000): TREC-5 Confusion Track OCR corruption evaluation on scanned collections.
  - McNamee & Mayfield (2004): Character $n$-gram tokenization for error-resilient text retrieval.
- **Corrected JiWER Citation:** Mapped citation key `vaessen2024jiwer` to author Nik Vaessen (Python Package Index, 2024).
- **Cleaned Bibliography:** Removed unused reference `piczak2015esc50`. Verified 43 total bibitems and 43 cited keys with 0 unused and 0 missing.

### Phase 5: IJCDS Submission Template Synchronization
- **Synchronized `ijcds/ijcds_body.tex`:** Updated manuscript body with all calibrated numbers, softened claims, and classic literature citations.
- **Verified Tectonic Compilation:** `ijcds/ijcds_master_doc.pdf` (2.08 MB) compiles cleanly with 0 undefined references and 0 undefined citations.

### Phase 6: Reviewer Risk Report
- **Created `reviewer_risk_report.md`:** Detailed the Top 10 vulnerabilities a hostile reviewer would target, their severity, and pre-computed empirical defenses embedded in the manuscript.

---

## 3. Verification Sign-Off

The reproduction harness (`./reproduce_v10.sh`) executes and verifies all 5 quality gates:
1. `[1/5]` Frozen dataset inputs, provenance CSVs, and Kaggle run scripts exist.
2. `[2/5]` `check_numbers.py` reports 100% verification across all paper numbers.
3. `[3/5]` Abstract contains exactly 199 words (limit: 200).
4. `[4/5]` Tectonic compilation builds `IEEE_Paper_v10.pdf` cleanly.
5. `[5/5]` Log audit confirms 0 undefined references and 0 undefined citations.

