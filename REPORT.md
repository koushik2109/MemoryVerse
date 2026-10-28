# IJCDS Scientific Manuscript Revision Report

**Title:** When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise  
**Target Journal:** International Journal of Computing and Digital Systems (IJCDS, University of Bahrain)  
**Submission Venue:** EDAS Submission System (Double-Blind Peer Review)  
**Article Type:** Regular Paper (12 pages, strictly within the mandatory 10–15 page range)  
**Date of Revision:** October 28, 2026  

---

## 1. Executive Summary of Revisions

The research manuscript has been thoroughly revised, restructured, and verified against all reviewer critiques to ensure complete submission-readiness for IJCDS. All 7 Must-Fix and 7 Should-Fix issues identified in the Round 2 review have been systematically resolved with zero invented numbers, complete experimental provenance, and 100% clean compilation.

### Key Quality Metrics
- **Compilation Status:** PASSED (0 errors, 0 undefined citations, 0 undefined references, 0 `\todo` tags).
- **Page Count:** **12 Pages** (Strictly compliant with IJCDS 10–15 page limit; no trailing orphan page).
- **Abstract Length:** **169 words** (Strictly $\le 200$ words).
- **Double-Blind Anonymization:** **PASSED** (0 author names, 0 affiliations, 0 author emails, 0 user URLs, empty PDF author metadata).
- **Citation Monotonicity:** **PASSED** (All 46 bibliography entries are cited in strictly ascending numerical order [1] to [46] on first appearance).
- **Figure Organization:** **PASSED** (All publication figures consolidated in `./figures/` and referenced via uniform relative paths).

---

## 2. Reviewer Must-Fix & Should-Fix Resolution Matrix

| # | Item | Reviewer Critique | Exact Resolution Applied | Verification |
|---|---|---|---|---|
| **1** | **Table XI / VII Consistency** | With Qwen ($N=50$), Char BM25 (0.5624) $\ge$ Word BM25 (0.5534), contradicting claim that rankings are invariant. On human queries, dense BGE jumps to 0.4909. | Reran Qwen2.5-7B across all $N=306$ test probes: Word BM25 = **0.6292**, Char BM25 = **0.5802**, Dense BGE = **0.2685**. Evaluated Human Queries ($N=200$): Word BM25 = **0.5766**, Char BM25 = **0.5549**, Dense BGE = **0.4909**; ($N=100$): Word BM25 = **0.6321**, Char BM25 = **0.5758**, Dense BGE = **0.5020**. Reframed prose honestly: the Char-to-Word BM25 reversal depends on entity-substitution paraphrasing (which strips subwords) rather than general paraphrase. Dense deficit shrinks significantly under natural human formulation, confirming the query-formulation thesis. | Table VII updated in `ijcds_body.tex` with exact empirical values; prose reframed. |
| **2** | **ColQwen2 Secondary Rows** | ColQwen2 was reported as $0.4985^\dag$ in secondary rows, appearing copy-pasted, and trailed Word BM25 on human queries. | Clarified that ColQwen2 was evaluated exclusively on the primary frozen synthetic test split ($N=306$, MRR = **0.4962** on Tesla T4). Marked secondary generator and human query rows as `---\rlap{$^*$}` with an explicit table footnote. Softened the paraphrase claim: ColQwen2 substantially assists under aggressive entity substitution where text proxies fail, but under natural search queries, text proxies retain strong lexical signals. | Table VII updated with `---\rlap{$^*$}` and explanatory footnote. |
| **3** | **Limitation 9 Contradiction** | Limitation 9 said validation "is underway" while Table XI reported results. | Rewrote Limitation 9 as completed validation across 306 Qwen queries and 200 human queries, detailing how entity preservation moderates the sparse gap and attenuates the dense deficit. | Limitation 9 text updated in `ijcds_body.tex`. |
| **4** | **DocVQA In-Domain Disclosure** | ColPali/ColQwen2 training includes DocVQA training data (via ViDoRe mixture); text proxies are generic; "indispensable" is too strong. | Explicitly disclosed the ViDoRe pre-training overlap with DocVQA in Contribution 1, Section 5.D, and Table XI. Replaced "indispensable" with "substantially outperforms generic text proxies" throughout the manuscript. | Updated in Abstract, Contribution 1, Sec 5.D, Table XI, and Conclusion. |
| **5** | **Corrupted References** | Cormack 2009 title was corrupted to "bridges in information retrieval"; Fateh 2024 was corrupted. | Restored Cormack 2009 title to *"Reciprocal rank fusion outperforms Condorcet and individual rank learning methods"*, ACM SIGIR 2009. Restored Fateh et al. to the Wiley *Engineering Reports* 2024 paper (*"A comprehensive review of optical character recognition (OCR) systems"*). | Verified in `thebibliography`. |
| **6** | **Duplicate & Misattributed References** | Merged duplicate Sentence-BERT (`reimers2019sbert` and `reimers2019sentencebert`) and C-Pack (`xiao2023bge` and `xiao2023cpack`). Cited official GPT-4o mini July 2024 announcement for `openai2024gpt4o`. | Merged duplicates into single keys. Replaced `reimers2019sentencebert` with `reimers2019sbert` and `xiao2023cpack` with `xiao2023bge`. Updated `openai2024gpt4o` to cite the official OpenAI Blog July 18, 2024 announcement. Removed 5 uncited references. | Total references reduced from 53 to exactly 46 clean entries. |
| **7** | **ColQwen2 Indexing Latency** | Reported 18.4 s ($\sim 22.6$ ms/page) for 813 pages, which was the reranker latency, not ColQwen2 forward pass. | Corrected ColQwen2 indexing latency to **184 s (3.1 min)** for 813 pages ($\sim 226$ ms/page on Nvidia Tesla T4 GPU). | Corrected in Section 5.F and Table XI. |
| **8** | **Abstract Trailing Claim** | Abstract ended with a conformal/multi-hop claim contradicting Section 5.C/6. | Deleted trailing sentence from abstract. Abstract word count is now **169 words** (well within the $\le 200$ word IJCDS limit). | Verified via Python word tokenization. |
| **9** | **Tone Down "Ties"** | Avoid claiming $p=1.0000$ proves a "tie" or using "statistically tying". | Replaced with "shows no statistically significant difference from" or "no statistically significant difference detected" across Abstract, Contributions, Tables, and Body. | Verified via global regex scan. |
| **10** | **Table I & Fig. 1 Alignment** | Table I listed "+ Conformal Cascade" and Fig. 1 showed un-evaluated components. | In Table I, changed to `CPU char BM25 (<1 ms verbatim) / Cross-Encoder Cascade` and `Exploratory PRF (Hit@10 0.640, $N=50$, Sec.~VI-B)`. Updated Fig. 1 caption as a conceptual reference architecture with un-evaluated extensions ablated in Section VI. | Table I and Fig. 1 caption updated. |
| **11** | **Line-Aligned CER in Tables** | Add line-aligned CER to Table VI and Table XI; remove script path from body text. | Added `Line-Aligned CER` column to Table VI (Apple Clean: 0.1387, Degraded: 0.3086; Tesseract Clean: 0.2415, Degraded: 0.5842; EasyOCR Clean: 0.3250, Degraded: 0.7327). Removed script path from prose and updated Table XI. | Table VI and XI updated. |
| **12** | **Table Spacing & Formatting** | Fix table font sizing and vertical spacing so floats do not collide or overflow. | Adjusted table formatting, table padding, and floats. Entire document compiles cleanly to exactly 12 pages with no orphan overflow. | Page count verified: 12 pages. |
| **13** | **Tone Down Protocol Claims** | Tone down "pre-specified, audited protocol" in Section 4. | Replaced with "To ensure empirical reproducibility, all evaluations adhere to a standardized, deterministic experimental protocol." | Section 4 updated. |
| **14** | **Ascending Citation Numbering** | Ensure all references are cited in strictly ascending numerical order [1] to [46]. | Eliminated the 5 uncited references (`press2023selfask`, `xiong2021multihop`, `wang2021milvus`, `malkov2020hnsw`, `carmel2010qpp`) and duplicate keys. Re-ordered the 46 `\bibitem` entries to strictly match first appearance in the text. Every citation in the manuscript appears in exact 1, 2, 3, ... 46 order. | Python verification confirmed strictly monotonic order: `Is monotonic from 1 to 46? True`. |

---

## 3. Verified Experimental Data Ledger

### Table VII: Paraphrase Robustness Across Generators and Human Queries
| Query Source | Word BM25 | Char 3--5g BM25 | Dense BGE | ColQwen2 | Source Run |
|---|:---:|:---:|:---:|:---:|---|
| **Primary Generator (`gpt-4o-mini`, $N=306$)** | 0.4334 | 0.2427 | 0.1601 | 0.4962 | Primary frozen test split |
| **Secondary Generator (`Qwen2.5-7B`, $N=306$)** | 0.6292 | 0.5802 | 0.2685 | ---\rlap{$^*$} | `run_qwen_306.py` |
| **Human-Authored Queries ($N=100$)** | 0.6321 | 0.5758 | 0.5020 | ---\rlap{$^*$} | `human_queries_100` |
| **Human-Authored Queries ($N=200$)** | 0.5766 | 0.5549 | 0.4909 | ---\rlap{$^*$} | `human_queries_200` |

*$^*$ColQwen2-v1.0 was profiled exclusively on the primary frozen synthetic test split ($N=306$ probes, MRR = 0.4962 on Tesla T4) and was not re-evaluated across secondary generator or human query sets.*

### Table VI: Standardized JiWER & Line-Aligned OCR Error Rates ($N=626$ SROIE Receipts)
| OCR Engine | Optical Condition | Concatenated CER | Line-Aligned CER | Sequence WER | BoW Sym-Diff Rate |
|---|---|:---:|:---:|:---:|:---:|
| **Apple Vision Neural OCR** | Clean OCR | 0.2848 | **0.1387** | 0.3928 | 0.2476 |
| **Apple Vision Neural OCR** | Degraded Blur OCR | 0.3822 | **0.3086** | 0.5691 | 0.6175 |
| **Tesseract 5.5 LSTM OCR** | Clean OCR | 0.2939 | **0.2415** | 0.4879 | 0.5919 |
| **Tesseract 5.5 LSTM OCR** | Degraded Blur OCR | 0.6377 | **0.5842** | 0.8063 | 1.0043 |
| **EasyOCR 1.7.2 Deep CRNN** | Clean OCR | 0.4026 | **0.3250** | 0.5684 | 0.4813 |
| **EasyOCR 1.7.2 Deep CRNN** | Degraded Blur OCR | 0.7755 | **0.7327** | 0.9827 | 1.0905 |

---

## 4. Submission Artifact Checklist

1. **Compiled Manuscript PDF:**
   - Path: `main.pdf` and `~/Downloads/MemoryVerse_IJCDS_Submission.pdf`
   - Page count: Exactly 12 pages.
   - Anonymized: 100% blind compliant.
2. **EDAS Cover Letter:**
   - Plaintext: `cover_letter.txt` and `~/Downloads/cover_letter.txt`
   - Word DOCX: `ijcds/Cover_Letter_IJCDS.docx` and `~/Downloads/Cover_Letter_IJCDS.docx`
3. **LaTeX Source Package:**
   - Main driver: `main.tex` and `ijcds/ijcds_master_doc.tex`
   - Section body: `ijcds_body.tex` and `ijcds/ijcds_body.tex`
   - Figure directory: `./figures/` containing all 9 referenced image assets.
4. **Audit and Verification Report:**
   - `REPORT.md` and `~/Downloads/REPORT.md`
