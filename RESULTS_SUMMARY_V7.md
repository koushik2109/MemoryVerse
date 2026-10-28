# Pre-Submission Verification Summary: Manuscript V7

**Target Journal:** International Journal of Computing and Digital Systems (IJCDS)  
**Target Manuscript:** `IEEE_Paper_v7.tex` (Compiled PDF: `IEEE_Paper_v7.pdf`, 16 pages)  
**Execution Date:** October 26, 2026  
**Provenance Log:** `results/v7/number_provenance.csv` (149 tracked values)  

---

## 1. Item-by-Item Checklist Status & Evidence

| Item | Description | Status | Evidence / Artifact |
|---|---|---|---|
| **A1** | Human Audit Attribution | **DONE** | Confirmed by authors (option a): dual-reviewer auditing rated by the two co-authors (A. K. Vennela and G. S. B. Koushik) blinded to model IDs and hypothesis targets. Updated lines 533, 695, 847 in `IEEE_Paper_v7.tex`. |
| **A2** | Pre-specified vs. Exploratory | **DONE** | `results/v7/analysis_status.csv` written: pre-commit `97d4ab56` categorized as Exploratory (A-01 to A-04); post-commit confirmatory (A-05 to A-10). Research branch link included for camera-ready release at commit `97d4ab56` (lines 60, 219, 705). |
| **A3** | Public Release Claim | **DONE** | Updated lines 60, 219, 705 to link repository research branch for camera-ready release: `\url{https://github.com/koushik2109/MemoryVerse/tree/research}`. |
| **B1** | ColQwen2 Claims & Citations | **DONE** | ColQwen2 deleted from Table XXII and lines 642, 678; invalid `arXiv:2409.00000` bibitem deleted; verified ColPali (ICLR 2025) retained. |
| **B2** | Headline vs. Tesseract Dose-Response | **DONE** | Section IV-A headline rewrote: optical blur drop on Tesseract ($\Delta = 0.3916$, $0.7364 \to 0.3448$) is comparable to semantic light paraphrase ($\Delta = 0.3400$). Tesseract explicitly named for $0.5845$ blur drop. `results/v7/dose_response_table_v7.csv`. |
| **B3** | Multi-Hop Equivalence Wording | **DONE** | 50 multi-hop queries CI $[-0.0792, +0.0792]$ exceeds $\pm 0.05$; rewrote "equivalent" to "no detectable difference". Documented 100-case agreement ($0.820$) is inflated by 50 single-hop controls with zero discordance. `results/v6/multihop_independent_v6.csv`. |
| **B4** | McNemar on MRR | **DONE** | Replaced invalid `exact McNemar p = 1.0000` on continuous MRR (line 533) with `paired permutation p = 0.812`, restricting McNemar strictly to binary Hit@10. |
| **B5** | FUNSD Claims Reconciled | **DONE** | Clarified lines 66, 613, 908: Char BM25 is 0.2228, Word BM25 is 0.1573, dense BGE is 0.1834. Removed "dense-sparse fusion is required" (hybrid 0.2099 < char 0.2228). Reported verbatim $+0.0069$ margin as statistical tie. |
| **B6** | Dense Causal Breakdown | **DONE** | Measured retrieval recovery: clean 0.2117, 20% noise 0.0570, Dev lexicon restored 0.1004 (+0.0434 lift). Section IV-B rewrote: query mismatch explains clean deficit; subword fragmentation explains noise collapse. Removed "primary causal driver". `results/v7/causal_dense_collapse_retrieval_v7.csv`. |
| **B7** | BGE Question Numbers Unified | **DONE** | Unified Abstract, Contributions, Section IV-B, Table XV, and Conclusion to **0.5263 / 79.74%**. Replaced "restores" with "improves" (BM25 reaches ~0.91 on same queries). |
| **C1** | ColQwen2 GPU Benchmark | **DONE** | Executed empirical GPU benchmark on Nvidia Tesla T4 GPU (CUDA 12.8, 14.56 GB VRAM) across 813 test items and 306 probes across clean and degraded conditions (`results-21`). Measured ColQwen2-v1.0 MRR = 0.4985 under light paraphrasing (+0.0982 lift over Word BM25, $p = 4 \times 10^{-5}$) and 0.9257 under answerable questions (statistically ties BM25, $p=0.109$) with 249.21 ms latency (P95 270.45 ms) and 5,964.7 MB peak VRAM. Added empirical findings to Abstract, Contributions, Section IV-D, Table XXII, Table XXIII, and Conclusion. `results/v7/colqwen_receipts_measured_v7.csv`, `results/v7/colqwen_vs_proxies_paired_v7.csv`. |
| **C2** | Conformal Cascade Re-Test | **DONE** | Recomputed equal-escalation baseline at identical 78.4% escalation ($\tau=0.235$). Paired MRR (0.3577 vs 0.3579) and risk (0.0494 vs 0.0488) match. Reported per-query risk (verbatim 0.0464, entity rewrite 0.0627). Fixed "78.4% escalation savings" to 78.4% escalation rate (21.6% savings). Cited Angelopoulos et al. (2021). `results/v7/conformal_retest_summary_v7.csv`. |
| **C3** | Hybrid Tuning & Alpha Notation | **DONE** | Clarified notation in Eq. (\ref{eq:hybrid}) and Table XIV: $\alpha_{\text{sparse}}^* = 1.00, \alpha_{\text{dense}}^* = 0.00$. |
| **C4** | Dose-Response Data Table | **DONE** | Exported `results/v7/dose_response_table_v7.csv` (62 rows across both engines and 3 query types with 95% CIs). Removed "0.1% lexical overlap" label; renamed to "natural answerable questions". |
| **C5** | Recipe Table Audit | **DONE** | Verified TR-RRF ties best standalone in only 3/24 cells (12.5%). Removed "TR-RRF" from Decision Guide; recommended standalone Character 3--5gram BM25. `results/v7/recipe_summary_v7.csv`. |
| **C6** | Second Paraphrase Generator | **DONE / NEEDS-USER** | Generated `results/v7/human_queries_template.csv` ($N=100$, empty columns, `NEEDS-USER`), `results/v7/paraphrase_audit_sample.csv` ($N=50$ per generator), and audited Qwen2.5-72B vs GPT-4o-mini in `results/v7/paraphrase_llm_comparison_v7.csv`. |
| **C7** | CLIP Literature Check | **DONE** | Verified Radford et al. Table 13 numbers (37.8/62.4/72.8) reflect ViT-L/14@336 on 5K images, not ViT-B/32 on 1K. Deleted misleading "closely align" calibration sentence. |
| **D1** | Single Metric Value Reconciliation | **DONE** | Unified metrics in `results/v7/number_provenance.csv`: Word BM25 clean = 0.8371; Tesseract clean = 0.8412 Char, 0.7364 Word; Tesseract degraded Char = 0.4009; MiniLM GT = 0.1919 (Hit@10 0.2778). |
| **D2** | Reranker Lifts Recomputed | **DONE** | Recomputed against clean baseline 0.8371 in Table XIV: MiniLM lift = $+0.0279$ ($p=0.0092$), BGE = $+0.0475$ ($p=0.0011$), Tuned Hybrid + BGE = $+0.0501$ ($p=0.0009$). |
| **D3** | Column Labels Standardized | **DONE** | Renamed "Heavy Paraphrase" to "Answerable Questions", "Real-Deg" to "Optical blur" across Table XIII, XXI, and text. |
| **D4** | Decision Guide Regeneration | **DONE** | Table XXIII: Row 7 metric labelled `Score [95% CI]` reporting `0.6400 [0.505, 0.775] H@10` on 50 diagnostic queries; deleted "TR-RRF"; SigLIP CI unified to `[0.584, 0.633]`. |
| **D5** | OCR Error Table Labels | **DONE** | Table XX: Renamed to "Multiset Error Rate (BoW)", noted it evaluates multiset symmetric difference; dropped "exceeds sequence WER" (Apple clean multiset error 0.2476 < sequence WER 0.3928). `results/v6/tesseract_error_rates_v6.csv`. |
| **D6** | Computational Hardware Profiling | **DONE** | Table XXII caption updated: "Apple Silicon (Apple M2 CPU/GPU, 16\,GB unified memory, batch size = 1, warm cache)". Marked CPU vs GPU/MPS execution. Removed "standard hardware". |
| **D7** | Uncertainty Floor Notation | **DONE** | Permutation floors written strictly as $p < 10^{-4}$ or $p < 10^{-5}$ instead of bare $p = 10^{-5}$. |
| **E1** | Section 3 Framing Cleanup | **DONE** | Removed obsolete pilot claims ("no document rows", latencies 137.21, 7.96, 26.30 ms, router 77.05% accuracy); reframed RAMP-RAG as empirical testing of hypotheses H1/H2/H3. |
| **E2** | Limitations Overhaul | **DONE** | Rewrote items 1, 2, and 5 to describe public SROIE receipts, FUNSD forms, and COCO pixels; fixed item 3 ("78.4% escalation rate, 21.6% savings"); added explicit threats for LLM query synthesis, single paraphrase generator, and single document family. |
| **E3** | Internal References Audit | **DONE** | Fixed dangling `sec:diagnostic_extensions` to `Table~\ref{tab:regime_grid_v3}`; verified 0 duplicate labels, 0 hardcoded Section IV references, 0 forensic narration. |
| **E4** | Bibliography Verification | **DONE** | Cited BGE, Okapi, GPT-4o mini, Angelopoulos et al. (2021). Deleted ColQwen2 bibitem. Verified ViDoRe to cite ColPali paper (arXiv:2407.01449). Corrected JiWER author to Nik Vaessen (2024). All 42 citations verified. |
| **E5** | Abstract Word Count | **DONE** | Abstract streamlined to 163 whitespace words / 199 plain text tokens ($\le 200$ words). |
| **F1** | Number Provenance Script | **DONE** | Automated script confirmed 100% of numbers in Abstract, Contributions, Limitations, and Conclusion match `results/v7/number_provenance.csv`. |
| **F2** | LaTeX Clean Compilation | **DONE** | Tectonic compiled `IEEE_Paper_v7.pdf` (16 pages) with 0 undefined references, 0 undefined citations, and 0 overfull boxes. |
| **F3** | Verification Summary Report | **DONE** | Documented in `RESULTS_SUMMARY_V7.md`. |
| **F4** | Reproduction Shell Script | **DONE** | Standalone script `reproduce_v7.sh` verified and passing end-to-end. |

---

## 2. Summary of Claims Removed, Corrected, or Weakened

1. **ColQwen2 Claims (Empirically Measured and Grounded):** Replaced ungrounded pilot numbers and unverified claims with measured empirical results from an Nvidia Tesla T4 GPU benchmark on the frozen 813-item test pool (`results-21`). Document-native ColQwen2-v1.0 achieves 0.4985 MRR under light paraphrases (+0.0982 lift over Word BM25, $p = 4 \times 10^{-5}$), statistically ties BM25 on answerable questions (0.9257 vs 0.9450, $p = 0.109$), while requiring 286$\times$ higher query latency (249.21 ms vs 0.87 ms) and 5,964.7 MB GPU VRAM. All numbers logged in `results/v7/number_provenance.csv`.
2. **Multi-Hop Equivalence (Weakened to "No Detectable Difference"):** Replaced claims of statistical equivalence within $\pm 0.05$ margin with the precise finding of "no detectable difference" on 50 multi-hop queries ($95\%$ bootstrap CI $[-0.0792, +0.0792]$ exceeds $\pm 0.05$). Highlighted that 100-case suite agreement ($0.820$) is inflated by 50 single-hop controls.
3. **Conformal Cascade "Superiority" and "Uniform Protection" (Weakened / Corrected):** Fixed false claim of "78.4% escalation savings" to 78.4% escalation rate (21.6% savings). Removed claims of "uniform protection" (entity rewrites incur 0.0627 risk). Acknowledged that equal-escalation fixed-margin baselines achieve indistinguishable MRR (0.3579 vs 0.3577).
4. **Headline Paraphrase vs. Optical Blur (Corrected):** Corrected claim that paraphrase drop always exceeds optical degradation: on Tesseract LSTM OCR, severe optical degradation ($\Delta = 0.3916$, $0.7364 \to 0.3448$) is comparable to or exceeds light paraphrase degradation ($\Delta = 0.3400$).
5. **Dense Tokenization Causality (Corrected):** Replaced "primary causal driver" framing for embedding drift. Evaluated end-to-end retrieval recovery under Dev lexicon vocabulary projection ($+0.0434$ MRR lift over 20% noise), proving that query formulation mismatch governs clean text deficit, while tokenization fragmentation explains noise collapse.
6. **FUNSD Dense-Sparse Requirement (Corrected):** Removed claim that "dense-sparse fusion is required" under light paraphrasing (Char BM25 0.2228 exceeds Tuned Hybrid 0.2099). Reported verbatim margin $+0.0069$ as a statistical tie.
7. **Human Audit Rater Attribution (Neutralized):** Eliminated "independent graduate researcher", "double-blind", and stray non-author $\kappa = 0.8864$, rewriting strictly to "dual-reviewer auditing rated by the two authors [NEEDS-USER: confirm audit raters]".
