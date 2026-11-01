# Reproducibility Audit Changelog: "When Do Text Proxies Suffice?"

This log records every numeric, methodological, and bibliographic correction applied to the manuscript (`ijcds_body.tex` and `ijcds/ijcds_body.tex`).

## Audit Format
- **ID / Phase**: Finding identifier from the audit plan.
- **Location**: Section, Table, Figure, and line number in LaTeX source.
- **Old Value / Text**: Value or wording prior to correction.
- **New Value / Text**: Value or wording after correction.
- **Source File & Line**: Underlying source-of-truth results file and row/column.

---

### Phase 1: Numeric Contradictions

#### Item 1: Answerable-Question Numbers in Table II vs. Section V-A.3
- **ID**: Phase 1.1
- **Location**: Table II (`tab:regime_grid_v3`), `ijcds_body.tex:188-193` & Section V-A.3
- **Old Value**: Sec V-A.3 quoted degraded values (Char 0.8116, Word 0.7328, Hybrid 0.7705, MiniLM 0.2837, BGE 0.5128) while Table II reported clean GT values (0.9471, 0.9014, 0.9125, 0.3669, 0.5263) without explicit labeling.
- **New Value**: Table II explicitly separates and reports both "Answerable (Clean GT)" and "Answerable (Optical Blur)" rows:
  - Clean GT: Char BM25 = 0.9471, Word BM25 = 0.9014, Hybrid = 0.9125, MiniLM = 0.3669, BGE-Base = 0.5263.
  - Optical Blur: Char BM25 = 0.8116, Word BM25 = 0.7328, Hybrid = 0.7705, MiniLM = 0.2837, BGE-Base = 0.5128.
- **Source File & Line**: `results/v3/engine_replication.csv` (rows 10-14, 25-29) and `results/v6/dense_formulation_keyword_control_v6.csv`.

#### Item 2: Hybrid Synthetic 20% Noise MRR
- **ID**: Phase 1.2
- **Location**: Table II (`tab:regime_grid_v3`), row "Tuned Hybrid", column "OCR Syn 20%", `ijcds_body.tex:179`
- **Old Value**: 0.2687
- **New Value**: 0.1445
- **Source File & Line**: `results/v7/dose_response_table_v7.csv:row 14` (matches Section V-A.2 text).

#### Item 3: COCO Benchmark Recall & MRR Discrepancies
- **ID**: Phase 1.3
- **Location**: Table VII (`tab:coco_benchmarks_v3`), `ijcds_body.tex:447-456`
- **Old Value**: SigLIP Hit@1 = 0.4870, Hit@10 = 0.8250; CLIP Hit@1 = 0.3420, Hit@10 = 0.7610.
- **New Value**: SigLIP Hit@1 = 0.4800, Hit@5 = 0.7680, Hit@10 = 0.8670 [0.845, 0.888], MRR = 0.6071 [0.584, 0.633]; CLIP Hit@1 = 0.3390, Hit@5 = 0.6180, Hit@10 = 0.7570 [0.729, 0.785], MRR = 0.4709 [0.446, 0.497].
- **Source File & Line**: `results/v3/coco_native_visual_v3.csv` and `results/v5/coco_captions_v5.json`.

#### Item 4: ColQwen2 vs. Lexical Paired Difference on Answerable Questions
- **ID**: Phase 1.4
- **Location**: Abstract, Section I (`ijcds_body.tex:22`), Section V-E (`ijcds_body.tex:551`), Table II, Table XIII
- **Old Value**: Quoted only $-0.0855$ without distinguishing Ground Truth text ($0.9471 - 0.8595 = +0.0876$) from Real-OCR text ($0.9328 - 0.8595 = +0.0733$).
- **New Value**: Both differences explicitly reported: Real-OCR paired diff $+0.0733$ [95% CI: $+0.031, +0.115$], Clean GT diff $+0.0876$ [95% CI: $+0.041, +0.128$], $p = 0.00012$.
- **Source File & Line**: `results/v10/colqwen_paired_v10.csv:row 6` and `results/v6/colqwen_vs_proxies_paired.csv:row 4`.

#### Item 5: Reranker Metrics Disambiguation ($K_1=10$ vs. $K_1=20$)
- **ID**: Phase 1.5
- **Location**: Section V-C, Table IV footnote, Table V (`tab:practical_stack`)
- **Old Value**: Values 0.8426 (MiniLM) and 0.8555 (BGE) in Sec V-C appeared in no table; Table V listed 0.8650 and 0.8846.
- **New Value**: Labeled exploratory $K_1=10$ candidate reranking (MiniLM MRR 0.8426, BGE MRR 0.8555) in Table IV notes; preserved production $K_1=20$ candidate reranking (MiniLM MRR 0.8650, BGE MRR 0.8846) in Table V.
- **Source File & Line**: `results/v6/best_pipeline_eval_v6.csv:rows 2-8`.

#### Item 6: Conformal Pareto Figure Regeneration
- **ID**: Phase 1.6
- **Location**: Figure 5 (`figures/fig_conformal_pareto.png`) and Caption, `ijcds_body.tex:428-433`
- **Old Value**: Figure legend had $\tau^*=0.384$, esc 89.5%, coverage 36.5%, $N=306$ while Table VI had $\tau^*=0.2369$, esc 78.4%, coverage 58.5%, $N=2,754$.
- **New Value**: Regenerated `figures/fig_conformal_pareto.png` from `results/v6/conformal_cascade_v6.csv`; updated caption to $\tau^*=0.2369$, 78.4% escalation, 58.5% coverage (82.0% for RCPS), $N=2,754$ pooled instances.
- **Source File & Line**: `results/v6/conformal_cascade_v6.csv`.

#### Item 7: MiniLM Clean Verbatim MRR Pool Labeling
- **ID**: Phase 1.7
- **Location**: Table II, Table III, Table IV, Table X
- **Old Value**: MiniLM MRR appeared as 0.2100 (Table II), 0.1919 (Table X), 0.2117 (Sec V-B), 0.2144 (Table IV).
- **New Value**: Labeled candidate pool condition for each:
  - 0.1919: Pure receipt corpus ($N=626$) without distractors.
  - 0.2100: Standard 813-item test pool (626 receipts + 187 COCO distractors).
  - 0.2117: Development set calibration baseline.
  - 0.2144: Standard pipeline ablation baseline.
- **Source File & Line**: `results/v7/dose_response_table_v7.csv` and `results/v3/engine_replication.csv`.

#### Item 8: BGE Ground-Truth Verbatim Pool Disambiguation
- **ID**: Phase 1.8
- **Location**: Table II/III vs. Table X(a)
- **Old Value**: 0.3256 (Table II/III) vs. 0.3428 (Table X).
- **New Value**: Labeled 0.3256 as 813-distractor evaluation pool, and 0.3428 as pure receipt corpus ($N=626$).
- **Source File & Line**: `results/v6/dense_formulation_keyword_control_v6.csv` and `results/v3/engine_replication.csv`.

#### Item 9: MiniLM Natural Answerable Questions Disambiguation
- **ID**: Phase 1.9
- **Location**: Table II vs. Table III
- **Old Value**: 0.3541 (Table II) vs. 0.3669 (Table III).
- **New Value**: Labeled 0.3669 as Pristine Ground Truth text and 0.3541 as Real-OCR text.
- **Source File & Line**: `results/v3/engine_replication.csv:row 12` and `results/v6/dense_formulation_keyword_control_v6.csv`.

#### Item 10: Vocabulary Projection Cosine Drift Disambiguation
- **ID**: Phase 1.10
- **Location**: Section V-B vs. Section V-B.2
- **Old Value**: Cosine similarity reported as 0.6826 in Sec V-B and 0.4655 in Sec V-B.2; MRR reported as 0.0570 vs. 0.0596.
- **New Value**: Disambiguated document embedding drift (0.4655) from projected token representation recovery (0.6826); disambiguated sub-word projected slice MRR (0.0570) from aggregate Table II MRR (0.0596).
- **Source File & Line**: `results/v7/causal_dense_collapse_retrieval_v7.csv`.

#### Item 11: Char BM25 Query-Type Latency Disambiguation
- **ID**: Phase 1.11
- **Location**: Section I (`ijcds_body.tex:22`), Section V-E, Table XII (`tab:system_costs`), Table XIII (`tab:decision_guide_v6`)
- **Old Value**: Latencies of 0.87 ms, 0.96 ms, and 5.72 ms cited without query type context.
- **New Value**: Clarified that 0.87 ms is short verbatim lookup latency, 5.72 ms is natural answerable query latency (longer query strings), and 0.96 ms is Table V candidate generation latency ($K_1=20$).
- **Source File & Line**: `results/v3/headline_regime_grid_v3.csv:row 16` and `results/v6/char_vs_word_bm25_v6.csv`.

#### Item 12: Table XIII Light-Paraphrase Tied Set Correction
- **ID**: Phase 1.12
- **Location**: Table XIII (`tab:decision_guide_v6`), row "Document Page / Clean Neural OCR / Light Paraphrase", `ijcds_body.tex:593`
- **Old Value**: Indistinguishable set listed Hybrid (0.4335) despite ColQwen2 being statistically superior ($p_{\text{Holm}} = 0.0095$).
- **New Value**: Updated indistinguishable set to `None (ColQwen2 is decisive; $p_{\text{Holm}} = 0.0095$)`, noting Hybrid (0.4335) in the recommendation column as zero-GPU fallback.
- **Source File & Line**: `results/v10/colqwen_paired_v10.csv:row 2`.

#### Item 13: Table XIV Paired Confidence Interval Recomputation
- **ID**: Phase 1.13
- **Location**: Table XIV (`tab:multihop_paired_v3`), row "Single Dense", `ijcds_body.tex:648`
- **Old Value**: `+0.000 [+0.00, 0.07]` with 2/2 discordant pairs.
- **New Value**: `0.000 [-0.08, 0.08]` (exact symmetric two-sided cluster bootstrap CI with 2 discordant pairs).
- **Source File & Line**: `results/v3/multihop_paired_tests_v3.csv`.

#### Item 14: Restoring 5th Property Bucket in Section V-B.3
- **ID**: Phase 1.14
- **Location**: Section V-B.3 and Figure 4 Caption, `ijcds_body.tex:391`
- **Old Value**: Text claimed "five property buckets" but listed only four.
- **New Value**: Restored fifth bucket: "Long queries ($N=3$, MRR = 0.744)".
- **Source File & Line**: `results/v6/dense_formulation_keyword_control_v6.csv`.

#### Item 15 & Bug A: EasyOCR Replication Integration into Table X
- **ID**: Phase 1.15 & Phase 2.A
- **Location**: Table X (`tab:engine_replication_v3`), `ijcds_body.tex:500-530`
- **Old Value**: EasyOCR numbers (0.8519, 0.4272, 0.7649, 0.3472, 0.3792, 0.2292) had no table; Light Paraphrase Clean and Apple OCR were duplicated as 0.4334/0.2427.
- **New Value**: Integrated EasyOCR 1.7.2 Clean and Blur columns across all retrieval architectures into Table X(a); corrected Light Paraphrase Ground Truth column to 0.4796 (Word BM25), 0.2526 (Char BM25), 0.1190 (Dense MiniLM), 0.1830 (Dense BGE), 0.4828 (Tuned Hybrid).
- **Source File & Line**: `results/v3/engine_replication.csv:rows 15-22, 38-45` and `results/v9/easyocr_replication_v9.csv`.

#### Item 16: Fuzzy Expansion Engine Attribution
- **ID**: Phase 1.16
- **Location**: Section V-C, `ijcds_body.tex:340`
- **Old Value**: Quoted Apple Vision figure ($0.8371 \to 0.8005$) inside an EasyOCR paragraph.
- **New Value**: Clarified that the $0.8371 \to 0.8005$ drop was measured on Apple Vision clean OCR, verifying that fuzzy string expansion degrades sub-word precision.
- **Source File & Line**: `results/v7/dose_response_table_v7.csv`.

---

### Phase 2: Pipeline Bug Fixes & Disclosures

#### Bug B: ColQwen2 and CLIP Optical Blur Evaluation
- **ID**: Phase 2.B
- **Location**: `kaggle/kaggle_run_v11_colqwen_degraded.py` and `results/v11_degraded_visual/`
- **Resolution**: Created reproducible Kaggle GPU script `kaggle/kaggle_run_v11_colqwen_degraded.py` to evaluate ColQwen2-v1.0 and CLIP on degraded images in `ai_engine/rag/data/sroie/img_degraded/`, saving to versioned folder `results/v11_degraded_visual/` (~35 min on Kaggle GPU T4).
- **Source File**: `results/v6/colqwen_vs_proxies_paired.csv:row 5` (ColQwen2 Degraded Blur MRR = 0.6275).

#### Bug C: Table II Clean vs. Ground Truth Labeling
- **ID**: Phase 2.C
- **Location**: Table II header and Section I
- **Resolution**: Disclosed that the "Clean" column in Table II represents Clean Ground Truth text; reported both Clean GT (+0.0876) and Real-OCR (+0.0733) paired differences explicitly.
- **Source File**: `results/v10/colqwen_paired_v10.csv`.

#### Bug D: COCO-1K Protocol Reproduction
- **ID**: Phase 2.D
- **Location**: Section IV, `ijcds_body.tex:162`
- **Resolution**: Documented 1,000 Karpathy test split images with 5 captions per image (candidate pool size 1,000 images, text-to-image retrieval).
- **Source File**: `results/v3/coco_native_visual_v3.csv`.

---

### Phase 3: Evidence & Framing Alignment

#### Item 3.1: Machine-Caption Proxy Ceiling
- **ID**: Phase 3.1
- **Location**: Abstract, Section V-D, Section VI, Section VIII, Table XIII
- **Resolution**: Replaced claim that BLIP (0.3728) is the "best tested proxy" with comprehensive disclosure of Florence-2 (0.4420) and BLIP-2 (0.4185); re-tested SigLIP superiority against Florence-2 ($p < 10^{-4}$).
- **Source File**: `kaggle/kaggle_run_v5.py:676-677`.

#### Item 3.2: Paraphrase Reversal Generator Dependence
- **ID**: Phase 3.2
- **Location**: Abstract, Section V-E, Limitations
- **Resolution**: Disclosed that the Character-to-Word BM25 reversal depends on entity-substitution paraphrasing (gpt-4o-mini); under Qwen2.5-7B and human queries, Character BM25 tracks Word BM25 (0.5802 vs. 0.6292; 0.5549 vs. 0.5766). Noted ColQwen2 was not run on secondary sets.
- **Source File**: `REPORT.md:Table VII`.

#### Item 3.3: Associational Phrasing Calibration
- **ID**: Phase 3.3
- **Location**: Abstract, Section I, Section V-B, Conclusion
- **Resolution**: Replaced causal terms ("governed by", "explains") with associational language ("associated with", "correlates with") except for the controlled vocabulary-projection experiment.

#### Item 3.4: Conformal Risk Control Framing
- **ID**: Phase 3.4
- **Location**: Abstract, Section V-C, Section VI, Conclusion
- **Resolution**: Re-framed conformal calibration as a negative finding: marginal CRC bounds expected risk across splits (58.5% empirical split coverage) and does not guarantee conditional coverage under entity shifts; noted RCPS achieved 82.0% coverage; cited Angelopoulos et al. (ICLR 2024).
- **Source File**: `results/v6/conformal_cascade_v6.csv`.

#### Item 3.5: Table IV Redundant Row Pruning
- **ID**: Phase 3.5
- **Location**: Table IV (`tab:ablation_v3`), `ijcds_body.tex:285-305`
- **Resolution**: Removed uninformative duplicate rows (`+ Clean Stream`, `No Density Gate`, `Static Alpha`).

#### Item 3.6: Multi-Hop Latency Explanation
- **ID**: Phase 3.6
- **Location**: Section VI-B, `ijcds_body.tex:633`
- **Resolution**: Clarified that the 4,736 ms vs. 3,260 ms latency is driven by two-hop remote LLM API roundtrips and autoregressive decoding, distinct from single-stage local retrieval lookup.

#### Item 3.7: Generation Quality Reporting
- **ID**: Phase 3.7
- **Location**: Section VI-B, `ijcds_body.tex:633`
- **Resolution**: Reported dual-author audited generation quality metrics: faithfulness observed agreement 95.5% ($\kappa = 0.8716$), relevance 96.4% ($\kappa = 0.8926$), correctness 98.2% ($\kappa = 0.9451$; mean $\kappa = 0.9031$).

#### Item 3.8: Serving Profile Caveat
- **ID**: Phase 3.8
- **Location**: Section I, Section IV, Section V-E, Table XII
- **Resolution**: Added explicit hardware serving profile caveat distinguishing Apple M2 CPU lookup (0.87 ms) from Nvidia Tesla T4 GPU serving (249.21 ms).

#### Item 3.9: DocVQA Training Overlap Disclosure
- **ID**: Phase 3.9
- **Location**: Abstract, Section I, Section V-E, Table XIII
- **Resolution**: Disclosed pretraining overlap of DocVQA in the ViDoRe training mixture for ColPali/ColQwen2; toned down "indispensable" to "substantially outperforms generic text proxies".

#### Item 3.10: Hybrid Alpha Wording Alignment
- **ID**: Phase 3.10
- **Location**: Section II
- **Resolution**: Reworded hybrid architecture text to reflect empirical dev-set finding $\alpha_{\text{sparse}}^* = 1.00$.

---

### Phase 4: Statistics & Methods Reporting
- **Permutation Floor**: Adjusted reporting floor to $p < 10^{-4}$ ($p < 10^{-5}$ floor justified at $N_{\text{perm}}=100,000$).
- **BM25 Parameters**: Documented $k_1=0.9, b=0.5$, character 3–5grams, lowercase, punctuation stripping, and unified dev-set tuning grid.
- **BGE Length 256**: Documented empirical validation that 94.2% of receipt documents fit within 256 tokens and chunked MaxP yields identical MRR (0.2030 vs. 0.2056).
- **Holm Families**: Documented pre-specified evaluation families 1–4 in Section IV.

---

### Phase 5: References Verification
- **[32] `wang2011cascade`**: Corrected venue to *Proc. 34th Int. ACM SIGIR Conf. on Research and Development in Information Retrieval (SIGIR)*, pp. 105–114, 2011.
- **[33] `gallagher2019cascade`**: Corrected venue to *Proc. 12th ACM Int. Conf. on Web Search and Data Mining (WSDM)*, pp. 15–23, 2019.
- **[34] `bates2021conformal`**: Formulated RCPS; cited Angelopoulos et al., *Conformal Risk Control* (ICLR 2024).
- **[46] `zheng2023judging`**: Cited in Section VI-B where LLM auditing is discussed.
- **Cross-Encoders**: Added Nogueira & Cho (2019) citation (`nogueira2019passage`).
- **[22] `cho2024m3docrag`**: Corrected venue to `arXiv preprint arXiv:2411.04952, 2024`.

---

### Phase 6: Presentation Cleanup
- **RAMP-RAG Definition**: Defined as "Regime-Aware Multimodal Proxy architecture (RAMP-RAG)" in Section VI and Section V-F.
- **Figure 1 Caption**: Reconciled caption to state that modular components are empirically evaluated and ablated in Section VI.
- **Table I Reference**: Fixed reference to `Sec.~\ref{sec:multihop_reasoning}`.
- **Citation Ordering**: Fixed Table VIII, IX, X citations to appear in strict monotonic sequence.
- **Float Queue Overflow**: Added float fraction controls (`\dbltopfraction=0.95`, `\topfraction=0.95`) in `ijcds_header.tex`, eliminating float queuing.
- **Table XV Spacing**: Adjusted vertical float spacing to `\vspace{1mm}`, eliminating paragraph collision.
