# CHANGELOG_V3.md: Systematic Audit & Reconciliation Log

This log records every mechanical and fully traced edit applied to the LaTeX manuscript in `paper_v3/` in adherence to Ground Rules 1, 2, 3, and 7.

---

### Phase 4 Manuscript Edits (`paper_v3/ijcds_body.tex`)

1. **Table I (Systematic SOTA Comparison, line 52):**
   - *Old:* `M3DocRAG \cite{cho2024m3docrag} & EMNLP 2024`
   - *New:* `M3DocRAG \cite{cho2024m3docrag} & arXiv 2024`
   - *Source:* Verified arXiv preprint publication (`arXiv:2411.04952`) on DBLP (item `B-13`).

2. **Table II (Regime Analysis Grid, lines 164–210):**
   - *Old:* Visual models Syn-20% column showed unmeasured synthetic numbers with undefined asterisks (`0.7812*`, `0.0482*`).
   - *New:* Replaced with `---^\dagger` and added footnote: `$^\dagger$Visual models operate directly on unperturbed raw document images; synthetic character perturbation noise is not applicable.` (item `B-14`, `A-22`).
   - *Old:* Light Paraphrase Syn-20% Hybrid cell was `0.1445 [0.111, 0.183]`.
   - *New:* Updated to true retuned hybrid `0.2687 [0.229, 0.311]` from `results/v6/` (item `A-12`).
   - *Old:* Answerable Syn-20% and Blur Hybrid cells were swapped (`0.7705` and `0.5346`).
   - *New:* Corrected to `0.4672 [0.432, 0.502]` (Syn-20%) and `0.7865 [0.746, 0.824]` (Blur) from `results/v6/heavy_hybrid_reconciled_v6.csv` (item `A-12`).
   - *Old:* ColQwen2 Answerable row was `0.0870` across columns.
   - *New:* Corrected to measured `0.8595 [0.825, 0.892]` (Clean) and `0.7410 [0.697, 0.781]` (Blur) from `results/v6/native_visual_receipts_results.csv` (item `A-01`, `R-01`).

3. **Section 5.A.3 (Condition Labels and Narrative Text, line 223):**
   - *Old:* Citing `0.9471` and `0.9014` as "clean OCR", and citing `0.7705` as "clean OCR".
   - *New:* Correctly labeled `0.9471` and `0.9014` as Clean GT (Apple Clean OCR is `0.9328` and `0.8841`); cited real Apple Clean Hybrid as `0.8984` and Syn-20% Hybrid as `0.4672` (item `A-09`).

4. **Table III (Dense Sensitivity to Query Formulation, line 255):**
   - *Old:* Row 1 mixed 626 pure receipt pool for MiniLM (`0.1919`, Hit@10 `0.2778`) with 813 pool for BGE (`0.3256`, Hit@10 `0.3824`).
   - *New:* Unified row 1 to the 813-distractor pool matching rows 2–5: MiniLM is `0.2100` [0.1709, 0.2524] (Hit@10 `0.2843`) and BGE is `0.3256` [0.2764, 0.3757] (Hit@10 `0.3824`) (item `A-15`, `R-04`).

5. **Table IV (Architectural Ablation, lines 277–295):**
   - *Old:* Caption and prose did not explicitly distinguish $K_1=10$ candidate truncation from full-depth Table II BM25 (`0.8371`).
   - *New:* Added explicit notation in caption and Section 5.C prose clarifying that Table IV operates at candidate depth $K_1=10$, while production $K_1=20$ cascade over standalone BM25 is reported in Table V (`0.8650` and `0.8846`) (item `A-14`).

6. **Table V (Production Reranker Cascade, lines 299–320):**
   - *Old:* Standalone Char 3–5g BM25 printed `0.9437` (rounding drift).
   - *New:* Standardized to `0.9436` throughout (item `B-15`).

7. **Section 5.C (Conformal Cascade Risk Evaluation, lines 322–326):**
   - *Old:* Verbose partition-level narrative detailing expected risk vs per-split risk over two long paragraphs.
   - *New:* Surgically condensed into a single rigorous paragraph explicitly framing conformal risk control as a notable negative result (empirical split coverage 58.5% vs 0.05 target, matched by equal-escalation fixed margin baseline `0.3579` vs `0.3577`, $p>0.05$; conditional risk on entity rewrites exceeds target at `0.0627`) (item `B-16`).

8. **Section 5.D (SigLIP vs Human Gold TOST, line 397):**
   - *Old:* Mixed single-gold p-values ($p=0.142$, $p_{\text{TOST}}=0.0502$) with 5-reference gold mean (`0.6210`, diff `$-0.0139$`), printing an inconsistent 90% CI `[-0.0384, -0.0088]`.
   - *New:* Recomputed exact statistics: 5-reference human gold multi-captions (`0.6210`, diff `$-0.0139$`, paired $p=0.321$, 90% CI `[-0.0369, +0.0091]`, $p_{\text{TOST}}=0.0050$ at $\Delta=\pm 0.05$); single-gold comparison (`0.5835`, diff `$+0.0236$`, $p=0.142$, 90% CI `[-0.0029, +0.0501]`, $p_{\text{TOST}}=0.0502$ at $\Delta=\pm 0.05$, equivalent at $\Delta \ge \pm 0.06$ with $p_{\text{TOST}}=0.0118$) (item `A-05`, `R-04`).

9. **Table IX (DocVQA Benchmark, lines 433–455):**
   - *Old:* ColQwen2 nDCG@10 labeled as `0.6120`, lower than MRR `0.6450`.
   - *New:* Reconciled metric: official ViDoRe DocVQA leaderboard reported score `0.6120` is strictly nDCG@5; labeled true official ViDoRe nDCG@10 as `0.6333` with explanatory footnote (item `A-13`).
   - *Old:* Text in line 455 cited non-existent `Table X(c)`.
   - *New:* Corrected citation to `Table X(a)` (item `A-16`).

10. **Table X(b) (Cross-Engine Replication on FUNSD, lines 485–504):**
    - *Old:* EasyOCR Blur near-zero MRR (`0.02--0.05`) lacked physical explanation; BGE paraphrase performance on forms contradicted blanket "dense collapse" claims.
    - *New:* Added footnote and text explaining that EasyOCR blur collapse represents an optical CRAFT text detection failure threshold (67.8% blank transcriptions) rather than an IR ranking deficit; qualified that Dense BGE matches or outperforms Word BM25 on administrative forms (`0.2345` vs `0.1775` EasyOCR, `0.1834` vs `0.1573` Apple) under natural questions due to repeated form header ambiguity (item `A-16`, `A-19`, `A-20`, `B-08`).

11. **Section 5.E & Key Contributions (lines 22–27, 529):**
    - *Old:* Contribution 1 and Conclusion asserted formal TOST equivalence on entity-abstracted controls (`0.0546` vs `0.0495`, $+0.0051$). Contribution 2 claimed syntactic form "has no causal effect" based on permutation tests.
    - *New:* Applied Option A (approved): Demoted entity-abstracted TOST equivalence to a technical remark in Section 5.E noting mutual floor collapse at $\sim 0.05$ MRR; rephrased Contribution 2 to non-causal statistical language ("produces no statistically significant difference, $p>0.05$", with gains driven by entity mass $+0.2126$); updated Contribution 3 with reconciled SigLIP TOST bounds (item `A-06`, `A-07`, `B-02`).

12. **Table XIII (Empirical Decision Guide, lines 562–582):**
    - *Old:* Row 8 reported $N=50$ exploratory multi-hop queries (`0.6400`).
    - *New:* Standardized row 8 to $N=100$ relational queries (`0.8200 [0.742, 0.898]`), noting extractive PRF matches generative LLM bridging under TOST ($p_{\text{TOST}}=0.0062$ at $\Delta=\pm 0.05$) while eliminating 394$\times$ latency; clarified $N=50$ pure 2-hop baseline (`0.6400`) in footnote (item `B-03`, `B-12`).

13. **Section 6.C (Dual-Tier Feasibility Pilots, lines 633–634):**
    - *Old:* Included large, redundant Table XV feasibility pilots table.
    - *New:* Demoted Table XV into a single concise prose paragraph summarizing Tier 1 and Tier 2 feasibility pilots, adding citation to Karol J. Piczak ESC-50 dataset (`piczak2015esc50`) (item `B-17`).

14. **Section 7 (Limitations and Threats to Validity, lines 636–654):**
    - *Old:* Verbose, multi-paragraph items taking 1.75 columns of text.
    - *New:* Surgically pruned and condensed all 10 items into punchy, high-impact statements strictly preserving all empirical metrics, models, and confidence intervals. Contributed ~30 lines of savings to pull bibliography back onto Page 15 (item `B-09`, `B-10`, `B-12`).

15. **Bibliography (`\begin{thebibliography}`, lines 665–764):**
    - *Old:* References were unordered with respect to visual appearance, causing inverted citation callouts in text; Wang et al. 2011 and Gallagher et al. 2019 titles and authors were inaccurate; ESC-50 citation was missing.
    - *New:* Corrected Wang et al. 2011 and Gallagher et al. 2019 per DBLP; added `piczak2015esc50`; reordered all 49 entries in exact visual reading order, guaranteeing strictly monotonic citation numbering ($1, 2, 3, \dots, 49$) across the compiled PDF (item `P-01`, `P-02`, `P-04`).
