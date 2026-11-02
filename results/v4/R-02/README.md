# R-02 Execution Report: Recomputed TOST, Multi-Hop Subsets, DocVQA nDCG, and Holm Family Table

## Summary of Recomputed Statistical Artifacts
All statistical recomputations and verified outputs are archived in `results/v4/R-02/`:
1. `coco_siglip_tost_recomputed.csv`: MS-COCO SigLIP vs 4-Ref Multi-Caption Gold ($N=1,000$) TOST recomputation across margins $\Delta = \pm 0.03, \pm 0.05, \pm 0.06$.
2. `multihop_subsets_recomputed.csv`: Multi-hop per-subset recomputed metrics ($N=50$ hard pure 2-hop, $N=50$ easy relational ceiling, $N=100$ pooled) with exact paired standard error, 90% CIs, and TOST at $\Delta = \pm 0.05$ and $\pm 0.07$.
3. `docvqa_metrics_reconciled.csv`: DocVQA nDCG@5 and nDCG@10 metrics across all text-proxy systems and ColQwen2, clarifying ViDoRe full benchmark leaderboard score ($0.6120$) vs our 500-sample split ($0.6695$).
4. `holm_family_complete_table.csv`: Exhaustive 30-entry Holm-Bonferroni correction table mapping EVERY reported p-value across all 6 hypothesis testing families.
5. `holm_family_complete_table.tex`: Formatted publication-ready LaTeX table for inclusion in Section 4.E / Appendix.

---

## 1. MS-COCO SigLIP vs Gold TOST Recomputation ($N=1,000$)
- **Empirical Difference:** $\text{Diff} = -0.0139$, $\text{SE} = 0.0140$, $t(999) = -0.9919$, paired $p = 0.3214$.
- **90% Confidence Interval:** $[-0.0369, +0.0092]$
- **95% Confidence Interval:** $[-0.0413, +0.0136]$
- **TOST at $\Delta = \pm 0.05$:** $t_1 = +2.5825$ ($p_1 = 0.0050$), $t_2 = -4.5662$ ($p_2 = 0.0000$) $\implies p_{\text{TOST}} = 0.0050 < 0.01$ (**FORMALLY EQUIVALENT**, power $97.3\%$).
- **TOST at $\Delta = \pm 0.06$:** $p_{\text{TOST}} = 0.0005 < 0.001$ (**FORMALLY EQUIVALENT**, power $99.6\%$).
- **TOST at $\Delta = \pm 0.03$:** $t_1 = +1.1527$ ($p_1 = 0.1246$), $t_2 = -3.1365$ ($p_2 = 0.0009$) $\implies p_{\text{TOST}} = 0.1246 > 0.05$ (**INCONCLUSIVE / UNDERPOWERED**, power $69.1\%$).

---

## 2. Multi-Hop Relational Retrieval Subsets ($N=100$)
- **Hard 2-Hop Subset ($N=50$, pure sequential bridging):**
  - Extractive PRF Hit@10 = $0.6400$, Generative LLM Hit@10 = $0.6400$.
  - Mean Diff = $0.0000$, $\text{SE} = 0.0404$, $90\%$ CI $[-0.0677, +0.0677]$.
  - $p_{\text{TOST}}$ at $\Delta = \pm 0.05$: $0.1109 > 0.05$ (**UNDERPOWERED / INCONCLUSIVE**).
  - $p_{\text{TOST}}$ at $\Delta = \pm 0.07$: $0.0448 < 0.05$ (**EQUIVALENT** at $\Delta = \pm 0.07$).
- **Easy Relational Subset ($N=50$, direct multi-item co-occurrence):**
  - Extractive PRF Hit@10 = $1.0000$, Generative LLM Hit@10 = $1.0000$ (ceiling).
- **Pooled Benchmark ($N=100$ relational probes):**
  - Extractive PRF Hit@10 = $0.8200$, Generative LLM Hit@10 = $0.8200$.
  - Mean Diff = $0.0000$, $\text{SE} = 0.0201$, $90\%$ CI $[-0.0334, +0.0334]$.
  - $p_{\text{TOST}}$ at $\Delta = \pm 0.05$: $0.0073 < 0.01$ (**FORMALLY EQUIVALENT**, power $60.1\%$).

---

## 3. DocVQA Leaderboard nDCG@5 Provenance Reconciled
- Official ViDoRe published leaderboard score across full DocVQA collection: $\text{nDCG}@5 = 0.6120$, $\text{nDCG}@10 = 0.6333$.
- Evaluated on our 500-sample test split:
  - ColQwen2: Hit@1 = $0.5280$, Hit@5 = $0.7820$, Hit@10 = $0.8410$, $\text{MRR} = 0.6450$, $\text{nDCG}@5 = 0.6695$, $\text{nDCG}@10 = 0.6886$.
  - Tuned Hybrid (Tesseract): $\text{MRR} = 0.3507$, $\text{Hit}@10 = 0.4680$, $\text{nDCG}@10 = 0.3724$.
  - Char BM25 (Tesseract): $\text{MRR} = 0.3141$, $\text{Hit}@10 = 0.3940$, $\text{nDCG}@10 = 0.3261$.
  - Word BM25 (Tesseract): $\text{MRR} = 0.3107$, $\text{Hit}@10 = 0.3800$, $\text{nDCG}@10 = 0.3195$.
  - Dense BGE-base (Tesseract): $\text{MRR} = 0.2737$, $\text{Hit}@10 = 0.3940$, $\text{nDCG}@10 = 0.2948$.
  - Dense MiniLM (Tesseract): $\text{MRR} = 0.2063$, $\text{Hit}@10 = 0.3060$, $\text{nDCG}@10 = 0.2232$.

