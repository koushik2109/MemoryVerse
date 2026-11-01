# Proposed Edit: A-06 Entity-Abstracted TOST Formal Equivalence and Floor Effect

## Issue Summary
In Contribution 1 (line 22), Section 5.E (line 529), and Conclusion (line 691), the manuscript claims:
> "in entity-abstracted controls, ColQwen2 0.0546 and lexical proxies 0.0495 are formally equivalent under TOST with $\Delta = \pm 0.05$, diff $+0.0051$, $p_{\text{TOST}} = 1.66 \times 10^{-7}$"

At scores near $0.050$, an absolute margin of $\Delta = \pm 0.05$ is as large as the scores themselves (100% relative margin). Equivalence is trivially satisfied because both architectures suffer a floor collapse when all entity anchors are removed.

## Query Set & Evidence
- **Query Set:** $N=306$ `heavy_paraphrase` probes from `ai_engine/rag/data/v2/frozen_probes_with_paraphrases.json`.
- **Evidence File:** `results/gapfill_v1/T3-08/tost_equivalence_summary.csv` row 1; `results/v10/colqwen_paired_v10.csv` row 5.
- **Scores:** ColQwen2 MRR = $0.0546$, Char BM25 MRR = $0.0495$, Diff = $+0.0051$, $90\%$ CI $[-0.0088, +0.0192]$, $p_{\text{NHST}} = 0.5574$, $p_{\text{TOST}} = 1.66 \times 10^{-7}$ at $\Delta = \pm 0.05$.

## Proposed Options for User Decision

### Option A: Clean Removal from Headline Claims (Recommended)
Remove the parenthetical mention of entity-abstracted TOST equivalence from Contribution 1 (line 22) and Conclusion (line 691), and discuss it only in Section 5.E as a diagnostic control. Headline claims should focus exclusively on the primary headline finding where lexical retrieval beats ColQwen2 on natural answerable questions ($0.9471$ vs $0.8595$, diff $+0.0876$, $p = 0.00012$).

### Option B: Explicit Floor-Effect Caveat
Keep the sentence in Contribution 1, Section 5.E, and Conclusion, but add an explicit caveat qualifying the floor effect:
> "in entity-abstracted controls, both architectures experience a floor collapse (ColQwen2 0.0546 vs. lexical 0.0495; diff $+0.0051$, $p_{\text{TOST}} = 1.66 \times 10^{-7}$ at $\Delta = \pm 0.05$, reflecting a mutual floor effect rather than functional parity)"

