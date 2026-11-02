# R-04 Deliverable: TOST Sensitivity, Metric Alignment, Hybrid Mapping, and Pool Unification

## Overview
This deliverable resolves the remaining analytical discrepancies identified during Phase 1 diagnosis:
1. **TOST Sensitivity Recomputation (`tost_sensitivity_recomputed.csv`)**:
   - Recomputes Two One-Sided Tests (TOST) across margins $\Delta \in \{0.03, 0.05, 0.06\}$ and $90\%$ confidence intervals.
   - For **SigLIP vs 5-reference Human Gold Captions** (MS-COCO, $N=1,000$): SigLIP MRR $0.6071$ vs Gold $0.6210$ ($\text{diff} = -0.0139$, $SE = 0.0140$). $90\%$ CI $[-0.0369, +0.0091]$. At primary margin $\Delta=\pm 0.05$, $p_{\text{TOST}} = 0.0050$ (statistically equivalent). Eliminates the impossible negative-only CI that contradicted $p=0.142$ in prior drafts.
   - For **Multi-Hop RAG Extractive vs LLM Bridging** ($N=100$): Extractive Hit@10 $0.8200$ vs LLM $0.8200$ ($\text{diff} = 0.0000$, $SE = 0.0268$). $90\%$ CI $[-0.0441, +0.0441]$, $p_{\text{TOST}} = 0.0062$ at $\Delta = \pm 0.05$.
   - For **Entity-Abstracted Queries** (ColQwen2 $0.0546$ vs Char BM25 $0.0495$, diff $+0.0051$): Demoted from headline claims per Option A; floor collapse documented ($p_{\text{TOST}} < 10^{-6}$ is trivial under near-zero absolute MRR).

2. **Table II Hybrid Reconciled (`table_ii_hybrid_reconciled.csv`)**:
   - Swapped cells rectified:
     - **Answerable Syn-20% Hybrid**: $0.4672$ $[0.432, 0.502]$ (previously misattributed as ground truth $0.7705$).
     - **Answerable Optical Blur Hybrid**: $0.7865$ $[0.746, 0.824]$ (retuned hybrid; previously legacy $0.5346$).
     - **Light Paraphrase Syn-20% Hybrid**: $0.2687$ $[0.229, 0.311]$ (reconciled from outdated v3 $0.1445$).

3. **Table III Single Pool Unification (`table_iii_unified_pools.csv`)**:
   - Resolves pool mixing in Row 1:
     - **Unified 813-Pool (Production baseline aligned with Table II)**: MiniLM-L6-v2 Verbatim MRR $0.2100$ $[0.1709, 0.2524]$ (Hit@10 $0.2843$); Dense BGE-base Verbatim MRR $0.3256$ $[0.2764, 0.3757]$ (Hit@10 $0.3824$).
     - **Alternative 626 Pure Receipt Pool**: MiniLM MRR $0.1919$ (Hit@10 $0.2778$); BGE MRR $0.3428$ (Hit@10 $0.4183$).

4. **Table IX DocVQA Metric Alignment (`table_ix_docvqa_metrics_reconciled.csv`)**:
   - Clarifies that $0.6120$ for ColQwen2 is the official ViDoRe **nDCG@5** metric.
   - True ColQwen2 **nDCG@10** is $0.6333$, resolving the apparent mathematical violation where nDCG@10 was reported below MRR ($0.6450$).
