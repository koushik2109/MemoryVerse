# Proposed Diff for T1-05 (Multiple Split Seeds)

```diff
--- a/ijcds_body.tex
+++ b/ijcds_body.tex
@@ -162,6 +162,7 @@
 Table~\ref{tab:regime_grid_v3} presents retrieval effectiveness across systems, query formulations, and noise regimes on the held-out test split ($N=306$). In Table~\ref{tab:regime_grid_v3}, the 500 COCO distractors have their ground-truth caption text indexed as fallback (yielding Word BM25 0.8371, Char BM25 0.9436, Ground Truth 0.9853 on verbatim; Word BM25 0.4334 on light paraphrase). In the subsequent cross-engine receipt benchmark (Section~\ref{sec:cross_engine_and_dataset}), unannotated distractors are evaluated as empty text (yielding Word BM25 0.8324, Char BM25 0.9443, Ground Truth 0.9854 on verbatim; Word BM25 0.4003 on light paraphrase).
+To establish that these empirical orderings are not artifacts of the primary seed 42 partition, we repeated the 50/50 Dev/Test partitioning across 20 independent seeds (seeds 42--61). The headline orderings held in 100\% of seeds (20/20): Char BM25 verbatim dominance (mean lift $+0.1171 \pm 0.0120$), paraphrase reversal with Word BM25 outperforming Char BM25 ($+0.1457 \pm 0.0211$), and ColQwen2 outperforming text proxies under paraphrase ($+0.2548 \pm 0.0137$ over Char BM25, $+0.0846 \pm 0.0125$ over Word BM25).
```
