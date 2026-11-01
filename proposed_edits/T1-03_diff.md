# Proposed Manuscript Diff: T1-03 ColQwen2 and CLIP Degraded Visual Evaluation Suite

## Target Location
- `ijcds_body.tex` (Table II `tab:regime_grid_v3`, Table X, Section IV-A, Section IV-B)

## Context & Motivation
In the initial manuscript, visual encoders (\texttt{ColQwen2-v1.0} and \texttt{CLIP ViT-B/32}) were evaluated exclusively on uncorrupted clean document images, serving as an upper-bound visual control. Consequently, identical numbers appeared across all four noise columns (Clean GT, Synthetic 20%, Apple Clean, Optical Blur).
Reviewers noted this as Bug B: degraded images were not fed to visual encoders.
Experiment **T1-03** (Kaggle v11 GPU execution, Tesla T4, 100,000 permutations, $B=2,000$ cluster bootstrap) directly evaluated ColQwen2 and CLIP on degraded images ($N=813$ pages in `ai_engine/rag/data/sroie/img_degraded`).

## Measured Results Summary
- **ColQwen2 on Clean Receipts:**
  - Verbatim MRR: 0.8417 (Hit@1: 0.7908, Hit@10: 0.9444)
  - Light Paraphrase MRR: 0.4985 (Hit@1: 0.4052, Hit@10: 0.6569)
  - Answerable Question MRR: 0.0870 (Hit@1: 0.0458, Hit@10: 0.1438)
- **ColQwen2 on Optical Blur Degraded:**
  - Verbatim MRR: 0.7207 (Hit@1: 0.6536, Hit@10: 0.8497)
  - Light Paraphrase MRR: 0.4449 (Hit@1: 0.3595, Hit@10: 0.5915)
  - Answerable Question MRR: 0.0701 (Hit@1: 0.0327, Hit@10: 0.1209)
- **CLIP ViT-B/32:**
  - Clean MRR: 0.0590 (verbatim), 0.0381 (paraphrase), 0.0338 (answerable)
  - Degraded Blur MRR: 0.0443 (verbatim), 0.0284 (paraphrase), 0.0303 (answerable)
- **Paired Permutation Tests vs. Char 3–5g BM25 ($N_{\text{perm}}=100{,}000$):**
  - Clean Verbatim: ColQwen2 0.8417 vs. Char BM25 0.9854 (Diff: $-0.1436$ `[-0.1834, -0.1082]`, $p_{\text{Holm}} < 10^{-4}$)
  - Clean Paraphrase: ColQwen2 0.4985 vs. Char BM25 0.2526 (Diff: $+0.2459$ `[+0.2014, +0.2915]`, $p_{\text{Holm}} < 10^{-4}$)
  - Optical Blur Verbatim: ColQwen2 0.7207 vs. Char BM25 0.9443 (Diff: $-0.2236$ `[-0.2704, -0.1779]`, $p_{\text{Holm}} < 10^{-4}$)
  - Optical Blur Paraphrase: ColQwen2 0.4449 vs. Char BM25 0.2365 (Diff: $+0.2084$ `[+0.1660, +0.2506]`, $p_{\text{Holm}} < 10^{-4}$)
  - Optical Blur Answerable: ColQwen2 0.0701 vs. Char BM25 0.0495 (Diff: $+0.0207$ `[+0.0005, +0.0423]`, $p_{\text{Holm}} = 0.0508$)

## Proposed LaTeX Fragment Diff
```diff
--- a/ijcds_body.tex (Table II tab:regime_grid_v3, lines 180-201)
+++ b/ijcds_body.tex (Table II tab:regime_grid_v3, lines 180-201)
@@ -180,3 +180,3 @@
- & CLIP Visual             & \makecell{0.0436\\{\scriptsize [0.028, 0.062]}} & \makecell{0.0436\\{\scriptsize [0.028, 0.062]}} & \makecell{0.0436\\{\scriptsize [0.028, 0.062]}} & \makecell{0.0436\\{\scriptsize [0.028, 0.062]}} & 12.50 \\
- & ColQwen2-v1.0 Native    & \makecell{0.8529\\{\scriptsize [0.817, 0.887]}} & \makecell{0.8529\\{\scriptsize [0.817, 0.887]}} & \makecell{0.8529\\{\scriptsize [0.817, 0.887]}} & \makecell{0.8529\\{\scriptsize [0.817, 0.887]}} & 249.21 \\
+ & CLIP Visual             & \makecell{0.0590\\{\scriptsize [0.041, 0.079]}} & \makecell{0.0590\\{\scriptsize [0.041, 0.079]}} & \makecell{0.0590\\{\scriptsize [0.041, 0.079]}} & \makecell{0.0443\\{\scriptsize [0.028, 0.063]}} & 12.50 \\
+ & ColQwen2-v1.0 Native    & \makecell{0.8417\\{\scriptsize [0.805, 0.876]}} & \makecell{0.8417\\{\scriptsize [0.805, 0.876]}} & \makecell{0.8417\\{\scriptsize [0.805, 0.876]}} & \makecell{0.7207\\{\scriptsize [0.677, 0.763]}} & 249.21 \\
@@ -189,3 +189,3 @@
- & CLIP Visual             & \makecell{0.0303\\{\scriptsize [0.020, 0.042]}} & \makecell{0.0303\\{\scriptsize [0.020, 0.042]}} & \makecell{0.0303\\{\scriptsize [0.020, 0.042]}} & \makecell{0.0303\\{\scriptsize [0.020, 0.042]}} & 12.50 \\
- & ColQwen2-v1.0 Native    & \makecell{0.4962\\{\scriptsize [0.448, 0.547]}} & \makecell{0.4962\\{\scriptsize [0.448, 0.547]}} & \makecell{0.4962\\{\scriptsize [0.448, 0.547]}} & \makecell{0.4962\\{\scriptsize [0.448, 0.547]}} & 249.21 \\
+ & CLIP Visual             & \makecell{0.0381\\{\scriptsize [0.024, 0.054]}} & \makecell{0.0381\\{\scriptsize [0.024, 0.054]}} & \makecell{0.0381\\{\scriptsize [0.024, 0.054]}} & \makecell{0.0284\\{\scriptsize [0.016, 0.042]}} & 12.50 \\
+ & ColQwen2-v1.0 Native    & \makecell{0.4985\\{\scriptsize [0.450, 0.549]}} & \makecell{0.4985\\{\scriptsize [0.450, 0.549]}} & \makecell{0.4985\\{\scriptsize [0.450, 0.549]}} & \makecell{0.4449\\{\scriptsize [0.395, 0.494]}} & 249.21 \\
@@ -198,3 +198,3 @@
- & CLIP Visual             & \makecell{0.0388\\{\scriptsize [0.024, 0.057]}} & \makecell{0.0388\\{\scriptsize [0.024, 0.057]}} & \makecell{0.0388\\{\scriptsize [0.024, 0.057]}} & \makecell{0.0388\\{\scriptsize [0.024, 0.057]}} & 12.50 \\
- & ColQwen2-v1.0 Native    & \makecell{0.8595\\{\scriptsize [0.824, 0.893]}} & \makecell{0.8595\\{\scriptsize [0.824, 0.893]}} & \makecell{0.8595\\{\scriptsize [0.824, 0.893]}} & \makecell{0.8595\\{\scriptsize [0.824, 0.893]}} & 249.21 \\
+ & CLIP Visual             & \makecell{0.0338\\{\scriptsize [0.021, 0.049]}} & \makecell{0.0338\\{\scriptsize [0.021, 0.049]}} & \makecell{0.0338\\{\scriptsize [0.021, 0.049]}} & \makecell{0.0303\\{\scriptsize [0.018, 0.045]}} & 12.50 \\
+ & ColQwen2-v1.0 Native    & \makecell{0.0870\\{\scriptsize [0.061, 0.116]}} & \makecell{0.0870\\{\scriptsize [0.061, 0.116]}} & \makecell{0.0870\\{\scriptsize [0.061, 0.116]}} & \makecell{0.0701\\{\scriptsize [0.047, 0.096]}} & 249.21 \\
```

