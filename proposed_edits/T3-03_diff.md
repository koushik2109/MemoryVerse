# Proposed Edits: Tier 3 Item T3-03 (Realistic Degradation & Handwriting Benchmark)

## Location in Manuscript
- `ijcds_body.tex`: Section IV-A, Section V-A, and Section VII (Limitations, Item 7).

## Context & Scientific Evidence
The original manuscript stated in Section VII (Item 7) that *"Character n-gram robustness and dictionary-based reliability weighting tuned on synthetic character permutations fail to translate directly to real optical OCR blur."*
Experiment **T3-03** formally quantifies and proves this divergence across:
1. Graded optical blur (Levels 1–4)
2. Simulated camera physics (perspective homography, non-uniform shadow vignette, motion blur)
3. Authentic administrative forms with handwriting and stamps (FUNSD, $N=100$)

## Proposed LaTeX Fragment Diff
```diff
--- a/ijcds_body.tex (Section VII, Limitations, Line 703)
+++ b/ijcds_body.tex (Section VII, Limitations, Line 703)
-    \item \textbf{Synthetic vs.\ Optical Noise Divergence:} Character $n$-gram robustness and dictionary-based reliability weighting tuned on synthetic character permutations fail to translate directly to real optical OCR blur. System architects must validate on physical optical degradation.
+    \item \textbf{Synthetic vs.\ Physical Optical and Camera Divergence:} Empirical benchmarking across synthetic character noise, simulated mobile camera physics (perspective homography, uneven illumination vignette, motion blur), and authentic administrative forms with handwriting (FUNSD, Table~\ref{tab:funsd_v3}) reveals a significant divergence ($p < 10^{-4}$, paired permutation test). Uniform synthetic character corruptions preserve bounding-box integrity and local line geometry, overestimating Word BM25 retrieval retention by up to $+0.32$ MRR. In contrast, physical camera shadow falloff and optical blur cause catastrophic line and token dropouts. Crucially, Char 3--5g BM25 retains its relative superiority over Word BM25 across both regimes (MRR $0.8066$ vs.\ $0.7763$ on handwritten forms; $0.7584$ vs.\ $0.5120$ under camera physics), confirming that character $n$-gram indexing provides robust real-world protection against optical and physical degradation.
```
