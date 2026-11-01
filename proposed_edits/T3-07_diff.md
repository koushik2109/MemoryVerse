# Proposed Edit Diff for T3-07: Stronger Image-Caption Proxies

## Description
Systematically evaluates modern detailed VLMs (Florence-2 Detailed) and Multi-Caption concatenation indexing against native pixel representations (CLIP and SigLIP) on $N=1,000$ MS-COCO Karpathy-test images. Formally defines the Best Machine-Caption Proxy ceiling ($0.4580$ MRR) and re-estimates the persistent native-visual advantage with Holm-Bonferroni correction ($p_{\text{Holm}} < 10^{-15}$).
Data source: `results/gapfill_v1/T3-07/caption_proxies_summary.csv` and `caption_proxies_paired_tests.csv`.

```diff
--- a/ijcds/ijcds_body.tex
+++ b/ijcds/ijcds_body.tex
@@ -419,2 +419,2 @@
-Scaling captioning models does not close the gap: BLIP-2 (\texttt{opt-2.7b}) \cite{li2023blip2} achieves MRR = 0.4185 (gap to SigLIP: $-0.1886$), while Florence-2 detailed captioning \cite{xiao2024florence2} achieves MRR = 0.4420 (gap: $-0.1651$). Native visual embeddings exhibit no significant difference from human gold captions (MRR = 0.5835 vs.\ SigLIP 0.6071, diff $-0.0236$, paired permutation $p = 0.142$, equivalence not tested). Visual scene semantics cannot be completely compressed into synthetic machine tokens (Fig.~\ref{fig:coco_native_vs_proxy}).
+Scaling machine captioning models fails to bridge the modality divide: BLIP-2 (\texttt{opt-2.7b}) \cite{li2023blip2} reaches MRR = 0.4185, while detailed Florence-2 visual descriptions \cite{xiao2024florence2} reach MRR = 0.4420. Exhaustive multi-caption concatenation indexing (combining multi-perspective machine captions) defines the empirical machine proxy ceiling at \textbf{0.4580 MRR} (Hit@10 = 75.40\%). Relative to this best machine-caption proxy, Native SigLIP Base Patch16 retains a statistically decisive advantage of \textbf{$+0.1491$ MRR} ($p_{\text{Holm}} < 10^{-15}$, Hit@10 = 86.70\%, Table~\ref{tab:coco_benchmarks_v3}). Native SigLIP approaches the upper-bound performance of exhaustive 5-reference human gold multi-captions (0.6071 vs.\ 0.6210, diff $-0.0139$) without incurring the latency and cost of generative image captioning. Visual scene semantics cannot be completely compressed into synthetic machine tokens (Fig.~\ref{fig:coco_native_vs_proxy}).
```

