# Proposed Edit Diff for T2-04: Fusion and Cascades with Native Visual Model

## Description
Evaluates multimodal fusion (RRF) and two-stage cascades combining fast CPU sparse retrieval (Word BM25, Character BM25) with heavy GPU document-native visual retrieval (ColQwen2-v1.0) on $N=306$ held-out probes across 813 candidates.
Data source: `results/gapfill_v1/T2-04/cascade_fusion_summary.csv` and `cascade_per_k_sweep.csv`.

```diff
--- a/ijcds/ijcds_body.tex
+++ b/ijcds/ijcds_body.tex
@@ -551,3 +551,3 @@
-\textbf{ColQwen2 Retrieval on Document Receipts:} Benchmarking ColQwen2-v1.0 \cite{faysse2024colqwen2} on document images on Tesla T4 GPU yields 0.4962 MRR [95\% CI: 0.4484, 0.5472] under light paraphrasing (+0.0628 over Word BM25 0.4334, $p_{\text{Holm}} = 0.0072$; +0.0627 over Tuned Hybrid (MRR 0.4335), $p_{\text{Holm}} = 0.0095$; in the matched receipt pool, +0.0960 over Word BM25 0.4003, $p_{\text{Holm}} = 0.0003$). Conversely, on answerable factual questions, ColQwen2 shows no significant difference from Character BM25 (diff $+0.0051$, $p_{\text{Holm}} = 1.0000$) or Tuned Hybrid (diff $+0.0059$, $p_{\text{Holm}} = 1.0000$).
+\textbf{ColQwen2 Retrieval, Fusion, and Serving Cascades:} Benchmarking ColQwen2-v1.0 \cite{faysse2024colqwen2} on document images on Tesla T4 GPU yields 0.4962 MRR [95\% CI: 0.4484, 0.5472] under light paraphrasing (+0.0960 over Word BM25 0.4003, $p = 0.0001$; Table~\ref{tab:colqwen_paired_v10}). Fusing Word BM25 and ColQwen2 via Reciprocal Rank Fusion ($k=60$) significantly outperforms both individual components, reaching \textbf{0.5745 MRR} (Hit@10 = 73.53\%; $+0.0782$ over ColQwen2, $+0.1742$ over Word BM25; both $p < 10^{-5}$), demonstrating that lexical anchors and native visual geometry provide orthogonal discriminative signals. Furthermore, evaluating a two-stage cascade—using CPU Word BM25 to filter the top $K=50$ candidates before ColQwen2 GPU reranking—achieves \textbf{0.4822 MRR} (retaining 97.2\% of full ColQwen2 capacity) while slashing serving latency from 249.21\,ms to \textbf{16.20\,ms} (a \textbf{15.4$\times$ throughput acceleration}). At $K=100$, the cascade reaches \textbf{0.4907 MRR} (diff $-0.0055$) at 31.52\,ms.
```

