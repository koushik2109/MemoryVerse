# Proposed Edits: Tier 3 Item T3-06 (Broader Document Types Regime Benchmark)

## Location in Manuscript
- `ijcds_body.tex`: Section I (Introduction, Line 28), Section IV-B (Cross-Dataset Evaluation, Line 370), Table I (System Comparison), and Table IX (DocVQA Benchmark).

## Context & Scientific Evidence
The paper addresses the central question: *"When Do Text Proxies Suffice?"*
Experiment **T3-06** formalizes the architectural boundary across 6 distinct document layout classes ($N=906$ queries across SROIE, FUNSD, and DocVQA).
The empirical evidence proves:
1. **1D Linear Sequences (Receipts & Forms):** Text proxies (Char 3–5g BM25) dominate native visual models by $+0.23$ to $+0.45$ MRR at $<1$ ms latency.
2. **2D Coordinate & Non-Linear Layouts (Charts, Tables, Infographics, Memos):** Text proxies collapse ($0.16$–$0.45$ MRR) due to OCR serialization destroying 2D spatial relationships; ColQwen2 dominates ($0.64$–$0.71$ MRR, $+0.23$ to $+0.55$ lift).

## Proposed LaTeX Fragment Diff
```diff
--- a/ijcds_body.tex (Section IV-B, Line 370)
+++ b/ijcds_body.tex (Section IV-B, Line 370)
+\subsection{Per-Document-Type Regime Boundaries: Linear vs.\ Non-Linear 2D Layouts}
+\label{sec:document_type_regimes}
+Evaluating retrieval architectures across the structural continuum of document layouts---ranging from 1D sequential entity streams (SROIE Receipts) to semi-structured forms (FUNSD) and complex 2D corporate reports, tables, and charts (DocVQA, $N=906$ total queries; Table~\ref{tab:document_type_regimes} and Fig.~\ref{fig:document_type_regime_map})---delineates the exact operational boundary of text proxies:
+\begin{enumerate}
+    \item \textbf{Linear and Semi-Structured Sufficiency:} On 1D sequential documents (Receipts) and bounding-box administrative forms (FUNSD), text proxies not only suffice but decisively outperform native visual retrieval (Char 3--5g BM25 MRR $0.9442$ vs.\ ColQwen2 $0.4962$ on receipts, $+0.4480$ lift; Tuned Hybrid $0.8461$ vs.\ $0.6120$ on forms, $+0.2341$ lift) at less than 1\% of serving latency.
+    \item \textbf{2D Spatial Collapse:} Conversely, text proxies experience severe failure on non-linear 2D layouts where reading order serialization destroys spatial adjacency: on structured financial tables, Char BM25 achieves only $0.4225$ vs.\ ColQwen2 $0.6820$ ($+0.2595$ visual lift); on coordinate charts and plots, text proxies collapse to $0.1493$ vs.\ ColQwen2 $0.7140$ (a massive $+0.5647$ visual lift). In these 2D graphical regimes, native visual late-interaction is indispensable.
+\end{enumerate}
```
