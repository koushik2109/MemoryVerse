# Proposed Edits: G-05 (Cross-Encoder Candidate Depth Sweep & Learned-Sparse / ColBERTv2)

### Target File: `paper_v3/ijcds_body.tex`
### Location: Section 5.C (Production Cross-Encoder Reranker Cascade Ablation) & Limitation 10

```diff
@@ -310,6 +310,12 @@
 However, evaluating two-stage cascades over first-stage Character 3--5gram BM25 reveals an architectural ceiling: standalone Char-BM25 achieves MRR = \textbf{0.9436}, and applying MiniLM or BGE cross-encoders on top-20 candidates reduces MRR to 0.9175 and 0.9141 (lifts of $-0.0262$ and $-0.0296$, Table~\ref{tab:practical_stack}). Furthermore, tuning hybrid fusion between Char-BM25 and dense BGE on Dev selects $\alpha_{\text{sparse}}^* = 1.00$ ($\alpha_{\text{dense}}^* = 0.00$, 0.9495 Dev MRR), confirming that dense representations provide zero additive signal over sub-word inverted indexes on verbatim receipts.
 
+\textbf{Systematic Candidate Depth Sweep ($K_1 \in [5..100]$):} Sweeping candidate pool depth $K_1$ across 5, 10, 15, 20, 30, 50, and 100 on degraded optical blur (CER $\approx 20\%$) reveals that cross-encoder reranking cannot rescue verbatim entity precision: over first-stage Word BM25, reranked MRR remains below standalone baseline at all depths ($0.6695$ at $K_1=5$, $0.6561$ at $K_1=20$, $0.6472$ at $K_1=100$ vs.\ $0.6950$ stage-1 MRR, $\Delta = -0.0391$, $p < 0.001$). Over Char 3--5g BM25, cross-encoding imposes a persistent penalty ($-0.0427$ at $K_1=20$, $0.7844$ vs.\ $0.8271$ stage-1 MRR, $p_{\text{Holm}} < 0.001$) that deepens as distractor candidate depth expands, while scaling serving latency from $4.7$\,ms to $92.1$\,ms. Cross-encoding provides a positive lift only on semantic paraphrases ($+0.0363$ lift at $K_1=20$, $0.2509$ vs.\ $0.2146$, $p = 0.02$).
+
+\textbf{Benchmarking Learned-Sparse (SPLADE) and Late Interaction (ColBERTv2):} Under optical degradation, learned-sparse term expansion (SPLADE) suffers catastrophic collapse to $0.0597$ MRR (trailing Char BM25 by $-0.7674$ MRR, $p_{\text{Holm}} < 10^{-4}$) because OCR noise fragments character sequences into out-of-vocabulary WordPiece subwords that fail to trigger vocabulary projection activations. Token-level contextual late interaction (ColBERTv2) achieves $0.3896$ clean and $0.3145$ degraded MRR, significantly outperforming dense BGE-base ($0.2796$, $p < 0.01$) by preserving fine-grained token alignments, but trails sub-word Char 3--5g BM25 by $-0.5126$ MRR ($p_{\text{Holm}} < 10^{-4}$) at $29.7\times$ higher latency ($28.5$\,ms vs.\ $0.96$\,ms). Standalone CPU Char 3--5g BM25 strictly Pareto-dominates all tested neural paradigms in OCR-degraded retrieval accuracy, latency, and index size.
+
 \begin{table}[htbp]
 \centering
 \footnotesize
```
