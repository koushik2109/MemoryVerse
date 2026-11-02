# Proposed Edits: G-06 (Real-World Retrieval Pool Contamination & Near-Duplicate Distractor Collisions)

### Target File: `paper_v3/ijcds_body.tex`
### Location: Section 5.B (Empirical Retrieval Regimes) & Limitation 10

```diff
@@ -260,6 +260,12 @@
 In contrast, character n-gram tokenization preserves subword structural anchors across optical noise regimes, establishing consistent superiority on structured documents.
 
+\textbf{Retrieval Pool Contamination and Near-Duplicate Collisions:} In real-world personal photo and document archives, indexes accumulate near-duplicates (recurring same-vendor receipts sharing store templates and camera multi-shot photo bursts). Sweeping near-duplicate contamination rate $\rho \in [0\%, 5\%, 10\%, 20\%, 40\%]$ over the 813-document pool (up to $N=1{,}138$ documents) under degraded optical blur (CER $\approx 20\%$) reveals that dense bi-encoders (BGE-base) suffer severe distractor collisions ($22.22\%$ collision rate on specific transaction queries, MRR decaying from $0.4391$ to $0.3073$), because single-vector pooling collapses fine-grained numeric differences into vendor-level centroids. Word BM25 degrades by $-0.1866$ MRR ($0.8162 \to 0.6296$, $p < 0.0001$) as character noise forces fallback to shared template tokens. Conversely, Character 3--5g BM25 retains $80.5\%$ of its baseline accuracy ($0.7003$ MRR at $\rho=40\%$, $+0.0707$ lead over Word BM25, $p_{\text{Holm}} < 0.001$), maintaining a low $4.25\%$ collision rate. Furthermore, ingestion-time MinHash LSH deduplication ($\tau=0.85$) collapses burst captures to boost degraded MRR from $0.7003$ to \textbf{0.8610} ($+0.1607$ lift, $p_{\text{Holm}} < 10^{-4}$) and restore Hit@1 from $51.96\%$ to $81.70\%$, while shrinking index memory by $19.9\%$ ($38.6$\,MB vs.\ $48.2$\,MB) and reducing query serving latency to $0.82$\,ms.
+
 \subsection{Multimodal Vision vs. Text Proxies}
```
