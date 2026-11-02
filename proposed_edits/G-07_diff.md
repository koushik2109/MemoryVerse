# Proposed Edits: G-07 (Adversarial OCR Obfuscation & Security Boundaries)

### Target File: `paper_v3/ijcds_body.tex`
### Location: Section 5.C (Production Cascade & Conformal Risk) & Limitation 10

```diff
@@ -325,6 +325,12 @@
 Marginal risk calibration guarantees average coverage across instances but exhibits conditional sensitivity across heterogeneous query partitions.
 
+\textbf{Adversarial OCR Obfuscation and Security Boundaries:} Real-world document pipelines face adversarial evasion via Unicode homoglyphs (e.g., Latin 'a' $\to$ Cyrillic '\u0430') and invisible zero-width spaces (ZWSP $\backslash$u200B). Under 50\% homoglyph substitution, raw Word BM25 collapses to $0.8170$ MRR and Dense BGE collapses by $-66.2\%$ ($0.5393 \to 0.1825$ MRR; $64.05\%$ evasion) as foreign codepoints shatter lexical and WordPiece tokenization. Under a combined multi-vector attack (homoglyphs, ZWSP, and leetspeak), raw Word BM25 drops by $-34.0\%$ ($0.9573 \to 0.6319$ MRR), whereas subword Char 3--5g BM25 retains $0.8329$ MRR ($+0.2010$ lift over Word BM25, $p_{\text{Holm}} < 10^{-4}$). In contrast, native visual late-interaction (ColQwen2) processes pixel rasters and is $100\%$ immune to text-level Unicode obfuscations ($0.8824$ MRR sustained). Ingestion-time confusable folding and zero-width stripping completely restore Char BM25 to \textbf{0.9528} MRR ($+0.1199$ lift, $p_{\text{Holm}} < 10^{-4}$) and Word BM25 to $0.9110$. Furthermore, when undefended text attacks suppress first-stage score margins, the conformal cascade's non-conformity gate ($\tau=0.235$) autonomously flags the anomaly ($100\%$ detection recall on evaded queries), safely escalating to visual inspection to retain $0.8513$ cascade MRR.
+
 \subsection{Multimodal Modality Deficit}
```
