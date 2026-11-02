# Proposed Edits: G-16 (Privacy-Preserving Collaborative Memory Generation)

### Target File: `paper_v3/ijcds_body.tex`
### Location: Section 6.C (Dual-Tier Feasibility Pilots) / Architectural Principles for Multi-Party Personal RAG

```diff
@@ -638,3 +638,5 @@
 \subsection{Dual-Tier Feasibility Pilots}
 Prior to establishing the frozen benchmark, exploratory pilots evaluated multimodal proxies across text ($N=334$) and physical media ($N=151$ probes: $N=85$ scene text, $N=13$ ESC-50 audio recordings \cite{piczak2015esc50} via LAION-CLAP \cite{wu2023clap}, and $N=53$ album photos via CLIP ViT-B/32 \cite{radford2021clip}). On physical media, sparse BM25 achieved MRR = 0.7916 on scene text, CLAP achieved MRR = 1.0000 on audio, and CLIP achieved Hit@10 = 0.792 (MRR = 0.6182) on raw photos, confirming modality-specific proxy bounds.
+
+\textbf{Privacy-Preserving Collaborative Memory Construction:} Multi-user shared vaults (collaborative trip journals, group video recaps) demand cryptographic policy governance and selective obfuscation. Evaluating $N=300$ collaborative queries across a 6-participant shared vault ($N=1{,}000$ multimodal records) under four user consent policies demonstrates that unconstrained retrieval incurs a 23.00\% unconsented privacy exposure rate (leaking 69.0\% of face-opt-out participant appearances). Naive blanket redaction eliminates leaks but collapses generation coherence from 94.58\% down to 52.03\% ANLS, destroying narrative readability. Conversely, MemoryVerse Consent-Aware Privacy Filtering (CAPF)—employing policy-gated access control, selective bounding-box face blurring, and contextual role preservation—guarantees 0.00\% unconsented privacy exposure across all categories while retaining 93.56\% generation coherence ($+41.53\%$ ANLS lift over blanket redaction, $p_{\text{Holm}} < 0.0001$) and matching unconstrained retrieval MRR (0.3258 vs.\ 0.3241, diff $+0.0017$, $p_{\text{Holm}} < 0.0001$), with only 0.7\,ms policy check latency (3.9\,ms total serving time).
```

