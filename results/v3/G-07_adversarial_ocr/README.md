# G-07: Retrieval Robustness under Adversarial OCR Obfuscation & Homoglyphs

## Overview
This directory contains experimental results for **G-07**, systematically investigating document retrieval robustness
and security boundaries when text is deliberately obfuscated:
1. **Unicode Homoglyph Substitution (25% and 50%):** Replacing Latin characters with visually indistinguishable Cyrillic codepoints.
2. **Invisible Zero-Width Space Injection (ZWSP):** Injecting `\u200B` to shatter subword and word tokenizers.
3. **Leetspeak Obfuscation:** Common evasion substitutions (`0` $\leftrightarrow$ `O`, `1` $\leftrightarrow$ `l`, `$` $\leftrightarrow$ `S`, `@` $\leftrightarrow$ `a`).
4. **Visual Watermark Bleed:** Diagonal security stamp ("CONFIDENTIAL") merging into OCR tokens.
5. **Combined Multi-Vector Attack:** Simultaneous homoglyph, ZWSP, and leetspeak attack.

- **Corpus:** $N=813$ standard benchmark documents (313 SROIE receipts + 500 COCO distractors)
- **Evaluation Probes:** $N=306$ held-out probes across Verbatim and Specific Transaction queries ($N=2{,}142$ evaluated queries per system)
- **Paradigms Evaluated:** Word BM25 (Raw vs. Defended), Char 3–5g BM25 (Raw vs. Defended), Dense BGE-base (Raw vs. Defended), ColBERTv2, Native Visual ColQwen2, and Conformal Cascade with Visual Escalation.
- **Statistical Rigor:** Paired permutation tests ($B=10{,}000$) with Holm-Bonferroni correction and $B=2{,}000$ bootstrap confidence intervals.

---

## Headline Findings

### 1. The Vulnerability Hierarchy under Text Obfuscation:
- **Catastrophic Collapse of Raw Word BM25:**
  Under 25% Homoglyph substitution, raw Word BM25 drops from $0.9573$ to **$0.8452$ MRR**; under 50% homoglyphs it drops to **$0.8170$ MRR**; under Zero-Width Space injection it drops to **$0.8261$ MRR** (with Hit@10 collapsing to $90.85\%$ and evasion jumping to $9.15\%$). Under the Combined Multi-Vector attack, Word BM25 collapses to **$0.6319$ MRR** (a **$-34.0\%$ drop**, Hit@1 falling to $51.31\%$ and evasion to $15.03\%$).
- **Subword Character Resilience vs. Homoglyph Vulnerability:**
  Character 3–5g BM25 retains decisive superiority over Word BM25 across all attacks:
  - 25% Homoglyphs: Char BM25 achieves **$0.9196$ MRR** vs. Word BM25 $0.8452$ ($+0.0744$ lift, $p < 0.0001$).
  - 50% Homoglyphs: Char BM25 achieves **$0.8869$ MRR** vs. Word BM25 $0.8170$ ($+0.0700$ lift, $p < 0.0001$).
  - ZWSP Injection: Char BM25 achieves **$0.9337$ MRR** vs. Word BM25 $0.8261$ ($+0.1077$ lift, $p < 0.0001$).
  - Leetspeak: Char BM25 achieves **$0.9343$ MRR** vs. Word BM25 $0.8514$ ($+0.0829$ lift, $p < 0.0001$).
  - Combined Attack: Char BM25 achieves **$0.8329$ MRR** vs. Word BM25 $0.6319$ ($+0.2010$ lift, $p < 0.0001$).
- **Dense Model Vocabulary Fracture:**
  Dense BGE-base and ColBERTv2 suffer severe vocabulary fragmentation:
  - BGE-base collapses from $0.5393$ baseline to **$0.3171$** (25% homoglyphs), **$0.1825$** (50% homoglyphs, $-66.2\%$ drop, $64.05\%$ evasion), and **$0.2053$** (Combined attack, $63.07\%$ evasion).
  - ColBERTv2 collapses from $0.5276$ baseline to **$0.3411$** (25% homoglyphs), **$0.2044$** (50% homoglyphs, $-61.3\%$ drop, $60.46\%$ evasion), and **$0.2694$** (Combined attack, $50.65\%$ evasion).

### 2. Native Visual Immunity:
- **Pixel-Level Invariance (ColQwen2):**
  Because ColQwen2 operates on raw document pixel rasters rather than OCR Unicode codepoints, it is **100% immune** to Unicode Homoglyphs, Zero-Width Space attacks, and Leetspeak obfuscations, sustaining its baseline **$0.8824$ MRR** across all text attacks. Under physical Watermark Bleed, it gracefully retains **$0.8382$ MRR**.

### 3. Dual-Layer Defense Strategy:
- **Layer 1: Ingestion Confusable Normalization:**
  Applying NFKD decomposition, Cyrillic lookalike folding, and zero-width codepoint stripping fully neutralizes text attacks:
  - Defended Char 3–5g BM25 recovers back to **$0.9761$ MRR** on Homoglyphs ($+0.0565$ lift on 25%, $+0.0891$ lift on 50%, $p < 0.0001$) and **$0.9528$ MRR** on Combined attack ($+0.1199$ lift, $p < 0.0001$).
  - Defended Word BM25 recovers from $0.6319$ to **$0.9110$ MRR** on Combined attack ($+0.2791$ lift, $p < 0.0001$).
- **Layer 2: Conformal Safety Cascade with Visual Escalation Gate:**
  When text-level adversarial attacks degrade first-stage retrieval confidence, the non-conformity score triggers an **autonomous escalation** to the visual model (`ColQwen2`). Under all attack regimes, the conformal gate achieves **$100\%$ detection recall** on evaded queries (escalation rates $98.4\% - 99.3\%$), sustaining cascade MRR at **$0.8513 - 0.8873$** without requiring manual intervention.
