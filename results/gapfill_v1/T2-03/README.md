# T2-03: Sub-Word vs Character Effects Ablation Study

## Purpose & Scope
This experiment systematically tests which representation drives the sparse inverted index gain under OCR noise:
1. **Character n-gram length:** n=2, 3, 4, 5, 6 separately and 3–5 combined.
2. **Subword tokenizers:** BERT WordPiece and GPT-2 BPE.
3. **Linguistic expansions:** Porter stemming and Levenshtein fuzzy expansion.

## Key Empirical Findings
1. **Why WordPiece & BPE Fail under Noise:**
   - On **Apple Clean OCR (Verbatim)**: WordPiece achieves MRR = **0.8289** and BPE achieves **0.6690**.
   - On **Optical Blur Degraded (Verbatim)**: WordPiece drops to **0.6608** and BPE drops to **0.5612**, lagging Char 3–5gram BM25 (**0.8086**) by over **-0.20 MRR**.
   - Under **20% Synthetic Noise (Verbatim)**: WordPiece achieves only **0.4095** and BPE achieves **0.3930**, whereas Char 3–5gram BM25 retains **0.6723**.
   - **Mechanism:** Learned subwords (WordPiece/BPE) rely on specific vocabulary prefix/suffix splits (e.g. `##n`, `##hd`). A single character corruption shatters the subword segmentation, causing out-of-vocabulary fallback to single characters with skewed IDF statistics.

2. **Optimal Character n-gram Length:**
   - Under optical blur, Char 3-gram (**0.7972**) and Char 4-gram (**0.8070**) are the primary drivers of resilience.
   - Char 2-gram suffers from severe discriminability loss (high collision rate across receipt lines).
   - Char 5-gram and 6-gram lose matching power when character errors occur within words.
   - Combining **n=3..5** achieves the best balance across all noise regimes (**0.9437** clean, **0.8086** blur).

3. **Linguistic Word Expansions (Stemming & Fuzzy):**
   - **Stemming:** Has virtually no benefit on receipts (clean verbatim 0.8367 vs Word 0.8371) because entity names, serial numbers, and vendor names are proper nouns unaffected by morphological inflection.
   - **Fuzzy Expansion:** Drops MRR (clean verbatim 0.8005 vs 0.8371) due to false positive collisions across short numeric and code tokens.

## Verdict on Scientific Label
- **"Character n-gram" is the accurate, necessary scientific label.**
- "Sub-word" is misleading if interpreted as learned subwords (WordPiece or BPE), because WordPiece and BPE tokenizers collapse under optical noise almost as severely as whole words. The robust indexing unit is strictly contiguous character n-grams.

## Provenance
- **Git Commit:** `6b9f2aaa8a873049f7d95fd831e22f868cd5e636`
- **Runtime:** `38.12s`
