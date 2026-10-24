# Pre-Registration Addendum: When Do Text Proxies Suffice?

**Title:** When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise  
**Date:** October 25, 2026  
**Status:** Pre-Registered Prior to Addendum Data Generation and Result Execution  
**Protocol Version:** 2.0 (Journal Revision Addendum)

---

## 1. Study Reframing & Purpose
This addendum formally reframes the manuscript from an algorithmic promotion of RAMP-RAG to an empirical **regime analysis** addressing the fundamental question: *Under what query intents and proxy noise conditions do text proxies suffice, and where do native multi-vector or pixel representations remain indispensable?*

RAMP-RAG components (query routing, reliability weighting, margin-triggered cascades) are evaluated as candidate mechanisms under pre-registered hypotheses, with negative and null results reported as primary empirical contributions.

---

## 2. Experimental Specifications

### Phase 1: Public Scene-Text and Document Benchmark
- **Corpus Target:** $\ge 1,000$ index items and $\ge 300$ single-target probes drawn from public scene-text and document datasets (TextVQA / TextCaps / ICDAR / FUNSD / SROIE).
- **Split Protocol:** Fixed 50/50 Dev/Test partition stratified by item ID (seeded with Seed 42). All hyperparameter tuning ($\alpha^*$, $\gamma^*$, $\tau$) is restricted exclusively to the Dev split.
- **Probe Construction:** Each probe is a factual text span uniquely grounded in exactly one target item's OCR/ground-truth text.
- **Cluster Definition:** Cluster ID = unique target document ID to account for repeated probe sampling.
- **Proxy Extraction:** Real OCR using Tesseract / EasyOCR, alongside ground-truth text upper bounds.

---

### Phase 2: Query-Type $\times$ Noise Regime Grid (Core Study)
- **Design:** Full factorial grid across:
  - **Query Types (3 levels):**
    1. *Verbatim:* Original extracted text span.
    2. *Light Paraphrase:* Semantic rephrasing with 30%–50% token overlap with verbatim probe, preserving numbers/identifiers.
    3. *Heavy Paraphrase:* Rephrasing with $<15\%$ token overlap, eliminating verbatim lexical shortcuts while preserving semantic query intent.
  - **Noise Conditions ($\ge 5$ levels):**
    1. Clean (0% noise)
    2. Synthetic 10% character corruption
    3. Synthetic 20% character corruption
    4. Synthetic 40% character corruption
    5. Real OCR noise from image captures
    6. Optical degradation variants (Gaussian blur, JPEG Q20, downsampling, rotation)
  - **Systems (12 architectures):**
    1. BM25 (word-level Okapi)
    2. Character $n$-gram BM25 ($n=3..5$)
    3. Fuzzy / edit-distance lexical matcher
    4. Dense MiniLM (`all-MiniLM-L6-v2`)
    5. Dense BGE-base (`BAAI/bge-base-en-v1.5`)
    6. Multilingual / Dense BGE-M3 (`BAAI/bge-m3` or `multilingual-MiniLM`)
    7. Tuned Hybrid ($\alpha^*$ tuned on Dev split, not fixed 0.70)
    8. All-Stream RRF (BM25 + Dense + CLIP)
    9. CLIP ViT-B/32 (raw pixels)
    10. BM25 + CLIP Clean Stream
    11. Hybrid + CLIP Clean Stream
    12. Proposed RAMP-RAG
- **Metrics:** MRR, Hit@10, Hit@1.
- **Hypotheses:**
  - **H2.1 (Confirmatory - Lexical Dominance on Verbatim):** On verbatim queries under $\ge 20\%$ character noise, lexical methods (BM25 / char $n$-gram) achieve higher MRR than dense bi-encoders.
  - **H2.2 (Confirmatory - Semantic Paraphrase Reversal):** Under light and heavy paraphrasing, dense/hybrid representations outperform BM25 even under moderate noise, demonstrating that lexical dominance is contingent on verbatim token overlap.
  - **H2.3 (Exploratory - Crossover Thresholds):** The noise level at which lexical and dense retrieval switch rank superiority monotonically decreases as semantic token overlap decreases.
- **Statistical Tests:** Cluster bootstrap 95% CIs ($B=2,000$ resamples over target item clusters), paired permutation tests ($N_{\text{perm}}=10,000$) against tuned hybrid, and step-down Holm-Bonferroni correction within each query type family.
- **Decision Rule:** Claim of improvement or degradation requires $p_{\text{Holm}} < 0.05$ and paired difference 95% CI excluding zero.

---

### Phase 3: Mechanism Analysis (Exploratory)
- **Tokenization Fragmentation:** Wordpiece tokens per word across clean vs. corrupted text for WordPiece / Byte-Pair Tokenizers.
- **Embedding Drift:** Mean cosine similarity $\cos(\mathbf{e}_{\text{clean}}, \mathbf{e}_{\text{corrupt}})$ across models.
- **Error Attribution:** Sub-group breakdown across probe categories: numeric-heavy, short ($\le 4$ tokens), rare-token, named entity, long.

---

### Phase 4: Rerankers and RAMP-RAG Ablation on Public Corpus (Confirmatory)
- **H1 (Reliability-Weighted Fusion):** Re-test whether reliability weighting outperforms tuned hybrid and BM25 on the larger public benchmark.
  - *Decision Rule:* Reject H1 if BM25 or unweighted hybrid achieves higher MRR or if difference CI includes zero.
- **H2 (Learned Query Router):** Re-test whether learned query routers outperform unweighted All-Stream RRF on the larger benchmark.
  - *Decision Rule:* Reject H2 if All-Stream RRF $\ge$ best router MRR.
- **Strong Reranker:** Benchmark `BAAI/bge-reranker-base` alongside `ms-marco-MiniLM-L-6-v2`.

---

### Phase 5: Conformal Cascade Calibration (Confirmatory)
- **Dataset:** $\ge 1,000$ queries pooled from Phase-1 evaluation material.
- **Calibration Protocol:** 200 random seeded 50/50 Dev/Test splits.
- **Target Risk:** Non-escalated miss rate bound $\epsilon \le 0.05$.
- **Evaluation:** Empirical miss rate on held-out test split, percentage of splits satisfying the bound, mean escalation rate, and latency Pareto curve.

---

### Phase 6: Multi-Hop Equivalence Testing (Confirmatory)
- **Hypothesis:** Extractive PRF bridging matches generative LLM decomposition.
- **Statistical Test:** Two One-Sided Tests (TOST) for equivalence with pre-specified equivalence margin $\Delta = \pm 0.05$ Target Hit@10.
- **Decision Rule:** Claim "matches" if and only if the 90% confidence interval of $(\text{Hit@10}_{\text{extractive}} - \text{Hit@10}_{\text{LLM}})$ falls strictly within $[-0.05, +0.05]$.

---

### Phase 7: COCO Native-Visual Claim Hardening (Confirmatory)
- **Hypothesis:** Native pixel retrieval (CLIP ViT-B/32) significantly outperforms caption text proxies across Hit@1, Hit@5, Hit@10, and MRR.
- **Statistical Test:** Paired permutation tests with step-down Holm-Bonferroni correction ($m=4$ metrics).

---

## 3. Data Freezing and Non-Interference
- All newly generated corpora, probes, paraphrases, and splits are frozen with SHA-256 hashes in `MANIFEST.json`.
- Zero files in `results/` or existing baseline probe datasets will be modified or overwritten. All addendum outputs reside under `results/v2/`.
