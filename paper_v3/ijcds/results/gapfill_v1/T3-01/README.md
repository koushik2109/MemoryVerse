# T3-01: Controlled Test of Query-Formulation Claim

## Overview
This experiment isolates whether the observed dense retrieval deficit is causally driven by **syntactic formulation** (interrogative question vs. declarative sentence vs. keyword bag) or by **information mass / entity co-specification** (isolated entity snippet vs. multi-anchor context).

- **Git Commit:** `6b9f2aaa8a873049f7d95fd831e22f868cd5e636`
- **Evaluation Split:** Test probes ($N=306$) against frozen Test corpus ($N=813$).
- **Seed:** `42`
- **Permutation Tests:** $B=100,000$ paired permutations per comparison.

## Controlled Query Formulations (Identical Ground-Truth Need)
1. **Short Verbatim Snippet:** Isolated substring/single entity (e.g., `"BOOK TA .K (TAMAN DAYA) SDN BHD"`).
2. **Keyword Bag:** Exact multi-entity co-specification (e.g., `"ONE ONE THREE SEAFOOD RESTAURANT SDN BHD 179.50 23-06-2018"`).
3. **Declarative Sentence:** Full grammatical sentence with identical anchors (e.g., `"Receipt from ONE ONE THREE SEAFOOD RESTAURANT SDN BHD totaling 179.50 on 23-06-2018."`).
4. **Interrogative Question:** Natural question syntax with identical anchors (e.g., `"What was the purchase from ONE ONE THREE SEAFOOD RESTAURANT SDN BHD totaling 179.50 on 23-06-2018?"`).
5. **Conversational Imperative:** User command syntax with identical anchors (e.g., `"Find the receipt for ONE ONE THREE SEAFOOD RESTAURANT SDN BHD with total 179.50 on date 23-06-2018."`).

## Headline Findings

### 1. Syntactic Form Has No Causal Effect on Dense Retrieval
When entity anchors are held strictly identical:
- **BGE-small on Real OCR:**
  - Keyword Bag: MRR = `0.5187`
  - Declarative: MRR = `0.5210`
  - Interrogative: MRR = `0.5073`
  - Paired difference Keyword vs. Question: $+0.0114$ ($p = 0.6617$).
  - Paired difference Declarative vs. Question: $+0.0137$ ($p = 0.5316$).
- **MiniLM on Real OCR:**
  - Keyword Bag: MRR = `0.3382`
  - Declarative: MRR = `0.3081`
  - Interrogative: MRR = `0.3091`
  - Paired difference Keyword vs. Question: $+0.0291$ ($p = 0.0073$, favoring keywords!).
  - Paired difference Declarative vs. Question: $-0.0011$ ($p = 0.5516$).

### 2. Entity Anchor Mass Drives Retrieval Gains
Moving from an isolated snippet to multi-entity anchors lifts MRR by:
- **BGE:** $+0.2126$ (from $0.3061$ to $0.5187$, $p < 10^{-5}$).
- **MiniLM:** $+0.1144$ (from $0.2238$ to $0.3382$, $p < 10^{-5}$).
- **Char BM25:** $+0.0984$ (from $0.8524$ to $0.9508$, $p < 10^{-5}$).
- **Word BM25:** $+0.1345$ (from $0.7821$ to $0.9166$, $p < 10^{-5}$).

### 3. Policy Recommendation on Causal Language
> [!IMPORTANT]
> **Causal Language Recommendation:** The manuscript **MUST NOT** assert that dense bi-encoder failure is caused by query formulation syntax (i.e. not being in question format).
> The data shows that keyword bags achieve identical or superior MRR to grammatical questions when entity anchors match.
> The paper should use **observational language** regarding syntactic form, and clarify that the causal driver of dense recovery is **entity co-specification mass**, not interrogative phrasing.
