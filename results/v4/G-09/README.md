# G-09: Controlled Query-Formulation Experiments

## Executive Summary & Statistical Findings
This study resolves Reviewer Objection **G-09 / T3-01** by isolating whether the dense retrieval deficit observed in document RAG is caused by **syntactic phrasing** (interrogative question vs. declarative sentence vs. keyword bag) or by **information mass / entity co-specification** (isolated entity snippet vs. multi-anchor context).

### 1. Syntactic Syntax Has No Causal Effect on Dense Retrieval
Holding entity anchors strictly identical across 306 queries:
- **Dense BGE-small on Real OCR:**
  * Keyword Bag: MRR = **0.5187** (Hit@10: 0.8137)
  * Declarative Sentence: MRR = **0.5210** (Hit@10: 0.7908)
  * Interrogative Question: MRR = **0.5073** (Hit@10: 0.8072)
  * Paired difference Keyword vs. Question: $+0.0114$ ($p = 0.6617$).
  * Paired difference Declarative vs. Question: $+0.0137$ ($p = 0.5316$).
- **Dense MiniLM on Real OCR:**
  * Keyword Bag: MRR = **0.3382** (Hit@10: 0.5425)
  * Declarative Sentence: MRR = **0.3081** (Hit@10: 0.4477)
  * Interrogative Question: MRR = **0.3091** (Hit@10: 0.4771)
  * Paired difference Keyword vs. Question: $+0.0291$ ($p = 0.0073$, favoring keywords!).
  * Paired difference Declarative vs. Question: $-0.0011$ ($p = 0.5516$).

### 2. Entity Anchor Mass Drives the Observed Dense Recovery
When queries provide multiple entity anchors (vendor name, total amount, date), dense models recover substantial performance regardless of syntax:
- **BGE-small:** Lifts from **0.3061** (single snippet) to **0.5187** (multi-anchor bag), $\Delta = +0.2126$ ($p < 10^{-5}$).
- **MiniLM:** Lifts from **0.2238** to **0.3382**, $\Delta = +0.1144$ ($p < 10^{-5}$).
- **Char 3-5g BM25:** Lifts from **0.9399** to **0.9644**, $\Delta = +0.0245$ ($p < 10^{-4}$).
- **Word BM25:** Lifts from **0.8091** to **0.9426**, $\Delta = +0.1335$ ($p < 10^{-5}$).

### 3. Manuscript Policy & Wording Rule
The manuscript **MUST NOT assert causal claims** that dense bi-encoders collapse because queries lack question syntax. The data demonstrates that keyword bags match or exceed interrogative questions. The paper will adopt **strictly observational language** regarding syntactic form and state that dense performance scales with **entity co-specification mass**.

## Provenance
- **Git Commit:** `ebc19dd07d31fe5b854467a62310cd3a0ff1fb32`
- **Source Files:** `results/gapfill_v1/T3-01/query_formulation_summary.csv`
- **Output Artifacts:** `results/v4/G-09/query_formulation_summary.csv`, `results/v4/G-09/table_query_formulation_controlled.tex`
