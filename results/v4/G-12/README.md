# G-12: Query-Side Noise Robustness & Diagnostic Failure Attribution

## Executive Summary & Statistical Findings
This study addresses Reviewer Objection **G-12** by evaluating:
1. **Query-Side Corruptions (T3-04):** Robustness against realistic keyboard-adjacent QWERTY typos, transpositions, and character omissions ($r \in [0\%, 30\%]$).
2. **Diagnostic Error Taxonomy (T3-10):** Mechanistic failure attribution of every retrieval miss across $N=306$ held-out probes.

### 1. Asymmetric Query-Side Noise Tolerance
- **Char 3–5g BM25:** Retains **0.7328 MRR** at 20% query typo rate (**77.7% retention** relative to clean 0.9436). Contiguous $n$-grams preserve matching mass when internal characters are transposed or substituted.
- **Word BM25:** Collapses to **0.4820 MRR** at 20% typo rate (**57.6% retention**). Single character typos invalidate entire word tokens in the inverted dictionary.
- **Word + Fuzzy Levenshtein Expansion (dist $\le 1$):** Recovers Word BM25 to **0.6012 MRR** (75.1% retention), but still trails Char BM25 (0.7328) by over $-0.13$ MRR while multiplying query evaluation latency.
- **Dense MiniLM:** Starts from a crippled baseline (0.2040) and degrades to 0.1582.

### 2. Mechanistic Root Cause of Dense Retrieval Collapse
Diagnostic analysis of 306 queries reveals why dense bi-encoders collapse on document text:
- Under Real OCR verbatim queries:
  * **BGE-small** suffers 192 failures (Hit@10 = 37.25%). **186 of 192 failures (96.9%)** are attributable to **`dense_numeric_short_dispersion`**. Dense pooling maps short numeric strings (dates, totals, invoice IDs) to diffuse, overlapping embedding spaces that collide with hundreds of unrelated documents.
  * **MiniLM** suffers 210 failures (Hit@10 = 31.37%), with **202 of 210 failures (96.2%)** driven by numeric/short dispersion.
- **Why Character BM25 Dominates:**
  * Suffers only **5 failures out of 306** (98.37% Hit@10).
  * 4 of the 5 failures are caused by **total OCR omission** (the scanner dropped the line entirely), representing an insurmountable physical target ceiling rather than an IR failure.
- **Why Whole-Word BM25 Fails on Verbatim Text:**
  * Suffers 26 failures (91.50% Hit@10). **20 of 26 failures (76.9%)** are caused by **`distractor_collision`** resulting from OCR word fragmentation (e.g. `SDN BHD` split into `S DN BH D`).

## Provenance
- **Git Commit:** `ebc19dd07d31fe5b854467a62310cd3a0ff1fb32`
- **Source Files:** `results/gapfill_v1/T3-04/query_noise_summary.csv`, `results/gapfill_v1/T3-10/error_taxonomy_breakdown.csv`
- **Output Artifacts:** `results/v4/G-12/table_query_noise_and_error_taxonomy.tex`, `results/v4/G-12/fig_query_noise_robustness.png`
