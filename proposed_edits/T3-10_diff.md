# Proposed Edit Diff for T3-10: Granular Error Analysis & Failure Mode Attribution

## Description
Adds a granular diagnostic failure taxonomy across $N=306$ held-out probes and 813 test corpus records, categorizing every retrieval failure (Rank > 10 / Hit@10 = 0) into:
1. `OCR_Target_Absence`
2. `Sparse_Distractor_Collision`
3. `Dense_Numeric_Short_Dispersion`
4. `Paraphrase_Entity_Gap` / `Partial_Mismatch`

Data source: `results/gapfill_v1/T3-10/error_taxonomy_breakdown.csv` and `per_query_failure_attribution.csv`.

```diff
--- a/ijcds/ijcds_body.tex
+++ b/ijcds/ijcds_body.tex
@@ -237,3 +237,3 @@
-    \item \textbf{Diagnostic Error Attribution across Probe Properties:} Stratifying test probes into property buckets under 20\% noise exposes distinct vulnerability profiles across four primary buckets (and a minor long-query bucket, $N=3$, MRR = 0.744). Dense bi-encoders collapse on \emph{Numeric-Heavy} probes ($N=217$, MiniLM MRR = 0.033, BGE = 0.081) and \emph{Short} entity probes ($N=194$, MiniLM MRR = 0.068, BGE = 0.135). Conversely, character $n$-gram BM25 retains high accuracy across all buckets: Numeric-Heavy (MRR = 0.415), Short entities (0.473), Rare Tokens ($N=167$, MRR = 0.643), and Named Entities ($N=248$, MRR = 0.504).
+    \item \textbf{Diagnostic Error Attribution and Failure Taxonomy:} Automated failure attribution across $N=306$ held-out probes and 813 corpus items categorizes retrieval failures (Rank $> 10$) into physical root causes (Table~\ref{tab:error_taxonomy_breakdown}). On Real OCR verbatim lookups, Character 3--5gram BM25 achieves 98.37\% Hit@10 with only 5 failures; 4 of these 5 failures represent complete OCR target dropout where text was omitted by the scanner, with only 1 distractor collision. Conversely, Word BM25 suffers 26 failures (91.50\% Hit@10), with 20 of 26 (76.9\%) caused by OCR token fragmentation driving spurious distractor collisions. For dense bi-encoders, over 96\% of verbatim failures (186/192 for BGE, 202/210 for MiniLM) are attributed to \emph{Dense Numeric/Short Dispersion}, where dual-encoders fail to discriminate short vendor names, transaction dates, and currency totals. Under light paraphrasing, Character BM25 failures surge to 192 (62.75\% failure rate), with 119 failures driven by spurious subword matching across distractor text, whereas Word BM25 confines failures to 131 (42.81\% failure rate) by leveraging word-level IDF against anchorless subwords.
```

