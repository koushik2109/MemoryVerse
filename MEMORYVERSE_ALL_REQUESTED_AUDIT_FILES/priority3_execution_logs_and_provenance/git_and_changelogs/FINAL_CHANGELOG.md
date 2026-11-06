# Final Changelog — Rapid Experimental Strengthening for IJCDS Submission

**Date:** November 2026  
**Repository:** MemoryVerse (`IJCDS` submission branch)  
**Status:** COMPLETE, AUDITED, & FULLY SYNCHRONIZED  

---

## 1. Summary of Executed Work

To address reviewer scrutiny regarding query authoring, within-entity transaction confusion, and cascade candidate sensitivity, we completed three targeted, scientifically rigorous evaluations:

1. **Experiment 1 (Transaction-Date Ablation, $N=56$):**
   - Executed paired ablation testing on all 56 date-bearing human queries against the frozen $N=813$ test corpus.
   - Identified that excising explicit transaction dates significantly reduces Word BM25 MRR ($-0.1158$, $p=0.0023$) and Character BM25 MRR ($-0.1041$, $p=0.0031$), while Dense BGE-Base is insensitive ($-0.0199$, $p=0.3902$).
   - In multi-receipt merchant stores ($N=36$), date ablation dropped Character BM25 Hit@1 from 27.8% down to 13.9% and increased same-merchant false-target confusion from 11.1% to 16.7%.
   - Generated: [`TRANSACTION_DATE_ABLATION.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/TRANSACTION_DATE_ABLATION.csv) and [`NEW_EXPERIMENT_SCRIPTS/run_experiment_1_date_ablation.py`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/NEW_EXPERIMENT_SCRIPTS/run_experiment_1_date_ablation.py).

2. **Experiment 2 (Same-Merchant Hard Negatives, $N=224$):**
   - Evaluated 224 queries targeting merchants with multiple receipts in the test corpus across Word BM25, Char BM25, and Dense BGE-Base.
   - **Sample Size Traceability (276 Full-Pool vs. 224 Test-Split):** Across the entire 1,626-document pool (626 total receipts), 276 human query instances target multi-receipt merchants. However, under the leak-free 50/50 Dev/Test partition ($N=813$ test index), exactly 26 merchants have duplicate receipts partitioned into the Dev split ($26 \times 2 = 52$ queries). Because cross-split leakage is strictly prevented, these 52 queries have zero same-merchant distractors within the 813-item test index. The released evaluation reports the exact, uninvented $N=224$ queries with active same-merchant distractors in the test index.
   - Demonstrated that under verbatim queries, **68.8% to 69.6%** of all evaluated queries returned an alternate receipt from the same merchant at rank one, while unrelated-receipt errors accounted for less than 10% of queries.
   - Measured within-merchant discrimination MRR: Word BM25 reaches 0.4608 and Char BM25 reaches 0.4303 under human paraphrase.
   - Generated: [`SAME_MERCHANT_HARD_NEGATIVES.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/SAME_MERCHANT_HARD_NEGATIVES.csv) and [`NEW_EXPERIMENT_SCRIPTS/run_experiment_2_same_merchant_hard_negatives.py`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/NEW_EXPERIMENT_SCRIPTS/run_experiment_2_same_merchant_hard_negatives.py).

3. **Experiment 3 (Cascade Candidate-Depth Sensitivity, $K \in [5, 200]$):**
   - Synthesized candidate-depth sweeps with live measured hardware latency on Apple M2 CPU and Apple Silicon MPS.
   - Confirmed that the cost-quality Pareto frontier is broad and robust: $K=50$ captures 97.2% of full ColQwen2 MRR with a $15.5\times$ CPU speedup (12.18 ms) and $4.4\times$ MPS speedup (1.26 ms), while $K=100$ recovers 98.9% MRR with an $8.0\times$ CPU speedup (23.78 ms).
   - Generated: [`CASCADE_DEPTH_RESULTS.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/CASCADE_DEPTH_RESULTS.csv) and [`NEW_EXPERIMENT_SCRIPTS/extract_experiment_3_cascade_depth.py`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/NEW_EXPERIMENT_SCRIPTS/extract_experiment_3_cascade_depth.py).

---

## 2. Manuscript & Document Updates

- **`ijcds_body.tex`:**
  - Added paragraph `\textbf{Transaction-Date Disambiguation and Same-Merchant Hard Negatives:}` reporting the paired statistical test results of EXP-01 and EXP-02.
  - Maintained strictly **15.0 pages** with $12.02\text{ pt}$ of clean clearance above the footer line on Page 15 (max text $Y = 779.78\text{ pt}$ vs. footer line at $791.8\text{ pt}$).
- **`STATISTICAL_RESULTS.md`:** Comprehensive reporting of sample sizes, paired bootstrap 95% CIs, permutation tests, McNemar's tests, and Holm-Bonferroni corrections.
- **`EXPERIMENT_INVENTORY.md` & `EXPERIMENT_REUSE_MATRIX.csv`:** Full traceability of existing vs. new evidence.

---

## 3. Package Synchronization & SHA-256 Manifest

All files, sources, PDFs, and archives synchronized across repository trees:
- `ijcds_body.tex` synchronized across all 5 directories.
- `main.pdf` compiled to strictly 15.0 pages and mirrored to `FINAL_IJCDS_SUBMISSION.pdf`.
- `FINAL_IJCDS_REPRODUCIBILITY_PACKAGE.zip` updated and mirrored to `~/Downloads/`.

