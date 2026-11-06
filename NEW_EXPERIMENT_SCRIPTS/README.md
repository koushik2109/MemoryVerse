# NEW_EXPERIMENT_SCRIPTS

This directory contains standalone, deterministic replication scripts for the three targeted experimental evaluations conducted for the IJCDS submission:

1. **`run_experiment_1_date_ablation.py`**:
   - Evaluates the effect of transaction dates in human-authored queries by comparing original queries against versions with excised date expressions.
   - Evaluates Word BM25, Character 3–5gram BM25, and Dense BGE-Base against the frozen 813-item SROIE test corpus.
   - Computes paired bootstrap 95% confidence intervals, 100,000-iteration paired permutation tests, McNemar's test for Hit@1, and same-merchant wrong-transaction rates.
   - Outputs: `TRANSACTION_DATE_ABLATION.csv`, `TRANSACTION_DATE_ABLATION_SUMMARY.csv`.

2. **`run_experiment_2_same_merchant_hard_negatives.py`**:
   - Evaluates retrieval precision and failure attribution when queries target retail merchants with multiple receipts in the corpus.
   - Classifies Top-1 errors into same-merchant hard negatives versus random distractors.
   - Computes within-merchant discrimination MRR isolating within-store ranking from corpus retrieval.
   - Outputs: `SAME_MERCHANT_HARD_NEGATIVES.csv`, `SAME_MERCHANT_HARD_NEGATIVES_SUMMARY.csv`.

3. **`extract_experiment_3_cascade_depth.py`**:
   - Synthesizes candidate-depth cutoff sweeps ($K \in \{5, 10, 20, 30, 50, 75, 100, 150, 200\}$) with live measured hardware latency on Apple M2 CPU and Apple Silicon MPS.
   - Outputs: `CASCADE_DEPTH_RESULTS.csv`.

## Execution

All scripts use the project virtual environment and run offline without external network dependencies:

```bash
# Run Experiment 1 (Transaction-Date Ablation)
.venv/bin/python NEW_EXPERIMENT_SCRIPTS/run_experiment_1_date_ablation.py

# Run Experiment 2 (Same-Merchant Hard Negatives)
.venv/bin/python NEW_EXPERIMENT_SCRIPTS/run_experiment_2_same_merchant_hard_negatives.py

# Run Experiment 3 (Cascade Depth Synthesis)
python3 NEW_EXPERIMENT_SCRIPTS/extract_experiment_3_cascade_depth.py
```

