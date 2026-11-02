# R-04: Rebuilt Numeric Claims Ledger (Phase 2)

- **Total Claims in Manuscript:** 1742
- **Verified Unchanged:** 1713
- **Phase 2 Corrected (Scheduled for Phase 5 Patch):** 7
- **Unverified / Structural Exclusions:** 22

## Phase 2 Correction Diff
The following numbers in the current manuscript will be updated during Phase 5:

| Current MS Value | Corrected Value | Location | System | Corrected Source | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0.00012` | `0.0011` | Introduction (line 116) | Char BM25 | `results/v4/N-03/paired_answerable_stats.md` | Real-OCR paired permutation p-value (+0.0733 diff). Clean GT remains p=0.00012 (+0.0876 diff). |
| `0.6120` | `0.6695` | Empirical Regime Analysis and Results > Regime A: Text-Dominated Document and Receipt Retrieval > Micro-Mechanisms of Dense Collapse and Query Formulation Controls > Production Cascades and Risk-Controlled Escalation > Regime B: Visual-Dominated Natural Image Search > Cross-Engine OCR Replication and Document-Native Visual Retrieval (line 529) | ColQwen2 | `results/v4/R-02/docvqa_metrics_reconciled.csv:3` | Own 500-split nDCG@5 is 0.6695; 0.6120 is official full-benchmark leaderboard score. |
| `0.6120` | `0.6695` | Empirical Regime Analysis and Results > Regime A: Text-Dominated Document and Receipt Retrieval > Micro-Mechanisms of Dense Collapse and Query Formulation Controls > Production Cascades and Risk-Controlled Escalation > Regime B: Visual-Dominated Natural Image Search > Cross-Engine OCR Replication and Document-Native Visual Retrieval (line 550) | Char BM25 | `results/v4/R-02/docvqa_metrics_reconciled.csv:3` | Own 500-split nDCG@5 is 0.6695; 0.6120 is official full-benchmark leaderboard score. |
| `0.00012` | `0.0011` | Empirical Regime Analysis and Results > Regime A: Text-Dominated Document and Receipt Retrieval > Micro-Mechanisms of Dense Collapse and Query Formulation Controls > Production Cascades and Risk-Controlled Escalation > Regime B: Visual-Dominated Natural Image Search > Cross-Engine OCR Replication and Document-Native Visual Retrieval (line 624) | Char BM25 | `results/v4/N-03/paired_answerable_stats.md` | Real-OCR paired permutation p-value (+0.0733 diff). Clean GT remains p=0.00012 (+0.0876 diff). |
| `0.00012` | `0.0011` | Table XIII (line 667) | Char BM25 | `results/v4/N-03/paired_answerable_stats.md` | Real-OCR paired permutation p-value (+0.0733 diff). Clean GT remains p=0.00012 (+0.0876 diff). |
| `0.00012` | `0.0011` | Empirical Regime Analysis and Results > Regime A: Text-Dominated Document and Receipt Retrieval > Micro-Mechanisms of Dense Collapse and Query Formulation Controls > Production Cascades and Risk-Controlled Escalation > Regime B: Visual-Dominated Natural Image Search > Cross-Engine OCR Replication and Document-Native Visual Retrieval > Computational Resource and Latency Profile > Empirical Decision Guide and Practical Implications (line 685) | Char BM25 | `results/v4/N-03/paired_answerable_stats.md` | Real-OCR paired permutation p-value (+0.0733 diff). Clean GT remains p=0.00012 (+0.0876 diff). |
| `0.00012` | `0.0011` | Conclusion and Future Work (line 747) | Char BM25 | `results/v4/N-03/paired_answerable_stats.md` | Real-OCR paired permutation p-value (+0.0733 diff). Clean GT remains p=0.00012 (+0.0876 diff). |
