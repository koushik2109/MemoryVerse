# Forensic Audit Report: Multi-Hop TOST Equivalence Discrepancy & Resolution

**Date:** November 6, 2026  
**Audit Target:** Family 6 Multi-Hop Relational Reasoning TOST Equivalence  
**Raw Source File:** `results/v6/multihop_100_cases_v6.csv` (SHA-256: `10554 bytes`)

---

## 1. Discrepancy Root-Cause Analysis

The discrepancy between the previous master ledger and manuscript arose from a synthesized placeholder row in `scripts/generate_all_master_audit_ledgers.py`:
- **Previous Master Ledger Row:** Reported $N=50$, $\text{Mean Diff} = -0.0200$, and $p_{\text{TOST}} = 0.0381$. This row was hardcoded with synthetic parameters and did not match any underlying observation file.
- **Manuscript Text (Section VI-B, line 628):** Reported $N=100$, $\text{Mean Diff} = 0.0000$, and $p_{\text{TOST}} = 0.0073 < 0.01$ at $\Delta = \pm 0.05$. It also explicitly noted that the $N=50$ hard subset was underpowered ($p_{\text{TOST}} = 0.1109$).

Inspection of `results/v6/multihop_100_cases_v6.csv` reveals that **all 100 paired query observations exist** and have been preserved:
- Exactly 50 queries are hard multi-hop queries (`is_multihop == True`), where Extractive Hit@10 = 0.6400 (32/50) and LLM Hit@10 = 0.6400 (32/50).
- Exactly 50 queries are relational ceiling lookups (`is_multihop == False`), where Extractive Hit@10 = 1.0000 (50/50) and LLM Hit@10 = 1.0000 (50/50).
- Across the pooled $N=100$ benchmark, Extractive Hit@10 = 0.8200 and LLM Hit@10 = 0.8200.
- Exactly 96 queries are concordant ties (both hit or both miss), 2 are Extractive wins, and 2 are LLM wins.
- The true mean paired difference is **exactly 0.0000** ($SE = 0.0201$).

---

## 2. Definitive Recomputed Results

| Subset | $N$ | Extractive Hit@10 | LLM Hit@10 | Mean Diff | $SE$ | Margin $\Delta$ | $t_1$ | $p_1$ | $t_2$ | $p_2$ | $p_{\text{TOST}}$ | 90% CI | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hard (Pure 2-Hop) | 50 | 0.6400 | 0.6400 | 0.0000 | 0.0404 | $\pm 0.03$ | 0.7425 | 0.2307 | -0.7425 | 0.2307 | 0.2307 | [-0.0677, +0.0677] | Inconclusive |
| Hard (Pure 2-Hop) | 50 | 0.6400 | 0.6400 | 0.0000 | 0.0404 | $\pm 0.05$ | 1.2374 | 0.1109 | -1.2374 | 0.1109 | 0.1109 | [-0.0677, +0.0677] | Inconclusive |
| Hard (Pure 2-Hop) | 50 | 0.6400 | 0.6400 | 0.0000 | 0.0404 | $\pm 0.07$ | 1.7324 | 0.0448 | -1.7324 | 0.0448 | 0.0448 | [-0.0677, +0.0677] | Equivalent ($p < 0.05$) |
| Hard (Pure 2-Hop) | 50 | 0.6400 | 0.6400 | 0.0000 | 0.0404 | $\pm 0.10$ | 2.4749 | 0.0084 | -2.4749 | 0.0084 | 0.0084 | [-0.0677, +0.0677] | Equivalent ($p < 0.01$) |
| **Pooled Relational** | **100** | **0.8200** | **0.8200** | **0.0000** | **0.0201** | **$\pm 0.03$** | 1.4925 | 0.0694 | -1.4925 | 0.0694 | 0.0694 | [-0.0334, +0.0334] | Inconclusive |
| **Pooled Relational** | **100** | **0.8200** | **0.8200** | **0.0000** | **0.0201** | **$\pm 0.05$** | 2.4875 | 0.0073 | -2.4875 | 0.0073 | **0.0073** | [-0.0334, +0.0334] | **Equivalent ($p < 0.01$)** |
| **Pooled Relational** | **100** | **0.8200** | **0.8200** | **0.0000** | **0.0201** | **$\pm 0.07$** | 3.4825 | 0.0004 | -3.4825 | 0.0004 | **0.0004** | [-0.0334, +0.0334] | **Equivalent ($p < 0.001$)** |
| **Pooled Relational** | **100** | **0.8200** | **0.8200** | **0.0000** | **0.0201** | **$\pm 0.10$** | 4.9749 | <1e-4 | -4.9749 | <1e-4 | **<1e-4** | [-0.0334, +0.0334] | **Equivalent ($p < 0.001$)** |

---

## 3. Editorial & Scientific Conclusion

1. **The manuscript text was correct:** The reported values in the manuscript ($N=100$, diff $=0.0000$, $p_{\text{TOST}} = 0.0073 < 0.01$ at $\Delta = \pm 0.05$, and $N=50$ underpowered at $p=0.1109$) are the genuine empirical results calculated from `results/v6/multihop_100_cases_v6.csv`.
2. **The previous master ledger row was invalid:** The synthetic row ($N=50$, diff $=-0.0200$, $p=0.0381$) has been permanently expunged.
3. **Master Ledger Update:** `MASTER_TOST_RESULTS.csv` and `MASTER_PAIRED_STATISTICS.csv` have been synchronized to report the verified $N=100$ and $N=50$ TOST metrics.
