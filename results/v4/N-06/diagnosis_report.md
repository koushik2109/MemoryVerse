# N-06 Diagnostic Report: Multi-Hop Subset Breakdown, Statistical Power, and Latency Reconciliation

## 1. Executive Summary
- **Concerns Raised:**
  1. Pooled $N=100$ relational multi-hop queries consists of 50 hard queries (Hit@10 = 0.640 for both) and 50 easy queries (Hit@10 = 1.000 for both, ceiling). Need separate TOST, 90%/95% CIs, and statistical power reported for the hard subset alone versus pooled.
  2. Latency numbers appear conflicting: 3,260 ms single-stage baseline in Section 6.B vs. $\le 50$ ms in Table XII (Table XI in earlier numbering), and 45% end-to-end penalty vs. $394\times$ latency multiplier.
  3. Reconcile Section 5.G.4 (citing 0.640) vs. Table I / Table XIII (citing 0.820).

## 2. Subset Breakdown and Equivalence Statistics (TOST)
Across the $N=100$ relational multi-hop queries (`results/v6/multihop_100_cases_v6.csv`):

| Subset | $N$ | Extractive PRF Hit@10 | Generative LLM Hit@10 | Mean Diff | $SE$ | 90% CI | 95% CI | $p_{\text{TOST}}$ ($\Delta=\pm 0.05$) | Power ($\Delta=\pm 0.05$) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Hard (Pure 2-Hop Bridging)** | 50 | 0.6400 (32/50) | 0.6400 (32/50) | 0.0000 | 0.0404 | $[-0.0677, +0.0677]$ | $[-0.0812, +0.0812]$ | $0.1109$ | 0.0% (underpowered) | Inconclusive at $\pm 0.05$ (equivalent at $\pm 0.07$) |
| **Easy (Relational Linking)** | 50 | 1.0000 (50/50) | 1.0000 (50/50) | 0.0000 | 0.0000 | $[0.0000, 0.0000]$ | $[0.0000, 0.0000]$ | $< 10^{-6}$ | 100.0% (ceiling) | Formally Equivalent |
| **Pooled Relational Queries** | 100 | 0.8200 (82/100) | 0.8200 (82/100) | 0.0000 | 0.0201 | $[-0.0334, +0.0334]$ | $[-0.0399, +0.0399]$ | $0.0073$ | 60.1% | Formally Equivalent ($p < 0.01$) |

### Statistical Insight:
- On the **Hard subset alone ($N=50$)**, both methods reach identical Hit@10 = 0.6400, but with $N=50$ and 4 discordant pairs (2 wins for PRF, 2 wins for LLM), the 90% CI $[-0.0677, +0.0677]$ extends past $\pm 0.05$. Thus, TOST at $\Delta = \pm 0.05$ is inconclusive ($p = 0.1109$).
- Adding the 50 relational queries brings the pooled sample to $N=100$, narrowing the 90% CI to $[-0.0334, +0.0334]$, which strictly fits within $[-0.05, +0.05]$ and confirms equivalence ($p_{\text{TOST}} = 0.0073$).
- The manuscript must report both: state honestly that the hard subset achieves equivalence at $\Delta = \pm 0.07$ and that $N=100$ is required for $\Delta = \pm 0.05$.

## 3. Latency Reconciliation (45% vs. 394x and 3,260 ms)
- **Why 3,260 ms vs. Table XII ($\le 50$ ms)?**
  - Table XII reports **isolated local retrieval lookup latencies** (e.g. Char BM25 lookup = 0.87 ms, Dense BGE vector dot product = 0.01 ms / query encoding = 28 ms).
  - The $3,260\,\text{ms}$ in Section 6.B is the **end-to-end multi-hop pipeline execution time** (document text parsing, hop-1 retrieval, entity extraction, and hop-2 retrieval).
- **Why 45% vs. 394x?**
  - **45% penalty (End-to-End Pipeline):** Extractive PRF pipeline ($3,260\,\text{ms}$) vs. Generative LLM pipeline ($4,736\,\text{ms}$): $\frac{4736 - 3260}{3260} = +45.3\%$.
  - **394x penalty (Isolated Bridging Step):** Extractive entity heuristic lookup ($12.0\,\text{ms}$) vs. Generative LLM API roundtrip + autoregressive decoding ($4,736\,\text{ms}$): $\frac{4736}{12.0} \approx 394.7\times$.
- **Fix:** Clarify the exact scope in Section 6.B to avoid confusion between end-to-end pipeline runtime and isolated module execution.

## 4. Manuscript Location Reconciliations (0.640 vs. 0.820)
1. **Section 5.G.4 (line 593):** Update `"0.640 vs. 0.640 Hit@10"` to `"0.6400 on pure 2-hop, 0.8200 pooled on N=100 relational queries"`.
2. **Table I (line 55):** Already accurately cites both: `Exploratory PRF (Hit@10 0.640, N=50; 0.820, N=100, Sec. 6.B)`.
3. **Table XIII (line 576 & footnote 580):** Already cites 0.8200 with footnote explaining $N=50$ pure 2-hop is 0.6400.
4. **Section 6.B (line 612):** Add the subset breakdown table and state the power limitations explicitly.

