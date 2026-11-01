# T3-09: Larger Multi-Hop Evaluation ($N=100$)

## Overview
This experiment extends the exploratory $N=50$ multi-hop evaluation to the full $N=100$ query suite from `results/v6/multihop_100_cases_v6.csv`. It resolves the sample size power limitation noted in the manuscript (Section IV-D and Limitations Item 3).

- **Git Commit:** `6b9f2aaa8a873049f7d95fd831e22f868cd5e636`
- **Sample Size:** $N=100$ multi-hop relational queries
- **Baselines Compared:** Rule-Based Extractive Pseudo-Relevance Feedback (PRF) vs. Generative LLM Bridging (2-Hop decomposition).

## Headline Findings

| Dataset Scope | Sample Size $N$ | Extractive Hit@10 | LLM Hit@10 | Paired Diff [95% CI] | Discordant Pairs (Ext / LLM / Tie) | Exact McNemar $p$ | TOST $p$ ($\Delta=\pm 0.05$) | Verdict |
|---|---|---|---|---|---|---|---|---|
| **Exploratory Pilot** | $N=50$ | 0.6400 (32/50) | 0.6400 (32/50) | $0.0000$ [$-0.0792, +0.0792$] | 2 / 2 / 46 | 1.0000 | 0.1109 | **Underpowered** ($95\%\text{ CI} > \pm 0.05$) |
| **Expanded Evaluation** | $N=100$ | **0.8200** (82/100) | **0.8200** (82/100) | **$0.0000$ [$-0.0392, +0.0392$]** | 2 / 2 / 96 | **1.0000** | **0.0073** | **EQUIVALENT** ($p < 0.01$) |

### 1. Formal Equivalence Established at $N=100$
- At $N=50$, the 95% confidence interval $[-0.0792, +0.0792]$ exceeded the $\pm 0.05$ equivalence margin, precluding an equivalence claim.
- At $N=100$, the 95% confidence interval shrinks to $[-0.0392, +0.0392]$, which falls entirely inside the $[-0.05, +0.05]$ margin.
- The Two One-Sided Tests (TOST) reject the null hypothesis of non-equivalence ($p_{\text{TOST}} = 0.0073 < 0.01$).
- **Conclusion:** Extractive PRF and Generative LLM bridging achieve statistically equivalent retrieval accuracy on relational multi-hop queries.

### 2. Operational & Cost Trade-Offs
- **Extractive PRF:** 0 API calls, $0.00 cost, 12ms CPU execution latency.
- **Generative LLM:** 2 sequential API calls, ~\$0.004 per query, 4,736ms latency (+45% latency penalty, 394x slower than PRF).
- Because accuracy is formally equivalent, deploying generative LLM bridging for multi-hop retrieval introduces substantial cost and latency overhead without measurable empirical benefit.
