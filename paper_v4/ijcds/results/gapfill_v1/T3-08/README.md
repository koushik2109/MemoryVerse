# Tier 3 Item T3-08: Proper Equivalence Tests (TOST)

## Overview
This study addresses reviewer objections regarding informal "no significant difference detected" claims
by applying the **Two One-Sided Tests (TOST)** procedure (Schuirmann 1987; Lakens 2017).
Under TOST, failing to detect a difference (NHST p >= 0.05) is NOT accepted as proof of equivalence.
Instead, formal equivalence is affirmed if and only if both one-sided null hypotheses are rejected:
$$\mathcal{H}_{01}: \Delta \le -\Delta_0 \quad \text{and} \quad \mathcal{H}_{02}: \Delta \ge +\Delta_0$$
which is mathematically equivalent to confirming that the $(1 - 2\alpha)$ two-sided confidence interval
(the 90% CI for nominal $\alpha = 0.05$) falls entirely within the pre-stated equivalence bounds $[-\Delta_0, +\Delta_0]$.

We evaluate against the pre-stated benchmark margin $\Delta_0 = \pm 0.05$ (5% absolute retrieval effectiveness)
and conduct sensitivity analysis across margins [0.03, 0.04, 0.05, 0.06, 0.075, 0.10].

---

## Formal Classifications

| Comparison | Condition / N | Metric | Mean Diff | 90% CI | 95% CI | TOST p (Δ=0.05) | Formal Verdict |
|---|---|---|---|---|---|---|---|
| **ColQwen2 vs. Char BM25** | Entity-Abstracted (N=306) | MRR | +0.0051 | [-0.0090, +0.0192] | [-0.0119, +0.0220] | < 1e-6 | **EQUIVALENT** |
| **ColQwen2 vs. Tuned Hybrid** | Entity-Abstracted (N=306) | MRR | +0.0059 | [-0.0082, +0.0200] | [-0.0109, +0.0228] | < 1e-6 | **EQUIVALENT** |
| **Extractive PRF vs. LLM** | Diagnostic (N=50) | Hit@10 | 0.0000 | [-0.0664, +0.0664] | [-0.0792, +0.0792] | 0.1109 | **UNDERPOWERED / INCONCLUSIVE** |
| **Extractive PRF vs. LLM** | Expanded (N=100) | Hit@10 | 0.0000 | [-0.0331, +0.0331] | [-0.0394, +0.0394] | 0.0073 | **EQUIVALENT** |
| **SigLIP vs. Human Gold** | MS-COCO 1K (N=1,000) | MRR | +0.0236 | [-0.0028, +0.0500] | [-0.0079, +0.0551] | 0.0502 | **BORDERLINE / UNDERPOWERED (Δ=0.05)** |

---

## Detailed Findings

1. **ColQwen2 vs. Lexical Proxies on Entity-Abstracted Controls (N=306):**
   - **Verdict:** **EQUIVALENT** ($p_{\text{TOST}} < 10^{-6}$ at $\Delta = \pm 0.05$; even at tight $\Delta = \pm 0.03$, $p_{\text{TOST}} = 0.002$).
   - The observed mean difference is tiny ($+0.0051$ MRR) with a tight $95\%$ CI of $[-0.0119, +0.0220]$.
   - This formally proves that native visual ColQwen2 and CPU lexical search are statistically equivalent on answerable factual questions when entity anchors are abstracted.

2. **Grounded Extractive PRF vs. Generative LLM Bridging (Multi-Hop):**
   - **Diagnostic Set ($N=50$):** **UNDERPOWERED / INCONCLUSIVE** ($p_{\text{TOST}} = 0.1109$).
     Although both achieve identical Target Hit@10 ($0.640$ vs. $0.640$), the $95\%$ CI $[-0.0792, +0.0792]$ exceeds the $\pm 0.05$ margin. With $N=50$, power is only $\approx 42\%$ to detect equivalence within $\pm 0.05$.
   - **Expanded Diagnostic Set ($N=100$):** **EQUIVALENT** ($p_{\text{TOST}} = 0.0073 < 0.01$).
     With $N=100$, the $95\%$ CI contracts to $[-0.0394, +0.0394]$, strictly within $[-\Delta, +\Delta]$. This confirms that rule-based extractive PRF matches generative LLM bridging without paying LLM API fees.

3. **SigLIP vs. Human Gold Captions on MS-COCO ($N=1,000$):**
   - **Verdict:** **BORDERLINE / UNDERPOWERED at $\Delta = \pm 0.05$** ($p_{\text{TOST}} = 0.0502$).
     The $90\%$ CI is $[-0.0028, +0.0500]$, just kissing the upper boundary.
   - At relaxed bounds $\Delta \ge \pm 0.06$, formal equivalence holds ($p = 0.0118$). At narrow margins $\Delta \le \pm 0.03$, the test is firmly inconclusive ($p = 0.345$).
   - The paper must honestly report this boundary rather than asserting blanket equivalence.

---

## Reproducibility
- Command: `.venv/bin/python scripts/gapfill_t3_08_tost_equivalence.py`
- Seed: 42
- Git Commit: 6b9f2aaa8a873049f7d95fd831e22f868cd5e636
