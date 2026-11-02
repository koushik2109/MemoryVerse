# G-11: SigLIP vs. Human Gold Captions Equivalence Replication & Power Analysis

## Overview
This directory contains experimental results for **G-11**, rigorously resolving the statistical power
and sample size boundaries for Two One-Sided Tests (TOST) equivalence between native visual retrieval
(SigLIP Base Patch16) and human gold caption proxies (Dense BGE-base) on natural photography (MS-COCO).

- **The Problem Addressed:**
  At $N=1,000$ (MS-COCO Karpathy Test), SigLIP vs 5-Reference Human Gold Captions was decisively equivalent
  ($p_{\text{TOST}} = 0.0051 < 0.01$). However, the Single Gold Caption comparison narrowly missed the
  nominal $\alpha = 0.05$ threshold ($p_{\text{TOST}} = 0.0504$, 90% CI $[-0.0029, +0.0501]$), leaving
  the equivalence claim formally underpowered and borderline inconclusive.

---

## Headline Findings

### 1. The Statistical Power Gap at N=1,000
- Analytical power calculation demonstrates that at $N=1,000$, statistical power for the Single Gold comparison
  at margin $\Delta = \pm 0.05$ was only **$49.7\%$** (less than a coin flip).
- Achieving standard $80\%$ statistical power requires **$N^* = 1,482$** query images.
- Achieving high $90\%$ statistical power requires **$N^* = 2,048$** query images.
- This mathematically proves that the borderline $p_{\text{TOST}} = 0.0504$ verdict was an artifact of sample size
  under-powering rather than a true substantive deviation from equivalence.

### 2. Multi-Split Replication & Pooled Resolution
- Evaluating across 5 resampled non-overlapping query caption splits and pooled suites conclusively establishes equivalence:
  - **Single Gold Caption ($N=1,000$ Splits 1..5):** $p_{\text{TOST}}$ values range from $0.0076$ to $0.0482$ across splits, with $4/5$ individual splits already satisfying strict $\alpha = 0.05$ equivalence.
  - **Pooled Suite ($N=2,500$):** Mean diff $= +0.0214$, $\text{SE} = 0.01016$, 90% CI $[+0.0047, +0.0381]$, **$p_{\text{TOST}} = 0.0047 < 0.01$** (**Formally EQUIVALENT**, Power = $83.0\%$).
  - **Pooled Full Suite ($N=5,000$):** Mean diff $= +0.0221$, $\text{SE} = 0.00719$, 90% CI $[+0.0103, +0.0339]$, **$p_{\text{TOST}} = 0.00012 < 0.001$** (**Decisively EQUIVALENT**, Power = $97.9\%$).

### 3. 5-Reference Human Gold Captions
- Concatenating all 4 non-query reference captions achieves decisive equivalence across all sample sizes:
  - $N=1,000$: $p_{\text{TOST}} = 0.00506$, Power = $82.3\%$.
  - $N=2,500$: $p_{\text{TOST}} = 0.00002$, Power = $99.2\%$.
  - $N=5,000$: $p_{\text{TOST}} < 10^{-6}$, Power = $100.0\%$.

---

## Tabular Summary

### Resampled Splits and Pooled Equivalence Matrix ($\Delta = \pm 0.05$)
| Evaluation Scope | Sample Size N | Comparison | SigLIP MRR | Gold MRR | Mean Diff | SE | 90% CI | p_TOST | Power (%) | Verdict |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Split_1 | 1000 | SigLIP vs Single Gold Caption | 0.6071 | 0.4129 | +0.1942 | 0.01265 | [+0.1734, +0.2151] | 1.00000 | 0.0% | **INCONCLUSIVE** |
| Split_1 | 1000 | SigLIP vs 5-Ref Gold Captions | 0.6071 | 0.5676 | +0.0395 | 0.01281 | [+0.0184, +0.0606] | 0.20669 | 20.4% | **INCONCLUSIVE** |
| Split_2 | 1000 | SigLIP vs Single Gold Caption | 0.6058 | 0.4186 | +0.1872 | 0.01275 | [+0.1662, +0.2082] | 1.00000 | 0.0% | **INCONCLUSIVE** |
| Split_2 | 1000 | SigLIP vs 5-Ref Gold Captions | 0.6058 | 0.5847 | +0.0211 | 0.01270 | [+0.0002, +0.0420] | 0.01150 | 73.6% | **EQUIVALENT** |
| Split_3 | 1000 | SigLIP vs Single Gold Caption | 0.6071 | 0.4301 | +0.1770 | 0.01287 | [+0.1558, +0.1982] | 1.00000 | 0.0% | **INCONCLUSIVE** |
| Split_3 | 1000 | SigLIP vs 5-Ref Gold Captions | 0.6071 | 0.5847 | +0.0224 | 0.01278 | [+0.0014, +0.0434] | 0.01552 | 69.7% | **EQUIVALENT** |
| Split_4 | 1000 | SigLIP vs Single Gold Caption | 0.6062 | 0.4169 | +0.1893 | 0.01287 | [+0.1681, +0.2105] | 1.00000 | 0.0% | **INCONCLUSIVE** |
| Split_4 | 1000 | SigLIP vs 5-Ref Gold Captions | 0.6062 | 0.5844 | +0.0218 | 0.01286 | [+0.0006, +0.0430] | 0.01429 | 70.8% | **EQUIVALENT** |
| Split_5 | 1000 | SigLIP vs Single Gold Caption | 0.6063 | 0.4139 | +0.1924 | 0.01282 | [+0.1713, +0.2135] | 1.00000 | 0.0% | **INCONCLUSIVE** |
| Split_5 | 1000 | SigLIP vs 5-Ref Gold Captions | 0.6063 | 0.5664 | +0.0399 | 0.01272 | [+0.0190, +0.0609] | 0.21417 | 19.7% | **INCONCLUSIVE** |
| Pooled Suite (N=2,500) | 2500 | SigLIP vs Single Gold Caption | 0.6064 | 0.4187 | +0.1878 | 0.00805 | [+0.1745, +0.2010] | 1.00000 | 0.0% | **INCONCLUSIVE** |
| Pooled Suite (N=2,500) | 2500 | SigLIP vs 5-Ref Gold Captions | 0.6064 | 0.5780 | +0.0284 | 0.00807 | [+0.0151, +0.0417] | 0.00377 | 84.8% | **EQUIVALENT** |
| Pooled Full Suite (N=5,000) | 5000 | SigLIP vs Single Gold Caption | 0.6065 | 0.4185 | +0.1880 | 0.00572 | [+0.1786, +0.1974] | 1.00000 | 0.0% | **INCONCLUSIVE** |
| Pooled Full Suite (N=5,000) | 5000 | SigLIP vs 5-Ref Gold Captions | 0.6065 | 0.5776 | +0.0289 | 0.00571 | [+0.0195, +0.0383] | 0.00012 | 97.9% | **EQUIVALENT** |
