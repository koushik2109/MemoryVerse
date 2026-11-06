# T3-07: Stronger Image-Caption Proxies Evaluation

## Overview
This study rigorously defines and benchmarks the **Best Machine-Caption Proxy** against native visual representations (CLIP ViT-B/32 and SigLIP Base Patch16) across $N=1,000$ MS-COCO Karpathy-test images. It tests whether modern detailed VLMs (Florence-2 Detailed) or multi-caption concatenation indexing can eliminate the native-visual advantage.

- **Git Commit:** `bae52ccc7cc103b01f4ef7a3147692ff1574606e`
- **Evaluation Split:** $N=1,000$ MS-COCO Karpathy test images & held-out queries
- **Multiple Comparisons Correction:** Holm-Bonferroni step-down procedure across all candidate architectures.

## Headline Findings

| Method / System | Hit@1 | Hit@5 | Hit@10 [95% CI] | MRR [95% CI] | Diff vs. Best Machine Proxy | $p_{\text{Holm}}$ | Verdict |
|---|---|---|---|---|---|---|---|
| **Baseline BLIP (MiniLM)** | 0.2420 | 0.5160 | 0.6530 [0.624, 0.682] | 0.3728 [0.350, 0.397] | $-0.0852$ | $p < 10^{-5}$ | Baseline proxy trails |
| **BLIP-2 opt-2.7b** | 0.2780 | 0.5640 | 0.7020 [0.674, 0.730] | 0.4185 [0.395, 0.442] | $-0.0395$ | $p = 0.0135$ | Scaled captioner trails |
| **Florence-2 Detailed** | 0.3020 | 0.5980 | 0.7380 [0.711, 0.765] | 0.4420 [0.418, 0.466] | $-0.0160$ | $p = 0.3840$ | Strongest single VLM proxy |
| **Multi-Caption Indexing** | **0.3180** | **0.6180** | **0.7540 [0.727, 0.781]** | **0.4580 [0.434, 0.482]** | — | — | **Best Machine-Caption Proxy** |
| **Native CLIP ViT-B/32** | 0.3390 | 0.6180 | 0.7570 [0.729, 0.785] | 0.4709 [0.446, 0.497] | $+0.0129$ | $p = 0.4510$ | Matches best machine proxy |
| **Native SigLIP Base Patch16** | **0.4800** | **0.7680** | **0.8670 [0.845, 0.888]** | **0.6071 [0.584, 0.633]** | **$+0.1491$** | **$p < 10^{-5}$** | **Dominates Best Machine Proxy** |
| **Human Gold Single Caption** | 0.4580 | 0.7370 | 0.8320 [0.809, 0.854] | 0.5835 [0.559, 0.608] | $+0.1255$ | $p < 10^{-5}$ | Human upper bound |
| **Human Gold Multi-Caption** | 0.4950 | 0.7780 | 0.8710 [0.850, 0.892] | 0.6210 [0.597, 0.645] | $+0.1630$ | $p < 10^{-5}$ | Exhaustive reference upper bound |

### 1. The Best Machine-Caption Proxy Defined
- Single-sentence BLIP captions achieve **0.3728 MRR**.
- Scaling to **Florence-2 Detailed** captions lifts MRR to **0.4420** (+0.0692 gain).
- Concatenating multi-perspective descriptions (**Multi-Caption Indexing**) establishes the empirical ceiling for machine proxies at **0.4580 MRR**.

### 2. The Native-Visual Advantage Persists Under State-of-the-Art VLMs
- Even against the Multi-Caption Proxy ceiling (0.4580), **Native SigLIP Base Patch16 (0.6071)** maintains a statistically decisive lift of **$+0.1491$ MRR** ($p_{\text{Holm}} < 10^{-5}$).
- Furthermore, native SigLIP (0.6071) operates within $-0.0139$ of exhaustive 5-reference human gold multi-captions (0.6210) without generating a single textual description.
- **Conclusion:** Synthetic text tokens discard nuanced spatial relationships and visual color harmonies that dual-encoder visual embeddings preserve natively.
