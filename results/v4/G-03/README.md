# G-03: ColQwen2 and CLIP Visual Retrieval under Optical Noise and Query Shift

## 1. Executive Summary & Provenance Audit
This package resolves Reviewer Gap **G-03** and the user's specific inquiries regarding:
1. `—^\dagger` in Table II (Native visual models under Syn 20% and optical blur)
2. `—*` in Table XI (ColQwen2 under Qwen2.5-7B and human query sets)

### Ground Rule 1 & 2 Verification
A provenance audit of `working/results/v13_colqwen_robustness/colqwen_robustness_summary.csv` revealed it was generated under `Mode: SMOKE` (CPU mock random perturbation). **Under Ground Rule 1 & 2, SMOKE numbers cannot be reported in the manuscript.**

## 2. Table II Verified Measured Results (Stage v11 on Tesla T4)
Physical optical blur was evaluated on raw document images across $N=306$ queries and $N=813$ corpus items:

| Model | Clean Verbatim | Optical Blur Verbatim | Clean Paraphrase | Optical Blur Paraphrase | Optical Blur Answerable | Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ColQwen2-v1.0** | 0.8417 | **0.7207** [0.672, 0.768] | 0.4985 | **0.4449** [0.388, 0.501] | **0.0701** [0.043, 0.098] | **MEASURED** |
| **CLIP ViT-B/32** | 0.0590 | **0.0443** [0.030, 0.059] | 0.0381 | **0.0284** [0.016, 0.041] | **0.0303** [0.017, 0.043] | **MEASURED** |

### Key Physical Insights:
1. **ColQwen2 is NOT Noise-Invariant:** Physical optical blur drops ColQwen2 verbatim MRR from 0.8417 to 0.7207 ($-0.1210$, $p < 10^-4$).
2. **Lexical Dominance under Verbatim Blur:** Under optical blur, Character 3-5g BM25 (0.8276 MRR) decisively outperforms Native ColQwen2 (0.7207 MRR, diff $+0.1069$, $p < 10^-4$).
3. **ColQwen2 Retains Superiority under Paraphrase Blur:** On light paraphrases under blur, ColQwen2 (0.4449) decisively outperforms Word BM25 (0.3497) and Char BM25 (0.2385).
4. **Syn 20% Footnote Justification:** Character permutation operates on string characters, which cannot be directly fed to pixel encoders without an artificial document re-rendering engine. The footnote $\dagger$ is scientifically sound.

## 3. Table XI Execution Protocol for the Three `—*` Dashes
To replace the three dashes in Table XI with 100% authentic measured numbers:
- **Notebook:** `kaggle/kaggle_run_v13_colqwen_robustness.ipynb`
- **Settings:** Accelerator: **GPU T4 x2**, Internet: **ON**, Dataset: `memoryverse-frozen-inputs-v11`.
- **Runtime:** ~12 minutes on Kaggle.
- **Output:** Produces `colqwen_robustness_summary.csv` with `Mode: MEASURED`.
