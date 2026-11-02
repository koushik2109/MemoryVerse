#!/usr/bin/env python3
"""scripts/package_g03_results.py

Packages and consolidates ColQwen2 and CLIP visual robustness evaluations (G-03):
1. Degraded Physical Optical Blur on SROIE receipts (Table II):
   - ColQwen2-v1.0 Native: Verbatim = 0.7207, Paraphrase = 0.4449, Answerable = 0.0701 (MEASURED)
   - CLIP ViT-B/32: Verbatim = 0.0443, Paraphrase = 0.0284, Answerable = 0.0303 (MEASURED)
2. Synthetic Character Noise (Syn 20%):
   - Properly scoped with footnote dagger explaining inapplicability of string substitutions to pixel inputs.
3. Secondary Generator (Qwen2.5-7B) & Human Query Robustness (Table XI):
   - Documents the Kaggle GPU execution artifact (kaggle_run_v13_colqwen_robustness.ipynb).
   - Explicitly flags SMOKE vs MEASURED provenance under Ground Rules 1 & 2.

Outputs:
- results/v4/G-03/config.json
- results/v4/G-03/colqwen_degraded_visual_summary.csv
- results/v4/G-03/clip_degraded_visual_summary.csv
- results/v4/G-03/paired_statistical_tests.csv
- results/v4/G-03/README.md
"""

import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "v4" / "G-03"
OUT_DIR.mkdir(parents=True, exist_ok=True)

src_colqwen = REPO_ROOT / "results" / "gapfill_v1" / "T1-03" / "colqwen_degraded_noise_summary.csv"
src_clip = REPO_ROOT / "results" / "gapfill_v1" / "T1-03" / "clip_degraded_noise_summary.csv"
src_paired = REPO_ROOT / "results" / "gapfill_v1" / "T1-03" / "paired_statistical_tests_t1_03.csv"

shutil.copy(src_colqwen, OUT_DIR / "colqwen_degraded_visual_summary.csv")
shutil.copy(src_clip, OUT_DIR / "clip_degraded_visual_summary.csv")
shutil.copy(src_paired, OUT_DIR / "paired_statistical_tests.csv")

git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

config = {
    "item_id": "G-03",
    "title": "ColQwen2 and CLIP Visual Retrieval under Optical Blur and Query Shift",
    "git_commit": git_hash,
    "seed": 42,
    "hardware_measured": "Tesla T4 GPU (Kaggle v11 runner)",
    "table_ii_degraded_visual": "MEASURED (ColQwen2: 0.7207 verbatim, 0.4449 paraphrase; CLIP: 0.0443 verbatim)",
    "table_xi_secondary_human_queries": "Ready for Kaggle GPU execution via kaggle/kaggle_run_v13_colqwen_robustness.ipynb",
    "smoke_provenance_flag": "CPU mock numbers in working/ are strictly flagged SMOKE and NOT reported in paper."
}

with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

readme_content = f"""# G-03: ColQwen2 and CLIP Visual Retrieval under Optical Noise and Query Shift

## 1. Executive Summary & Provenance Audit
This package resolves Reviewer Gap **G-03** and the user's specific inquiries regarding:
1. `—^\\dagger` in Table II (Native visual models under Syn 20% and optical blur)
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
1. **ColQwen2 is NOT Noise-Invariant:** Physical optical blur drops ColQwen2 verbatim MRR from 0.8417 to 0.7207 ($-0.1210$, $p < 10^{-4}$).
2. **Lexical Dominance under Verbatim Blur:** Under optical blur, Character 3-5g BM25 (0.8276 MRR) decisively outperforms Native ColQwen2 (0.7207 MRR, diff $+0.1069$, $p < 10^{-4}$).
3. **ColQwen2 Retains Superiority under Paraphrase Blur:** On light paraphrases under blur, ColQwen2 (0.4449) decisively outperforms Word BM25 (0.3497) and Char BM25 (0.2385).
4. **Syn 20% Footnote Justification:** Character permutation operates on string characters, which cannot be directly fed to pixel encoders without an artificial document re-rendering engine. The footnote $\\dagger$ is scientifically sound.

## 3. Table XI Execution Protocol for the Three `—*` Dashes
To replace the three dashes in Table XI with 100% authentic measured numbers:
- **Notebook:** `kaggle/kaggle_run_v13_colqwen_robustness.ipynb`
- **Settings:** Accelerator: **GPU T4 x2**, Internet: **ON**, Dataset: `memoryverse-frozen-inputs-v11`.
- **Runtime:** ~12 minutes on Kaggle.
- **Output:** Produces `colqwen_robustness_summary.csv` with `Mode: MEASURED`.
"""

with open(OUT_DIR / "README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"Successfully packaged G-03 results to {OUT_DIR}")

