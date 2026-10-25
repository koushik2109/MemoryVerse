# Kaggle Execution Guide for MemoryVerse v5 Study

**Manuscript:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Execution Target:** Kaggle GPU Notebook (Nvidia Tesla T4 x2 or P100, 16 GB VRAM)  
**Output Bundle:** `results_v5_final.zip` (and `checkpoint1_native_visual.zip` after Run A)  

---

## 1. Prerequisites & Bundle Preparation

1. **Build the Frozen Input Dataset (Locally):**
   ```bash
   bash kaggle/build_frozen_dataset.sh
   ```
   This generates `kaggle/frozen_inputs_v5.zip` (~480–500 MB) containing all code, frozen benchmark splits, probes, Apple Vision OCR texts (clean and degraded blur), COCO distractors, and SROIE images.

2. **Upload Dataset to Kaggle:**
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) -> Click **New Dataset**.
   - Title: `memoryverse-frozen-inputs-v5` (Slug: `memoryverse-frozen-inputs-v5`).
   - Drag & drop `kaggle/frozen_inputs_v5.zip`.
   - Click **Create**.
   - Kaggle will mount this dataset at `/kaggle/input/memoryverse-frozen-inputs-v5/` (the runner script also auto-discovers the dataset if named differently).

---

## 2. Notebook Configuration on Kaggle

1. **Create Notebook:**
   - Go to [kaggle.com/code](https://www.kaggle.com/code) -> Click **New Notebook**.
   - Or upload `kaggle/kaggle_run_v5.ipynb` directly via **File > Upload Notebook**.

2. **Attach the Dataset:**
   - In the right-hand sidebar under **Data**, click **+ Add Data**.
   - Select **Your Datasets** -> choose `memoryverse-frozen-inputs-v5`.

3. **Configure Environment Settings (CRITICAL):**
   - **Accelerator:** Under **Session options** in the right sidebar, change Accelerator to **GPU T4 x2** (or **GPU P100**).
   - **Internet:** Toggle **Internet ON**. This is required for `pip install -r requirements_kaggle.txt` and downloading open-weights model backbones.
   - **Secrets (Optional):** If using gated HuggingFace checkpoints, add your token under **Add-ons > Secrets** with label `HF_TOKEN`. For standard non-gated ColQwen2 (`vidore/colqwen2-v1.0`), no token is required.

---

## 3. Execution & Workflow Profiles

### Profile A: Dual-Session Review Workflow (Recommended)
- **Run A (Tasks 3, 4, 2, 1):**
  - In Cell 3 of the notebook, set `cmd = [sys.executable, str(runner_path), "--run", "A"]`.
  - Click **Save Version** -> Select **Save & Run All (Commit)** -> Click **Save**.
  - When the background commit completes (~15–20 minutes), open the notebook's **Output** tab and download `checkpoint1_native_visual.zip` (contains `CHECKPOINT1.md`, comparison tables, and effect sizes).
- **Run B (Tasks 6, 5):**
  - After reviewing Checkpoint 1, create a secondary run with `cmd = [sys.executable, str(runner_path), "--run", "B"]` to generate the final decision guide and supporting cost/power analyses.

### Profile B: Complete End-to-End Run (`--run all`)
- To execute all six tasks in one continuous session:
  - In Cell 3 of the notebook, keep `cmd = [sys.executable, str(runner_path), "--run", "all"]`.
  - Click **Save Version** -> **Save & Run All (Commit)**.
  - The script executes Tasks 3, 4, 2, 1, 6, and 5 sequentially, saving `results_v5_partial.zip` after every stage and producing `results_v5_final.zip` at the end.

> [!IMPORTANT]
> **Persistence Notice:** Outputs in `/kaggle/working` are **only preserved** when the notebook is run via **"Save Version" -> "Save & Run All (Commit)"**. Interactive sessions discard generated files upon timeout.

---

## 4. Expected Per-Stage Runtimes and VRAM Estimates

Measured from baseline benchmarking on Nvidia T4 GPU:

| Stage | Task Name | Expected Runtime | Peak VRAM | Key Output Files |
| :--- | :--- | :---: | :---: | :--- |
| **Stage 3** | Query-Side Typo Noise vs OCR Noise | ~1.5 min | < 1.0 GB | `results/v5/query_side_noise_results.csv` |
| **Stage 4** | Paraphrase Robustness (Qwen2.5) | ~3.0 min | ~3.5 GB | `results/v5/paraphrase_llm_comparison.csv` |
| **Stage 2** | Proxy-Quality Axis (COCO VLMs) | ~5.0 min | ~6.5 GB | `results/v5/proxy_quality_axis_results.csv`, `fig_proxy_quality_vs_native.png` |
| **Stage 1** | ColQwen2 Document-Native Visual Retriever | ~8.0 min | ~9.2 GB | `results/v5/native_visual_receipts_results.csv`, `checkpoint1_native_visual.zip` |
| **Stage 6** | Supporting Analyses (Cost, Power, Stack) | ~1.5 min | ~2.0 GB | `results/v5/system_cost_benchmark.csv`, `practical_stack_ablation.csv` |
| **Stage 5** | Automated Decision Guide Matrix | ~0.5 min | < 0.5 GB | `results/v5/decision_guide.csv`, `latex_fragments/table_decision_guide.tex` |
| **Final** | Archive & SHA-256 Sums Verification | ~0.5 min | < 0.2 GB | `results_v5_final.zip` (< 2 GB) |
| **Total** | **Full `--run all` Pipeline** | **~20 min** | **~9.2 GB** | (Safely within Kaggle 9-hour ceiling) |

---

## 5. Artifact Inspection & Verification

Once the commit finishes, the **Output** tab will contain:
- `results_v5_final.zip`: Complete publication archive containing:
  - `results/v5/`: All CSV score files, JSON frozen queries, figures, and failure cases.
  - `latex_fragments/`: `table_decision_guide.tex`, `table_cost.tex`, `table_practical_stack.tex`.
  - `DECISION_GUIDE.md`: Synthesized decision guide text.
  - `PREREG_V5.md`: Committed study pre-registration.
  - `CONFIG_LOG.md`: Forking-paths configuration registry (CFG-001 through CFG-020).
  - `RUNLOG.md`: Detailed timestamps, wall-clock measurements, and tracebacks.
  - `env_report.txt`: GPU model, CUDA driver, and toolchain versions.
  - `SHA256SUMS.txt`: Cryptographic checksums for every artifact.
- `checkpoint1_native_visual.zip`: Run A intermediate checkpoint.
