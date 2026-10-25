# Kaggle Execution Guide for MemoryVerse v6 Study

**Manuscript:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Execution Target:** Kaggle GPU Notebook (Nvidia Tesla T4 x2 or P100, 16 GB VRAM)  
**Output Bundle:** `results_v6_final.zip` (containing `results/v6/`, `latex_fragments_v6/`, `SHA256SUMS.txt`, `CHECKPOINT2.md`, etc.)  

---

## 1. Prerequisites & Bundle Preparation

1. **Build the Frozen Input Dataset (Locally):**
   ```bash
   bash kaggle/build_frozen_dataset.sh
   ```
   This generates `kaggle/frozen_inputs_v6.zip` (~480–500 MB) containing all code, frozen benchmark splits, probes, Apple Vision OCR texts (clean and degraded blur), Tesseract OCR sweeps, COCO distractors, SROIE images, and results from Phases 0–7.

2. **Upload Dataset to Kaggle:**
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) -> Click **New Dataset**.
   - Title: `memoryverse-frozen-inputs-v6` (Slug: `memoryverse-frozen-inputs-v6`).
   - Drag & drop `kaggle/frozen_inputs_v6.zip`.
   - Click **Create**.
   - Kaggle will mount this dataset at `/kaggle/input/memoryverse-frozen-inputs-v6/` (the runner script also auto-discovers the dataset if named differently).

---

## 2. Notebook Configuration on Kaggle

1. **Create Notebook:**
   - Go to [kaggle.com/code](https://www.kaggle.com/code) -> Click **New Notebook**.
   - Or upload `kaggle/kaggle_run_v6.ipynb` directly via **File > Upload Notebook**.

2. **Attach the Dataset:**
   - In the right-hand sidebar under **Data**, click **+ Add Data**.
   - Select **Your Datasets** -> choose `memoryverse-frozen-inputs-v6`.

3. **Configure Environment Settings (CRITICAL):**
   - **Accelerator:** Under **Session options** in the right sidebar, change Accelerator to **GPU T4 x2** (or **GPU P100**).
   - **Internet:** Toggle **Internet ON**. This is required for `pip install -r requirements_kaggle.txt` and downloading open-weights model backbones (`vidore/colqwen2-v1.0`).
   - **Secrets (Optional):** If using gated HuggingFace checkpoints, add your token under **Add-ons > Secrets** with label `HF_TOKEN`. For standard non-gated ColQwen2 (`vidore/colqwen2-v1.0`), no token is required.

---

## 3. Execution & Workflow Profiles

### Profile A: Standard Run B Execution
- In Cell 3 of the notebook, keep `cmd = [sys.executable, str(runner_path), "--run", "all", "--data_dir", str(dataset_root)]`.
- Click **Save Version** -> Select **Save & Run All (Commit)** -> Click **Save**.
- When the background commit completes (~15–20 minutes), open the notebook's **Output** tab and download `results_v6_final.zip`.

> [!IMPORTANT]
> **Persistence Notice:** Outputs in `/kaggle/working` are **only preserved** when the notebook is run via **"Save Version" -> "Save & Run All (Commit)"**. Interactive sessions discard generated files upon timeout.

---

## 4. Expected Per-Stage Runtimes and VRAM Estimates

Measured from baseline benchmarking on Nvidia T4 GPU:

| Stage | Task Name | Expected Runtime | Peak VRAM | Key Output Files |
| :--- | :--- | :---: | :---: | :--- |
| **Stage 3** | Phase 3: ColQwen2 Document-Native Retrieval | ~8.0 min | ~9.2 GB | `results/v6/native_visual_receipts_results.csv`, `colqwen_vs_proxies_paired.csv` |
| **Stage 4** | Phase 4: Recognized Benchmark (ViDoRe) | ~6.0 min | ~6.5 GB | `results/v6/vidore_benchmark_results.csv` |
| **Stage 5** | Phase 5: Production Cross-Encoder Stack Ablation | ~2.5 min | ~3.0 GB | `results/v6/practical_stack_ablation.csv`, `table_practical_stack.tex` |
| **Stage 6** | Phase 6: System Cost & Engineering Analyses | ~1.5 min | ~2.0 GB | `results/v6/system_cost_benchmark.csv`, `table_cost.tex`, `failure_cases.md` |
| **Final** | Checkpoint 2 Synthesis & Final Archive | ~0.5 min | < 0.2 GB | `CHECKPOINT2.md`, `results_v6_final.zip` (< 2 GB) |
| **Total** | **Full Pipeline** | **~18 min** | **~9.2 GB** | (Safely within Kaggle 9-hour ceiling) |

---

## 5. Artifact Inspection & Verification

Once the commit finishes, the **Output** tab will contain:
- `results_v6_final.zip`: Complete publication archive containing:
  - `results/v6/`: All CSV score files, JSON frozen queries, figures, and failure cases.
  - `latex_fragments_v6/`: All publication-ready LaTeX fragments including abstract, contributions, and tables.
  - `CHECKPOINT1.md`: Problems (a)–(i) resolutions.
  - `CHECKPOINT2.md`: Native-visual vs. text-proxy comparison across physical receipts and recognized benchmark.
  - `PREREG_V6.md`: Committed study pre-registration.
  - `CONFIG_LOG.md`: Forking-paths configuration registry (CFG-001 through CFG-035).
  - `RUNLOG.md`: Detailed timestamps, wall-clock measurements, and tracebacks.
  - `env_report.txt`: GPU model, CUDA driver, and toolchain versions.
  - `SHA256SUMS.txt`: Cryptographic checksums for every artifact.
