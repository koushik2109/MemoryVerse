# Kaggle Execution Guide for MemoryVerse v9 Study

**Manuscript:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Execution Target:** Kaggle GPU Notebook (Nvidia Tesla T4 x2 or P100, 16 GB VRAM)  
**Output Bundle:** `results_v9_final.zip` (containing `results/v9/`, `SHA256SUMS.txt`, `CHECKPOINT1.md`, `CHECKPOINT2.md`, etc.)  

---

## 1. Prerequisites & Bundle Preparation

1. **Build the Frozen Input Dataset (Locally):**
   ```bash
   bash kaggle/build_frozen_dataset.sh
   ```
   This generates `kaggle/frozen_inputs_v9.zip` (~480–500 MB) containing all code, frozen benchmark splits, probes, Apple Vision OCR texts (clean and degraded blur), Tesseract OCR sweeps, COCO distractors, SROIE images, and results from Phases 0–7.

2. **Upload Dataset to Kaggle:**
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) -> Click **New Dataset**.
   - Title: `memoryverse-frozen-inputs-v9` (Slug: `memoryverse-frozen-inputs-v9`).
   - Drag & drop `kaggle/frozen_inputs_v9.zip`.
   - Click **Create**.
   - Kaggle will mount this dataset at `/kaggle/input/memoryverse-frozen-inputs-v9/` (the runner script also auto-discovers the dataset if named differently).

---

## 2. Notebook Configuration on Kaggle

1. **Create Notebook:**
   - Go to [kaggle.com/code](https://www.kaggle.com/code) -> Click **New Notebook**.
   - In Cell 1:
     ```python
     !pip install -r /kaggle/input/memoryverse-frozen-inputs-v9/kaggle/requirements_kaggle.txt
     ```
   - In Cell 2:
     ```python
     !python /kaggle/input/memoryverse-frozen-inputs-v9/kaggle/kaggle_run_v9.py --run all --resume
     ```

2. **Attach the Dataset:**
   - In the right-hand sidebar under **Data**, click **+ Add Data**.
   - Select **Your Datasets** -> choose `memoryverse-frozen-inputs-v9`.

3. **Configure Environment Settings (CRITICAL):**
   - **Accelerator:** Under **Session options** in the right sidebar, change Accelerator to **GPU T4 x2** (or **GPU P100**).
   - **Internet:** Toggle **Internet ON**. This is required for `pip install -r requirements_kaggle.txt` and downloading open-weights model backbones (`vidore/colqwen2-v1.0`).
   - **Secrets (Optional):** If using gated HuggingFace checkpoints, add your token under **Add-ons > Secrets** with label `HF_TOKEN`. For standard non-gated ColQwen2 (`vidore/colqwen2-v1.0`), no token is required.

---

## 3. Execution & Workflow Profiles

### Profile A: Local CPU Smoke Test
Run quickly on local CPU to verify clean execution before uploading:
```bash
python3 kaggle/kaggle_run_v9.py --smoke --run all
```

### Profile B: Full Resumable Kaggle GPU Execution
Run all stages with resumable `.done` markers and error handling:
```bash
python3 kaggle/kaggle_run_v9.py --run all --resume
```

> [!IMPORTANT]
> **Persistence Notice:** Outputs in `/kaggle/working` are **only preserved** when the notebook is run via **"Save Version" -> "Save & Run All (Commit)"**. Interactive sessions discard generated files upon timeout.

---

## 4. Expected Per-Stage Runtimes and VRAM Estimates

Measured from baseline benchmarking on Nvidia T4 GPU:

| Stage | Task Name | Expected Runtime | Peak VRAM | Key Output Files |
| :--- | :--- | :---: | :---: | :--- |
| **Stage 3** | Phase 3: ColQwen2 Document-Native Retrieval | ~8.0 min | ~9.2 GB | `results/v9/native_visual_receipts_results.csv`, `colqwen_vs_proxies_paired.csv` |
| **Stage 4** | Phase 4: Recognized Benchmark (ViDoRe) | ~6.0 min | ~6.5 GB | `results/v9/vidore_benchmark_results.csv` |
| **Stage 5** | Phase 5: Production Cross-Encoder Stack Ablation | ~2.5 min | ~3.0 GB | `results/v9/practical_stack_ablation.csv`, `results/v9/frozen_recipe_v9.json` |
| **Stage 6** | Phase 6: System Cost & Engineering Analyses | ~1.5 min | ~2.0 GB | `results/v9/system_cost_benchmark.csv` |
| **Final** | Checkpoint Synthesis & Final Archive | ~0.5 min | < 0.2 GB | `CHECKPOINT1.md`, `results_v9_final.zip` (< 2 GB) |
| **Total** | **Full Pipeline** | **~18 min** | **~9.2 GB** | (Safely within Kaggle 9-hour ceiling) |

---

## 5. Artifact Inspection & Verification

Once the commit finishes, the **Output** tab will contain:
- `results_v9_final.zip`: Complete publication archive containing:
  - `results/v9/`: All CSV score files, JSON frozen queries, figures, and failure cases.
  - `CHECKPOINT1.md`: Phase 0 & Phase 1 forensic resolution report.
  - `AUDIT_V9.md`: Table inventory and dataset usage catalog.
  - `CHANGELOG_V9.md`: Comprehensive audit trail of all manuscript modifications.
  - `IEEE_Paper_v9.tex`: Verified manuscript copy.
  - `CONFIG_LOG.md`: Forking-paths configuration registry.
  - `RUNLOG.md`: Detailed timestamps, wall-clock measurements, and tracebacks.
  - `env_report.txt`: GPU model, CUDA driver, and toolchain versions.
  - `SHA256SUMS.txt`: Cryptographic checksums for every artifact.

---

## 6. Version 11 Suite: Degraded Visual Retrieval Benchmark

**Target Output:** `results_v11_final.zip` (containing `colqwen_degraded_eval_v11.csv`, `clip_degraded_eval_v11.csv`, `paired_degraded_visual_v11.csv`, `v11_provenance.txt`)

### Artifacts Delivered
1. **Dataset Archive:** `kaggle/frozen_inputs_v11.zip` (448.96 MB, SHA-256: `afd3e099287a8d5491c6e8d4b4f515f42c45e127d1fd7cbbbfd5160e6b105eff`)
   - Bundles all frozen benchmark splits, queries, clean and optical blur degraded images, and audit records.
2. **Kaggle Notebook:** `kaggle/kaggle_run_v11.ipynb`
   - Complete 10-cell self-contained interactive notebook ready for direct upload to Kaggle.
3. **Standalone Python Script:** `kaggle/kaggle_run_v11_colqwen_degraded.py`
   - Standalone CLI execution script.

### Kaggle Execution Commands
```bash
# In Kaggle Notebook (with Internet ON, GPU T4 x2 or P100):
!python kaggle_run_v11_colqwen_degraded.py --run B

# Or directly open and run all cells in kaggle_run_v11.ipynb
```
- **Expected Runtime:** ~35 min on GPU T4 (~18 min ColQwen2 encoding + MaxSim scoring, ~12 min CLIP image/text feature scoring, ~5 min for 100,000 paired permutation tests).
- **Smoke Test:** `python kaggle_run_v11_colqwen_degraded.py --smoke` (verifies entire statistical and IO pipeline in ~1 second).

---

## 7. Version 13 Suite: ColQwen2 Robustness on Secondary & Human Queries

**Goal:** Evaluates `vidore/colqwen2-v1.0` across the 3 remaining robustness query sets on held-out SROIE receipts to obtain genuine empirical numbers to replace the three `—*` dashes in **Table VIII** (`tab:generator2_robustness`).

### Query Sets Evaluated
1. **Primary Gen.** (`gpt-4o-mini`, $N=306$) — Ground-truth sanity anchor (MRR = 0.4962)
2. **Secondary Gen.** (`Qwen2.5-7B`, $N=306$) — From `results/v5/qwen_paraphrases_v5.json`
3. **Human-Authored Queries** ($N=100$) — From `data/human_queries_200.csv`
4. **Human-Authored Queries** ($N=200$) — From `data/human_queries_200.csv`

### Artifacts Delivered
1. **Interactive Notebook:** `kaggle/kaggle_run_v13_colqwen_robustness.ipynb`
   - Complete 10-cell turnkey Jupyter notebook with built-in embedded fallback queries (can run against ANY existing Kaggle dataset like `frozen-inputs-v11` without requiring a new upload).
2. **Standalone Runner:** `kaggle/kaggle_run_v13_colqwen_robustness.py`
   - Resumable command-line runner with GPU auto-detection and `--smoke` verification.
3. **Automatic Table Updater:** `scripts/update_table_viii_colqwen.py`
   - Automatically parses `colqwen_robustness_summary.csv`, updates Table VIII in `paper_v3/ijcds_body.tex`, recompiles via Tectonic, and verifies the strict 15-page ceiling.

### Kaggle Execution Steps
1. Create a new notebook at [kaggle.com/code](https://www.kaggle.com/code).
2. Right sidebar -> **Session options**:
   - **Accelerator:** `GPU T4 x2` or `GPU P100`
   - **Internet:** `ON`
3. Attach Dataset: Under **Data** -> click **+ Add Data** -> select `memoryverse-frozen-inputs-v11` (or upload `kaggle/frozen_inputs_v11.zip`).
4. Upload `kaggle/kaggle_run_v13_colqwen_robustness.ipynb` and click **Run All** (or run `!python kaggle_run_v13_colqwen_robustness.py --run all`).
5. Execution takes ~12–15 minutes on Tesla T4 GPU.
6. Once complete, download `results_v13_colqwen_final.zip` from `/kaggle/working` into the project root and run:
   ```bash
   unzip results_v13_colqwen_final.zip
   python3 scripts/update_table_viii_colqwen.py
   ```

---

## 7. Stage v14: Master Reproducibility Notebook (Tier A Gapfills)

**Notebook:** `kaggle/kaggle_run_v14_reproducibility.ipynb`  
**Purpose:** End-to-end interactive replication of Tier A high-compute evaluations:
- **G-01:** Modern Text Retrieval Baselines (BGE-M3 1024-dim, SPLADE-v3, ColBERTv2, Char BM25).
- **G-02:** DocVQA Layout & OCR Converter Robustness (Docling/Paddle format vs Native ColQwen2).
- **G-03:** Normalized Single-Hardware Latency Profiling (Batch=1, Tesla T4 GPU).

### Execution Instructions on Kaggle
1. Create a new notebook on [kaggle.com/code](https://www.kaggle.com/code).
2. Under **Session options** (right sidebar):
   - **Accelerator:** `GPU T4 x2`
   - **Internet:** `ON`
3. Upload `kaggle/kaggle_run_v14_reproducibility.ipynb` (or copy/paste cells) and click **Run All**.
4. Execution runs in ~10–15 minutes on a Tesla T4 GPU (or instantly verifies across all pre-computed baseline arrays).
5. All outputs, figures, and CSVs are written to `/kaggle/working/results/v4/`.


