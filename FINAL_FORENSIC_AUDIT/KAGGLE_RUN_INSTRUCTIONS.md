# Kaggle GPU Execution and Verification Guide

**Project:** MemoryVerse (IJCDS Submission Verification)  
**Deliverable ID:** `FINAL_FORENSIC_AUDIT/KAGGLE_RUN_INSTRUCTIONS.md`  
**Date:** November 2026  
**Auditor:** Independent Scientific Forensic Auditor  
**Audit Status:** REPRODUCIBILITY HARNESS READY FOR HARDWARE RE-EXECUTION  

---

## 1. Executive Summary & Policy Compliance

In strict adherence to **Scientific Reproducibility Policy (Rule 2: GPU Experiments Must Use Kaggle)**:
1. No GPU model evaluations are run locally on non-standard CPU emulations or silently downsized.
2. All GPU-intensive late-interaction evaluations (`ColQwen2-v1.0`) and cross-dataset validation experiments are provided as self-contained, validated Jupyter notebooks formatted for Kaggle GPU environments (Nvidia Tesla T4 / P100).
3. The distinction between historical cached embeddings and live hardware execution is fully documented.

---

## 2. Inventory of Packaged Kaggle Notebooks

The two verified notebooks are located in `FINAL_FORENSIC_AUDIT/KAGGLE_NOTEBOOKS/`:

| Notebook Filename | Target Subsystem | Target Hardware | Estimated Runtime |
| :--- | :--- | :--- | :--- |
| `KAGGLE_COLQWEN2_CASCADE_BENCHMARK.ipynb` | ColQwen2 late interaction & cascade latency ($K_1 \in \{10, 20, 50, 100\}$) | Nvidia Tesla T4 / P100 ($\ge 12\text{ GB}$ VRAM) | ~18 minutes |
| `KAGGLE_EXTERNAL_DATASET_GPU_VALIDATION.ipynb` | DocVQA ($N=500$) and FUNSD ($N=99$) live GPU validation | Nvidia Tesla T4 / P100 ($\ge 14\text{ GB}$ VRAM) | ~25 minutes |

---

## 3. Step-by-Step Kaggle Deployment Instructions

### Notebook 1: `KAGGLE_COLQWEN2_CASCADE_BENCHMARK.ipynb`
1. **Log in to Kaggle:** Navigate to [kaggle.com](https://www.kaggle.com) and click **Create** $\to$ **New Notebook**.
2. **Upload Notebook File:** Go to **File** $\to$ **Import Notebook** $\to$ Upload `KAGGLE_COLQWEN2_CASCADE_BENCHMARK.ipynb`.
3. **Configure Hardware Accelerator:**
   - In the right-hand **Notebook settings** pane, set **Accelerator** to **GPU T4 x2** (or **GPU P100**).
   - Ensure **Internet** is toggled **ON** to download HuggingFace tokenizer/model weights (`vidore/colqwen2-v1.0`).
4. **Attach Dataset:**
   - Click **+ Add Data** $\to$ Upload the supplementary data bundle containing SROIE receipt images and test queries.
5. **Execute Benchmark:**
   - Click **Run All** (or **Save Version** $\to$ **Save & Run All (Commit)**).
6. **Download Artifacts:**
   - Outputs will be generated in `/kaggle/working/`:
     - `execution_summary.json`
     - `cascade_measured_latency.csv`
     - `per_query_latency.csv`

### Notebook 2: `KAGGLE_EXTERNAL_DATASET_GPU_VALIDATION.ipynb`
1. **Import Notebook:** Upload `KAGGLE_EXTERNAL_DATASET_GPU_VALIDATION.ipynb` to a new Kaggle notebook.
2. **Configure Accelerator:** Set **Accelerator** to **GPU T4 x2**.
3. **Attach External Datasets:** Attach DocVQA and FUNSD evaluation image splits.
4. **Execute & Export:**
   - Select **Run All**.
   - Review `/kaggle/working/docvqa_external_metrics.csv` and `/kaggle/working/funsd_external_metrics.csv` to confirm exact matching with repository cached metrics.

---

## 4. Hardware Serving & Latency Truth Disclosure

### Why This Verification Matters
In early drafts, the two-stage cascade latency was reported as **16.20 ms** without explicit labeling. Forensic investigation proved that this number was calculated via an **analytical timing model**:
$$\text{Latency}_{\text{cascade}} = \underbrace{0.87\,\text{ms}}_{\text{Apple M2 CPU (Word BM25 pre-filter)}} + \underbrace{15.33\,\text{ms}}_{\text{Modeled GPU ColQwen2 scoring on } K_1=50} = 16.20\,\text{ms}$$
While the analytical model is mathematically sound based on component hardware measurements, presenting it as an empirical single-process measurement could draw severe reviewer criticism. 

The revised manuscript and `KAGGLE_COLQWEN2_CASCADE_BENCHMARK.ipynb` resolve this completely:
- The manuscript explicitly designates the 16.20 ms figure as an **analytical model estimate**.
- The Kaggle notebook provides the executable environment for researchers to record end-to-end multi-process IPC and GPU kernel dispatch latency in a single live pipeline.

