# KAGGLE GPU EXPERIMENT NOTEBOOKS
## MemoryVerse IJCDS Submission Package
**Directory:** `MEMORYVERSE_FINAL_AUDIT/KAGGLE_NOTEBOOKS/`  

---

## 1. PURPOSE & SCIENTIFIC NEED

In accordance with **Execution Rule 2 (GPU Experiments Must Use Kaggle)**:
- GPU models are never run locally or silently swapped with small CPU approximations.
- Missing live GPU evaluations are encapsulated into self-contained, valid Jupyter notebooks designed for execution on Kaggle GPU instances (Nvidia Tesla T4 / P100 / V100).
- Modeled estimates (such as the 16.20 ms cascade latency) are clearly distinguished from live measured benchmarks.

---

## 2. NOTEBOOK INVENTORY

### 2.1 `KAGGLE_COLQWEN2_CASCADE_BENCHMARK.ipynb`
- **Purpose:** Resolves the critical defect in the original cascade experiment, where the 16.20 ms latency was derived from an analytical mathematical model summing Apple M2 CPU time with Tesla T4 GPU time, rather than a live end-to-end benchmark. This notebook executes live two-stage retrieval: Stage 1 CPU BM25 pre-filtering ($K_1 \in \{10, 20, 50, 100\}$) followed by live GPU ColQwen2 late-interaction MaxSim scoring on the candidate set.
- **Accelerator Required:** GPU (Nvidia Tesla T4 / P100, $\ge 12\text{ GB}$ VRAM).
- **Required Kaggle Datasets:** `memoryverse-data` (or repository attached dataset containing SROIE receipts and MS-COCO distractors).
- **Inputs:** Candidate document images, query strings, and pre-computed lexical indices.
- **Expected Outputs (saved to `/kaggle/working/`):**
  - `execution_summary.json`
  - `cascade_measured_latency.csv`
  - `per_query_latency.csv`
- **Execution Instructions:**
  1. Upload `KAGGLE_COLQWEN2_CASCADE_BENCHMARK.ipynb` to Kaggle.
  2. In Notebook Settings, set **Accelerator** to **GPU T4 x2** or **GPU P100**.
  3. Attach the project dataset: `memoryverse-data`.
  4. Select **Run All** (or Save Version $\to$ Quick Save / Run & Save).
  5. Download output artifacts from the Output panel.
- **Known Limitations:** Evaluates batched candidate scoring; batch size $\le 32$ is recommended on 16GB GPUs to prevent out-of-memory errors during ColQwen2 late-interaction matrix multiplication.

---

### 2.2 `KAGGLE_EXTERNAL_DATASET_GPU_VALIDATION.ipynb`
- **Purpose:** Executes live GPU multi-vector late-interaction inference on the untouched testing splits of the DocVQA (502 pages) and FUNSD (administrative forms) benchmarks. Re-verifies MRR, Hit@1, and Hit@5 under live hardware conditions, ensuring complete transparency regarding ViDoRe pretraining overlap.
- **Accelerator Required:** GPU (Nvidia Tesla T4 / P100, $\ge 14\text{ GB}$ VRAM).
- **Required Kaggle Datasets:** `memoryverse-data` (containing `ai_engine/rag/data/docvqa/` and `funsd/`).
- **Inputs:** High-resolution page images, OCR transcripts, and test question lists.
- **Expected Outputs (saved to `/kaggle/working/`):**
  - `external_validation_summary.json`
  - `docvqa_external_metrics.csv`
  - `funsd_external_metrics.csv`
- **Execution Instructions:**
  1. Upload `KAGGLE_EXTERNAL_DATASET_GPU_VALIDATION.ipynb` to Kaggle.
  2. Set **Accelerator** to **GPU T4 x2**.
  3. Attach `memoryverse-data`.
  4. Click **Run All**.
  5. Inspect output logs to confirm zero split contamination.

---

## 3. COMPLIANCE & REPRODUCIBILITY GUARANTEE
Both notebooks strictly comply with Kaggle commit compatibility:
- Valid `nbformat` v4 JSON structure.
- Sequential execution from Cell 0 to the final summary cell.
- Explicit progress printing; zero reliance on interactive widgets.
- All persistent artifacts saved under `/kaggle/working/`.
- Clear exception handling for missing paths.

