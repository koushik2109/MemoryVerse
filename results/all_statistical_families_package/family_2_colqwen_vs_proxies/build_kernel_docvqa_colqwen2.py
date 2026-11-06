#!/usr/bin/env python3
"""
kaggle/build_kernel_docvqa_colqwen2.py
======================================
Builds the standalone Kaggle notebook to execute ColQwen2-v1.0
on the official DocVQA benchmark (vidore/docvqa_test_subsampled, N=500).
Generates exact per-query integer ranks, reciprocal ranks, top-10 retrieved doc IDs,
and 10,000-sample bootstrap confidence intervals on Tesla T4 GPU.
"""

import json
from pathlib import Path

dest_dir = Path("kaggle/run_colqwen_docvqa_kernel")
dest_dir.mkdir(parents=True, exist_ok=True)

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(src):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [src + "\n"]})

def add_code(src):
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in src.split("\n")]})

# Cell 0: Title
add_md("""# MemoryVerse: Live ColQwen2-v1.0 DocVQA Per-Query Evaluation
**Hardware Target:** Nvidia Tesla T4 GPU (Kaggle Cloud)
**Benchmark Split:** `vidore/docvqa_test_subsampled` ($N=500$ queries, $N=500$ document pages)
**Goal:** Generate genuine per-query integer target ranks, top-10 document ID rankings, and 10,000-resample bootstrap confidence intervals directly from live `vidore/colqwen2-v1.0` weights on Tesla T4.""")

# Cell 1: Environment Setup
add_code("""# CELL 1: Environment Setup & Dependencies
import os, sys, subprocess
from pathlib import Path

print("=== Setting up Environment & Dependencies ===", flush=True)
working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./working")
working_dir.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["TORCH_HOME"] = "/tmp/hf_cache"

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "colpali-engine>=0.3.0",
    "transformers>=4.45.0",
    "accelerate>=0.34.0",
    "datasets>=2.19.0",
    "tabulate", "scipy"
], check=False)
print("Dependencies installed successfully!", flush=True)""")

# Cell 2: Hardware Diagnostics
add_code("""# CELL 2: GPU Hardware Diagnostics
import torch

print("=== GPU Hardware Diagnostics ===", flush=True)
print(f"PyTorch Version: {torch.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"CUDA Available:  {cuda_avail}")
device = torch.device("cuda:0" if cuda_avail else "cpu")
if cuda_avail:
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    print(f"VRAM Total:      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"BF16 Supported:  {torch.cuda.is_bf16_supported()}")
dtype = torch.bfloat16 if cuda_avail and torch.cuda.is_bf16_supported() else torch.float16
print(f"Selected Torch Precision: {dtype}", flush=True)""")

# Cell 3: Load Dataset
add_code("""# CELL 3: Load Official ViDoRe DocVQA Test Split
from datasets import load_dataset
from PIL import Image

print("=== Loading vidore/docvqa_test_subsampled from Hugging Face ===", flush=True)
ds = load_dataset("vidore/docvqa_test_subsampled", split="test")
print(f"Loaded {len(ds)} evaluation items from ViDoRe DocVQA!", flush=True)

queries = [item["query"] for item in ds]
images = [item["image"].convert("RGB") for item in ds]
image_filenames = [item.get("image_filename", f"doc_{i}") for i, item in enumerate(ds)]
doc_ids = [str(item.get("docId", image_filenames[i])) for i, item in enumerate(ds)]
question_ids = [str(item.get("questionId", f"q_{i}")) for i, item in enumerate(ds)]

print(f"Sample 0 Query:     {queries[0]}")
print(f"Sample 0 Filename:  {image_filenames[0]}")
print(f"Sample 0 Doc ID:    {doc_ids[0]}")
print(f"Total Unique Images: {len(images)}", flush=True)""")

# Cell 4: Load ColQwen2 Model
add_code("""# CELL 4: Load ColQwen2-v1.0 Model and Processor
from colpali_engine.models import ColQwen2, ColQwen2Processor

MODEL_ID = "vidore/colqwen2-v1.0"
print(f"=== Loading {MODEL_ID} on {device} ({dtype}) ===", flush=True)
processor = ColQwen2Processor.from_pretrained(MODEL_ID)
model = ColQwen2.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map="cuda:0").eval()
print("Model loaded successfully!", flush=True)""")

# Cell 5: Live Document Image Encoding
add_code("""# CELL 5: Live GPU Encoding of 500 Document Page Images
import time

print(f"=== Encoding {len(images)} document images on GPU ===", flush=True)
t_start_img = time.time()
batch_size_img = 4
doc_embeddings = []

for b_idx in range(0, len(images), batch_size_img):
    b_imgs = images[b_idx : b_idx + batch_size_img]
    batch_inputs = processor.process_images(b_imgs).to(device)
    with torch.no_grad():
        b_embs = model(**batch_inputs)
        # Store on CPU/GPU as tensor list
        for emb in b_embs:
            doc_embeddings.append(emb.to(dtype).cpu())
    if (b_idx // batch_size_img) % 10 == 0 or b_idx + batch_size_img >= len(images):
        elapsed = time.time() - t_start_img
        print(f"  Encoded {min(b_idx + batch_size_img, len(images))}/{len(images)} images in {elapsed:.1f}s...", flush=True)

img_elapsed = time.time() - t_start_img
print(f"Completed image encoding of {len(images)} document pages in {img_elapsed:.1f}s!", flush=True)

# Save document embeddings
import torch
from pathlib import Path
working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./working")
working_dir.mkdir(parents=True, exist_ok=True)
torch.save(doc_embeddings, working_dir / "colqwen2_docvqa_doc_embeddings.pt")
print(f"Saved doc embeddings tensor list ({len(doc_embeddings)} docs) to disk.", flush=True)""")

# Cell 6: Live Query Text Encoding
add_code("""# CELL 6: Live GPU Encoding of 500 Query Texts
import time
import torch
from pathlib import Path

print(f"=== Encoding {len(queries)} human queries on GPU ===", flush=True)
t_start_q = time.time()
batch_size_q = 8
query_embeddings = []

for b_idx in range(0, len(queries), batch_size_q):
    b_queries = queries[b_idx : b_idx + batch_size_q]
    batch_inputs = processor.process_queries(b_queries).to(device)
    with torch.no_grad():
        b_embs = model(**batch_inputs)
        for emb in b_embs:
            query_embeddings.append(emb.to(dtype).cpu())
    if (b_idx // batch_size_q) % 15 == 0 or b_idx + batch_size_q >= len(queries):
        print(f"  Encoded {min(b_idx + batch_size_q, len(queries))}/{len(queries)} queries...", flush=True)

q_elapsed = time.time() - t_start_q
print(f"Completed query encoding of {len(queries)} queries in {q_elapsed:.1f}s!", flush=True)""")

# Cell 7: Multi-Vector MaxSim Late-Interaction Scoring & Per-Query Extraction
add_code("""# CELL 7: Multi-Vector Late-Interaction MaxSim Scoring & Per-Query Extraction
import math
import numpy as np
import pandas as pd
import torch
from pathlib import Path

print("=== Computing Late-Interaction MaxSim Scores across All Pairs ===", flush=True)
working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./working")
working_dir.mkdir(parents=True, exist_ok=True)

# Pre-load doc embeddings to GPU for accelerated scoring if VRAM permits (~150MB total)
try:
    doc_embeddings_gpu = [d.to(device) for d in doc_embeddings]
    print(f"Pre-loaded all {len(doc_embeddings_gpu)} document embeddings onto {device} for fast scoring.", flush=True)
except Exception as e:
    print(f"GPU cache fallback for doc embeddings: {e}", flush=True)
    doc_embeddings_gpu = None

per_query_results = []
rrs = []
hits1 = []
hits5 = []
hits10 = []
ndcgs10 = []

for i in range(len(queries)):
    q_emb = query_embeddings[i].to(device) # (L_q, D)
    target_idx = i # 1-to-1 ground truth correspondence in test_subsampled
    
    # Compute similarity against all documents
    scores = []
    if doc_embeddings_gpu is not None:
        for d_emb in doc_embeddings_gpu:
            sim = torch.matmul(q_emb, d_emb.T) # (L_q, L_d)
            max_sim = torch.max(sim, dim=1).values # (L_q,)
            score = float(torch.sum(max_sim).cpu())
            scores.append(score)
    else:
        chunk_size = 50
        for c_start in range(0, len(doc_embeddings), chunk_size):
            c_docs = [d.to(device) for d in doc_embeddings[c_start : c_start + chunk_size]]
            for d_emb in c_docs:
                sim = torch.matmul(q_emb, d_emb.T)
                max_sim = torch.max(sim, dim=1).values
                score = float(torch.sum(max_sim).cpu())
                scores.append(score)
            
    scores = np.array(scores)
    ranked_indices = np.argsort(-scores)
    target_rank = int(np.where(ranked_indices == target_idx)[0][0]) + 1
    
    rr = 1.0 / target_rank
    h1 = 1.0 if target_rank == 1 else 0.0
    h5 = 1.0 if target_rank <= 5 else 0.0
    h10 = 1.0 if target_rank <= 10 else 0.0
    ndcg10 = (1.0 / math.log2(target_rank + 1)) if target_rank <= 10 else 0.0
    
    rrs.append(rr)
    hits1.append(h1)
    hits5.append(h5)
    hits10.append(h10)
    ndcgs10.append(ndcg10)
    
    top10_fn = [image_filenames[idx] for idx in ranked_indices[:10]]
    
    per_query_results.append({
        "question_id": question_ids[i],
        "doc_id": doc_ids[i],
        "target_image_filename": image_filenames[i],
        "query": queries[i],
        "target_rank": target_rank,
        "reciprocal_rank": round(rr, 6),
        "hit1": int(h1),
        "hit5": int(h5),
        "hit10": int(h10),
        "ndcg10": round(ndcg10, 6),
        "top10_retrieved_filenames": ";".join(top10_fn)
    })
    
    if (i + 1) % 50 == 0 or i + 1 == len(queries):
        print(f"  Scored {i + 1}/{len(queries)} queries | Running MRR: {np.mean(rrs):.4f} | Running Hit@10: {np.mean(hits10):.4f}", flush=True)

df_per_query = pd.DataFrame(per_query_results)
df_per_query.to_csv(working_dir / "colqwen2_docvqa_per_query.csv", index=False)
print(f"\\nSaved raw per-query results: {working_dir / 'colqwen2_docvqa_per_query.csv'} ({len(df_per_query)} rows)", flush=True)""")

# Cell 8: Bootstrap Confidence Intervals
add_code("""# CELL 8: 10,000 Bootstrap Confidence Interval Resamples
from pathlib import Path
import numpy as np
import pandas as pd

print("=== Computing 10,000 Bootstrap Confidence Intervals ===", flush=True)
working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./working")
working_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
B = 10000
n = len(rrs)

rr_arr = np.array(rrs)
h1_arr = np.array(hits1)
h5_arr = np.array(hits5)
h10_arr = np.array(hits10)
ndcg_arr = np.array(ndcgs10)

def get_ci(arr):
    boots = [float(np.mean(rng.choice(arr, size=n, replace=True))) for _ in range(B)]
    mean_val = float(np.mean(arr))
    se_val = float(np.std(boots))
    return mean_val, se_val, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

mrr_mean, mrr_se, mrr_ci_l, mrr_ci_u = get_ci(rr_arr)
h1_mean, h1_se, h1_ci_l, h1_ci_u = get_ci(h1_arr)
h5_mean, h5_se, h5_ci_l, h5_ci_u = get_ci(h5_arr)
h10_mean, h10_se, h10_ci_l, h10_ci_u = get_ci(h10_arr)
ndcg_mean, ndcg_se, ndcg_ci_l, ndcg_ci_u = get_ci(ndcg_arr)

summary_rows = [{
    "Benchmark": "DocVQA (ViDoRe Subsampled, N=500)",
    "System": "ColQwen2-v1.0 (Live Measured Tesla T4)",
    "N_Queries": n,
    "MRR": round(mrr_mean, 4),
    "MRR_StdError": round(mrr_se, 4),
    "MRR_95%_CI": f"[{mrr_ci_l:.4f}, {mrr_ci_u:.4f}]",
    "Hit@1": round(h1_mean, 4),
    "Hit@5": round(h5_mean, 4),
    "Hit@10": round(h10_mean, 4),
    "nDCG@10": round(ndcg_mean, 4),
    "Mode": "MEASURED_LIVE_GPU"
}]

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(working_dir / "colqwen2_docvqa_summary.csv", index=False)

from tabulate import tabulate
print("\\n" + tabulate(df_summary, headers="keys", tablefmt="github", index=False))""")

# Cell 9: Master Packaging & Export
add_code("""# CELL 9: Master Packaging & Summary Diagnostics Export
import zipfile, json
from pathlib import Path
working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./working")

audit = {
    "model_id": MODEL_ID,
    "benchmark": "vidore/docvqa_test_subsampled",
    "queries_count": len(queries),
    "images_count": len(images),
    "device": torch.cuda.get_device_name(0),
    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "torch_version": torch.__version__,
    "precision": str(dtype),
    "image_encoding_time_s": round(img_elapsed, 1),
    "query_encoding_time_s": round(q_elapsed, 1),
    "measured_mrr": round(mrr_mean, 4),
    "measured_hit10": round(h10_mean, 4)
}

with open(working_dir / "colqwen2_docvqa_audit.json", "w") as f:
    json.dump(audit, f, indent=2)

zip_path = working_dir / "colqwen2_docvqa_results.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(working_dir / "colqwen2_docvqa_per_query.csv", arcname="colqwen2_docvqa_per_query.csv")
    zf.write(working_dir / "colqwen2_docvqa_summary.csv", arcname="colqwen2_docvqa_summary.csv")
    zf.write(working_dir / "colqwen2_docvqa_audit.json", arcname="colqwen2_docvqa_audit.json")

print(f"\\nAll DocVQA ColQwen2 evaluation artifacts packaged to: {zip_path}")
print(f"Archive size: {zip_path.stat().st_size / 1024:.1f} KB")
print("\\n=== EXECUTION SUCCESSFUL ===", flush=True)""")

dest_nb = dest_dir / "kaggle_colqwen2_docvqa.ipynb"
with open(dest_nb, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

# Kernel metadata
meta = {
    "id": "anuragkrishnavennela/colqwen2-robustness-evaluation",
    "title": "colqwen2-robustness-evaluation",
    "code_file": "kaggle_colqwen2_docvqa.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}

with open(dest_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Generated DocVQA ColQwen2 notebook at:", dest_nb)
print("Generated metadata at:", dest_dir / "kernel-metadata.json")

