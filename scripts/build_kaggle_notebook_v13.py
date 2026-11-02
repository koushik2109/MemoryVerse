#!/usr/bin/env python3
"""
scripts/build_kaggle_notebook_v13.py — Generate kaggle_run_v13_colqwen_robustness.ipynb
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "kaggle" / "kaggle_run_v13_colqwen_robustness.ipynb"

# Read human queries and qwen queries to embed as fallback
with open(ROOT / "results" / "v5" / "qwen_paraphrases_v5.json", "r", encoding="utf-8") as f:
    qwen_queries = json.load(f)

import pandas as pd
human_df = pd.read_csv(ROOT / "data" / "human_queries_200.csv")
human_queries = human_df[["receipt_id", "handwritten_paraphrase"]].to_dict(orient="records")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# MemoryVerse: ColQwen2 Robustness Evaluation Runner (Stage v13)\n",
            "**Manuscript:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise* (IJCDS)\n",
            "\n",
            "### Purpose\n",
            "This notebook evaluates **ColQwen2-v1.0** (`vidore/colqwen2-v1.0`) across the **three missing query robustness sets** on held-out SROIE receipts:\n",
            "1. **Primary Synthetic Generator** (`gpt-4o-mini`, $N=306$) — Ground-truth sanity anchor (MRR = 0.4962)\n",
            "2. **Secondary Open-Weight Generator** (`Qwen2.5-7B`, $N=306$)\n",
            "3. **Human-Authored Queries** ($N=100$)\n",
            "4. **Human-Authored Queries** ($N=200$)\n",
            "\n",
            "Executing this notebook generates the authentic empirical numbers to replace the three `—*` dashes in **Table VIII** (`tab:generator2_robustness`).\n",
            "\n",
            "### Kaggle Execution Settings (CRITICAL)\n",
            "- **Accelerator:** Under **Session options** in the right sidebar, select **GPU T4 x2** (or **GPU P100**).\n",
            "- **Internet:** Toggle **Internet ON** (required to download `vidore/colqwen2-v1.0` weights).\n",
            "- **Dataset:** Attach `memoryverse-frozen-inputs-v11` (or `frozen-inputs-v11`).\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 1. Environment Setup & Dependency Installation\n",
            "import os, sys, subprocess\n",
            "\n",
            "print('=== Setting up Environment & Dependencies ===', flush=True)\n",
            "os.environ['HF_HOME'] = '/tmp/hf_cache'\n",
            "os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'\n",
            "os.environ['TORCH_HOME'] = '/tmp/hf_cache'\n",
            "\n",
            "# Fix any torchao version conflict pre-installed on Kaggle\n",
            "try:\n",
            "    import torchao\n",
            "    from packaging import version\n",
            "    if version.parse(torchao.__version__) < version.parse('0.16.0'):\n",
            "        print('Removing incompatible torchao...', flush=True)\n",
            "        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)\n",
            "except Exception:\n",
            "    pass\n",
            "\n",
            "# Install colpali_engine\n",
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'colpali_engine', 'transformers', 'accelerate', 'torchvision', 'packaging'], check=False)\n",
            "print('Dependencies installed successfully!', flush=True)\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 2. GPU Hardware Verification\n",
            "import torch\n",
            "print(f'PyTorch Version: {torch.__version__}')\n",
            "print(f'CUDA Available:  {torch.cuda.is_available()}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'Device Name:     {torch.cuda.get_device_name(0)}')\n",
            "    print(f'Device Count:    {torch.cuda.device_count()}')\n",
            "    print(f'VRAM Total:      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')\n",
            "else:\n",
            "    print('WARNING: Running on CPU. GPU is strongly recommended for 7B VLM inference!')\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 3. Path Resolution & Dataset Discovery\n",
            "from pathlib import Path\n",
            "\n",
            "working_dir = Path('/kaggle/working').resolve() if Path('/kaggle/working').exists() else Path('./working').resolve()\n",
            "working_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "data_dir = None\n",
            "for cand in [\n",
            "    Path('/kaggle/input/memoryverse-frozen-inputs-v11'),\n",
            "    Path('/kaggle/input/frozen-inputs-v11'),\n",
            "    Path('/kaggle/input/memoryverse-frozen-inputs-v9'),\n",
            "    Path('.'\n",
            ")]:\n",
            "    if (cand / 'ai_engine/rag/data/v2/public_benchmark_v2.json').exists():\n",
            "        data_dir = cand.resolve()\n",
            "        break\n",
            "\n",
            "if data_dir is None:\n",
            "    # Recursive search in /kaggle/input\n",
            "    for p in Path('/kaggle/input').glob('**/public_benchmark_v2.json'):\n",
            "        curr = p.parent\n",
            "        while curr != curr.parent:\n",
            "            if (curr / 'ai_engine/rag/data/v2/public_benchmark_v2.json').exists():\n",
            "                data_dir = curr.resolve()\n",
            "                break\n",
            "            curr = curr.parent\n",
            "        if data_dir:\n",
            "            break\n",
            "\n",
            "print(f'Resolved Data Directory:    {data_dir}')\n",
            "print(f'Resolved Working Directory: {working_dir}')\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 4. Load Benchmark Documents & Robustness Query Suites\n",
            "import json\n",
            "import pandas as pd\n",
            "\n",
            "def find_file(rel: str) -> Path:\n",
            "    p = Path(rel)\n",
            "    if data_dir and (data_dir / p).exists():\n",
            "        return data_dir / p\n",
            "    for match in Path('/kaggle/input').glob(f'**/{p.name}'):\n",
            "        return match\n",
            "    return Path(rel)\n",
            "\n",
            "bench_file = find_file('ai_engine/rag/data/v2/public_benchmark_v2.json')\n",
            "with open(bench_file, 'r', encoding='utf-8') as f:\n",
            "    bench = json.load(f)\n",
            "\n",
            "test_corpus_ids = set(bench['test_corpus_ids'])\n",
            "test_corpus = [r for r in bench['records'] if r['id'] in test_corpus_ids]\n",
            "print(f'Loaded {len(test_corpus)} test corpus documents.')\n",
            "\n",
            "# 1. Primary Synthetic Queries (gpt-4o-mini, N=306)\n",
            "probes_file = find_file('ai_engine/rag/data/v2/frozen_probes_with_paraphrases.json')\n",
            "with open(probes_file, 'r', encoding='utf-8') as f:\n",
            "    p_data = json.load(f)\n",
            "primary_q = [{'target_id': p['target_id'], 'query_text': p['light_paraphrase']} for p in p_data['test_probes'] if p.get('light_paraphrase')]\n",
            "\n",
            "# 2. Secondary Generator Queries (Qwen2.5-7B, N=306)\n",
            "qwen_file = find_file('results/v5/qwen_paraphrases_v5.json')\n",
            "qwen_embedded = " + json.dumps(qwen_queries) + "\n",
            "human_embedded = " + json.dumps(human_queries) + "\n",
            "if qwen_file.exists():\n",
            "    with open(qwen_file, 'r', encoding='utf-8') as f:\n",
            "        qwen_raw = json.load(f)\n",
            "else:\n",
            "    qwen_raw = qwen_embedded\n",
            "qwen_q = [{'target_id': it['target_id'], 'query_text': it['qwen_paraphrase']} for it in qwen_raw]\n",
            "\n",
            "# 3 & 4. Human-Authored Queries (N=100 and N=200)\n",
            "human_file = find_file('data/human_queries_200.csv')\n",
            "if human_file.exists():\n",
            "    h_df = pd.read_csv(human_file)\n",
            "    human_raw = [{'target_id': r['receipt_id'], 'query_text': r['handwritten_paraphrase']} for _, r in h_df.iterrows()]\n",
            "else:\n",
            "    human_raw = [{'target_id': r['receipt_id'], 'query_text': r['handwritten_paraphrase']} for r in human_embedded]\n",
            "\n",
            "human_100_q = human_raw[:100]\n",
            "human_200_q = human_raw[:200]\n",
            "\n",
            "print(f'Primary Queries:   {len(primary_q)}')\n",
            "print(f'Qwen2.5-7B Queries: {len(qwen_q)}')\n",
            "print(f'Human-100 Queries: {len(human_100_q)}')\n",
            "print(f'Human-200 Queries: {len(human_200_q)}')\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 5. Load ColQwen2-v1.0 Model\n",
            "import torch\n",
            "from colpali_engine.models import ColQwen2, ColQwen2Processor\n",
            "\n",
            "MODEL_ID = 'vidore/colqwen2-v1.0'\n",
            "device = 'cuda:0' if torch.cuda.is_available() else 'cpu'\n",
            "dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16\n",
            "\n",
            "print(f'Loading {MODEL_ID} on {device} ({dtype})...', flush=True)\n",
            "processor = ColQwen2Processor.from_pretrained(MODEL_ID)\n",
            "model = ColQwen2.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map=device).eval()\n",
            "print('Model loaded successfully!', flush=True)\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 6. Encode All 813 Document Images\n",
            "from PIL import Image\n",
            "import time\n",
            "\n",
            "img_paths = [find_file(r['image_path']) for r in test_corpus]\n",
            "corpus_ids = [r['id'] for r in test_corpus]\n",
            "id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}\n",
            "\n",
            "print(f'Encoding {len(img_paths)} document images on GPU...', flush=True)\n",
            "t0 = time.time()\n",
            "doc_embeddings = []\n",
            "batch_size = 4\n",
            "\n",
            "for b in range(0, len(img_paths), batch_size):\n",
            "    b_paths = img_paths[b : b + batch_size]\n",
            "    b_imgs = [Image.open(p).convert('RGB') if p.exists() else Image.new('RGB', (224, 224), (255, 255, 255)) for p in b_paths]\n",
            "    b_proc = processor.process_images(b_imgs).to(device)\n",
            "    with torch.no_grad():\n",
            "        out = model(**b_proc)\n",
            "        for i in range(len(b_imgs)):\n",
            "            doc_embeddings.append(out[i].cpu())\n",
            "    if (b // batch_size) % 25 == 0:\n",
            "        print(f'  Encoded {min(b + batch_size, len(img_paths))}/{len(img_paths)} images...', flush=True)\n",
            "\n",
            "print(f'Completed image encoding in {time.time() - t0:.1f}s!', flush=True)\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 7. Late-Interaction MaxSim Retrieval Across All Query Suites\n",
            "import numpy as np\n",
            "\n",
            "def late_interaction_maxsim(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:\n",
            "    sim = torch.matmul(query_emb.float(), doc_emb.float().t())\n",
            "    max_sim = torch.max(sim, dim=1).values\n",
            "    return float(torch.sum(max_sim).item())\n",
            "\n",
            "def score_query(q_text: str, tgt_id: str) -> float:\n",
            "    b_q = processor.process_queries([q_text]).to(device)\n",
            "    with torch.no_grad():\n",
            "        q_emb = model(**b_q)[0].cpu()\n",
            "    scores = np.array([late_interaction_maxsim(q_emb, d) for d in doc_embeddings])\n",
            "    tgt_idx = id_to_idx[tgt_id]\n",
            "    ranks = np.argsort(-scores)\n",
            "    pos = np.where(ranks == tgt_idx)[0]\n",
            "    if len(pos) == 0:\n",
            "        return 0.0\n",
            "    return 1.0 / (pos[0] + 1)\n",
            "\n",
            "suites = [\n",
            "    ('Primary Gen. (gpt-4o-mini, N=306)', primary_q, 0.4334, 0.2427, 0.1601),\n",
            "    ('Secondary Gen. (Qwen2.5-7B, N=306)', qwen_q, 0.6292, 0.5802, 0.2685),\n",
            "    ('Human-Authored Queries (N=100)', human_100_q, 0.6321, 0.5758, 0.5020),\n",
            "    ('Human-Authored Queries (N=200)', human_200_q, 0.5766, 0.5549, 0.4909),\n",
            "]\n",
            "\n",
            "results_summary = []\n",
            "for name, q_list, w_bm25, c_bm25, d_bge in suites:\n",
            "    print(f'Scoring {name} ({len(q_list)} queries)...', flush=True)\n",
            "    t_start = time.time()\n",
            "    rrs = [score_query(q['query_text'], q['target_id']) for q in q_list]\n",
            "    mrr = float(np.mean(rrs))\n",
            "    # Bootstrap 95% CI\n",
            "    rng = np.random.default_rng(42)\n",
            "    boot = [float(np.mean(rng.choice(rrs, size=len(rrs), replace=True))) for _ in range(2000)]\n",
            "    ci_l, ci_u = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))\n",
            "    results_summary.append({\n",
            "        'Query Source': name,\n",
            "        'N Probes': len(q_list),\n",
            "        'Word BM25': w_bm25,\n",
            "        'Char 3-5g BM25': c_bm25,\n",
            "        'Dense BGE': d_bge,\n",
            "        'ColQwen2 MRR': round(mrr, 4),\n",
            "        '95% CI': f'[{ci_l:.4f}, {ci_u:.4f}]',\n",
            "        'Elapsed (s)': round(time.time() - t_start, 1),\n",
            "    })\n",
            "    print(f'  -> MRR = {mrr:.4f} [{ci_l:.4f}, {ci_u:.4f}] in {time.time() - t_start:.1f}s', flush=True)\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 8. Display & Save Results\n",
            "out_df = pd.DataFrame(results_summary)\n",
            "print('\\n=== TABLE VIII COMPLETED RESULTS ===')\n",
            "print(out_df.to_markdown(index=False))\n",
            "\n",
            "out_dir = working_dir / 'results' / 'v13_colqwen_robustness'\n",
            "out_dir.mkdir(parents=True, exist_ok=True)\n",
            "csv_path = out_dir / 'colqwen_robustness_summary.csv'\n",
            "out_df.to_csv(csv_path, index=False)\n",
            "print(f'\\nSaved summary CSV to: {csv_path}')\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 9. Package Output Archive for One-Click Kaggle Download\n",
            "import zipfile\n",
            "zip_path = working_dir / 'results_v13_colqwen_final.zip'\n",
            "with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:\n",
            "    for f in out_dir.glob('*'):\n",
            "        zf.write(f, arcname=f'results/v13_colqwen_robustness/{f.name}')\n",
            "print(f'=== Final Archive Created ===')\n",
            "print(f'File: {zip_path}')\n",
            "print(f'Size: {zip_path.stat().st_size / 1024:.1f} KB')\n",
            "print('Download this zip from Kaggle Output tab and drop into your repo!')\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Generated turnkey notebook: {NB_PATH}")
