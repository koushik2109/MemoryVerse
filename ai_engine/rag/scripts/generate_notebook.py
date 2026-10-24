import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"

nb = {
    "cells": [],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

add_md("""# RAMP-RAG: Heterogeneous Quality (H1) and Mixed-Intent Routing (H2) Benchmarks
**Paper:** *Multi-Stage Multimodal RAG with Text-Proxy Hybrid Retrieval*  
**Target:** IJCDS (International Journal of Computing and Digital Systems)  
**Pre-Specified Plan:** Frozen at commit `2e63e9e` (Oct 24, 2026). Seeds 42, 43, 44, strict 50/50 Dev/Test split by item ID, cluster bootstrap (B=2,000 resamples), exact Clopper-Pearson intervals, paired permutation tests with step-down Holm-Bonferroni correction, standardized 3-run latency profiling on warm cache.""")

add_code("""import json
from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown, Image

REPO_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
DATA_DIR = REPO_ROOT / 'ai_engine' / 'rag' / 'data'
RESULTS_JSON = DATA_DIR / 'heterogeneous_revision_results.json'

with open(RESULTS_JSON, encoding='utf-8') as f:
    results = json.load(f)

print('Heterogeneous and mixed routing evaluation results loaded successfully!')
print(f'Timestamp: {results.get("timestamp_utc")} | Commit: {results.get("commit_hash")}')""")

add_md("""## Part 0: Prerequisites & Ablation Audit
### 0.1 BM25 Alone vs. BM25 + Clean Stream (Uniform Noise Sweep)
Testing whether the gain of clean stream fusion over hybrid is simply dropping the corrupted dense stream.""")

add_code("""p01 = results['part0_1_bm25_vs_clean']
rows_p01 = []
for k, d in p01.items():
    noise_pct = k.replace('noise_', '').replace('pct', '%')
    rows_p01.append({
        'Noise': noise_pct,
        'BM25 Alone MRR': d['BM25_alone']['mrr'],
        'BM25 Alone H@10': d['BM25_alone']['hit10'],
        'BM25+Clean MRR': d['BM25_plus_Clean']['mrr'],
        'BM25+Clean H@10': d['BM25_plus_Clean']['hit10'],
        'Diff (BM25 - Clean)': d['diff_mrr_bm25_minus_clean'],
        'Paired Permutation p': d['paired_p_val']
    })
display(pd.DataFrame(rows_p01))""")

add_md("""### 0.2 Baseline Reconciliation: Distractor Pool Size
Reconciling 0%-noise numbers with full-corpus clean results on the same 55 probes.""")

add_code("""p02 = results['part0_2_reconciliation']
print('Explanation:', p02['explanation'])
rows_p02 = [
    {'Evaluation Index': 'Test-Split Distractor Pool (97 items)', **p02['test_split_pool_97_items']},
    {'Evaluation Index': 'Full-Corpus Distractor Pool (193 items)', **p02['full_corpus_pool_193_items']},
]
display(pd.DataFrame(rows_p02))""")

add_md("""### 0.3 MS-COCO Held-Out Test Audit (Genuine CLIP Image Vectors)
Auditing the duplicate 0.3975 MRR artifacts by evaluating with authentic precomputed CLIP ViT-B/32 image vectors.""")

add_code("""p03 = results['part0_3_coco_rerun']
rows_p03 = [{'Method': k, 'MRR': v['mrr'], 'MRR_95CI': str(v['mrr_ci95']), 'Hit@10': v['hit10']} for k, v in p03.items()]
display(pd.DataFrame(rows_p03).sort_values(by='MRR', ascending=False))""")

add_md("""### 0.4 AUROC of Observable Reliability $r_{\\text{ocr}}$
Evaluating discrimination ability of observable $r_{\\text{ocr}}$ in distinguishing corrupted vs. clean documents without oracle labels.""")

add_code("""p04 = results['part0_4_auroc_r_ocr']
for k, auc in p04.items():
    print(f'{k.replace("_", " ")}: Observable r_ocr AUROC = {auc:.4f}')""")

add_md("""---
## Part A: Experiment 1 - Heterogeneous-Quality Corpus (Hypothesis H1)
Corrupting document-side OCR text only ($f_{\\text{bad}} \\in \\{0, 0.1, 0.3, 0.5, 0.7\\}$ with 40% noise, plus Beta continuous). Queries stay clean. Gamma tuned on Dev only.""")

add_code("""exp1 = results['experiment_1_heterogeneous_results']
for cond in ['f_bad_10', 'f_bad_30', 'f_bad_50', 'f_bad_70', 'beta_continuous']:
    print(f'=== Condition: {cond} ===')
    rows_cond = []
    for m, d in exp1[cond].items():
        if m != 'hypothesis_h1_test':
            rows_cond.append({
                'Method': m,
                'Overall MRR': d['mrr'],
                'MRR 95CI': str(d['mrr_ci95']),
                'Hit@10': d['hit10'],
                'Clean Target MRR': d['clean_target_mrr'],
                'Corrupted Target MRR': d['corrupted_target_mrr'],
                'Holm p vs BM25': d['holm_p_vs_bm25'],
                'Holm p vs RRF': d['holm_p_vs_rrf'],
            })
    display(pd.DataFrame(rows_cond).sort_values(by='Overall MRR', ascending=False))

print('\\n=== HYPOTHESIS H1 VERDICT ===')
h1_v = results['hypothesis_h1_verdict']
print('Verdict:', h1_v['verdict'])
print('Detail:', h1_v['detail'])

img_h1 = DATA_DIR / 'mrr_vs_fbad.png'
if img_h1.exists():
    display(Image(filename=str(img_h1)))""")

add_md("""---
## Part B: Experiment 2 - Mixed-Intent Router (Hypothesis H2)
Unified 1,193-item index (COCO + Photos + Documents). Evaluated over 600 held-out test queries across 5 intents (Paraphrase, Scene-Text, Identifier, CJK, Numeric).""")

add_code("""exp2 = results['experiment_2_mixed_routing']
rows_exp2 = []
for s, d in exp2['scores'].items():
    rows_exp2.append({
        'System': s,
        'Overall MRR': d['mrr'],
        'MRR 95CI': str(d['mrr_ci95']),
        'Hit@10': d['hit10'],
        'Paraphrase MRR': d['per_intent_mrr'].get('paraphrase'),
        'Scene-Text MRR': d['per_intent_mrr'].get('scene_text'),
        'Identifier MRR': d['per_intent_mrr'].get('identifier'),
        'CJK MRR': d['per_intent_mrr'].get('cjk'),
        'Numeric MRR': d['per_intent_mrr'].get('numeric')
    })
display(pd.DataFrame(rows_exp2).sort_values(by='Overall MRR', ascending=False))

print('\\n=== Router Classification & Calibration Metrics ===')
rows_rm = []
for r_name, d in exp2['router_classification_metrics'].items():
    rows_rm.append({
        'Router': r_name,
        'Accuracy': f"{d['accuracy']*100:.1f}%",
        'Macro F1': d['macro_f1'],
        'ECE': d['ece'],
        'Oracle Gap Closed': f"{d['oracle_gap_closed_pct']:.1f}%",
        'Holm p vs Best Single': d['holm_p_vs_best_single'],
        'Holm p vs All-Stream RRF': d['holm_p_vs_all_stream_rrf'],
    })
display(pd.DataFrame(rows_rm))

print('\\n=== HYPOTHESIS H2 VERDICT ===')
h2_v = results['hypothesis_h2_verdict']
print('Verdict:', h2_v['verdict'])
print('Detail:', h2_v['detail'])

img_h2 = DATA_DIR / 'router_intent_breakdown.png'
if img_h2.exists():
    display(Image(filename=str(img_h2)))""")

add_md("""---
## Part C: Multi-Encoder Robustness (C2) & Paraphrased Probes (C3)""")

add_code("""print('=== C2: Multi-Encoder Uniform Noise Sweep ===')
c2 = results['part_c2_multi_encoder']
rows_c2 = [{'Noise': k.replace('noise_', '').replace('pct', '%'), **v} for k, v in c2.items()]
display(pd.DataFrame(rows_c2))

print('\\n=== C3: Paraphrased Scene-Text Probes (Verbatim vs. Non-Verbatim) ===')
c3 = results['part_c3_paraphrased_probes']
rows_c3 = [
    {'Probe Condition': 'Verbatim Lexical Overlap', **c3['verbatim_probes_clean']},
    {'Probe Condition': 'Semantic Paraphrase (No Lexical Match)', **c3['paraphrased_probes_clean']},
]
display(pd.DataFrame(rows_c3))""")

add_md("""---
## Standardized Latency Profile (Warm Cache, Mean & P95 over 3 Runs)
Scope includes query encoding, index retrieval, score fusion, and reranking.""")

add_code("""lat = results['latency_profiling']
rows_lat = [{'System': k, 'Mean Latency (ms)': v['mean_ms'], 'P95 Latency (ms)': v['p95_ms']} for k, v in lat.items()]
display(pd.DataFrame(rows_lat).sort_values(by='Mean Latency (ms)'))""")

out_path = REPO_ROOT / "notebooks" / "ramp_rag_clean_stream_eval.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"Jupyter notebook written to {out_path}")
