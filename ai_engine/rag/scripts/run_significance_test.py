"""
Paired Bootstrap Significance Testing & Uncertainty Profiling
=============================================================
Calculates 95% confidence intervals and reproducible paired sign-flip
significance p-values on the deterministic unified multimodal query set.
The retrieval metric reported here is HitRate@K: whether any relevant
document appears in the top K, not standard set-based Recall@K.
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

RAG_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = RAG_DIR / "data" / "unified_rag_dataset.jsonl"
OUTPUT_PATH = RAG_DIR / "data" / "significance_test_results.json"
MAX_QUERIES = 500
MAX_IMAGE_QUERIES = 350
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def compute_dcg(relevances: List[int], k: int = 10) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    return dcg

def compute_ndcg(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 10) -> float:
    relevances = [1 if doc_id in ground_truth_ids else 0 for doc_id in retrieved_ids[:k]]
    actual_dcg = compute_dcg(relevances, k)
    ideal_relevances = [1] * min(len(ground_truth_ids), k)
    ideal_dcg = compute_dcg(ideal_relevances, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg

def compute_mrr(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0

def compute_recall(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Compute standard Recall@K over the complete relevant-document set."""
    if not ground_truth_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return len(retrieved_set & ground_truth_ids) / len(ground_truth_ids)


def compute_hit_rate(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Return 1 when any relevant document appears in the top K, else 0."""
    return float(bool(set(retrieved_ids[:k]) & ground_truth_ids))


def ranked_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    """Rank scores with a stable corpus-order tie break."""
    return np.argsort(-scores, kind="stable")[:limit]


def select_eval_queries(
    image_query_map: Dict[str, Set[str]],
    audio_category_map: Dict[str, Set[str]],
) -> Tuple[List[Tuple[str, Set[str], str]], Dict[str, int]]:
    """Select the same deterministic, explicitly capped query mix as the benchmark."""
    rng = np.random.default_rng(RANDOM_SEED)
    image_queries = sorted(image_query_map)
    audio_categories = sorted(audio_category_map)

    audio_limit = min(len(audio_categories), MAX_QUERIES)
    selected_audio_categories = audio_categories[:audio_limit]
    image_limit = min(MAX_IMAGE_QUERIES, MAX_QUERIES - len(selected_audio_categories))
    if len(image_queries) > image_limit:
        selected_image_queries = sorted(
            rng.choice(image_queries, size=image_limit, replace=False).tolist()
        )
    else:
        selected_image_queries = image_queries[:image_limit]

    selected = [
        (query, image_query_map[query], "image")
        for query in selected_image_queries
    ]
    selected.extend(
        (f"{category} sound", audio_category_map[category], "audio")
        for category in selected_audio_categories
    )
    composition = {
        "image": len(selected_image_queries),
        "audio": len(selected_audio_categories),
    }
    return selected, composition


def paired_sign_flip_p_value(
    first: np.ndarray,
    second: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Return an observed mean difference and a valid paired two-sided p-value.

    Sign flipping tests the paired differences under a zero-centered null.  The
    plus-one correction makes the Monte Carlo p-value reproducible and prevents
    zero-valued estimates; the absolute comparison also handles discrete metrics
    and exact ties without producing p-values greater than one.
    """
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    observed = float(np.mean(differences))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, differences.size))
    null_means = np.mean(signs * differences, axis=1)
    extreme = np.count_nonzero(np.abs(null_means) >= abs(observed) - 1e-12)
    p_value = (extreme + 1) / (n_resamples + 1)
    return observed, float(min(1.0, max(0.0, p_value)))

def main():
    print("=" * 70)
    print(f"Running Paired Significance Testing (B={N_BOOTSTRAP}, seed={RANDOM_SEED})")
    print("=" * 70)

    # 1. Load Corpus once.  Use the corpus row as the evaluation identity so
    # basename collisions in older image exports cannot merge documents.
    corpus = []
    image_query_map = defaultdict(set)
    audio_category_map = defaultdict(set)

    with open(DATA_PATH, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            record["doc_idx"] = idx
            corpus.append(record)

            modality = record.get("modality", "")
            doc_id = str(idx)
            if modality == "image":
                queries_en = record.get("metadata", {}).get("queries_en", [])
                for q in queries_en:
                    q_clean = q.strip()
                    if len(q_clean) > 3:
                        image_query_map[q_clean].add(doc_id)
            elif modality == "audio":
                cat = record.get("metadata", {}).get("category", "")
                if cat:
                    audio_category_map[cat].add(doc_id)

    print(f"Total corpus documents: {len(corpus):,}")

    # 2. Query Selection (same policy and seed as run_real_benchmarks.py)
    eval_queries, query_composition = select_eval_queries(image_query_map, audio_category_map)
    print(
        f"Selected {len(eval_queries)} evaluation queries "
        f"({query_composition['image']} Image, {query_composition['audio']} Audio; "
        f"seed={RANDOM_SEED})."
    )

    # 3. Indexing
    print("Building BM25 sparse index...")
    corpus_tokens = [tokenize(doc.get("text_representation", "")) for doc in corpus]
    bm25 = BM25Okapi(corpus_tokens)

    print("Encoding dense representations (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_texts = [doc.get("text_representation", "") for doc in corpus]
    corpus_embeddings = encoder.encode(corpus_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    methods = [
        "BM25",
        "Dense Only",
        "Hybrid Convex (a=0.7)",
        "Hybrid Convex (a=0.2)",
        "Hybrid RRF (k=60)",
        "Multi-Stage",
    ]

    # Per-query metric matrices: N_queries x metric
    per_query = {
        m: {
            "HitRate@1": [],
            "HitRate@5": [],
            "HitRate@10": [],
            "MRR": [],
            "NDCG@10": [],
        }
        for m in methods
    }

    print(f"Evaluating {len(eval_queries)} queries across methods...")
    for q_idx, (q_text, gt_set, modality) in enumerate(eval_queries):
        # BM25
        q_tokens = tokenize(q_text)
        bm25_scores = np.array(bm25.get_scores(q_tokens))
        bm25_top = [str(corpus[i]["doc_idx"]) for i in ranked_indices(bm25_scores, 30)]

        # Dense
        q_embed = encoder.encode(q_text, normalize_embeddings=True)
        dense_scores = np.dot(corpus_embeddings, q_embed)
        dense_top = [str(corpus[i]["doc_idx"]) for i in ranked_indices(dense_scores, 30)]

        # Normalization
        norm_bm25 = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        norm_dense = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)

        # Hybrid Convex 0.7
        hyb07_scores = 0.7 * norm_dense + 0.3 * norm_bm25
        hyb07_top = [str(corpus[i]["doc_idx"]) for i in ranked_indices(hyb07_scores, 30)]

        # Hybrid Convex 0.2 (tuned)
        hyb02_scores = 0.2 * norm_dense + 0.8 * norm_bm25
        hyb02_top = [str(corpus[i]["doc_idx"]) for i in ranked_indices(hyb02_scores, 30)]

        # Hybrid RRF
        rrf_scores = np.zeros(len(corpus))
        k_rrf = 60
        for rank, idx in enumerate(ranked_indices(dense_scores, 100), start=1):
            rrf_scores[idx] += 1.0 / (k_rrf + rank)
        for rank, idx in enumerate(ranked_indices(bm25_scores, 100), start=1):
            rrf_scores[idx] += 1.0 / (k_rrf + rank)
        rrf_indices = ranked_indices(rrf_scores, 30)
        rrf_top = [str(corpus[i]["doc_idx"]) for i in rrf_indices]

        # Multi-Stage (Candidate pool from RRF top 30)
        reranked = []
        query_tokens = set(tokenize(q_text))
        for idx in rrf_indices:
            doc = corpus[idx]
            document_tokens = set(tokenize(doc.get("text_representation", "")))
            exact_overlap = len(query_tokens & document_tokens) / max(1, len(query_tokens))
            cos_sim = float(dense_scores[idx])
            final_score = 0.65 * cos_sim + 0.35 * exact_overlap
            reranked.append((final_score, str(doc["doc_idx"])))
        reranked.sort(key=lambda x: (-x[0], x[1]))
        multistage_top = [x[1] for x in reranked]

        cand_map = {
            "BM25": bm25_top,
            "Dense Only": dense_top,
            "Hybrid Convex (a=0.7)": hyb07_top,
            "Hybrid Convex (a=0.2)": hyb02_top,
            "Hybrid RRF (k=60)": rrf_top,
            "Multi-Stage": multistage_top,
        }

        for m, top_ids in cand_map.items():
            per_query[m]["HitRate@1"].append(compute_hit_rate(top_ids, gt_set, 1))
            per_query[m]["HitRate@5"].append(compute_hit_rate(top_ids, gt_set, 5))
            per_query[m]["HitRate@10"].append(compute_hit_rate(top_ids, gt_set, 10))
            per_query[m]["MRR"].append(compute_mrr(top_ids, gt_set))
            per_query[m]["NDCG@10"].append(compute_ndcg(top_ids, gt_set, 10))

    # Convert to numpy arrays
    N = len(eval_queries)
    if N == 0:
        raise ValueError("No evaluation queries were discovered in the dataset")
    metrics_list = ["HitRate@1", "HitRate@5", "HitRate@10", "MRR", "NDCG@10"]
    data_arrays = {m: {met: np.array(per_query[m][met]) for met in metrics_list} for m in methods}

    # 4. Bootstrap Confidence Intervals (B=1000)
    print(f"\nComputing {N_BOOTSTRAP} bootstrap resamples for 95% Confidence Intervals...")
    bootstrap_rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_indices = bootstrap_rng.integers(0, N, size=(N_BOOTSTRAP, N))
    bootstrap_ci = {}
    for m in methods:
        bootstrap_ci[m] = {}
        for met in metrics_list:
            vals = data_arrays[m][met]
            mean_val = float(np.mean(vals))
            # Resample
            boot_means = np.mean(vals[bootstrap_indices], axis=1)
            ci_low = float(np.percentile(boot_means, 2.5))
            ci_high = float(np.percentile(boot_means, 97.5))
            std_err = float(np.std(boot_means, ddof=1))
            bootstrap_ci[m][met] = {
                "mean": round(mean_val, 4),
                "ci95": [round(ci_low, 4), round(ci_high, 4)],
                "se": round(std_err, 4),
            }

    # 5. Paired sign-flip significance tests against Multi-Stage
    print("\nComputing paired sign-flip significance tests vs. Multi-Stage...")
    p_values = {}
    baseline_methods = ["BM25", "Dense Only", "Hybrid Convex (a=0.7)", "Hybrid RRF (k=60)"]
    significance_rng = np.random.default_rng(RANDOM_SEED + 1)
    for b in baseline_methods:
        p_values[b] = {}
        for met in metrics_list:
            diff_observed, p_val = paired_sign_flip_p_value(
                data_arrays["Multi-Stage"][met],
                data_arrays[b][met],
                N_BOOTSTRAP,
                significance_rng,
            )
            p_values[b][met] = {
                "diff": round(float(diff_observed), 4),
                "p_value": round(float(p_val), 4),
                "significant_05": bool(p_val < 0.05),
            }

    # Print Summary Table
    print("\n" + "=" * 90)
    print("EMPIRICAL BENCHMARK METRICS WITH 95% BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 90)
    print(f"{'Method':<25} | {'HitRate@1 (95% CI)':<22} | {'HitRate@10 (95% CI)':<22} | {'MRR (95% CI)':<22}")
    print("-" * 90)
    for m in methods:
        r1 = bootstrap_ci[m]["HitRate@1"]
        r10 = bootstrap_ci[m]["HitRate@10"]
        mrr = bootstrap_ci[m]["MRR"]
        print(f"{m:<25} | {r1['mean']:.3f} [{r1['ci95'][0]:.3f}, {r1['ci95'][1]:.3f}] | "
              f"{r10['mean']:.3f} [{r10['ci95'][0]:.3f}, {r10['ci95'][1]:.3f}] | "
              f"{mrr['mean']:.3f} [{mrr['ci95'][0]:.3f}, {mrr['ci95'][1]:.3f}]")

    print("\n" + "=" * 80)
    print("PAIRED SIGNIFICANCE TESTING (Multi-Stage vs. Baselines)")
    print("=" * 80)
    for b in baseline_methods:
        print(f"\n--- Multi-Stage vs. {b} ---")
        for met in ["HitRate@1", "HitRate@10", "MRR", "NDCG@10"]:
            info = p_values[b][met]
            sig = "*" if info["significant_05"] else "n.s."
            print(f"  {met:<8}: diff = {info['diff']:+.4f}, p = {info['p_value']:.4f} ({sig})")

    out_data = {
        "dataset_size": len(corpus),
        "dataset_composition": dict(sorted(Counter(doc.get("modality", "unknown") for doc in corpus).items())),
        "evaluation_scope": {
            "proxy_only": True,
            "image_text_source": "metadata.queries_en annotation strings",
            "audio_text_source": "metadata.category ESC-50 label",
            "raw_media_inference": False,
            "document_records": 0,
        },
        "available_query_composition": {
            "image": len(image_query_map),
            "audio": len(audio_category_map),
        },
        "test_queries_count": len(eval_queries),
        "test_query_composition": query_composition,
        "query_selection": {
            "random_seed": RANDOM_SEED,
            "max_queries": MAX_QUERIES,
            "max_image_queries": MAX_IMAGE_QUERIES,
        },
        "retrieval_metric": "HitRate@K (at least one relevant document in top K)",
        "bootstrap": {
            "resamples": N_BOOTSTRAP,
            "random_seed": RANDOM_SEED,
            "confidence_level": 0.95,
        },
        "significance_test": {
            "method": "paired_sign_flip",
            "alternative": "two-sided",
            "resamples": N_BOOTSTRAP,
            "random_seed": RANDOM_SEED + 1,
        },
        "bootstrap_ci": bootstrap_ci,
        "paired_significance": p_values,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults successfully written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
