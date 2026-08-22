"""
RAG Evaluation Script — PhotoBench Album1
==========================================
Evaluates all 5 RAG retrieval strategies using PhotoBench Album1 as ground truth.

RAG Types Evaluated
--------------------
1. Semantic RAG     → caption as query → search text_embedding  → Recall@10
2. Metadata RAG     → event/mood/object tags as filters         → Precision@10
3. Temporal RAG     → date extracted from filename/tag          → Date Accuracy@10
4. Entity RAG       → person/location tags as queries           → Recall@10
5. Multimodal RAG   → image itself as query → search image_embedding → Precision@10

Ground Truth Source
--------------------
PhotoBench album1_validation.json + L10 album1/query.json
Each entry: { query_en, ground_truth: [filenames], Location, Time, Person, Object, Genre }

The `memories` table (populated by ingest_photobench.py) is the search index.

Usage
-----
  python eval_rag.py                          # all 5 evaluations, album1 data
  python eval_rag.py --rag semantic           # single eval type
  python eval_rag.py --top-k 5               # adjust k
  python eval_rag.py --user-id <id>          # filter by user
  python eval_rag.py --offline               # no Supabase; uses local JSONL
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─── Path setup ───────────────────────────────────────────────────────────────
RAG_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(RAG_DIR.parent.parent))

from ai_engine.rag.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, TOP_K
from ai_engine.rag.audio.embedder import embed_text as bge_embed
from ai_engine.rag.video.siglip_embedder import embed_frames, embed_text_siglip
from supabase import create_client

# ─── Data paths ───────────────────────────────────────────────────────────────
ALBUM1_VALIDATION = RAG_DIR / "PhotoBench" / "data" / "validation" / "album1_validation.json"
ALBUM1_TEST       = RAG_DIR / "PhotoBench" / "data" / "test"       / "album1_test.json"
L10_QUERY         = RAG_DIR / "20260822_l10318e6g11p5pnb2" / "album1" / "query.json"
L10_IMAGES_DIR    = RAG_DIR / "20260822_l10318e6g11p5pnb2" / "album1" / "images"
UNIFIED_JSONL     = RAG_DIR / "data" / "unified_rag_dataset.jsonl"

MEMORIES_TABLE    = "memories"

# ─── PhotoBench dimension → semantic role ─────────────────────────────────────
# "fact"      = the query uses a factual (exact) tag
# "cognitive" = the query uses a cognitive/descriptive tag
LOCATION_DIMS = {"Location"}
TIME_DIMS     = {"Time"}
PERSON_DIMS   = {"Person"}
OBJECT_DIMS   = {"Object"}
GENRE_DIMS    = {"Genre"}
MOOD_DIMS     = {"Concept"}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"  [warn] Not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all_queries() -> List[Dict]:
    """Merge PhotoBench album1 validation + L10 album1 query entries."""
    entries: List[Dict] = []

    for path in (ALBUM1_VALIDATION, L10_QUERY):
        data = load_json(path)
        for entry in data:
            # Only keep entries that have at least one ground truth image
            if entry.get("ground_truth"):
                entries.append(entry)

    print(f"Loaded {len(entries)} query entries with ground truth")
    return entries


def load_offline_index() -> Dict[str, Dict]:
    """
    Build a filename → record index from the unified_rag_dataset.jsonl.
    Used in --offline mode (no Supabase required).
    """
    index: Dict[str, Dict] = {}
    if not UNIFIED_JSONL.exists():
        return index
    with open(UNIFIED_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            fname = Path(rec.get("file_path", "")).name.upper()
            if fname:
                index[fname] = rec
    return index


# ══════════════════════════════════════════════════════════════════════════════
# Supabase retrieval helpers
# ══════════════════════════════════════════════════════════════════════════════

def _db():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def supabase_text_search(
    query_embedding: List[float],
    top_k: int,
    user_id: Optional[str] = None,
) -> List[str]:
    """Returns list of filenames from text_embedding ANN search."""
    params = {
        "query_embedding": query_embedding,
        "match_count": top_k,
        "filter_user_id": user_id,
    }
    resp = _db().rpc("match_memories_by_text", params).execute()
    return [r.get("filename", "").upper() for r in (resp.data or [])]


def supabase_image_search(
    query_embedding: List[float],
    top_k: int,
    user_id: Optional[str] = None,
) -> List[str]:
    """Returns list of filenames from image_embedding ANN search."""
    params = {
        "query_embedding": query_embedding,
        "match_count": top_k,
        "filter_user_id": user_id,
    }
    resp = _db().rpc("match_memories_by_image", params).execute()
    return [r.get("filename", "").upper() for r in (resp.data or [])]


def supabase_metadata_filter(
    filters: Dict[str, str],
    top_k: int,
    user_id: Optional[str] = None,
) -> List[str]:
    """
    Filter memories by metadata columns (mood, event, objects, location).
    Returns list of filenames.
    """
    q = _db().table(MEMORIES_TABLE).select("filename")
    if user_id:
        q = q.eq("user_id", user_id)
    for col, val in filters.items():
        q = q.ilike(col, f"%{val}%")  # case-insensitive substring match
    resp = q.limit(top_k).execute()
    return [r.get("filename", "").upper() for r in (resp.data or [])]


# ══════════════════════════════════════════════════════════════════════════════
# Offline retrieval helpers (cosine sim on JSONL)
# ══════════════════════════════════════════════════════════════════════════════

def _cosine(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-8)


def offline_text_search(
    index: Dict[str, Dict],
    query_embedding: List[float],
    top_k: int,
) -> List[str]:
    scored = []
    for fname, rec in index.items():
        emb = rec.get("metadata", {}).get("text_embedding") or []
        if len(emb) == len(query_embedding):
            scored.append((fname, _cosine(query_embedding, emb)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scored[:top_k]]


def offline_image_search(
    index: Dict[str, Dict],
    query_embedding: List[float],
    top_k: int,
) -> List[str]:
    scored = []
    for fname, rec in index.items():
        emb = rec.get("metadata", {}).get("image_embedding") or []
        if len(emb) == len(query_embedding):
            scored.append((fname, _cosine(query_embedding, emb)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scored[:top_k]]


def offline_text_filter(
    index: Dict[str, Dict],
    keywords: List[str],
    top_k: int,
) -> List[str]:
    """Substring match on text_representation."""
    results = []
    for fname, rec in index.items():
        text = rec.get("text_representation", "").lower()
        if any(kw.lower() in text for kw in keywords):
            results.append(fname)
        if len(results) >= top_k:
            break
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Date extraction from filename
# ══════════════════════════════════════════════════════════════════════════════

_DATE_PATTERN = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")

def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract YYYY-MM-DD from filenames like 2018-12-30_132157.jpg"""
    m = _DATE_PATTERN.search(filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def recall_at_k(retrieved: List[str], relevant: Set[str]) -> float:
    """Recall@K = |retrieved ∩ relevant| / |relevant|"""
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved if r.upper() in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str]) -> float:
    """Precision@K = |retrieved ∩ relevant| / K"""
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r.upper() in relevant)
    return hits / len(retrieved)


def date_accuracy(retrieved: List[str], query_date: str) -> float:
    """
    Date Accuracy = fraction of retrieved images whose filename encodes the same date.
    """
    if not retrieved or not query_date:
        return 0.0
    hits = sum(1 for r in retrieved if extract_date_from_filename(r) == query_date)
    return hits / len(retrieved)


# ══════════════════════════════════════════════════════════════════════════════
# Per-RAG Evaluation Functions
# ══════════════════════════════════════════════════════════════════════════════

def eval_semantic_rag(
    queries: List[Dict],
    top_k: int,
    offline: bool,
    offline_index: Dict,
    user_id: Optional[str],
) -> Tuple[float, int]:
    """
    Semantic RAG: embed the query_en caption → text_embedding search → Recall@K
    Skips entries with empty ground_truth.
    """
    scores, count = [], 0
    for entry in queries:
        gt = {f.upper() for f in entry.get("ground_truth", [])}
        if not gt:
            continue
        query_text = entry.get("query_en", "")
        if not query_text:
            continue

        q_embed = bge_embed(query_text)
        if offline:
            retrieved = offline_text_search(offline_index, q_embed, top_k)
        else:
            retrieved = supabase_text_search(q_embed, top_k, user_id)

        scores.append(recall_at_k(retrieved, gt))
        count += 1

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, count


def eval_metadata_rag(
    queries: List[Dict],
    top_k: int,
    offline: bool,
    offline_index: Dict,
    user_id: Optional[str],
) -> Tuple[float, int]:
    """
    Metadata RAG: use Object/Genre/Concept tags as keyword filters → Precision@K
    Picks entries that have at least one non-null dimension.
    """
    scores, count = [], 0

    for entry in queries:
        gt = {f.upper() for f in entry.get("ground_truth", [])}
        if not gt:
            continue

        # Collect metadata keywords from the query text itself
        keywords: List[str] = []
        query_en = entry.get("query_en", "")
        if entry.get("Object") and query_en:
            keywords.append(query_en)
        if entry.get("Genre") and query_en:
            keywords.append(query_en)
        if not keywords:
            continue

        if offline:
            retrieved = offline_text_filter(offline_index, keywords, top_k)
        else:
            # Use ilike substring match on the `objects` / `mood` columns
            filters: Dict[str, str] = {}
            # map first keyword to column best guess
            filters["caption"] = keywords[0]
            retrieved = supabase_metadata_filter(filters, top_k, user_id)

        scores.append(precision_at_k(retrieved, gt))
        count += 1

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, count


def eval_temporal_rag(
    queries: List[Dict],
    top_k: int,
    offline: bool,
    offline_index: Dict,
    user_id: Optional[str],
) -> Tuple[float, int]:
    """
    Temporal RAG: query by date extracted from ground-truth filenames → Date Accuracy@K
    Uses Time-tagged entries where filenames encode dates (e.g., L10 dataset).
    """
    scores, count = [], 0

    for entry in queries:
        if not entry.get("Time"):
            continue
        gt_files = entry.get("ground_truth", [])
        if not gt_files:
            continue

        # Extract date from first ground-truth filename
        ref_date = extract_date_from_filename(gt_files[0])
        if not ref_date:
            # No date in filename; use query text as temporal hint
            query_text = entry.get("query_en", "")
            q_embed = bge_embed(query_text)
            if offline:
                retrieved = offline_text_search(offline_index, q_embed, top_k)
            else:
                retrieved = supabase_text_search(q_embed, top_k, user_id)
            gt = {f.upper() for f in gt_files}
            scores.append(recall_at_k(retrieved, gt))
        else:
            # Embed date string and search
            q_embed = bge_embed(f"photos from {ref_date}")
            if offline:
                retrieved = offline_text_search(offline_index, q_embed, top_k)
            else:
                retrieved = supabase_text_search(q_embed, top_k, user_id)
            scores.append(date_accuracy(retrieved, ref_date))

        count += 1

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, count


def eval_entity_rag(
    queries: List[Dict],
    top_k: int,
    offline: bool,
    offline_index: Dict,
    user_id: Optional[str],
) -> Tuple[float, int]:
    """
    Entity RAG: use Person/Location-tagged query_en text → text_embedding search → Recall@K
    Only evaluates entries with a Person or Location dimension set.
    """
    scores, count = [], 0

    for entry in queries:
        gt = {f.upper() for f in entry.get("ground_truth", [])}
        if not gt:
            continue
        if not (entry.get("Person") or entry.get("Location")):
            continue

        entity_query = entry.get("query_en", "")
        if not entity_query:
            continue

        q_embed = bge_embed(entity_query)
        if offline:
            retrieved = offline_text_search(offline_index, q_embed, top_k)
        else:
            retrieved = supabase_text_search(q_embed, top_k, user_id)

        scores.append(recall_at_k(retrieved, gt))
        count += 1

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, count


def eval_multimodal_rag(
    queries: List[Dict],
    top_k: int,
    offline: bool,
    offline_index: Dict,
    user_id: Optional[str],
) -> Tuple[float, int]:
    """
    Multimodal RAG: embed first ground-truth image with SigLIP → image_embedding search → Precision@K
    Skips entries whose images are not present on disk.
    """
    scores, count = [], 0

    for entry in queries:
        gt = {f.upper() for f in entry.get("ground_truth", [])}
        if not gt:
            continue

        # Find first ground-truth image on disk
        img_path = None
        for fname in entry.get("ground_truth", []):
            candidate = L10_IMAGES_DIR / fname
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue  # image not available locally

        try:
            import cv2
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            img_embed = embed_frames([frame])[0]
        except Exception as e:
            continue

        if offline:
            retrieved = offline_image_search(offline_index, img_embed, top_k)
        else:
            retrieved = supabase_image_search(img_embed, top_k, user_id)

        scores.append(precision_at_k(retrieved, gt))
        count += 1

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, count


# ══════════════════════════════════════════════════════════════════════════════
# Summary Table Printer
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: Dict[str, Dict[str, Any]], top_k: int) -> None:
    COLS = ["RAG Type", "Metric", "Score", "Queries Evaluated", "Time (s)"]
    col_w = [25, 18, 10, 22, 12]

    def row(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, col_w))

    sep = "─" * (sum(col_w) + 2 * (len(col_w) - 1))

    print(f"\n{'═' * len(sep)}")
    print(f"  RAG EVALUATION SUMMARY  ─  PhotoBench Album1  ─  top_k={top_k}")
    print(f"{'═' * len(sep)}")
    print(f"  {row(COLS)}")
    print(f"  {sep}")

    for name, res in results.items():
        score_pct = f"{res['score'] * 100:.2f}%"
        elapsed   = f"{res['elapsed']:.1f}"
        print(f"  {row([name, res['metric'], score_pct, res['count'], elapsed])}")

    print(f"{'═' * len(sep)}\n")

    # Quick verdict
    scores = [v["score"] for v in results.values()]
    best   = max(results, key=lambda k: results[k]["score"])
    worst  = min(results, key=lambda k: results[k]["score"])
    avg    = sum(scores) / len(scores) if scores else 0
    print(f"  Average score across all RAGs : {avg * 100:.2f}%")
    print(f"  Best  : {best}  ({results[best]['score'] * 100:.2f}%)")
    print(f"  Worst : {worst} ({results[worst]['score'] * 100:.2f}%)\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

RAG_CHOICES = ["semantic", "metadata", "temporal", "entity", "multimodal", "all"]

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all 5 RAG types on PhotoBench Album1."
    )
    parser.add_argument(
        "--rag", choices=RAG_CHOICES, default="all",
        help="Which RAG type to evaluate (default: all)"
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Retrieval depth K (default: 10)"
    )
    parser.add_argument(
        "--user-id", default=None,
        help="Filter Supabase memories by user_id (optional)"
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use local unified_rag_dataset.jsonl instead of Supabase"
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Save results to a JSON file"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PhotoBench RAG Evaluator")
    print(f"  mode={'OFFLINE' if args.offline else 'SUPABASE'}  top_k={args.top_k}")
    print(f"{'='*60}\n")

    # Load ground-truth queries
    queries = load_all_queries()
    if not queries:
        print("No queries loaded. Exiting.")
        sys.exit(1)

    # Load offline index if needed
    offline_index = {}
    if args.offline:
        offline_index = load_offline_index()
        print(f"Offline index: {len(offline_index)} records loaded\n")

    eval_map = {
        "semantic":   (eval_semantic_rag,   "Recall@K"),
        "metadata":   (eval_metadata_rag,   "Precision@K"),
        "temporal":   (eval_temporal_rag,   "Date Accuracy@K"),
        "entity":     (eval_entity_rag,     "Recall@K"),
        "multimodal": (eval_multimodal_rag, "Precision@K"),
    }

    run_keys = list(eval_map.keys()) if args.rag == "all" else [args.rag]

    results: Dict[str, Dict[str, Any]] = {}

    for key in run_keys:
        fn, metric = eval_map[key]
        print(f"  ▶  Evaluating [{key.upper()} RAG] ── {metric} ...")
        t0 = time.perf_counter()
        score, count = fn(
            queries      = queries,
            top_k        = args.top_k,
            offline      = args.offline,
            offline_index= offline_index,
            user_id      = args.user_id,
        )
        elapsed = time.perf_counter() - t0
        results[key.capitalize() + " RAG"] = {
            "score":   score,
            "metric":  metric,
            "count":   count,
            "elapsed": elapsed,
        }
        print(f"     {metric}: {score * 100:.2f}%  ({count} queries, {elapsed:.1f}s)\n")

    print_summary(results, top_k=args.top_k)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {k: {**v, "score": round(v["score"], 4)} for k, v in results.items()},
                f, indent=2
            )
        print(f"  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
