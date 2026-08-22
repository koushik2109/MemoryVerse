"""
Supabase Vector Store for Video Memories
Handles insert and dual-embedding retrieval.

Expected Supabase table (video_memories):
  id               UUID         PRIMARY KEY DEFAULT gen_random_uuid()
  user_id          TEXT         NOT NULL
  file_path        TEXT
  duration         FLOAT
  fps              FLOAT
  num_keyframes    INT
  auto_caption     TEXT
  image_embedding  VECTOR(768)  -- SigLIP averaged keyframe embedding
  text_embedding   VECTOR(1024) -- BGE-M3 caption embedding
  created_at       TIMESTAMPTZ  DEFAULT now()

Two pgvector RPC functions are used:
  match_video_by_text   → searches text_embedding  (BGE-M3 query)
  match_video_by_image  → searches image_embedding (SigLIP text-encoder query)
"""
from typing import Any, Dict, List, Optional

from supabase import create_client, Client

from ai_engine.rag.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    TOP_K,
)

VIDEO_TABLE = "video_memories"


def _client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ─── Ingest ────────────────────────────────────────────────────────────────────

def store_video_memory(record: Dict[str, Any]) -> Dict[str, Any]:
    """Upsert a video memory record. Returns the inserted row."""
    resp = _client().table(VIDEO_TABLE).insert(record).execute()
    return resp.data[0] if resp.data else {}


# ─── Retrieval ─────────────────────────────────────────────────────────────────

def search_by_text_embedding(
    query_embedding: List[float],
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    ANN search against text_embedding (BGE-M3 / 1024-dim).

    Supabase RPC: match_video_by_text
    See supabase_schema.sql for function definition.
    """
    resp = _client().rpc(
        "match_video_by_text",
        {"query_embedding": query_embedding, "match_count": top_k},
    ).execute()
    return resp.data or []


def search_by_image_embedding(
    query_embedding: List[float],
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    ANN search against image_embedding (SigLIP / 768-dim).

    Supabase RPC: match_video_by_image
    See supabase_schema.sql for function definition.
    """
    resp = _client().rpc(
        "match_video_by_image",
        {"query_embedding": query_embedding, "match_count": top_k},
    ).execute()
    return resp.data or []


def merge_and_deduplicate(
    text_results: List[Dict[str, Any]],
    image_results: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Merge two ranked result lists by taking the max similarity score per id,
    then re-rank and return top_k.
    """
    merged: Dict[str, Dict[str, Any]] = {}

    for r in text_results:
        rid = r["id"]
        merged[rid] = {**r, "similarity": r.get("similarity", 0.0), "match_source": "text"}

    for r in image_results:
        rid = r["id"]
        sim = r.get("similarity", 0.0)
        if rid in merged:
            if sim > merged[rid]["similarity"]:
                merged[rid]["similarity"] = sim
                merged[rid]["match_source"] = "image"
        else:
            merged[rid] = {**r, "similarity": sim, "match_source": "image"}

    sorted_results = sorted(merged.values(), key=lambda x: x["similarity"], reverse=True)
    return sorted_results[:top_k]
