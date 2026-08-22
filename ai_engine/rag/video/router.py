"""
Video RAG Pipeline - FastAPI Router
Endpoints:
  POST /ingest/video   → upload, extract keyframes, embed, caption, store
  POST /retrieve/video → dual-embedding semantic search
"""
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ai_engine.rag.video.keyframe_extractor import extract_keyframes
from ai_engine.rag.video.siglip_embedder import (
    embed_frames,
    embed_text_siglip,
    average_frame_embeddings,
    select_most_distinct_keyframes,
)
from ai_engine.rag.video.captioner import caption_keyframes
from ai_engine.rag.video.vector_store import (
    store_video_memory,
    search_by_text_embedding,
    search_by_image_embedding,
    merge_and_deduplicate,
)
from ai_engine.rag.audio.embedder import embed_text as bge_embed_text  # reuse BGE-M3
from ai_engine.rag.config import TOP_K

router = APIRouter(prefix="/api/v1", tags=["Video RAG"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


# ─── Request / Response Models ─────────────────────────────────────────────────

class IngestVideoResponse(BaseModel):
    memory_id: str
    duration: float
    fps: float
    num_keyframes: int
    auto_caption: str
    message: str


class RetrieveVideoRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = TOP_K


class VideoMemoryResult(BaseModel):
    id: str
    file_path: str
    duration: float
    fps: float
    num_keyframes: int
    auto_caption: str
    similarity: float
    match_source: str          # "text" | "image"
    created_at: str


class RetrieveVideoResponse(BaseModel):
    query: str
    results: List[VideoMemoryResult]


# ─── Helper ────────────────────────────────────────────────────────────────────

def _save_upload(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "video.mp4")[1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(upload.file, tmp)
    finally:
        tmp.close()
    return tmp.name


# ─── Ingest Endpoint ───────────────────────────────────────────────────────────

@router.post("/ingest/video", response_model=IngestVideoResponse)
async def ingest_video(
    file: UploadFile = File(..., description="Video file (mp4/mov/avi/mkv)"),
    user_id: str = Form(..., description="User ID from auth"),
):
    """
    Ingest a video file into the memory RAG store.

    Pipeline:
      1. Extract keyframes every 2 seconds (OpenCV)
      2. Embed each keyframe with SigLIP → average → video image embedding (768-dim)
      3. Select top-3 most distinct keyframes (max-marginal-relevance)
      4. Caption top-3 frames with Qwen3-VL (via Ollama)
      5. Concatenate captions → embed with BGE-M3 → text embedding (1024-dim)
      6. Store both embeddings + metadata in Supabase
    """
    tmp_path = _save_upload(file)

    try:
        # ── Step 1: Extract keyframes ─────────────────────────────────────────
        keyframes, duration, fps, num_keyframes = extract_keyframes(tmp_path, interval_sec=2.0)

        if not keyframes:
            raise HTTPException(status_code=422, detail="No keyframes could be extracted from video.")

        timestamps, frames = zip(*keyframes)
        frames = list(frames)

        # ── Step 2: SigLIP → image embedding ─────────────────────────────────
        frame_embeds = embed_frames(frames)
        image_embedding = average_frame_embeddings(frame_embeds)

        # ── Step 3: Select most distinct keyframes ────────────────────────────
        distinct_indices = select_most_distinct_keyframes(frame_embeds, top_k=3)

        # ── Step 4: Caption with Qwen3-VL ─────────────────────────────────────
        auto_caption = caption_keyframes(frames, distinct_indices)

        # ── Step 5: BGE-M3 → text embedding ──────────────────────────────────
        text_embedding = bge_embed_text(auto_caption)

        # ── Step 6: Store in Supabase ─────────────────────────────────────────
        memory_id = str(uuid.uuid4())
        record: Dict[str, Any] = {
            "id": memory_id,
            "user_id": user_id,
            "file_path": file.filename or tmp_path,
            "duration": duration,
            "fps": fps,
            "num_keyframes": num_keyframes,
            "auto_caption": auto_caption,
            "image_embedding": image_embedding,   # 768-dim (SigLIP)
            "text_embedding": text_embedding,     # 1024-dim (BGE-M3)
        }
        store_video_memory(record)

    finally:
        os.unlink(tmp_path)

    return IngestVideoResponse(
        memory_id=memory_id,
        duration=duration,
        fps=fps,
        num_keyframes=num_keyframes,
        auto_caption=auto_caption,
        message="Video memory ingested successfully.",
    )


# ─── Retrieve Endpoint ─────────────────────────────────────────────────────────

@router.post("/retrieve/video", response_model=RetrieveVideoResponse)
async def retrieve_video(req: RetrieveVideoRequest):
    """
    Retrieve video memories via dual-embedding semantic search.

    Pipeline:
      1. Embed query with BGE-M3      → search text_embedding  (1024-dim)
      2. Embed query with SigLIP text → search image_embedding (768-dim)
      3. Merge + deduplicate by max similarity score
      4. Return top_k ranked results
    """
    # ── Step 1: BGE-M3 text search ───────────────────────────────────────────
    text_query_embed = bge_embed_text(req.query)
    text_results = search_by_text_embedding(text_query_embed, top_k=req.top_k)

    # ── Step 2: SigLIP cross-modal search ────────────────────────────────────
    siglip_query_embed = embed_text_siglip([req.query])[0]
    image_results = search_by_image_embedding(siglip_query_embed, top_k=req.top_k)

    # ── Step 3: Merge and deduplicate ─────────────────────────────────────────
    merged = merge_and_deduplicate(text_results, image_results, top_k=req.top_k)

    # ── Step 4: Format response ────────────────────────────────────────────────
    results = [
        VideoMemoryResult(
            id=r.get("id", ""),
            file_path=r.get("file_path", ""),
            duration=r.get("duration", 0.0),
            fps=r.get("fps", 0.0),
            num_keyframes=r.get("num_keyframes", 0),
            auto_caption=r.get("auto_caption", ""),
            similarity=round(r.get("similarity", 0.0), 4),
            match_source=r.get("match_source", "text"),
            created_at=str(r.get("created_at", "")),
        )
        for r in merged
    ]

    return RetrieveVideoResponse(query=req.query, results=results)
