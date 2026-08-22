"""
Audio RAG Pipeline - FastAPI Router
Endpoints:
  POST /ingest/audio   → upload & index an audio file
  POST /retrieve/audio → semantic/mood-based retrieval
"""
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ai_engine.rag.audio.feature_extractor import extract_features
from ai_engine.rag.audio.transcriber import transcribe
from ai_engine.rag.audio.ambient_classifier import classify_ambient
from ai_engine.rag.audio.embedder import embed_text
from ai_engine.rag.audio.vector_store import store_audio_memory, semantic_search
from ai_engine.rag.audio.intent_extractor import extract_intent
from ai_engine.rag.config import TOP_K

router = APIRouter(prefix="/api/v1", tags=["Audio RAG"])

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


# ─── Request / Response Models ─────────────────────────────────────────────────

class IngestResponse(BaseModel):
    memory_id: str
    audio_type: str
    duration: float
    tempo: float
    mood_label: str
    transcript: Optional[str]
    esc50_category: Optional[str]
    environment: Optional[str]
    message: str


class RetrieveRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = TOP_K


class AudioMemoryResult(BaseModel):
    id: str
    audio_type: str
    duration: float
    mood_label: str
    transcript: Optional[str]
    environment: Optional[str]
    file_path: str
    similarity: float
    created_at: str


class RetrieveResponse(BaseModel):
    query: str
    intent_type: str
    results: List[AudioMemoryResult]


# ─── Helper ────────────────────────────────────────────────────────────────────

def _save_upload(upload: UploadFile) -> str:
    """Save uploaded file to a temp path and return the path."""
    suffix = os.path.splitext(upload.filename or "audio.wav")[1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(upload.file, tmp)
    finally:
        tmp.close()
    return tmp.name


# ─── Ingest Endpoint ───────────────────────────────────────────────────────────

@router.post("/ingest/audio", response_model=IngestResponse)
async def ingest_audio(
    file: UploadFile = File(..., description="Audio file (mp3/wav/m4a)"),
    user_id: str = Form(..., description="User ID from auth"),
    description: Optional[str] = Form(None, description="Optional description / lyrics for music"),
):
    """
    Ingest an audio file into the memory RAG store.

    Pipeline:
      1. Extract acoustic features (librosa)
      2. Classify audio type (speech / music / ambient / mixed)
      3. Transcribe if speech/mixed (Whisper)
      4. Classify ambient environment if ambient (ESC-50)
      5. Build text representation → embed with BGE-M3
      6. Store record in Supabase with vector embedding
    """
    tmp_path = _save_upload(file)

    try:
        # ── Step 1 & 2: Feature extraction + classification ──────────────────
        features = extract_features(tmp_path)

        transcript: Optional[str] = None
        esc50_category: Optional[str] = None
        environment: Optional[str] = None
        text_to_embed: str = ""

        # ── Step 3: Transcription (speech / mixed) ────────────────────────────
        if features.audio_type in ("speech", "mixed"):
            transcript = transcribe(tmp_path)
            text_to_embed = transcript or description or ""

        # ── Step 4: Ambient classification ────────────────────────────────────
        elif features.audio_type == "ambient":
            esc50_category, environment = classify_ambient(tmp_path)
            text_to_embed = description or f"{esc50_category} {environment}"

        # ── Step 5: Music ─────────────────────────────────────────────────────
        elif features.audio_type == "music":
            text_to_embed = description or f"music tempo:{features.tempo:.0f} mood:{features.mood_label}"

        # Fallback: use description or mood
        if not text_to_embed:
            text_to_embed = description or features.mood_label

        # ── Step 5: Embed ──────────────────────────────────────────────────────
        embedding = embed_text(text_to_embed)

        # ── Step 6: Store in Supabase ──────────────────────────────────────────
        memory_id = str(uuid.uuid4())
        record: Dict[str, Any] = {
            "id": memory_id,
            "user_id": user_id,
            "file_path": file.filename or tmp_path,
            "audio_type": features.audio_type,
            "duration": features.duration,
            "tempo": features.tempo,
            "energy": features.energy,
            "mood_label": features.mood_label,
            "esc50_category": esc50_category,
            "environment": environment,
            "transcript": transcript,
            "description": description,
            "text_embedding": embedding,
        }
        store_audio_memory(record)

    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        memory_id=memory_id,
        audio_type=features.audio_type,
        duration=features.duration,
        tempo=features.tempo,
        mood_label=features.mood_label,
        transcript=transcript,
        esc50_category=esc50_category,
        environment=environment,
        message="Audio memory ingested successfully.",
    )


# ─── Retrieve Endpoint ─────────────────────────────────────────────────────────

@router.post("/retrieve/audio", response_model=RetrieveResponse)
async def retrieve_audio(req: RetrieveRequest):
    """
    Retrieve audio memories by natural-language query.

    Pipeline:
      1. Extract intent (Ollama / fallback)
      2. Build metadata filters based on intent
      3. Embed query keywords with BGE-M3
      4. ANN vector search in Supabase with optional filters
      5. Return top_k results with similarity score
    """
    # ── Step 1: Intent extraction ─────────────────────────────────────────────
    intent = extract_intent(req.query)
    intent_type: str = intent.get("intent_type", "general")
    mood_label: Optional[str] = intent.get("mood_label")
    environment: Optional[str] = intent.get("environment")
    keywords: List[str] = intent.get("keywords", [req.query])

    # ── Step 2: Build filters ─────────────────────────────────────────────────
    filters: Dict[str, Any] = {}
    if intent_type == "mood" and mood_label:
        filters["mood_label"] = mood_label
    elif intent_type == "ambient" and environment:
        filters["environment"] = environment

    # ── Step 3: Embed the query ────────────────────────────────────────────────
    combined_query = " ".join(keywords)
    query_embedding = embed_text(combined_query)

    # ── Step 4: Semantic + filtered search ────────────────────────────────────
    raw_results = semantic_search(
        query_embedding=query_embedding,
        top_k=req.top_k,
        filters=filters if filters else None,
    )

    # ── Step 5: Format response ────────────────────────────────────────────────
    results = [
        AudioMemoryResult(
            id=r.get("id", ""),
            audio_type=r.get("audio_type", ""),
            duration=r.get("duration", 0.0),
            mood_label=r.get("mood_label", ""),
            transcript=r.get("transcript"),
            environment=r.get("environment"),
            file_path=r.get("file_path", ""),
            similarity=round(r.get("similarity", 0.0), 4),
            created_at=str(r.get("created_at", "")),
        )
        for r in raw_results
    ]

    return RetrieveResponse(
        query=req.query,
        intent_type=intent_type,
        results=results,
    )
