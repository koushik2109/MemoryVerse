"""
Supabase Vector Store for Audio Memories
Handles insert and retrieval of audio memory records.

Expected Supabase table (audio_memories):
  id               UUID  PRIMARY KEY DEFAULT gen_random_uuid()
  user_id          TEXT  NOT NULL
  file_path        TEXT
  audio_type       TEXT                         -- speech | music | ambient | mixed
  duration         FLOAT
  tempo            FLOAT
  energy           FLOAT
  mood_label       TEXT
  esc50_category   TEXT                         -- for ambient audio
  environment      TEXT                         -- rain | crowd | nature etc
  transcript       TEXT
  description      TEXT
  text_embedding   VECTOR(1024)                 -- bge-m3 dense dim
  created_at       TIMESTAMPTZ DEFAULT now()
"""
from typing import Any, Dict, List, Optional

from supabase import create_client, Client

from ai_engine.rag.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    AUDIO_TABLE,
    TOP_K,
)


def _get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ─── Ingest ────────────────────────────────────────────────────────────────────

def store_audio_memory(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upsert an audio memory record into Supabase.
    `record` must contain all required columns.
    Returns the inserted row.
    """
    client = _get_client()
    resp = client.table(AUDIO_TABLE).insert(record).execute()
    return resp.data[0] if resp.data else {}


# ─── Retrieval ─────────────────────────────────────────────────────────────────

def semantic_search(
    query_embedding: List[float],
    top_k: int = TOP_K,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform ANN vector search in Supabase using pgvector.
    Optionally apply metadata filters (mood_label, audio_type, environment).

    Calls a Supabase RPC function `match_audio_memories` which should be
    defined in Supabase as:

        CREATE OR REPLACE FUNCTION match_audio_memories(
            query_embedding VECTOR(1024),
            match_count     INT,
            filter          JSONB DEFAULT '{}'
        )
        RETURNS TABLE (
            id              UUID,
            user_id         TEXT,
            file_path       TEXT,
            audio_type      TEXT,
            duration        FLOAT,
            tempo           FLOAT,
            energy          FLOAT,
            mood_label      TEXT,
            esc50_category  TEXT,
            environment     TEXT,
            transcript      TEXT,
            description     TEXT,
            created_at      TIMESTAMPTZ,
            similarity      FLOAT
        )
        LANGUAGE plpgsql AS $$
        BEGIN
            RETURN QUERY
            SELECT
                am.id, am.user_id, am.file_path, am.audio_type,
                am.duration, am.tempo, am.energy, am.mood_label,
                am.esc50_category, am.environment,
                am.transcript, am.description, am.created_at,
                1 - (am.text_embedding <=> query_embedding) AS similarity
            FROM audio_memories am
            WHERE
                (filter->>'mood_label' IS NULL OR am.mood_label = filter->>'mood_label')
                AND (filter->>'audio_type' IS NULL OR am.audio_type = filter->>'audio_type')
                AND (filter->>'environment' IS NULL OR am.environment = filter->>'environment')
            ORDER BY am.text_embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
    """
    client = _get_client()
    params: Dict[str, Any] = {
        "query_embedding": query_embedding,
        "match_count": top_k,
        "filter": filters or {},
    }
    resp = client.rpc("match_audio_memories", params).execute()
    return resp.data or []
