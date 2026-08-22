-- ─────────────────────────────────────────────────
-- Enable pgvector
-- ─────────────────────────────────────────────────
-- (Already present in audio section below, kept here for clarity)

-- ─────────────────────────────────────────────────
-- Unified Memories Table (Images)
-- Used by the PhotoBench / L10 ingestion script
-- ─────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          TEXT          NOT NULL,
    media_type       TEXT          DEFAULT 'image' CHECK (media_type IN ('image', 'video', 'audio')),
    file_path        TEXT,
    file_hash        TEXT          UNIQUE,          -- SHA-256 for dedup
    filename         TEXT,
    album            TEXT,
    -- Semantic tags (from PhotoBench / user metadata)
    caption          TEXT,
    caption_cn       TEXT,
    objects          TEXT,
    entities         TEXT,
    mood             TEXT,
    event            TEXT,
    location         TEXT,
    timestamp_hint   TEXT,
    -- Embeddings
    image_embedding  VECTOR(768),                   -- SigLIP base patch-16
    text_embedding   VECTOR(1024),                  -- BGE-M3
    created_at       TIMESTAMPTZ    DEFAULT now()
);

-- IVFFlat index for image search (SigLIP 768-dim)
CREATE INDEX IF NOT EXISTS memories_image_embedding_idx
    ON memories
    USING ivfflat (image_embedding vector_cosine_ops)
    WITH (lists = 100);

-- IVFFlat index for text search (BGE-M3 1024-dim)
CREATE INDEX IF NOT EXISTS memories_text_embedding_idx
    ON memories
    USING ivfflat (text_embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS memories_user_id_idx     ON memories (user_id);
CREATE INDEX IF NOT EXISTS memories_file_hash_idx   ON memories (file_hash);
CREATE INDEX IF NOT EXISTS memories_media_type_idx  ON memories (media_type);
CREATE INDEX IF NOT EXISTS memories_album_idx       ON memories (album);

-- ─────────────────────────────────────────────────
-- RPC: match_memories_by_text  (BGE-M3 1024-dim)
-- ─────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION match_memories_by_text(
    query_embedding VECTOR(1024),
    match_count     INT,
    filter_user_id  TEXT DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    user_id         TEXT,
    media_type      TEXT,
    file_path       TEXT,
    filename        TEXT,
    album           TEXT,
    caption         TEXT,
    mood            TEXT,
    event           TEXT,
    location        TEXT,
    created_at      TIMESTAMPTZ,
    similarity      FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id, m.user_id, m.media_type, m.file_path,
        m.filename, m.album, m.caption, m.mood, m.event, m.location,
        m.created_at,
        1 - (m.text_embedding <=> query_embedding) AS similarity
    FROM memories m
    WHERE (filter_user_id IS NULL OR m.user_id = filter_user_id)
    ORDER BY m.text_embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ─────────────────────────────────────────────────
-- RPC: match_memories_by_image  (SigLIP 768-dim)
-- ─────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION match_memories_by_image(
    query_embedding VECTOR(768),
    match_count     INT,
    filter_user_id  TEXT DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    user_id         TEXT,
    media_type      TEXT,
    file_path       TEXT,
    filename        TEXT,
    album           TEXT,
    caption         TEXT,
    mood            TEXT,
    event           TEXT,
    location        TEXT,
    created_at      TIMESTAMPTZ,
    similarity      FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id, m.user_id, m.media_type, m.file_path,
        m.filename, m.album, m.caption, m.mood, m.event, m.location,
        m.created_at,
        1 - (m.image_embedding <=> query_embedding) AS similarity
    FROM memories m
    WHERE (filter_user_id IS NULL OR m.user_id = filter_user_id)
    ORDER BY m.image_embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ─────────────────────────────────────────────────
-- Audio Memories Table
-- Run this in your Supabase SQL Editor
-- ─────────────────────────────────────────────────

-- Enable pgvector if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Audio memories table
CREATE TABLE IF NOT EXISTS audio_memories (
    id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          TEXT          NOT NULL,
    file_path        TEXT,
    audio_type       TEXT          CHECK (audio_type IN ('speech', 'music', 'ambient', 'mixed')),
    duration         FLOAT,
    tempo            FLOAT,
    energy           FLOAT,
    mood_label       TEXT,
    esc50_category   TEXT,
    environment      TEXT,
    transcript       TEXT,
    description      TEXT,
    text_embedding   VECTOR(1024),   -- BGE-M3 dense dimension
    created_at       TIMESTAMPTZ    DEFAULT now()
);

-- Index for fast ANN search
CREATE INDEX IF NOT EXISTS audio_memories_embedding_idx
    ON audio_memories
    USING ivfflat (text_embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for user-scoped queries
CREATE INDEX IF NOT EXISTS audio_memories_user_id_idx
    ON audio_memories (user_id);

-- ─────────────────────────────────────────────────
-- RPC: match_audio_memories
-- Called by vector_store.py semantic_search()
-- ─────────────────────────────────────────────────
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
        am.id,
        am.user_id,
        am.file_path,
        am.audio_type,
        am.duration,
        am.tempo,
        am.energy,
        am.mood_label,
        am.esc50_category,
        am.environment,
        am.transcript,
        am.description,
        am.created_at,
        1 - (am.text_embedding <=> query_embedding) AS similarity
    FROM audio_memories am
    WHERE
        (filter->>'mood_label'  IS NULL OR am.mood_label  = filter->>'mood_label')
        AND (filter->>'audio_type' IS NULL OR am.audio_type = filter->>'audio_type')
        AND (filter->>'environment' IS NULL OR am.environment = filter->>'environment')
    ORDER BY am.text_embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
-- ─────────────────────────────────────────────────
-- Video Memories Table
-- ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS video_memories (
    id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          TEXT          NOT NULL,
    file_path        TEXT,
    duration         FLOAT,
    fps              FLOAT,
    num_keyframes    INT,
    auto_caption     TEXT,
    image_embedding  VECTOR(768),    -- SigLIP averaged keyframe embedding
    text_embedding   VECTOR(1024),   -- BGE-M3 caption embedding
    created_at       TIMESTAMPTZ    DEFAULT now()
);

-- IVFFlat index for image_embedding (SigLIP 768-dim)
CREATE INDEX IF NOT EXISTS video_memories_image_embedding_idx
    ON video_memories
    USING ivfflat (image_embedding vector_cosine_ops)
    WITH (lists = 100);

-- IVFFlat index for text_embedding (BGE-M3 1024-dim)
CREATE INDEX IF NOT EXISTS video_memories_text_embedding_idx
    ON video_memories
    USING ivfflat (text_embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS video_memories_user_id_idx
    ON video_memories (user_id);

-- ─────────────────────────────────────────────────
-- RPC: match_video_by_text (BGE-M3 1024-dim)
-- ─────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION match_video_by_text(
    query_embedding VECTOR(1024),
    match_count     INT
)
RETURNS TABLE (
    id              UUID,
    user_id         TEXT,
    file_path       TEXT,
    duration        FLOAT,
    fps             FLOAT,
    num_keyframes   INT,
    auto_caption    TEXT,
    created_at      TIMESTAMPTZ,
    similarity      FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        vm.id, vm.user_id, vm.file_path,
        vm.duration, vm.fps, vm.num_keyframes,
        vm.auto_caption, vm.created_at,
        1 - (vm.text_embedding <=> query_embedding) AS similarity
    FROM video_memories vm
    ORDER BY vm.text_embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ─────────────────────────────────────────────────
-- RPC: match_video_by_image (SigLIP 768-dim)
-- ─────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION match_video_by_image(
    query_embedding VECTOR(768),
    match_count     INT
)
RETURNS TABLE (
    id              UUID,
    user_id         TEXT,
    file_path       TEXT,
    duration        FLOAT,
    fps             FLOAT,
    num_keyframes   INT,
    auto_caption    TEXT,
    created_at      TIMESTAMPTZ,
    similarity      FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        vm.id, vm.user_id, vm.file_path,
        vm.duration, vm.fps, vm.num_keyframes,
        vm.auto_caption, vm.created_at,
        1 - (vm.image_embedding <=> query_embedding) AS similarity
    FROM video_memories vm
    ORDER BY vm.image_embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
