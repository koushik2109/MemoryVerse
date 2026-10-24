-- Preserve the legacy RAG memory records found in the source Supabase project.
-- This table is separate from public.memories used by the current app schema.

CREATE TABLE IF NOT EXISTS public.rag_memories (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL,
    media_url         TEXT,
    media_type        TEXT NOT NULL DEFAULT 'image'
        CHECK (media_type IN ('image', 'video', 'audio')),
    file_hash         TEXT,
    caption           TEXT,
    objects           TEXT[] NOT NULL DEFAULT '{}',
    entities          TEXT[] NOT NULL DEFAULT '{}',
    location          TEXT,
    mood_label        TEXT,
    "timestamp"       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    audio_type        TEXT
        CHECK (audio_type IN ('speech', 'music', 'ambient', 'mixed')),
    tempo             DOUBLE PRECISION,
    duration          DOUBLE PRECISION,
    metadata          JSONB NOT NULL DEFAULT '{}',
    text_embedding    extensions.vector(1024),
    image_embedding   extensions.vector(768),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_entities
    ON public.rag_memories USING gin (entities);
CREATE INDEX IF NOT EXISTS idx_rag_file_hash
    ON public.rag_memories (file_hash) WHERE file_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rag_image_embedding
    ON public.rag_memories USING ivfflat (image_embedding vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_rag_metadata
    ON public.rag_memories USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_rag_objects
    ON public.rag_memories USING gin (objects);
CREATE INDEX IF NOT EXISTS idx_rag_text_embedding
    ON public.rag_memories USING ivfflat (text_embedding vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_rag_timestamp
    ON public.rag_memories ("timestamp" DESC);
CREATE INDEX IF NOT EXISTS idx_rag_user_ts
    ON public.rag_memories (user_id, "timestamp" DESC);
CREATE INDEX IF NOT EXISTS idx_rag_user_type
    ON public.rag_memories (user_id, media_type);

ALTER TABLE public.rag_memories ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "rag_memories_service_all" ON public.rag_memories;
CREATE POLICY "rag_memories_service_all" ON public.rag_memories
    FOR ALL USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "rag_memories_user_select" ON public.rag_memories;
CREATE POLICY "rag_memories_user_select" ON public.rag_memories
    FOR SELECT USING (user_id = auth.uid());
