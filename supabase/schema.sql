-- ============================================================
--  MemoryVerse — Unified Perfect Schema
--  Run this in your Supabase SQL Editor on a FRESH database.
-- ============================================================

-- 1. EXTENSIONS
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- 2. HELPER FUNCTIONS
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


-- ============================================================
-- 3. TABLES
-- ============================================================

-- PROFILES
CREATE TABLE IF NOT EXISTS public.profiles (
    id              UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email           TEXT NOT NULL,
    full_name       TEXT,
    username        TEXT UNIQUE,
    avatar_url      TEXT,
    bio             TEXT,
    vault_count     INT NOT NULL DEFAULT 0,
    media_count     INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- VAULTS
CREATE TABLE IF NOT EXISTS public.vaults (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id        UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    name            TEXT NOT NULL CHECK (char_length(name) BETWEEN 1 AND 100),
    description     TEXT,
    cover_image_url TEXT,
    is_archived     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- VAULT MEMBERS
CREATE TABLE IF NOT EXISTS public.vault_members (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vault_id    UUID NOT NULL REFERENCES public.vaults(id) ON DELETE CASCADE,
    user_id     UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    role        TEXT NOT NULL DEFAULT 'editor' CHECK (role IN ('owner', 'editor', 'viewer')),
    joined_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(vault_id, user_id)
);

-- SECURITY DEFINER function to bypass RLS to check membership safely
CREATE OR REPLACE FUNCTION public.is_vault_member(v_id UUID)
RETURNS BOOLEAN
LANGUAGE sql SECURITY DEFINER SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM vault_members
    WHERE vault_id = v_id AND user_id = auth.uid()
  );
$$;

-- VAULT INVITATIONS
CREATE TABLE IF NOT EXISTS public.vault_invitations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vault_id    UUID NOT NULL REFERENCES public.vaults(id) ON DELETE CASCADE,
    created_by  UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    invite_code TEXT NOT NULL UNIQUE,
    invite_link TEXT,
    max_uses    INT,
    use_count   INT NOT NULL DEFAULT 0,
    expires_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- MEMORIES (Circular Dependency 1: Created first without cover_media_id FK)
CREATE TABLE IF NOT EXISTS public.memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id        UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    vault_id        UUID REFERENCES public.vaults(id) ON DELETE SET NULL,
    title           TEXT NOT NULL,
    description     TEXT,
    cover_media_id  UUID, -- FK added after media table is created
    memory_date     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    location_name   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- MEDIA (Circular Dependency 2: References memories)
CREATE TABLE IF NOT EXISTS public.media (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id        UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    vault_id        UUID REFERENCES public.vaults(id) ON DELETE SET NULL,
    memory_id       UUID REFERENCES public.memories(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    storage_path    TEXT NOT NULL,
    url             TEXT NOT NULL,
    thumbnail_url   TEXT,
    media_type      TEXT NOT NULL DEFAULT 'image' CHECK (media_type IN ('image', 'video')),
    file_size       BIGINT NOT NULL DEFAULT 0,
    mime_type       TEXT,
    width           INT,
    height          INT,
    duration        INT,         -- seconds for video
    taken_at        TIMESTAMPTZ, -- EXIF timestamp
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    location_name   TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add the missing Foreign Key for memories.cover_media_id
ALTER TABLE public.memories
    ADD CONSTRAINT memories_cover_media_id_fkey 
    FOREIGN KEY (cover_media_id) REFERENCES public.media(id) ON DELETE SET NULL;

-- VIDEO JOBS
CREATE TABLE IF NOT EXISTS public.video_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id       UUID NOT NULL REFERENCES public.memories(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'queued', -- queued, processing, completed, failed
    result_media_id UUID REFERENCES public.media(id) ON DELETE SET NULL,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- AI CONVERSATIONS
CREATE TABLE IF NOT EXISTS public.ai_conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT 'New Conversation',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- AI MESSAGES
CREATE TABLE IF NOT EXISTS public.ai_messages (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id     UUID NOT NULL REFERENCES public.ai_conversations(id) ON DELETE CASCADE,
    role                TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content             TEXT NOT NULL,
    related_memory_ids  UUID[] DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- MEDIA EMBEDDINGS (pgvector)
CREATE TABLE IF NOT EXISTS public.media_embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    media_id    UUID NOT NULL REFERENCES public.media(id) ON DELETE CASCADE UNIQUE,
    embedding   extensions.vector(1536),  -- OpenAI ada-002 dimension
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 4. INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_profiles_email ON public.profiles(email);

CREATE INDEX IF NOT EXISTS idx_vaults_owner_id   ON public.vaults(owner_id);
CREATE INDEX IF NOT EXISTS idx_vaults_updated_at ON public.vaults(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_vault_members_vault_id ON public.vault_members(vault_id);
CREATE INDEX IF NOT EXISTS idx_vault_members_user_id  ON public.vault_members(user_id);

CREATE INDEX IF NOT EXISTS idx_vault_invitations_code ON public.vault_invitations(invite_code);

CREATE INDEX IF NOT EXISTS idx_memories_owner_id   ON public.memories(owner_id);
CREATE INDEX IF NOT EXISTS idx_memories_vault_id   ON public.memories(vault_id);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON public.memories(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_media_owner_id      ON public.media(owner_id);
CREATE INDEX IF NOT EXISTS idx_media_vault_id      ON public.media(vault_id);
CREATE INDEX IF NOT EXISTS idx_media_memory_id     ON public.media(memory_id);
CREATE INDEX IF NOT EXISTS idx_media_created_at    ON public.media(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_media_taken_at      ON public.media(taken_at DESC);
CREATE INDEX IF NOT EXISTS idx_media_type          ON public.media(media_type);

CREATE INDEX IF NOT EXISTS idx_video_jobs_user_id   ON public.video_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_video_jobs_memory_id ON public.video_jobs(memory_id);

CREATE INDEX IF NOT EXISTS idx_ai_conversations_user_id ON public.ai_conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_messages_conversation_id ON public.ai_messages(conversation_id);

CREATE INDEX IF NOT EXISTS idx_media_embeddings_media_id ON public.media_embeddings(media_id);

-- ============================================================
-- 5. TRIGGERS
-- ============================================================

-- updated_at triggers
DROP TRIGGER IF EXISTS set_profiles_updated_at ON public.profiles;
CREATE TRIGGER set_profiles_updated_at BEFORE UPDATE ON public.profiles FOR EACH ROW EXECUTE PROCEDURE public.set_updated_at();

DROP TRIGGER IF EXISTS set_vaults_updated_at ON public.vaults;
CREATE TRIGGER set_vaults_updated_at BEFORE UPDATE ON public.vaults FOR EACH ROW EXECUTE PROCEDURE public.set_updated_at();

DROP TRIGGER IF EXISTS set_memories_updated_at ON public.memories;
CREATE TRIGGER set_memories_updated_at BEFORE UPDATE ON public.memories FOR EACH ROW EXECUTE PROCEDURE public.set_updated_at();

DROP TRIGGER IF EXISTS set_video_jobs_updated_at ON public.video_jobs;
CREATE TRIGGER set_video_jobs_updated_at BEFORE UPDATE ON public.video_jobs FOR EACH ROW EXECUTE PROCEDURE public.set_updated_at();

DROP TRIGGER IF EXISTS set_ai_conversations_updated_at ON public.ai_conversations;
CREATE TRIGGER set_ai_conversations_updated_at BEFORE UPDATE ON public.ai_conversations FOR EACH ROW EXECUTE PROCEDURE public.set_updated_at();

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', split_part(NEW.email, '@', 1))
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created AFTER INSERT ON auth.users FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();

-- Update vault_count trigger
CREATE OR REPLACE FUNCTION public.update_profile_vault_count()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE public.profiles SET vault_count = vault_count + 1 WHERE id = NEW.user_id AND NEW.role = 'owner';
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE public.profiles SET vault_count = GREATEST(0, vault_count - 1) WHERE id = OLD.user_id AND OLD.role = 'owner';
    END IF;
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS trg_vault_count ON public.vault_members;
CREATE TRIGGER trg_vault_count AFTER INSERT OR DELETE ON public.vault_members FOR EACH ROW EXECUTE PROCEDURE public.update_profile_vault_count();

-- Update media_count trigger
CREATE OR REPLACE FUNCTION public.update_profile_media_count()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE public.profiles SET media_count = media_count + 1 WHERE id = NEW.owner_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE public.profiles SET media_count = GREATEST(0, media_count - 1) WHERE id = OLD.owner_id;
    END IF;
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS trg_media_count ON public.media;
CREATE TRIGGER trg_media_count AFTER INSERT OR DELETE ON public.media FOR EACH ROW EXECUTE PROCEDURE public.update_profile_media_count();

-- ============================================================
-- 6. ROW LEVEL SECURITY
-- ============================================================

ALTER TABLE public.profiles           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vaults             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vault_members      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vault_invitations  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.memories           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.media              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.video_jobs         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_conversations   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_messages        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.media_embeddings   ENABLE ROW LEVEL SECURITY;

-- ── Profiles ──
DROP POLICY IF EXISTS "profiles_select_any" ON public.profiles;
CREATE POLICY "profiles_select_any" ON public.profiles FOR SELECT USING (auth.role() = 'authenticated');
DROP POLICY IF EXISTS "profiles_update_own" ON public.profiles;
CREATE POLICY "profiles_update_own" ON public.profiles FOR UPDATE USING (auth.uid() = id);

-- ── Vaults ──
DROP POLICY IF EXISTS "vaults_select_member" ON public.vaults;
CREATE POLICY "vaults_select_member" ON public.vaults FOR SELECT USING (owner_id = auth.uid() OR public.is_vault_member(id));
DROP POLICY IF EXISTS "vaults_insert_own" ON public.vaults;
CREATE POLICY "vaults_insert_own" ON public.vaults FOR INSERT WITH CHECK (owner_id = auth.uid());
DROP POLICY IF EXISTS "vaults_update_owner" ON public.vaults;
CREATE POLICY "vaults_update_owner" ON public.vaults FOR UPDATE USING (owner_id = auth.uid());
DROP POLICY IF EXISTS "vaults_delete_owner" ON public.vaults;
CREATE POLICY "vaults_delete_owner" ON public.vaults FOR DELETE USING (owner_id = auth.uid());

-- ── Vault Members ──
DROP POLICY IF EXISTS "vault_members_select" ON public.vault_members;
CREATE POLICY "vault_members_select" ON public.vault_members FOR SELECT USING (public.is_vault_member(vault_id));
DROP POLICY IF EXISTS "vault_members_insert" ON public.vault_members;
CREATE POLICY "vault_members_insert" ON public.vault_members FOR INSERT WITH CHECK (auth.role() = 'authenticated');
DROP POLICY IF EXISTS "vault_members_delete" ON public.vault_members;
CREATE POLICY "vault_members_delete" ON public.vault_members FOR DELETE USING (
    user_id = auth.uid()
    OR vault_id IN (SELECT id FROM public.vaults WHERE owner_id = auth.uid())
);

-- ── Vault Invitations ──
DROP POLICY IF EXISTS "invitations_select_vault_member" ON public.vault_invitations;
CREATE POLICY "invitations_select_vault_member" ON public.vault_invitations FOR SELECT USING (
    public.is_vault_member(vault_id) OR auth.role() = 'authenticated'
);
DROP POLICY IF EXISTS "invitations_insert_vault_owner" ON public.vault_invitations;
CREATE POLICY "invitations_insert_vault_owner" ON public.vault_invitations FOR INSERT WITH CHECK (
    vault_id IN (SELECT id FROM public.vaults WHERE owner_id = auth.uid())
);
DROP POLICY IF EXISTS "invitations_delete_vault_owner" ON public.vault_invitations;
CREATE POLICY "invitations_delete_vault_owner" ON public.vault_invitations FOR DELETE USING (
    vault_id IN (SELECT id FROM public.vaults WHERE owner_id = auth.uid())
);

-- ── Memories ──
DROP POLICY IF EXISTS "memories_select" ON public.memories;
CREATE POLICY "memories_select" ON public.memories FOR SELECT USING (
    owner_id = auth.uid() OR (vault_id IS NOT NULL AND public.is_vault_member(vault_id))
);
DROP POLICY IF EXISTS "memories_insert" ON public.memories;
CREATE POLICY "memories_insert" ON public.memories FOR INSERT WITH CHECK (owner_id = auth.uid());
DROP POLICY IF EXISTS "memories_update" ON public.memories;
CREATE POLICY "memories_update" ON public.memories FOR UPDATE USING (owner_id = auth.uid());
DROP POLICY IF EXISTS "memories_delete" ON public.memories;
CREATE POLICY "memories_delete" ON public.memories FOR DELETE USING (owner_id = auth.uid());

-- ── Media ──
DROP POLICY IF EXISTS "media_select" ON public.media;
CREATE POLICY "media_select" ON public.media FOR SELECT USING (
    owner_id = auth.uid() OR public.is_vault_member(vault_id)
);
DROP POLICY IF EXISTS "media_insert" ON public.media;
CREATE POLICY "media_insert" ON public.media FOR INSERT WITH CHECK (owner_id = auth.uid());
DROP POLICY IF EXISTS "media_delete" ON public.media;
CREATE POLICY "media_delete" ON public.media FOR DELETE USING (owner_id = auth.uid());

-- ── Video Jobs ──
DROP POLICY IF EXISTS "video_jobs_select" ON public.video_jobs;
CREATE POLICY "video_jobs_select" ON public.video_jobs FOR SELECT USING (user_id = auth.uid());
DROP POLICY IF EXISTS "video_jobs_insert" ON public.video_jobs;
CREATE POLICY "video_jobs_insert" ON public.video_jobs FOR INSERT WITH CHECK (user_id = auth.uid());
DROP POLICY IF EXISTS "video_jobs_update" ON public.video_jobs;
CREATE POLICY "video_jobs_update" ON public.video_jobs FOR UPDATE USING (user_id = auth.uid());
DROP POLICY IF EXISTS "video_jobs_delete" ON public.video_jobs;
CREATE POLICY "video_jobs_delete" ON public.video_jobs FOR DELETE USING (user_id = auth.uid());

-- ── AI Conversations ──
DROP POLICY IF EXISTS "ai_conversations_own" ON public.ai_conversations;
CREATE POLICY "ai_conversations_own" ON public.ai_conversations FOR ALL USING (user_id = auth.uid());

-- ── AI Messages ──
DROP POLICY IF EXISTS "ai_messages_own" ON public.ai_messages;
CREATE POLICY "ai_messages_own" ON public.ai_messages FOR ALL USING (
    conversation_id IN (SELECT id FROM public.ai_conversations WHERE user_id = auth.uid())
);

-- ── Media Embeddings ──
DROP POLICY IF EXISTS "embeddings_select" ON public.media_embeddings;
CREATE POLICY "embeddings_select" ON public.media_embeddings FOR SELECT USING (
    media_id IN (SELECT id FROM public.media WHERE owner_id = auth.uid())
);
DROP POLICY IF EXISTS "embeddings_insert" ON public.media_embeddings;
CREATE POLICY "embeddings_insert" ON public.media_embeddings FOR INSERT WITH CHECK (
    media_id IN (SELECT id FROM public.media WHERE owner_id = auth.uid())
);

-- ============================================================
-- 7. STORAGE BUCKETS
-- ============================================================
INSERT INTO storage.buckets (id, name, "public")
VALUES ('memories', 'memories', false)
ON CONFLICT (id) DO NOTHING;

DROP POLICY IF EXISTS "storage_memories_insert" ON storage.objects;
CREATE POLICY "storage_memories_insert" ON storage.objects FOR INSERT WITH CHECK (
    bucket_id = 'memories'
    AND auth.role() = 'authenticated'
    AND (storage.foldername(name))[1] = auth.uid()::text
);

DROP POLICY IF EXISTS "storage_memories_select" ON storage.objects;
CREATE POLICY "storage_memories_select" ON storage.objects FOR SELECT USING (
    bucket_id = 'memories'
    AND auth.role() = 'authenticated'
);

DROP POLICY IF EXISTS "storage_memories_delete" ON storage.objects;
CREATE POLICY "storage_memories_delete" ON storage.objects FOR DELETE USING (
    bucket_id = 'memories'
    AND auth.uid()::text = (storage.foldername(name))[1]
);
