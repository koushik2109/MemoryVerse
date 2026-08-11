-- ============================================================
--  MemoryVerse — Initial Schema Migration
--  Run this in your Supabase SQL Editor
-- ============================================================

-- Enable pgvector for AI embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- PROFILES
-- ============================================================
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

CREATE INDEX IF NOT EXISTS idx_profiles_email ON public.profiles(email);

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
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();

-- ============================================================
-- VAULTS
-- ============================================================
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

CREATE INDEX IF NOT EXISTS idx_vaults_owner_id   ON public.vaults(owner_id);
CREATE INDEX IF NOT EXISTS idx_vaults_updated_at ON public.vaults(updated_at DESC);

-- ============================================================
-- VAULT MEMBERS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.vault_members (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vault_id    UUID NOT NULL REFERENCES public.vaults(id) ON DELETE CASCADE,
    user_id     UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    role        TEXT NOT NULL DEFAULT 'editor' CHECK (role IN ('owner', 'editor', 'viewer')),
    joined_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(vault_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_vault_members_vault_id ON public.vault_members(vault_id);
CREATE INDEX IF NOT EXISTS idx_vault_members_user_id  ON public.vault_members(user_id);

-- ============================================================
-- VAULT INVITATIONS
-- ============================================================
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

CREATE INDEX IF NOT EXISTS idx_vault_invitations_code ON public.vault_invitations(invite_code);

-- ============================================================
-- MEDIA
-- ============================================================
CREATE TABLE IF NOT EXISTS public.media (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id        UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    vault_id        UUID REFERENCES public.vaults(id) ON DELETE SET NULL,
    filename        TEXT NOT NULL,
    storage_path    TEXT NOT NULL,
    url             TEXT NOT NULL,
    thumbnail_url   TEXT,
    media_type      TEXT NOT NULL DEFAULT 'image' CHECK (media_type IN ('image', 'video')),
    file_size       BIGINT NOT NULL DEFAULT 0,
    mime_type       TEXT,
    width           INT,
    height          INT,
    duration        INT,        -- seconds for video
    taken_at        TIMESTAMPTZ, -- EXIF timestamp
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    location_name   TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_media_owner_id   ON public.media(owner_id);
CREATE INDEX IF NOT EXISTS idx_media_vault_id   ON public.media(vault_id);
CREATE INDEX IF NOT EXISTS idx_media_created_at ON public.media(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_media_taken_at   ON public.media(taken_at DESC);
CREATE INDEX IF NOT EXISTS idx_media_type       ON public.media(media_type);

-- ============================================================
-- AI CONVERSATIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.ai_conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT 'New Conversation',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ai_conversations_user_id ON public.ai_conversations(user_id);

-- ============================================================
-- AI MESSAGES
-- ============================================================
CREATE TABLE IF NOT EXISTS public.ai_messages (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id     UUID NOT NULL REFERENCES public.ai_conversations(id) ON DELETE CASCADE,
    role                TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content             TEXT NOT NULL,
    related_memory_ids  UUID[] DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ai_messages_conversation_id ON public.ai_messages(conversation_id);

-- ============================================================
-- EMBEDDINGS (pgvector)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.media_embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    media_id    UUID NOT NULL REFERENCES public.media(id) ON DELETE CASCADE UNIQUE,
    embedding   vector(1536),  -- OpenAI ada-002 dimension
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_media_embeddings_media_id
    ON public.media_embeddings(media_id);

-- IVFFlat index for similarity search (requires at least some data to train)
-- Run manually after loading data:
-- CREATE INDEX ON public.media_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================
-- UPDATED_AT TRIGGERS
-- ============================================================
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS set_profiles_updated_at ON public.profiles;
CREATE TRIGGER set_profiles_updated_at
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE PROCEDURE public.set_updated_at();

DROP TRIGGER IF EXISTS set_vaults_updated_at ON public.vaults;
CREATE TRIGGER set_vaults_updated_at
    BEFORE UPDATE ON public.vaults
    FOR EACH ROW EXECUTE PROCEDURE public.set_updated_at();

DROP TRIGGER IF EXISTS set_ai_conversations_updated_at ON public.ai_conversations;
CREATE TRIGGER set_ai_conversations_updated_at
    BEFORE UPDATE ON public.ai_conversations
    FOR EACH ROW EXECUTE PROCEDURE public.set_updated_at();

-- ============================================================
-- UPDATE PROFILE COUNTERS
-- ============================================================
CREATE OR REPLACE FUNCTION public.update_profile_vault_count()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE public.profiles SET vault_count = vault_count + 1
        WHERE id = NEW.user_id AND NEW.role = 'owner';
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE public.profiles SET vault_count = GREATEST(0, vault_count - 1)
        WHERE id = OLD.user_id AND OLD.role = 'owner';
    END IF;
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS trg_vault_count ON public.vault_members;
CREATE TRIGGER trg_vault_count
    AFTER INSERT OR DELETE ON public.vault_members
    FOR EACH ROW EXECUTE PROCEDURE public.update_profile_vault_count();

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
CREATE TRIGGER trg_media_count
    AFTER INSERT OR DELETE ON public.media
    FOR EACH ROW EXECUTE PROCEDURE public.update_profile_media_count();

-- ============================================================
-- ROW LEVEL SECURITY
-- ============================================================

ALTER TABLE public.profiles           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vaults             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vault_members      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vault_invitations  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.media              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_conversations   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_messages        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.media_embeddings   ENABLE ROW LEVEL SECURITY;

-- ── Profiles ─────────────────────────────────────────────────
DROP POLICY IF EXISTS "profiles_select_own"   ON public.profiles;
DROP POLICY IF EXISTS "profiles_select_any"   ON public.profiles;
DROP POLICY IF EXISTS "profiles_update_own"   ON public.profiles;

-- Any authenticated user can read any profile (for vault member display)
CREATE POLICY "profiles_select_any" ON public.profiles
    FOR SELECT USING (auth.role() = 'authenticated');

-- Users can update only their own profile
CREATE POLICY "profiles_update_own" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

-- ── Vaults ───────────────────────────────────────────────────
DROP POLICY IF EXISTS "vaults_select_member"  ON public.vaults;
DROP POLICY IF EXISTS "vaults_insert_own"     ON public.vaults;
DROP POLICY IF EXISTS "vaults_update_owner"   ON public.vaults;
DROP POLICY IF EXISTS "vaults_delete_owner"   ON public.vaults;

-- Helper function to bypass RLS recursion
CREATE OR REPLACE FUNCTION public.is_vault_member(v_id UUID)
RETURNS BOOLEAN
LANGUAGE sql SECURITY DEFINER SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM vault_members
    WHERE vault_id = v_id AND user_id = auth.uid()
  );
$$;

CREATE POLICY "vaults_select_member" ON public.vaults
    FOR SELECT USING (
        owner_id = auth.uid() OR public.is_vault_member(id)
    );

CREATE POLICY "vaults_insert_own" ON public.vaults
    FOR INSERT WITH CHECK (owner_id = auth.uid());

CREATE POLICY "vaults_update_owner" ON public.vaults
    FOR UPDATE USING (owner_id = auth.uid());

CREATE POLICY "vaults_delete_owner" ON public.vaults
    FOR DELETE USING (owner_id = auth.uid());

-- ── Vault Members ────────────────────────────────────────────
DROP POLICY IF EXISTS "vault_members_select"  ON public.vault_members;
DROP POLICY IF EXISTS "vault_members_insert"  ON public.vault_members;
DROP POLICY IF EXISTS "vault_members_delete"  ON public.vault_members;

-- Members of a vault can see membership list
CREATE POLICY "vault_members_select" ON public.vault_members
    FOR SELECT USING (
        public.is_vault_member(vault_id)
    );

-- Anyone authenticated can become a member (controlled by service logic)
CREATE POLICY "vault_members_insert" ON public.vault_members
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- Vault owner OR the member themselves can remove a membership
CREATE POLICY "vault_members_delete" ON public.vault_members
    FOR DELETE USING (
        user_id = auth.uid()
        OR vault_id IN (SELECT id FROM public.vaults WHERE owner_id = auth.uid())
    );

-- ── Vault Invitations ─────────────────────────────────────────
DROP POLICY IF EXISTS "invitations_select_vault_member" ON public.vault_invitations;
DROP POLICY IF EXISTS "invitations_insert_vault_owner"  ON public.vault_invitations;
DROP POLICY IF EXISTS "invitations_delete_vault_owner"  ON public.vault_invitations;
DROP POLICY IF EXISTS "invitations_select_by_code"      ON public.vault_invitations;

CREATE POLICY "invitations_select_vault_member" ON public.vault_invitations
    FOR SELECT USING (
        public.is_vault_member(vault_id)
        OR auth.role() = 'authenticated'  -- allow reading invite info by code
    );

CREATE POLICY "invitations_insert_vault_owner" ON public.vault_invitations
    FOR INSERT WITH CHECK (
        vault_id IN (SELECT id FROM public.vaults WHERE owner_id = auth.uid())
    );

CREATE POLICY "invitations_delete_vault_owner" ON public.vault_invitations
    FOR DELETE USING (
        vault_id IN (SELECT id FROM public.vaults WHERE owner_id = auth.uid())
    );

-- ── Media ────────────────────────────────────────────────────
DROP POLICY IF EXISTS "media_select"  ON public.media;
DROP POLICY IF EXISTS "media_insert"  ON public.media;
DROP POLICY IF EXISTS "media_delete"  ON public.media;

-- User can see their own media OR media belonging to vaults they're in
CREATE POLICY "media_select" ON public.media
    FOR SELECT USING (
        owner_id = auth.uid()
        OR public.is_vault_member(vault_id)
    );

CREATE POLICY "media_insert" ON public.media
    FOR INSERT WITH CHECK (owner_id = auth.uid());

CREATE POLICY "media_delete" ON public.media
    FOR DELETE USING (owner_id = auth.uid());

-- ── AI Conversations ─────────────────────────────────────────
DROP POLICY IF EXISTS "ai_conversations_own" ON public.ai_conversations;
CREATE POLICY "ai_conversations_own" ON public.ai_conversations
    FOR ALL USING (user_id = auth.uid());

-- ── AI Messages ──────────────────────────────────────────────
DROP POLICY IF EXISTS "ai_messages_own" ON public.ai_messages;
CREATE POLICY "ai_messages_own" ON public.ai_messages
    FOR ALL USING (
        conversation_id IN (
            SELECT id FROM public.ai_conversations WHERE user_id = auth.uid()
        )
    );

-- ── Media Embeddings ─────────────────────────────────────────
DROP POLICY IF EXISTS "embeddings_select" ON public.media_embeddings;
DROP POLICY IF EXISTS "embeddings_insert" ON public.media_embeddings;

CREATE POLICY "embeddings_select" ON public.media_embeddings
    FOR SELECT USING (
        media_id IN (
            SELECT id FROM public.media WHERE owner_id = auth.uid()
        )
    );

CREATE POLICY "embeddings_insert" ON public.media_embeddings
    FOR INSERT WITH CHECK (
        media_id IN (
            SELECT id FROM public.media WHERE owner_id = auth.uid()
        )
    );

-- ============================================================
-- STORAGE BUCKETS
-- ============================================================
-- Run in Supabase dashboard Storage section OR via CLI:
-- supabase storage buckets create memories --public false
-- Then add RLS policies on the storage.objects table:

INSERT INTO storage.buckets (id, name, public)
VALUES ('memories', 'memories', false)
ON CONFLICT (id) DO NOTHING;

-- Storage RLS: allow authenticated users to upload to their own folder
DROP POLICY IF EXISTS "storage_memories_insert" ON storage.objects;
CREATE POLICY "storage_memories_insert" ON storage.objects
    FOR INSERT WITH CHECK (
        bucket_id = 'memories'
        AND auth.role() = 'authenticated'
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

DROP POLICY IF EXISTS "storage_memories_select" ON storage.objects;
CREATE POLICY "storage_memories_select" ON storage.objects
    FOR SELECT USING (
        bucket_id = 'memories'
        AND auth.role() = 'authenticated'
    );

DROP POLICY IF EXISTS "storage_memories_delete" ON storage.objects;
CREATE POLICY "storage_memories_delete" ON storage.objects
    FOR DELETE USING (
        bucket_id = 'memories'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );
-- ============================================================
--  MemoryVerse — Migration 002: Add Memories Table
--  This migration introduces the Memories concept and groups
--  existing single media files into individual memories to preserve data.
-- ============================================================

-- 1. Create the memories table
CREATE TABLE IF NOT EXISTS public.memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    vault_id UUID REFERENCES public.vaults(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    description TEXT,
    cover_media_id UUID, -- References media(id), added after media table is updated to avoid circular reference issues in some Postgres versions
    memory_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    location_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Add memory_id to media table
ALTER TABLE public.media
ADD COLUMN IF NOT EXISTS memory_id UUID REFERENCES public.memories(id) ON DELETE CASCADE;

-- 3. Data Migration: Preserve existing data
-- We will create a memory for every existing media file that doesn't have one
DO $$
DECLARE
    r RECORD;
    new_memory_id UUID;
BEGIN
    FOR r IN SELECT * FROM public.media WHERE memory_id IS NULL
    LOOP
        new_memory_id := gen_random_uuid();
        
        -- Insert a new memory for this media item
        INSERT INTO public.memories (id, owner_id, vault_id, title, memory_date, created_at, updated_at)
        VALUES (
            new_memory_id, 
            r.owner_id, 
            r.vault_id, 
            'Memory from ' || to_char(COALESCE(r.taken_at, r.created_at), 'Mon DD, YYYY'),
            COALESCE(r.taken_at, r.created_at),
            r.created_at,
            r.created_at
        );
        
        -- Update the media item to point to the new memory
        UPDATE public.media 
        SET memory_id = new_memory_id 
        WHERE id = r.id;
        
        -- Set the cover media of the memory to this media item
        UPDATE public.memories
        SET cover_media_id = r.id
        WHERE id = new_memory_id;
    END LOOP;
END $$;

-- 4. Add foreign key for cover_media_id now that data is migrated
-- First check if the constraint already exists
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'memories_cover_media_id_fkey'
  ) THEN
    ALTER TABLE public.memories
    ADD CONSTRAINT memories_cover_media_id_fkey
    FOREIGN KEY (cover_media_id) REFERENCES public.media(id) ON DELETE SET NULL;
  END IF;
END $$;

-- 5. Create Indexes
CREATE INDEX IF NOT EXISTS idx_memories_owner_id ON public.memories(owner_id);
CREATE INDEX IF NOT EXISTS idx_memories_vault_id ON public.memories(vault_id);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON public.memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_media_memory_id ON public.media(memory_id);

-- 6. Setup Updated At Trigger
DROP TRIGGER IF EXISTS set_memories_updated_at ON public.memories;
CREATE TRIGGER set_memories_updated_at
    BEFORE UPDATE ON public.memories
    FOR EACH ROW EXECUTE PROCEDURE public.set_updated_at();

-- 7. Setup RLS Policies for Memories
ALTER TABLE public.memories ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "memories_select" ON public.memories;
DROP POLICY IF EXISTS "memories_insert" ON public.memories;
DROP POLICY IF EXISTS "memories_update" ON public.memories;
DROP POLICY IF EXISTS "memories_delete" ON public.memories;

-- User can see their own memories OR memories belonging to vaults they're in
CREATE POLICY "memories_select" ON public.memories
    FOR SELECT USING (
        owner_id = auth.uid()
        OR (vault_id IS NOT NULL AND public.is_vault_member(vault_id))
    );

CREATE POLICY "memories_insert" ON public.memories
    FOR INSERT WITH CHECK (owner_id = auth.uid());

CREATE POLICY "memories_update" ON public.memories
    FOR UPDATE USING (owner_id = auth.uid());

CREATE POLICY "memories_delete" ON public.memories
    FOR DELETE USING (owner_id = auth.uid());
-- ============================================================
--  MemoryVerse — Migration 003: Add Video Jobs Table
--  This migration adds the video_jobs table to track async 
--  video stitching jobs in the backend.
-- ============================================================

CREATE TABLE IF NOT EXISTS public.video_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES public.memories(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued', -- queued, processing, completed, failed
    result_media_id UUID REFERENCES public.media(id) ON DELETE SET NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_video_jobs_user_id ON public.video_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_video_jobs_memory_id ON public.video_jobs(memory_id);

-- Updated At Trigger
CREATE TRIGGER set_video_jobs_updated_at
    BEFORE UPDATE ON public.video_jobs
    FOR EACH ROW EXECUTE PROCEDURE public.set_updated_at();

-- RLS Policies
ALTER TABLE public.video_jobs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "video_jobs_select" ON public.video_jobs;
DROP POLICY IF EXISTS "video_jobs_insert" ON public.video_jobs;
DROP POLICY IF EXISTS "video_jobs_update" ON public.video_jobs;
DROP POLICY IF EXISTS "video_jobs_delete" ON public.video_jobs;

-- Users can only see their own jobs
CREATE POLICY "video_jobs_select" ON public.video_jobs
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "video_jobs_insert" ON public.video_jobs
    FOR INSERT WITH CHECK (user_id = auth.uid());

CREATE POLICY "video_jobs_update" ON public.video_jobs
    FOR UPDATE USING (user_id = auth.uid());

CREATE POLICY "video_jobs_delete" ON public.video_jobs
    FOR DELETE USING (user_id = auth.uid());
-- Fix Infinite Recursion in Row Level Security (RLS)

-- 1. Create a SECURITY DEFINER function that bypasses RLS to check membership safely
CREATE OR REPLACE FUNCTION public.is_vault_member(v_id UUID)
RETURNS BOOLEAN
LANGUAGE sql SECURITY DEFINER SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM vault_members
    WHERE vault_id = v_id AND user_id = auth.uid()
  );
$$;

-- 2. Drop the broken recursive policies
DROP POLICY IF EXISTS "vault_members_select" ON public.vault_members;
DROP POLICY IF EXISTS "vaults_select_member"  ON public.vaults;
DROP POLICY IF EXISTS "invitations_select_vault_member" ON public.vault_invitations;
DROP POLICY IF EXISTS "media_select"  ON public.media;

-- 3. Recreate the policies using the safe function
CREATE POLICY "vault_members_select" ON public.vault_members
    FOR SELECT USING (
        public.is_vault_member(vault_id)
    );

CREATE POLICY "vaults_select_member" ON public.vaults
    FOR SELECT USING (
        owner_id = auth.uid() OR public.is_vault_member(id)
    );

CREATE POLICY "invitations_select_vault_member" ON public.vault_invitations
    FOR SELECT USING (
        public.is_vault_member(vault_id)
        OR auth.role() = 'authenticated'
    );

CREATE POLICY "media_select" ON public.media
    FOR SELECT USING (
        owner_id = auth.uid()
        OR public.is_vault_member(vault_id)
    );
