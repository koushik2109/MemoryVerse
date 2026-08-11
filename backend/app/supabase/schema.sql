-- ============================================================================
-- MemoryVerse Database Schema & Supabase Setup
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ----------------------------------------------------------------------------
-- 1. PROFILES TABLE (linked to auth.users)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    username TEXT UNIQUE,
    avatar_url TEXT,
    bio TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Trigger to create/update profile when user signs up
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name, avatar_url)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', SPLIT_PART(NEW.email, '@', 1)),
        NEW.raw_user_meta_data->>'avatar_url'
    )
    ON CONFLICT (id) DO UPDATE
    SET email = EXCLUDED.email,
        full_name = COALESCE(EXCLUDED.full_name, public.profiles.full_name),
        updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ----------------------------------------------------------------------------
-- 2. VAULTS TABLE
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.vaults (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,
    cover_image_url TEXT,
    is_archived BOOLEAN DEFAULT FALSE NOT NULL,
    owner_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- ----------------------------------------------------------------------------
-- 3. VAULT MEMBERS TABLE
-- ----------------------------------------------------------------------------
CREATE TYPE vault_role AS ENUM ('owner', 'editor', 'viewer');

CREATE TABLE IF NOT EXISTS public.vault_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vault_id UUID NOT NULL REFERENCES public.vaults(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    role vault_role DEFAULT 'editor' NOT NULL,
    joined_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE(vault_id, user_id)
);

-- ----------------------------------------------------------------------------
-- 4. MEDIA TABLE
-- ----------------------------------------------------------------------------
CREATE TYPE media_type_enum AS ENUM ('image', 'video');

CREATE TABLE IF NOT EXISTS public.media (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vault_id UUID REFERENCES public.vaults(id) ON DELETE SET NULL,
    owner_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    url TEXT NOT NULL,
    thumbnail_url TEXT,
    media_type media_type_enum DEFAULT 'image' NOT NULL,
    file_size BIGINT DEFAULT 0 NOT NULL,
    mime_type TEXT,
    width INT,
    height INT,
    duration INT, -- For video in seconds
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- ----------------------------------------------------------------------------
-- 5. INVITATIONS TABLE
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.invitations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vault_id UUID NOT NULL REFERENCES public.vaults(id) ON DELETE CASCADE,
    invite_code TEXT UNIQUE NOT NULL,
    created_by UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    role vault_role DEFAULT 'editor' NOT NULL,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- ----------------------------------------------------------------------------
-- 6. NOTIFICATIONS TABLE
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    type TEXT NOT NULL, -- 'vault_invite', 'member_joined', 'media_uploaded', 'vault_updated'
    data JSONB DEFAULT '{}'::jsonb,
    is_read BOOLEAN DEFAULT FALSE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- ----------------------------------------------------------------------------
-- 7. ACTIVITY LOGS TABLE
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.activity_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    vault_id UUID REFERENCES public.vaults(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- ----------------------------------------------------------------------------
-- INDEXES
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_vaults_owner ON public.vaults(owner_id);
CREATE INDEX IF NOT EXISTS idx_vault_members_vault ON public.vault_members(vault_id);
CREATE INDEX IF NOT EXISTS idx_vault_members_user ON public.vault_members(user_id);
CREATE INDEX IF NOT EXISTS idx_media_vault ON public.media(vault_id);
CREATE INDEX IF NOT EXISTS idx_media_owner ON public.media(owner_id);
CREATE INDEX IF NOT EXISTS idx_invitations_code ON public.invitations(invite_code);
CREATE INDEX IF NOT EXISTS idx_notifications_user ON public.notifications(user_id);

-- ----------------------------------------------------------------------------
-- SUPABASE ROW LEVEL SECURITY (RLS) POLICIES
-- ----------------------------------------------------------------------------
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vaults ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vault_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.media ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.invitations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.activity_logs ENABLE ROW LEVEL SECURITY;

-- Profiles: Anyone authenticated can read profiles; users can update own profile
CREATE POLICY "Public profiles are readable by authenticated users" ON public.profiles
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE TO authenticated USING (auth.uid() = id);

-- Vaults: Accessible if owner or member
CREATE POLICY "Users can access vaults they own or are members of" ON public.vaults
    FOR SELECT TO authenticated USING (
        owner_id = auth.uid() OR
        EXISTS (SELECT 1 FROM public.vault_members WHERE vault_id = vaults.id AND user_id = auth.uid())
    );

CREATE POLICY "Users can create vaults" ON public.vaults
    FOR INSERT TO authenticated WITH CHECK (owner_id = auth.uid());

CREATE POLICY "Owners can update vaults" ON public.vaults
    FOR UPDATE TO authenticated USING (owner_id = auth.uid());

CREATE POLICY "Owners can delete vaults" ON public.vaults
    FOR DELETE TO authenticated USING (owner_id = auth.uid());

-- Media: Accessible if vault member or media owner
CREATE POLICY "Users can access media in accessible vaults or own media" ON public.media
    FOR SELECT TO authenticated USING (
        owner_id = auth.uid() OR
        (vault_id IS NOT NULL AND EXISTS (
            SELECT 1 FROM public.vault_members WHERE vault_id = media.vault_id AND user_id = auth.uid()
        ))
    );

CREATE POLICY "Users can insert media" ON public.media
    FOR INSERT TO authenticated WITH CHECK (owner_id = auth.uid());

CREATE POLICY "Owners can delete own media" ON public.media
    FOR DELETE TO authenticated USING (owner_id = auth.uid());

-- Notifications: Users read/update own notifications
CREATE POLICY "Users can manage own notifications" ON public.notifications
    FOR ALL TO authenticated USING (user_id = auth.uid());

-- ----------------------------------------------------------------------------
-- SUPABASE STORAGE BUCKET INITIALIZATION
-- ----------------------------------------------------------------------------
INSERT INTO storage.buckets (id, name, public)
VALUES ('memories', 'memories', true)
ON CONFLICT (id) DO NOTHING;

CREATE POLICY "Memories Bucket Public Select" ON storage.objects
    FOR SELECT TO authenticated USING (bucket_id = 'memories');

CREATE POLICY "Memories Bucket Authenticated Upload" ON storage.objects
    FOR INSERT TO authenticated WITH CHECK (bucket_id = 'memories');
