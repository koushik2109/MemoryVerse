-- Migration: 20260922_video_jobs_metadata.sql
-- Description: Ensure script_metadata JSONB column exists on video_jobs table
-- to store structured VideoSpecification, current_task, dimension, mood, and stage telemetry.

ALTER TABLE public.video_jobs 
ADD COLUMN IF NOT EXISTS script_metadata JSONB DEFAULT '{}'::jsonb;

COMMENT ON COLUMN public.video_jobs.script_metadata IS 
'Stores structured video generation metadata including VideoSpecification, stage, current_task, and progress';
