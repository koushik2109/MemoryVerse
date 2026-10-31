# Authoritative MemoryVerse workflow

This is the implementation-based workflow. Older design documents and the
research paper may describe proposed or historical behavior; use this document
when operating or extending the application.

## 1. Authentication

1. Flutter loads `frontend/.env` or Dart defines through
   `frontend/lib/core/config/app_config.dart`.
2. Supabase Flutter initializes PKCE authentication.
3. `ApiClient` attaches the current Supabase bearer token to backend requests.
4. FastAPI validates the token through Supabase Auth. Local JWT validation is
   only allowed when explicitly enabled for isolated development.

## 2. Memory creation and media upload

1. The user creates a memory through `MemoryRepository`.
2. FastAPI creates an owner-scoped row in `memories`.
3. The user selects local media.
4. Flutter creates upload tasks with bounded concurrency and retries.
5. Each upload sends multipart data to `/api/v1/media/upload`.
6. The backend validates the upload, writes the original and thumbnail to the
   private `memories` bucket, creates a `media` row, then schedules extraction.
7. Flutter waits for the selected upload tasks before starting dependent video
   generation.

## 3. Media extraction and indexing

`AIExtractor.process_media_item` extracts technical metadata and EXIF where
available, creates thumbnails/tags/embeddings, and writes derived metadata and
vectors. The production backend search path combines metadata, tags, lexical
fields, and optional CLIP/vector retrieval.

The standalone RAG service is a separate path, not a replacement for the
backend path:

- Video: OpenCV keyframes, SigLIP image embedding, Qwen/Ollama captioning,
  BGE-M3 text embedding, Supabase/pgvector.
- Audio: Librosa features, Whisper base transcription when applicable,
  ambient classification, BGE-M3 text embedding, Supabase/pgvector.

The standalone service is private and requires the shared internal token.

## 4. Search and assistant

1. The client sends a query to FastAPI.
2. The backend applies user/vault authorization and metadata filters.
3. Candidate retrieval uses the configured production search path.
4. Optional AI/RAG services add modality-specific candidates.
5. The assistant must answer from authorized evidence and abstain when no
   evidence is available.
6. Conversation rows preserve messages and related memory IDs where configured.

## 5. Video generation

1. Flutter requests `/media/memory/{memory_id}/generate-video` after uploads are
   complete.
2. The backend creates an owner-scoped `video_jobs` row.
3. Redis or the configured worker queue processes the job; local journal mode
   is suitable only for single-process development.
4. The worker loads authorized media, scores quality, removes duplicates,
   creates a story/shot plan, renders with MoviePy/FFmpeg, and validates output.
5. TTS and external providers are optional and configuration-dependent.
6. The result is uploaded to private storage when available, with a generated
   `media` row. Local fallback URLs are resolved using the actual media ID.
7. Flutter observes job status through SSE or polling and plays the authorized
   signed stream URL.

## 6. Deletion and privacy

Deletion must remove the original object, generated derivatives, database row,
embeddings, and job references. Storage is private; clients should use
short-lived signed URLs or authorized backend streaming. Logs must not contain
media bytes, tokens, or personal captions unnecessarily.

## 7. What is not currently guaranteed

- Device-wide on-device camera-roll indexing.
- Beat-synchronized editing.
- Exact duration tolerance for every render.
- Production-grade multi-process durability without Redis/database support.
- Universal people identity or emotion recognition.
- The research-paper benchmark results.
