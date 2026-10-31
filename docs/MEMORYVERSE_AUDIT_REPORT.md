# MemoryVerse Full System & AI Architecture Audit Report

This report provides a comprehensive, production-grade audit of your version of the **MemoryVerse** application. It details the end-to-end workflow, full-stack architecture, backend services, AI/ML pipelines, multimodal RAG, automatic event clustering, narrative generation, video generation engines, database schema, security model, and frontend architecture.

---

## 1. Executive Summary & Architectural Overview

While your friend's version is primarily focused as a specialized retrieval-only RAG module, **your version is a complete, full-stack, multimodal personal memory operating system**. 

Your codebase combines:
1. **Interactive Client App**: Cross-platform Flutter frontend with rich animations, timeline views, vault collaboration, and conversational AI.
2. **Production Backend**: FastAPI REST backend with JWT/Supabase authentication, background task queues, and media streaming (HTTP 206 Partial Content).
3. **Multi-Signal Ingestion & Feature Extractor**: EXIF GPS/timestamp extraction, zero-shot scene/object/people tagging, and local 512-dim visual embeddings via **CLIP (ViT-B/32)**.
4. **VLM & Multimodal Curation Engine**: VLM-first local candidate curation with **Qwen-2.5-VL** / **GPT-4o-mini** to eliminate false positives.
5. **Multi-Stage Event Clustering**: 5-stage temporal, spatial (Haversine), and semantic grouping that transforms raw media into cohesive memories and multi-day trips.
6. **Memory Narrative Intelligence & Semantic Timeline**: 7-stage factual evidence manifest, chronological segmentation, key moment scoring, and mood/atmosphere inference without hallucination.
7. **Automated Memory-to-Video Synthesis**: AI Editorial Director story planner, Ken Burns dynamic motion effects, subtitle rendering, and ambient chord audio generation.
8. **Audio & Video Multimodal RAG**: Speech transcription with **faster-whisper**, acoustic classification with **Librosa / ESC-50**, **BGE-M3** dense text embeddings, and **SigLIP** keyframe embeddings.
9. **Multi-Tenant Vault Collaboration**: Relational database with fine-grained PostgreSQL Row Level Security (RLS) policies and invite codes.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            FLUTTER CLIENT (FRONTEND)                             │
│   • Timeline View (Year/Month/Day)  • AI Chat Assistant  • Memory Creation/Curation  │
│   • Vault Management & Sharing     • Video Player & Media Viewer • Dark/Light Theme│
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ REST API / JWT
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│                              FASTAPI BACKEND API                                 │
│  /auth  •  /profile  •  /vaults  •  /memories  •  /media  •  /timeline  •  /ai   │
└──────┬──────────────┬──────────────┬──────────────┬──────────────┬───────────────┘
       │              │              │              │              │
┌──────▼──────┐┌──────▼──────┐┌──────▼──────┐┌──────▼──────┐┌──────▼──────────────┐
│  AI Feature ││    Event    ││   Narrative ││   AI Video   ││  Multi-Provider    │
│  Extractor  ││ Clustering  ││ Intelligence││  Generation ││  Chat & RAG Filter │
│  • EXIF     ││ • Haversine ││ • Evidence  ││ • AI Story  ││ • Memory Context    │
│  • CLIP     ││ • Semantic  ││   Manifest  ││   Director  ││ • OpenRouter/OpenAI │
│  • Tags     ││ • Trips     ││ • Timeline  ││ • Ken Burns ││ • Gemini Fallback  │
│  • Whisper  ││ • Cover     ││ • Atmosphere││ • Ambient Sfx││ • VLM Curation    │
└──────┬──────┘└──────┬──────┘└──────┬──────┘└──────┬──────┘└──────┬──────────────┘
       │              │              │              │              │
┌──────▼──────────────▼──────────────▼──────────────▼──────────────▼──────────────┐
│                 SUPABASE / POSTGRESQL + PGVECTOR + STORAGE                       │
│  • Tables: profiles, vaults, vault_members, memories, media, video_jobs, ai_*    │
│  • Row Level Security (RLS) & Security Definer Functions                         │
│  • Storage: 'memories' bucket with Signed URLs & HTTP 206 Byte-Range Streaming   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Models, Libraries & Technology Stack

### 2.1. AI / ML Models Used

| Model | Source / Framework | Purpose in Your Codebase |
| :--- | :--- | :--- |
| **CLIP (`clip-ViT-B-32`)** | `sentence-transformers` | Zero-shot scene/object/people tagging, 512-dim visual embeddings, and candidate semantic filtering. |
| **faster-whisper (`base` / int8)** | `faster-whisper` + FFmpeg | High-speed, local video & audio speech transcription with VAD (Voice Activity Detection). |
| **Qwen-2.5-VL-72B / GPT-4o-mini** | OpenRouter / OpenAI API | Vision-Language Model (VLM) for fine-grained image scene analysis, visual relevance verification, and local media curation. |
| **LLM Reasoning (OpenRouter / OpenAI / Gemini)** | Multiple Providers | Multi-provider fallback orchestration (Ling-3.0, LFM-2.5, MiniMax, GPT-4o-mini, Gemini 2.0 Flash) for memory chat, story planning, and narrative synthesis. |
| **BGE-M3 (`BAAI/bge-m3`)** | `sentence-transformers` | 1024-dim dense text embeddings for semantic audio/video RAG search. |
| **Google SigLIP** | `transformers` | Multimodal visual keyframe embedder (768-dim) for text-to-video and image-to-video retrieval. |
| **Librosa & ESC-50** | `librosa` / `scipy` | Audio feature extraction (tempo, spectral centroid, zero-crossing rate) and environmental acoustic classification. |

### 2.2. Core Backend Technologies (`backend/`)
- **Web Framework**: FastAPI `>=0.111.0`, Uvicorn, Pydantic v2 (`pydantic-settings`).
- **Database & Cloud Storage**: Supabase Python SDK `>=2.5.0` (PostgreSQL with `pgvector`, Storage buckets, RLS).
- **Authentication & Security**: Python-Jose (JWT), Passlib (Bcrypt), OAuth2 Bearer tokens.
- **Media & Computer Vision Processing**:
  - `Pillow` (`PIL`): Thumbnail extraction, image quality scoring, aspect-ratio resizing.
  - `MoviePy 2.x` & `imageio-ffmpeg`: Frame grabbing, video stream validation, dynamic Ken Burns rendering, audio-video muxing.
  - `exifread`: EXIF metadata extraction (Camera make/model, timestamps, GPS latitude/longitude).
  - `scipy` & `numpy`: Laplacian variance blur detection, luminance calculation, Haversine geospatial calculations.

### 2.3. Core Frontend Technologies (`frontend/`)
- **Framework**: Flutter (Dart `>=3.0.0 <4.0.0`).
- **State Management**: `flutter_riverpod` (`^2.5.1`).
- **Navigation & Routing**: `go_router` (`^14.2.7`).
- **Network & API**: `dio` (`^5.4.3+1`), `supabase_flutter` (`^2.16.0`).
- **Media Playback & Picker**: `chewie` (`^1.8.5`), `video_player` (`^2.9.2`), `video_player_win`, `image_picker`, `file_picker`.
- **UI & Animations**: `flutter_animate` (`^4.5.0`), `google_fonts`, `cached_network_image`, `carousel_slider`, `qr_flutter`.

---

## 3. End-to-End Workflows & Key Sub-Systems

### 3.1. Multi-Signal Media Ingestion Pipeline (`media.py`, `ai_extractor.py`)
When a user uploads photos or videos:
1. **Validation**: Enforces allowed MIME types (`jpeg`, `png`, `webp`, `gif`, `heic`, `mp4`, `mov`, `avi`, `webm`, `3gpp`) and 500MB size limit.
2. **Dual Upload**:
   - Original file uploaded to Supabase Storage private bucket `memories`.
   - Automatic thumbnail generation (~30–50KB JPEG for images, first-frame capture at 10% duration for videos) uploaded to `thumbs/` path.
3. **Asynchronous Background Processing (`AIExtractor`)**:
   - **EXIF Extraction**: Parses camera make, model, GPS coordinates, and native capture time (`taken_at`).
   - **CLIP Visual Embedding**: Generates a 512-dim embedding.
   - **Zero-Shot Tagging**: Softmax cosine scoring across scene labels (*beach, cityscape, mountains, sunset, party*), objects (*food, car, dog, laptop*), and people presence (*no people, one person, group of people*).
   - **Whisper Speech Extraction** (for videos): Transcribes spoken audio into timestamped segments.

### 3.2. VLM-First Local Media Curation (`ai_service.py:curate_local_media`)
Allows users to create memories from un-uploaded local device photos using natural language prompts (e.g., *"Our dinner at the Eiffel tower"*):
- Dispatches concurrent threads to **Vision-Language Models (Qwen-2.5-VL / GPT-4o-mini)**.
- Analyzes visible landmarks, subjects, and activities vs generic scenery false positives (e.g., distinguishing an actual landmark from generic sky/hotel room).
- Returns structured JSON with `relevance_score`, `is_subject_visible`, `evidence`, and recommended selections.

### 3.3. Multi-Signal Event Clustering Engine (`event_clustering_service.py`)
An automated 5-stage algorithm that groups unassociated photos and videos into cohesive Memories:
- **Stage 1 (Temporal + Spatial Grouping)**: Adaptive clustering based on time gaps and geospatial Haversine distance thresholding.
- **Stage 2 (Cheap Embedding Refinement)**: Cosine similarity clustering using pre-computed 512-dim CLIP vectors to separate unrelated themes in the same timeframe.
- **Stage 3 (VLM & Tag Refinement)**: Semantic grouping using scene tags, objects, and VLM descriptions.
- **Stage 4 (Parent Trip Detection & LLM Coherence)**: Identifies multi-day travel itineraries and resolves ambiguous cluster boundaries with LLMs.
- **Stage 5 (Synthesis & Cover Selection)**: Deterministically chooses the highest-quality, representative cover photo based on sharpness, exposure, and composition score.

### 3.4. Memory Narrative Intelligence & Semantic Timeline (`memory_narrative_service.py`)
Transforms a memory's media collection into a structured, grounded narrative:
- **Stage 1 (Evidence Manifest)**: Builds a factual manifest separating objective evidence (locations, times, detected objects, speech transcripts) from interpretation.
- **Stage 2 (Deterministic Timeline Segmentation)**: Segments the event into 3–8 logical phases (e.g., *Arrival, Main Activity, Golden Hour, Celebration*).
- **Stage 3 (Diversity-Aware Key Moment Selection)**: Ranks items using normalized multi-signal scoring (quality score + story value + temporal diversity).
- **Stage 4 (Atmosphere & Mood Inference)**: Grounded mood detection (*Joyful, Serene, Adventurous, Nostalgic*) strictly derived from visual/auditory evidence without emotional hallucination.
- **Stage 5 (Grounded Synthesis)**: Generates AI titles, summaries, and narrative highlights with automated deterministic fallback when offline.

### 3.5. Automated AI Video Generation Pipeline (`video_service.py`, `media_intelligence.py`)
Creates cinematic memory recap videos from raw photos and clips:
1. **Pre-Filtering & Scoring**: Laplacian variance sharpness scoring, luminance exposure analysis, and CLIP near-duplicate suppression.
2. **AI Editorial Director**: Generates a structured shot-list JSON matching pacing, scene duration, and transitions to the story arc.
3. **Motion Engine (Ken Burns Effect)**: Smooth dynamic zooming (`zoom_in`, `zoom_out`), panning (`pan_left`, `pan_right`), and cross-dissolves.
4. **Cinematic Title Cards & Subtitles**: Custom PIL rendering with font fallbacks and speech subtitle overlay.
5. **Ambient Audio Synthesizer**: Generates an in-memory 4-chord progression (*Cmaj7 → Am7 → Fmaj7 → G*) with stereo panning, soft attack/decay envelopes, and audio fade-ins/outs.

### 3.6. Memory Assistant Chat & RAG (`ai_service.py`)
- **Security-First Context Retrieval**: Enforces user authentication and ownership before memory context is compiled for the LLM.
- **Multi-Provider Resilience**: Auto-detects and seamlessly fails over across OpenRouter models, OpenAI, and Google Gemini.
- **Visual Card Association**: The LLM output directly correlates mentioned memories with interactive media cards displayed in the UI.

### 3.7. Audio & Video Multimodal RAG Engine (`ai_engine/rag/`)
- **Video RAG (`ai_engine/rag/video/`)**:
  - Keyframe extraction via OpenCV (sampled every 2 seconds or scene cuts).
  - Keyframe captioning via Vision-Language Models.
  - Dual vector indexing: Dense text embeddings (**BGE-M3**) + Multimodal keyframe embeddings (**SigLIP**).
  - Merged retrieval combining semantic text queries and visual image-to-video searches.
- **Audio RAG (`ai_engine/rag/audio/`)**:
  - Acoustic feature extraction (tempo, spectral centroid, RMS energy) with `librosa`.
  - Audio classification into *Speech*, *Music*, *Ambient*, or *Mixed*.
  - Environmental classification using **ESC-50** taxonomy.
  - Timestamped transcription indexing with **Whisper**.

---

## 4. Database Schema & Security Architecture

The database is built on PostgreSQL (Supabase) with `pgvector` enabled and strict Row Level Security (RLS).

### 4.1. Core Database Tables
1. **`profiles`**: User profiles with auto-updating counters (`vault_count`, `media_count`).
2. **`vaults`**: Collaborative memory containers with cover images and archive flags.
3. **`vault_members`**: Role-based access control (`owner`, `editor`, `viewer`).
4. **`vault_invitations`**: Unique invite codes with expiration dates and usage counters.
5. **`memories`**: Logical memory collections with `memory_date`, `location_name`, and `cover_media_id`.
6. **`media`**: Photos and videos with `storage_path`, `thumbnail_url`, `taken_at`, GPS coordinates, and rich `metadata` JSONB.
7. **`video_jobs`**: Asynchronous video creation queue (`queued`, `processing`, `completed`, `failed`).
8. **`ai_conversations` & `ai_messages`**: Chat history with linked `related_memory_ids`.
9. **`media_embeddings`**: Vector store for similarity searches.

### 4.2. Security & Media Streaming
- **Row Level Security (RLS)**: Enforced across all tables. Vault members can view shared memories, while editing/deleting is restricted to owners and editors via `is_vault_member()` security definer functions.
- **Storage Protection**: Private storage bucket `memories` accessible only via short-lived signed URLs.
- **HTTP 206 Partial Content Streaming**: The `/api/v1/media/stream/{media_id}` endpoint supports byte-range requests for smooth seeking in native desktop and mobile video players.

---

## 5. Direct Comparison: Your Version vs. Your Friend's Version

| Dimension | Your Friend's Version | Your Version (This Codebase) |
| :--- | :--- | :--- |
| **System Scope** | Standalone RAG retrieval pipeline / script audit. | **Complete Full-Stack Application** (Flutter UI + FastAPI Backend + AI Pipelines + DB + Storage). |
| **Frontend UI** | None / Not included in report. | **Full Cross-Platform Flutter App** (Timeline, Vault Sharing, Video Player, AI Assistant Chat, Local Media Curators). |
| **Storage & Media** | Local/Supabase file paths. | **Supabase Storage + Dual Thumbnails + HTTP 206 Byte-Range Streaming Engine**. |
| **Image & Video Ingestion** | Script-based indexing. | **Background Tasks with EXIF parser, Laplacian Blur Scoring, CLIP Zero-Shot Tagging, Whisper VAD**. |
| **Event Grouping** | Query-time retrieval only. | **5-Stage Multi-Signal Event Clustering Engine** (Haversine Spatial + Temporal + Semantic + Trips). |
| **Narrative Generation** | None. | **7-Stage Memory Narrative Intelligence Engine** (Timeline Phases, Key Moments, Grounded Atmosphere). |
| **Video Creation** | None. | **Automated Memory Recap Video Generator** (AI Editorial Director, Ken Burns Effects, Ambient Audio Synthesizer). |
| **RAG & Search** | LangGraph orchestrator (Text/Audio/Video branches) + BGE-M3 + SigLIP + BGE-Reranker-v2. | **Dual Layer**: 1. High-level Context RAG in Chat & Curation. 2. Dedicated Video/Audio RAG Pipelines with BGE-M3, SigLIP, Whisper, and Librosa. |
| **Collaboration** | Single-user retrieval queries. | **Multi-Tenant Vault Sharing** (Role-Based Access Control, Invite Codes, Granular Postgres RLS). |
| **LLM / VLM Stack** | Qwen3-VL, Qwen3 (Ollama). | **Multi-Provider Resilient Gateway** (OpenRouter, OpenAI GPT-4o, Google Gemini, Qwen-2.5-VL). |

---

## 6. Next Steps & Recommendations

1. **Unify the RAG Orchestrator**: Integrate your friend's LangGraph fusion & BGE-reranker pipeline directly into your FastAPI `ai_service.py` / `search_service.py` to give your Flutter frontend the highest possible retrieval recall.
2. **Database Migration**: Ensure `supabase/schema.sql` is deployed to your Supabase instance to enable all RLS policies, vector columns, and triggers.
3. **VLM Weight Caching**: For local development, pre-cache the CLIP (`clip-ViT-B-32`) and faster-whisper models to ensure low-latency media ingestion.
