# MemoryVerse: Backend & AI Engine Architecture

This document provides a comprehensive technical explanation of the **Backend** and **AI Engine** systems powering **MemoryVerse**.

---

## 1. System Topology & Process Separation

MemoryVerse employs a decoupled, production-grade microservice architecture separating web API orchestration from compute-intensive machine learning workloads.

```
                              ┌─────────────────────────────────────────┐
                              │           Flutter Client (App)          │
                              └───────────────────┬─────────────────────┘
                                                  │ HTTP / SSE / REST
                                                  ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │               FastAPI Backend (Port 8000)               │
                    │  - API Gateways & Tenant Isolation Auth                 │
                    │  - Layer 1 Home AI Assistant & Intent Dispatcher        │
                    │  - Memory Readiness Service                             │
                    │  - Media Processing & SHA-256 Deduplication             │
                    │  - JobStateManager & Durable Video Worker Pool          │
                    │  - Video Rendering Engine & AspectRatioComposer         │
                    └──────────────┬───────────────────────────┬──────────────┘
                                   │ Internal HTTP             │
                                   ▼                           │ SQL / pgvector
                    ┌───────────────────────────────┐          │
                    │    AI Engine (Port 8001)      │          │
                    │  - SigLIP Multimodal Vision   │          │
                    │  - BGE-M3 Text Embeddings     │          │
                    │  - Audio Beat/Rhythm Analysis │          ▼
                    │  - Whisper Transcription      │   ┌──────────────────────┐
                    │  - Multimodal RAG Indexing    │   │  Supabase PostgreSQL │
                    └───────────────────────────────┘   │  + pgvector Storage  │
                                                        └──────────────────────┘
```

---

## 2. FastAPI Backend (`backend/app/`)

### 2.1 Core Modules & Responsibilities

| Module | Location | Primary Responsibility |
| :--- | :--- | :--- |
| **Home AI Assistant** | `app/services/home_assistant_service.py` | Layer 1 conversational controller. Resolves natural language user queries into deterministic actions and tool calls. |
| **Memory Readiness Service** | `app/services/memory_readiness_service.py` | Inspects memories for media counts, processing state, embedding completeness, and emotion readiness. |
| **Durable Job State Manager** | `app/services/job_state_manager.py` | Tracks video generation job lifecycles across in-memory state and PostgreSQL/Upstash Redis cache. |
| **Video Worker Pool** | `app/services/video_queue.py` | Bounded thread pool executor running durable background jobs with heartbeats and crash recovery. |
| **AI Extractor & Media Processor**| `app/services/ai_extractor.py` | Parallel media intake, SHA-256 content deduplication, EXIF extraction, thumbnail generation, and emotion scoring. |
| **Video Rendering Engine** | `app/services/video_generator.py` | FFmpeg-based multi-track compositor with Ken Burns motion, beat alignment, and `AspectRatioComposer`. |
| **AI Director Service** | `app/services/ai_director_service.py` | Layer 2 conversational storyboard director for custom memory reel generation. |

---

### 2.2 Layer 1: Home AI Assistant Service

The **Home AI Assistant** provides conversational command dispatch on the home screen. Unlike raw chat models, it acts as a deterministic controller:

1. **Deterministic Intent Classifier**:
   - Parses incoming user messages against system intents: `CREATE_MEMORY`, `OPEN_MEMORY`, `CREATE_VAULT`, `OPEN_VAULT`, `LIST_MEMORIES`, `LIST_VAULTS`, `GET_MEMORY_MEDIA_COUNT`, `CHECK_MEMORY_READINESS`, `GENERATE_MEMORY_VIDEO`, `CHECK_GENERATION_STATUS`, `CANCEL_GENERATION`, `OPEN_MEMORY_VIDEO`, `OPEN_AI_DIRECTOR`.
2. **Controlled Tool Execution**:
   - Executes database and queue queries using verified user credentials (`user_id`).
   - Never exposes internal chain-of-thought traces to users.
3. **Structured Action Dispatching**:
   - Returns both a user-friendly conversational response and a typed structured action payload:
   ```json
   {
     "message": "Swarnagiri Temple is ready with 23 photos and 1 completed video reel!",
     "intent": "CHECK_MEMORY_READINESS",
     "structured_action": {
       "action": "OPEN_MEMORY",
       "params": {
         "memory_id": "8de35b27-f516-4aa2-8d30-6ea41988b22f",
         "title": "Swarnagiri Temple"
       }
     },
     "suggested_chips": ["Generate Video", "Open Video Reel", "Show Rooms"]
   }
   ```
4. **Action Handlers in Flutter**:
   - `OPEN_MEMORY`: Navigates directly to `MemoryDetailScreen`.
   - `OPEN_VAULT`: Opens `VaultDetailScreen`.
   - `WATCH_VIDEO_JOB`: Binds live progress to `VideoJobController`.
   - `OPEN_VIDEO`: Launches full-screen `VideoPlayerScreen`.
   - `OPEN_AI_DIRECTOR`: Initializes Layer 2 `AiDirectorScreen`.

---

### 2.3 Memory Readiness Service

Before generating memory reels, the **MemoryReadinessService** checks the underlying media health:

- **Total Media Count**: Minimum of 3 photos or videos required for video generation.
- **Processing Completeness**: Verifies whether all uploaded items have thumbnails and processed metadata.
- **RAG Embedding Readiness**: Checks that visual (SigLIP) and textual (BGE-M3) vectors are populated in `media_embeddings`.
- **Emotion Distribution Readiness**: Confirms that emotion tags (joy, nostalgia, calmness, etc.) have been computed for scene pacing.
- **Active Video Jobs**: Checks if another generation job is already running for the given memory.
- **Output Metrics**:
  - `status`: `"ready"`, `"partially_ready"`, or `"not_ready"`.
  - `suitability_score`: `0.0` to `1.0`.
  - `blockers`: List of blocking reasons (e.g., `"Need at least 3 photos or videos"`).
  - `recommendations`: Suggestions for enhancing video quality.

---

### 2.4 Bounded Parallel Media Processing & SHA-256 Deduplication

When users upload batches of photos and videos:

1. **SHA-256 Fingerprinting**:
   - Computes SHA-256 hash of media file bytes.
   - If an identical file has previously been processed for the user, existing embeddings and emotion scores are reused immediately, avoiding redundant GPU/CPU inference.
2. **Bounded Concurrency**:
   - Background tasks run via `ThreadPoolExecutor(max_workers=6)` preventing resource exhaustion or lock contention.
3. **Incremental Indexing**:
   - Ingests only new items. Unmodified items in existing memories retain their cached vector representation.
4. **Failure Isolation**:
   - Individual upload or indexing errors do not crash or abort the batch; each item reports status independently (`waiting`, `uploading`, `completed`, `error`).

---

### 2.5 Job State Manager & Video Worker Pool

Video generation is inherently asynchronous and long-running:

1. **Job States**:
   - `QUEUED` -> `PROCESSING` -> `VALIDATING` -> `COMPLETED` / `FAILED` / `CANCELLED` / `RETRYING`.
2. **Monotonic Progress Tracking**:
   - Progress advances monotonically from 0% to 100% across normalized stages (RAG retrieval 0-15%, Storyboard generation 15-35%, Media compilation 35-70%, Audio sync 70-85%, Validation 85-100%).
3. **SSE Live Streaming**:
   - `GET /api/v1/media/jobs/{id}/stream` streams real-time Server-Sent Events (`data: {"progress": 0.45, "stage": "rendering", ...}`) directly to Flutter.
4. **Crash Resilience & Watchdog**:
   - Active jobs report regular heartbeats. If a worker process crashes, stale jobs are detected and recovered or marked retryable.

---

### 2.6 Video Rendering Pipeline & Quality Engine

1. **AspectRatioComposer (Never Stretch)**:
   - When photos (e.g., 4:3 or 3:4 portrait) are rendered into a 9:16 vertical reel or 16:9 landscape reel, source media is **never stretched or distorted**.
   - Applies an adaptive, cinematic blurred background generated from the source frame, with the sharp original media centered and properly framed.
2. **Ken Burns Motion**:
   - Applies smooth pan-and-zoom curves tailored to the image focal point and emotional tone.
3. **Audio-Visual Beat Synchronization**:
   - Transitions occur on detected musical beats and downbeats.
4. **Post-Render Quality Validation**:
   - Validates that the rendered MP4 file has:
     - Non-zero byte size and valid container header.
     - Target resolution and valid frame rate (>= 24 fps).
     - Valid audio stream with matching duration.
     - No corrupted or black frames.

---

## 3. AI Engine (`ai_engine/`)

The **AI Engine** runs as an isolated FastAPI microservice on port 8001 dedicated to multimodal cognition, vector search, and storyboard planning.

### 3.1 Multimodal Retrieval-Augmented Generation (RAG)

The RAG pipeline enables semantic and emotional discovery over user memories:

```
[ User Media: Photo / Video ]
           │
           ├──► SigLIP Vision Model ────► 768-d Visual Embedding ────┐
           │                                                         │
           ├──► BGE-M3 Text Model   ────► 1024-d Semantic Embedding ─┼──► pgvector
           │                                                         │    Storage
           └──► Audio/Librosa Engine ───► Tempo, Energy, Beats ──────┘
```

1. **Visual Semantics (SigLIP / CLIP)**:
   - Encodes image frames into dense vector spaces for content recognition (e.g., "temple sunset", "beach hike", "family dinner").
2. **Narrative & Text Embeddings (BGE-M3)**:
   - Encodes memory titles, descriptions, speech transcriptions, and contextual tags.
3. **Audio Rhythm & Emotion**:
   - Analyzes background audio tempo, key, energy, and beat timestamps using Librosa.
   - Speech transcription via Whisper for spoken memories.
4. **Tenant Isolation in Vector Space**:
   - Every vector query enforces `WHERE user_id = :current_user_id` at the database level.

---

### 3.2 Layer 2: AI Memory Director

While Layer 1 (Home Assistant) handles quick actions across memories, **Layer 2 (AI Memory Director)** is the specialized creative co-pilot for a specific memory:

1. **Conversational Curation**:
   - Engages with the user in natural language (e.g., "Make it feel cinematic and slow", "Focus more on the temple carvings").
2. **Candidate Selection via RAG**:
   - Uses RAG retrieval to pick the most relevant and high-quality shots matching the user's intent.
3. **Pydantic Structured Storyboard Output**:
   - Generates and validates complete scene breakdowns (duration, zoom, transition, caption) against structured schemas (`VideoSpecification`).
4. **Revision Loops**:
   - Updates only modified scenes, invalidating only affected stages without re-rendering unchanged scenes from scratch.

---

## 4. Security, Tenant Isolation & Storage Rules

1. **Strict Server-Side Ownership**:
   - Every endpoint verifies `user_id == current_user.id`. Users cannot query, view, modify, or delete another user's memories, media, or video jobs.
2. **Local Staging Before Cloud Storage**:
   - Rendered video reels stage locally to `storage/staging/reels/{job_id}.mp4` before upload.
   - If an upload fails, the local artifact is preserved and retried directly without re-rendering.
3. **No Secrets in Code**:
   - All credentials, API keys, and database tokens load via `.env` configuration.

---

## 5. Summary of Key Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/ai/home-assistant` | Layer 1 Home AI Assistant command interpreter. |
| `GET` | `/api/v1/memories/{id}/readiness` | Memory video readiness assessment. |
| `GET` | `/api/v1/media/jobs` | Lists active and recent video generation jobs for user. |
| `GET` | `/api/v1/media/jobs/{id}` | Status of a specific video generation job. |
| `GET` | `/api/v1/media/jobs/{id}/stream` | SSE live progress stream for a video job. |
| `POST` | `/api/v1/media/memory/{id}/generate-video` | Starts durable background video creation. |
| `POST` | `/api/v1/media/upload-multiple` | Bounded parallel batch upload. |
