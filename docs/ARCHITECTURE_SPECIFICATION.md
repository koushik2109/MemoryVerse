# MemoryVerse System Architecture Specification

> **Status: architecture snapshot and target reference.** This is not a fresh runtime verification. Confirm implementation details against source code and current check output. Existing video-rendering paths still contain Ken Burns motion and procedural audio code; the product requirement is to remove both from production generation. Do not describe either removal as complete until every active renderer and fallback has been updated and checked.

## 1. Architectural Summary

MemoryVerse has a **Flutter** frontend and a **FastAPI** backend, with **Supabase** (PostgreSQL, pgvector, Storage) for persistence and optional **Upstash Redis** support for job management and caching. AI and media work is split across backend services and the `ai_engine/` package. Provider and active call paths vary by feature and configuration, so inspect the implementation before relying on a provider name in this snapshot. The desired video direction is deliberate composition without Ken Burns motion or procedural synthesized music.

## 2. Mermaid Architecture Diagram

```mermaid
flowchart TD
    %% ==========================================
    %% VISUAL STYLING & CLASSES (Pastel & Clean)
    %% ==========================================
    classDef frontend fill:#E3F2FD,stroke:#90CAF9,stroke-width:2px,color:#0D47A1
    classDef backend fill:#FFF3E0,stroke:#FFCC80,stroke-width:2px,color:#E65100
    classDef aiEngine fill:#F3E5F5,stroke:#CE93D8,stroke-width:2px,color:#4A148C
    classDef storage fill:#E8F5E9,stroke:#A5D6A7,stroke-width:2px,color:#1B5E20
    classDef extService fill:#ECEFF1,stroke:#B0BEC5,stroke-width:2px,color:#263238
    classDef edgeLabel fill:white,stroke:none,color:#424242,font-size:11px

    %% ==========================================
    %% 1. USER & FRONTEND LAYER
    %% ==========================================
    subgraph ClientLayer ["Client Interface"]
        Flutter["📱 Flutter Mobile App<br/><br/>• Select gallery media<br/>• Request memory search<br/>• View generated video"]:::frontend
    end

    %% ==========================================
    %% 2. BACKEND & GATEWAY LAYER
    %% ==========================================
    subgraph BackendLayer ["Orchestration & Gateway"]
        FastAPI["⚙️ FastAPI Backend<br/><br/>• API Routes & Auth<br/>• Request Validation<br/>• Background Worker/Jobs"]:::backend
        Upstash["⚡ Upstash Redis<br/><br/>• Cache (L1/L2, Embed, Emotion)<br/>• Job Fingerprints/Dedup<br/>• Rate Limits & OTP"]:::backend
    end

    %% ==========================================
    %% 3. EXTERNAL AI SERVICES
    %% ==========================================
    subgraph ExternalAI ["External AI Services"]
        OpenRouter["🧠 OpenRouter LLM/VLM<br/><br/>• Story Planning<br/>• Scene Selection<br/>• Structured Output"]:::extService
    end

    %% ==========================================
    %% 4. AI ENGINE (PROCESSING & MEDIA)
    %% ==========================================
    subgraph AILayer ["AI Engine Pipeline"]
        
        MediaIngest["🔍 Media Ingestion<br/><br/>• EXIF / GPS Parsing<br/>• Metadata Normalization"]:::aiEngine
        
        CLIP["👁️ CLIP Visual Intelligence<br/><br/>• Zero-shot extraction<br/>• Image Embeddings<br/>• Semantic Representation"]:::aiEngine
        
        EventDiscovery["🗺️ Event Discovery<br/><br/>• Temporal/Spatial (Haversine)<br/>• Duplicate Detection (Cosine)<br/>• Media Selection & Relevance"]:::aiEngine
        
        EmotionAnalysis["🎭 Emotion Intelligence<br/><br/>• Context & Media Analysis<br/>• Mood Inference<br/>• Emotion Profiles"]:::aiEngine

        StorySynthesis["📝 Narrative Synthesis<br/><br/>• Memory Context Prep<br/>• Narrative Prompting<br/>• Scene Choreography"]:::aiEngine

        AudioTTS["🗣️ Narration / Audio<br/><br/>• Provider varies by path<br/>• Verify delivery controls"]:::aiEngine

        VideoGen["🎬 Video Generation<br/><br/>• Scene composition<br/>• Transitions & captions<br/>• FFmpeg MP4 render"]:::aiEngine
    end

    %% ==========================================
    %% 5. STORAGE & PERSISTENCE
    %% ==========================================
    subgraph StorageLayer ["Supabase Storage Architecture"]
        Postgres["🗄️ PostgreSQL<br/><br/>• Users & Media Metadata<br/>• Memory & Event Records<br/>• Job State tracking"]:::storage
        PGVector["🔢 pgvector<br/><br/>• CLIP Embeddings<br/>• Vector Retrieval<br/>• Similarity Ops"]:::storage
        SupaStorage["🗂️ Supabase Storage<br/><br/>• Original Media<br/>• Staged Thumbnails<br/>• Generated MP4 Videos"]:::storage
    end

    %% ==========================================
    %% DATA FLOW / ARROWS
    %% ==========================================
    
    %% Client to Backend
    Flutter -- "Media requests & Prompts" --> FastAPI
    
    %% Backend to Cache/Queue
    FastAPI <--> "Cache, Jobs, Rate Limits" Upstash
    
    %% Backend to AI Engine (Ingest)
    FastAPI -- "Raw media bytes & metadata" --> MediaIngest
    
    %% AI Pipeline internal flow
    MediaIngest -- "Processed frames" --> CLIP
    CLIP -- "Embeddings & Semantics" --> EventDiscovery
    EventDiscovery -- "Selected Media Context" --> EmotionAnalysis
    EmotionAnalysis -- "Emotional Context" --> StorySynthesis
    
    %% AI Engine to External AI
    StorySynthesis -- "Context + Story Prompt" --> OpenRouter
    OpenRouter -- "Structured Story Plan" --> StorySynthesis
    
    %% Story to Video Flow
    StorySynthesis -- "Emotion + Narration" --> AudioTTS
    AudioTTS -- "Narration Audio (.mp3)" --> VideoGen
    StorySynthesis -- "Scene Order & Captions" --> VideoGen
    
    %% AI to Storage
    MediaIngest -- "Save metadata & EXIF" --> Postgres
    CLIP -- "Store vectors" --> PGVector
    EventDiscovery <-- "Retrieve candidate media" --> PGVector
    VideoGen -- "Final MP4 Artifact" --> SupaStorage
    
    %% Storage to Client
    SupaStorage -- "Stream Video URL" --> Flutter
    Postgres -- "Job Status" --> FastAPI
```

## 3. Strict Compliance Notes for Final Rendering

If this diagram is manually redesigned in a tool like Figma, Draw.io, or Illustrator for a presentation, you **must** follow these verified constraints:

### Redis
*   **Must** be represented as **Upstash Redis (Serverless Redis)** connected directly to the FastAPI Backend block.
*   **Must not** be shown as the primary database or media storage. Its exact roles are L1/L2 caching (embedding/emotion/story caches), job fingerprinting/deduplication, and rate limiting (OTP throttling, password-reset throttling, API/security rate limiting).

### Primary Storage
*   **Must** be grouped under a unified **Supabase** section.
*   **Must not** include MongoDB, Milvus, or Neo4j. The code relies strictly on PostgreSQL (metadata, users, jobs, events), pgvector (CLIP embeddings, similarity operations), and Supabase Storage (original media, generated video, persistent artifacts).

### AI Engine Components
*   **Visual Intelligence:** Represents `CLIP` model usage for embeddings and semantic zero-shot classification. 
*   **Event Discovery:** Implements temporal clustering and spatial grouping using **Haversine** distance (no DBSCAN). Handles near-duplicate filtering (via cosine similarity) and media selection based on relevance, quality, and EXIF/time.
*   **Emotion Intelligence:** Receives media and context to perform emotion analysis, producing emotional context for the story, and eventually influencing TTS pacing and mood profiles.
*   **Narrative AI:** Driven strictly by **OpenRouter**, producing a structured JSON story plan that acts as the blueprint for video generation.
*   **Audio/TTS:** Trace active narration and music providers in code. Do not assume a diagram proves a provider is configured or reachable. Procedural music synthesis exists in the repository and is not an approved production direction; replace production use with an approved, properly licensed source.
*   **Video Generation:** MoviePy/FFmpeg rendering code exists in the repository. Ken Burns motion is present in current code and is slated for removal from all production call paths. The desired composition preserves source media without stretching and uses reviewed scene layouts and approved audio.

### Layout & Style
*   Preserve the strong left-to-right/top-to-bottom logical pipeline shown in the Mermaid structure.
*   Use pastel colors (e.g., light blue for frontend, soft orange for backend orchestration, pale purple for AI tasks, soft green for persistent storage).
*   Keep bullet points inside boxes limited to 2-3 words per bullet for 15-second visual parseability.
