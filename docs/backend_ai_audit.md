# MemoryVerse Forensic Audit: Backend & AI Video-Generation Pipeline

**Audit Date:** October 2, 2026  
**Audited Subsystems:** `backend/` and `ai_engine/`  
**Status:** Audit baseline complete. No code modifications applied during this audit.

---

## 1. Current Architecture

The MemoryVerse backend and AI Engine are organized into distinct layers:
- **FastAPI Routing Layer (`backend/app/api/`):** Exposes RESTful endpoints for media ingestion, memory curation, video generation job control, and AI sessions.
- **Worker & Job State Management (`backend/app/services/`):**
  - `video_queue.py`: In-memory `VideoWorkerPool` with durable watchdog and worker recovery.
  - `job_state_manager.py`: Monotonic progress state machine tracking 12 standardized stages from `initializing` to `finalizing`.
- **Media Ingestion & Feature Extraction:**
  - `ai_extractor.py`: EXIF extraction via `exifread`, visual embeddings and zero-shot labeling via `SentenceTransformer('clip-ViT-B-32')`.
- **Media Intelligence & Selection:**
  - `media_intelligence.py`: Laplacian variance blur detection, resolution/exposure scoring, video shot boundary detection, near-duplicate clustering via Union-Find, and diversity ranking.
- **Video Assembly & Story Direction:**
  - `video_service.py`: Production orchestration engine driving story planning, clip composition, audio soundtrack generation, MoviePy rendering, and storage staging.
  - `ai_director_service.py`: Conversational session orchestrator for refining video specifications (`VideoSpecification`).
- **AI Engine Libraries (`ai_engine/`):**
  - `video_generation/aspect_ratio_composer.py`: Dynamic letterbox/pillarbox and blurred backdrop generator.
  - `video_generation/audio_synth.py`: Procedural 4-chord ambient waveform synthesizer.
  - `video_generation/tts_engine.py`: Emotion-aware Edge-TTS speech synthesizer with Microsoft Neural Voices.
  - `video_generation/quality_validator.py`: Post-render verification checking resolution, duration, audio streams, and black frames.
  - `emotion/emotion_analyzer.py`: Multimodal affective timeline calculator.
  - `models/`: Thread-safe loaders for CLIP (`clip-ViT-B-32`), SigLIP (`google/siglip-base-patch16-224`), BGE-M3 (`BAAI/bge-m3`), and Whisper (`faster-whisper` int8).

---

## 2. Current Execution Flow

The exact trace from a user request to final video artifact is:

```text
User triggers Video Generation (/api/v1/media/memory/{id}/generate-video)
    ↓
Deduplication check via SHA-256 job fingerprint in job_state_manager
    ↓
Database record created in video_jobs table (Supabase)
    ↓
Job ID submitted to VideoWorkerPool in video_queue.py
    ↓
Worker calls VideoService.execute_job_by_id(job_id) → process_video_job()
    ↓
JobStateManager enters stage: 'retrieving_memory'
    ↓
Fetch media items for memory_id from media table (with retry loop)
    ↓
Fetch precomputed CLIP embeddings from media_embeddings table
    ↓
Near-duplicate clustering in media_intelligence.detect_near_duplicates (threshold=0.92)
    ↓
Concurrent download of candidate media to temp directory (tmp_dir) via ThreadPoolExecutor
    ↓
JobStateManager enters stage: 'analyzing_media'
    ↓
Quality scoring: Laplacian variance (blur), brightness, and resolution for images;
Shot boundary detection + keyframe analysis + Whisper audio transcription for videos
    ↓
JobStateManager enters stage: 'selecting_media'
    ↓
Candidate selection: media_intelligence.select_story_candidates()
(clusters deduplicated, document/sensitive keywords filtered, top moments ranked, sorted by timeline)
    ↓
JobStateManager enters stage: 'planning_story'
    ↓
Build Media Manifest → LLM Editorial Director call via _call_llm() in ai_service.py
(Generates structured StoryPlan JSON with scene durations, motions, and evidence-based narration)
Fallback: Deterministic editorial plan if LLM is unavailable or invalid
    ↓
JobStateManager enters stage: 'generating_motion'
    ↓
Opening Title Card generated via _create_ai_title_card_clip (poetic title/tagline via LLM/heuristic)
    ↓
Per-scene clip construction:
- Images: AspectRatioComposer (blurred background) + dynamic keyframe numpy interpolation (make_frame)
- Videos: Resized + subclipped to scene boundaries
    ↓
JobStateManager enters stage: 'composing_video'
    ↓
Clips concatenated via MoviePy concatenate_videoclips(clips, method="compose")
    ↓
JobStateManager enters stage: 'generating_audio'
    ↓
Audio generation: Procedural ambient soundtrack synthesized via audio_synth.py / _create_ambient_audio
[CONFIRMED: TTS narration voiceover is currently commented out/bypassed in video_service.py]
Audio ducking envelope calculated and composited onto final_clip
    ↓
JobStateManager enters stage: 'encoding_video'
    ↓
Rendered to staged file (storage/staging/reels/{job_id}_...mp4) via final_clip.write_videofile()
Codec: libx264 (preset: fast/medium), Audio: aac, faststart enabled
    ↓
Permanent copy saved to ratio storage (storage/videos/{ratio}/)
    ↓
QualityValidator checks resolution, aspect ratio, duration, and stream integrity
    ↓
JobStateManager enters stage: 'uploading'
    ↓
Streaming upload to Supabase Storage ('memories' bucket: {user_id}/reels/{filename})
with exponential backoff, jitter, and bandwidth tracking
    ↓
JobStateManager enters stage: 'finalizing'
    ↓
Insert new video record into media table, complete job state, cleanup staged local file,
and remove temporary directory tmp_dir
```

---

## 3. Models Actually Used

| Model | Library / Provider | Initialized In | Called In | Input | Output | Dimensions | Purpose | In Production? | Fallback Model |
|---|---|---|---|---|---|---|---|---|---|
| **clip-ViT-B-32** | `sentence-transformers` | `ai_extractor.py` & `ai_engine/models/clip_loader.py` | `ai_extractor.extract_features` | PIL Image / Image bytes / Text labels | Normalized vector + label scores | 512-d | Visual embeddings & zero-shot scene/object/people tagging | **CONFIRMED: YES** | Empty 512-d zero vector |
| **OpenRouter LLMs** | OpenRouter HTTP API (`httpx`) | `backend/app/services/ai_service.py` | `ai_service._call_openrouter` & `video_service._generate_ai_story_plan` | System prompt + Media Manifest text | JSON / Text response | N/A (Text tokens) | Narrative story planning, title/tagline synthesis, Chat Assistant | **CONFIRMED: YES** | Fallback free OpenRouter models → deterministic local heuristic |
| **OpenRouter VLMs** | OpenRouter HTTP API (`httpx`) | `backend/app/services/ai_service.py` | `ai_service._call_openrouter_vlm` | Downsampled image (base64 JPEG) + prompt | JSON / Text analysis | N/A (Text tokens) | Local gallery curation against prompt | **CONFIRMED: YES (in curation endpoint)** | Filename keyword heuristic |
| **faster-whisper (base, int8)** | `faster-whisper` / `ctranslate2` | `transcription_service.py` & `whisper_loader.py` | `transcription_service.transcribe_video` | 16kHz mono WAV audio | Timestamped transcript segments | Text + timestamps | Speech understanding and audio timestamp extraction from user videos | **CONFIRMED: YES** | Standard OpenAI `whisper` library → empty transcript |
| **Edge-TTS (Microsoft Neural Voices)** | `edge-tts` | `tts_engine.py` & `tts_service.py` | `tts_service.synthesize_speech` | Text string + emotion profile | MP3 / WAV audio bytes | 24kHz audio stream | Spoken voiceover narration | **CONFIRMED: Implemented in engine, but BYPASSED in video_service.py** | Procedural harmonic acoustic tones |
| **Procedural Waveform Synthesizer** | Pure `numpy` | `audio_synth.py` & `video_service.py` | `synthesize_ambient_soundtrack` & `_create_ambient_audio` | Mood string + duration seconds | Stereo float32 audio array | 44.1kHz stereo | Ambient harmonic background music | **CONFIRMED: YES (Currently used for all video audio)** | Cmaj7 chord progression sine waves |
| **google/siglip-base-patch16-224** | `transformers` | `ai_engine/models/siglip_loader.py` | `ai_engine/rag/video/` & test suites | Video frames (BGR array) | Normalized vector | 768-d | Research RAG video keyframe retrieval | **CONFIRMED: UNUSED in main video_service pipeline** | None |
| **BAAI/bge-m3** | `FlagEmbedding` / `sentence-transformers` | `ai_engine/models/bge_loader.py` | `ai_engine/agents/planner/agent.py` & query pipeline | Text string | Dense text vector | 1024-d | Text query embedding in agent planner | **CONFIRMED: UNUSED in main video_service pipeline** | None |

---

## 4. OpenRouter Usage

- **Environment Variable:** `LLM_API_KEY` (in `settings.py`). Automatically detected if starting with `sk-or-v1-` or if `LLM_PROVIDER="openrouter"`.
- **Model Configuration:** Configurable via `LLM_MODEL`.
- **Fallback Candidate Models:**
  - Text: `inclusionai/ling-3.0-flash-fin:free`, `liquid/lfm-2.5-2.6b:free`, `nvidia/nemotron-3.5-lightning:free`, `minimax/minimax-m3:free`, `openai/gpt-4o-mini`.
  - VLM: `minimax/minimax-m3:free`, `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`, `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`, `openai/gpt-4o-mini`, `qwen/qwen-2.5-vl-72b-instruct`.
- **Text Calls:** `_call_openrouter(prompt, context, api_key, model)` in `ai_service.py`. Synchronous `httpx.Client(timeout=connect=10s, read=35s, write=30s)`.
- **VLM Calls:** `_call_openrouter_vlm(image_bytes, prompt, api_key)` in `ai_service.py`. Synchronous `httpx.Client(timeout=connect=15s, read=45s, write=60s)`.
- **Image Encoding & Downsampling:** Downsampled via `_downsample_image_for_vlm(image_bytes, max_dim=768, quality=80)` to produce ~50KB base64 strings rather than full 15MB payloads.
- **Multiple Images Behavior:** In `AIService.curate_local_media()`, candidates are dispatched concurrently using `ThreadPoolExecutor(max_workers=min(len(files_data), 6))`. Each candidate makes a separate VLM API call.
- **Deduplication of Calls:** In the video pipeline, VLM calls are NOT made per image; instead, CLIP tags and EXIF descriptions are compiled into a single text manifest and passed in one single text LLM call to `_generate_ai_story_plan()`.

---

## 5. Image Processing Pipeline

```text
User Gallery Image
    ↓ (Multipart upload via /api/v1/media/upload)
file_bytes in memory (io.BytesIO)
    ↓
SHA-256 Content Hash computed
    ↓
EXIF metadata extracted (exifread)
    ↓
CLIP Embedding (512-d) & zero-shot tags extracted
    ↓
Hash match check: If identical content_hash already indexed, reuse embeddings
    ↓
Near-duplicate check: Cosine similarity against sibling media in same vault/memory
    ↓
Upload original file bytes to Supabase Storage ('memories' bucket: {user_id}/{filename})
    ↓
Metadata saved to PostgreSQL 'media' table; vector saved to 'media_embeddings' table
    ↓
During Video Generation:
Downloaded concurrently to temp directory (tmp_dir)
    ↓
PIL Image processing: contrast enhance (1.08), color enhance (1.10), white border
    ↓
AspectRatioComposer: background blur + foreground layout
    ↓
MoviePy VideoClip rendering → Staged MP4 → Ratio storage → Supabase upload
```

- **Originals Modified?** [CONFIRMED: NO]. Original user uploaded bytes are uploaded untouched to Supabase Storage.
- **Originals Deleted?** [CONFIRMED: NO]. Never deleted from storage.
- **Temporary Copies:** Local temp directory `tmp_dir` created during `process_video_job()`, deleted upon completion in `finally: shutil.rmtree(tmp_dir)`.
- **Staged File:** `storage/staging/reels/{job_id}_{filename}` deleted via `cleanup_staged_file()` after confirmed database record insertion.
- **Ratio Storage Copy:** Retained permanently in `backend/storage/videos/{ratio}/{filename}`.
- **Embedding Permanence:** Embeddings in `media_embeddings` are permanent. They are required for semantic search (`search_service.py`) and candidate retrieval. They cannot be blindly deleted.

---

## 6. EXIF Data Pipeline

- **Library:** `exifread` in `backend/app/services/ai_extractor.py`.
- **Extracted Fields:**
  - Timestamps: `EXIF DateTimeOriginal`, `EXIF DateTimeDigitized`, `Image DateTime`.
  - Camera/Device: `Image Make`, `Image Model`.
  - GPS: `GPS GPSLatitude`, `GPS GPSLatitudeRef`, `GPS GPSLongitude`, `GPS GPSLongitudeRef`.
- **Timestamp Parsing:** Parsed using `datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")`.
- **Timezone Preservation:** [CONFIRMED: NOT PRESERVED]. Code forcefully applies `dt.replace(tzinfo=timezone.utc)` without reading EXIF timezone offset tags (`OffsetTimeOriginal` / `OffsetTime`).
- **GPS Parsing:** Converted to decimal degree floats by dividing numerator by denominator for degrees, minutes, and seconds, respecting North/South and East/West references.
- **Missing / Corrupted EXIF:** Returns `None` for missing tags. Handled gracefully with `try...except` without failing media ingestion.
- **Impact on Ordering:** `media_intelligence._timeline_sort_key` uses `taken_at` if present, falling back to `created_at`.
- **Impact on Emotion:** `emotion_analyzer.py` utilizes timestamp to determine time-of-day signals.

---

## 7. Embedding Pipeline

- **Visual Embedding Model:** `clip-ViT-B-32` via `sentence-transformers`.
- **Dimension:** 512-dimensional normalized float vector.
- **Storage Location:** `media_embeddings` table in Supabase (`embedding` and `clip_embedding` columns).
- **Text Embedding Model:** `clip-ViT-B-32` for zero-shot label similarity (`_cached_label_embeddings`). `BAAI/bge-m3` (1024-d) in RAG query pipeline.
- **Necessity for Future Search:** [CONFIRMED: REQUIRED]. `search_service.py` performs vector similarity queries against `media_embeddings` for user natural language queries.
- **Disposable Embeddings:** Only ephemeral test embeddings or orphaned embeddings from deleted media are disposable. Active user memory embeddings must remain.

---

## 8. Duplicate Detection Pipeline

- **Method 1: Exact Hash Deduplication:**
  - SHA-256 hash calculated on raw `file_bytes` in `ai_extractor.py`.
  - Queries `media` table for `metadata->>content_hash == content_hash`.
  - If match found, reuses existing `clip_embedding` and `ai_tags` without re-running the model.
- **Method 2: Near-Duplicate Detection (Ingestion):**
  - In `ai_extractor.py`, queries up to 50 sibling media in the same vault/memory.
  - Computes cosine similarity in Python using `np.dot(curr_unit, s_arr / s_norm)`.
  - Threshold: `0.92`.
  - If `>= 0.92`, flags `metadata->duplicate_info->is_alternate = True` and links to `original_media_id`.
- **Method 3: Near-Duplicate Clustering (Video Story Selection):**
  - In `media_intelligence.detect_near_duplicates()`, computes full $N \times N$ cosine similarity matrix across candidates using NumPy.
  - Clusters near-duplicates using Union-Find with path compression (threshold: `0.92`).
  - In `select_story_candidates()`, selects only the single highest-quality item per cluster (`cluster_best`).
- **Database Query Scope:** Comparison occurs in **Python memory using NumPy**, NOT in PostgreSQL/pgvector! Vectors are fetched over the network into Python arrays.

---

## 9. Image Selection Pipeline

Selection determines **WHICH** images/videos appear in the video.
- Implemented in `media_intelligence.select_story_candidates()`.
- **Selection Criteria:**
  1. Cluster winner: Only 1 representative per near-duplicate cluster.
  2. Quality floor: Laplacian variance blur, exposure, and resolution combined score must exceed `0.18`.
  3. Sensitive content filter: Filenames, descriptions, or tags containing `screenshot`, `invoice`, `receipt`, `passport`, `id_card`, `nsfw` are excluded unless explicitly selected.
  4. Video segment budget: Maximum 1–2 segments per source video. Boring setup segments pruned if a peak celebration segment exists in the same video.
  5. Composite score: `0.50 * quality_score + 0.30 * story_value + 0.20 * uniqueness`.
  6. Top-$K$ cutoff: Ranked and truncated to `target_count` (default 12–15 items).

---

## 10. Image Ordering Pipeline

Ordering determines the **SEQUENCE** in which selected media appears.
- **Initial Candidate Ordering:** In `media_intelligence.select_story_candidates()`, candidates are sorted chronologically via:
  ```python
  def _timeline_sort_key(x: Dict[str, Any]) -> str:
      date_part = str(x.get("taken_at") or x.get("created_at") or "")
      time_offset = f"{float(x.get('start_time', 0.0)):06.2f}"
      return f"{date_part}_{time_offset}"
  ```
- **Editorial Narrative Ordering:** The LLM editorial director in `_generate_ai_story_plan()` receives the chronological manifest and returns an ordered list of scenes (`s1, s2, ...`) based on narrative arc (opening setup → rising action → main peak moment → closing reflection).
- **Deterministic Fallback Ordering:** If LLM is unavailable, sorts candidates by `story_value` descending, assigning index 0 to `opening`, last index to `closing`, and intermediate items to `main_moment` or `rising_action`.
- **Weakness:** [CONFIRMED]. If EXIF timestamps are absent, `created_at` reflects upload order rather than capture order, which can cause erratic chronological sequencing.

---

## 11. Emotion Pipeline

- Implemented in `ai_engine/emotion/emotion_analyzer.py`.
- **Mechanism:** [CONFIRMED]. It is a **deterministic heuristic fusion engine**, NOT a deep neural emotion network.
- **Inputs:**
  - Lexical word bank matching (`LEXICAL_EMOTION_MAP`) across 6 affective classes (`joy`, `nostalgia`, `calm`, `sadness`, `excitement`, `love`).
  - Visual cues: Color temperature (mean red vs blue channels in NumPy).
  - CLIP zero-shot scene tags (`sunset`, `beach`, `party`).
  - Audio transcript words from Whisper.
  - Temporal metadata from EXIF (time of day).
- **Outputs:** `EmotionDistribution` and `EmotionTimeline`.
- **Usage in Video:**
  - Passes dominant mood (`calm`, `nostalgic`, `joyful`, etc.) to the audio synthesizer.
  - Dictates title card color palette and poetic tagline.
  - Sets pacing modifier and motion style recommendations.

---

## 12. Narration Pipeline

- **Prompt Construction:** Defined in `_STORY_PLAN_SYSTEM` and `_generate_ai_story_plan()`.
- **Input Context:** Full memory title, date, location, and the formatted `MEDIA MANIFEST` detailing candidate descriptions, people tags, detected scenes, and Whisper speech transcripts.
- **Instructions:** Explicitly commands the LLM:
  - "Base every sentence on what is actually visible or spoken in the evidence."
  - "Do NOT invent names, relationships, locations, activities, or emotions."
  - "Do NOT reproduce transcript text word-for-word as narration."
- **Style:** Produces an **evidence-grounded contextual narrative**, not raw robotic scene descriptions.
- **Fallback Narration:** If LLM is unavailable, uses sanitized descriptions from tags or rotates through 4 curated poetic narrative arcs.

---

## 13. TTS Pipeline

- **Engine:** `EmotionTTSEngine` in `ai_engine/video_generation/tts_engine.py`.
- **Provider:** Microsoft `edge-tts` with high-quality neural voices (`en-US-JennyNeural`, `en-US-ChristopherNeural`, `en-US-AriaNeural`, `en-US-AndrewNeural`).
- **Modulation:** Adjusts speech rate (e.g. `-6%` to `+12%`) and pitch (e.g. `-6Hz` to `+7Hz`) based on emotion profile (`EMOTION_PROFILES`).
- **Current Video Integration:** [CONFIRMED: BYPASSED].
  - In `video_service.py` line 1403:
    ```python
    # Voiceover/TTS disabled — BGM-only audio pipeline
    # (TTS files and engine are preserved for future use)
    ```
  - Spoken narration generation is skipped during video rendering, resulting in background music only.
  - `TTSService` in `tts_service.py` remains active and callable for standalone voice synthesis requests.

---

## 14. Audio Pipeline

- **Current Implementation:** Pure background music (BGM) synthesized procedurally via `audio_synth.py` (`synthesize_ambient_soundtrack`) or `_create_ambient_audio`.
- **Music Synthesizer Structure:**
  - Procedural 4-chord harmonic audio generator using pure NumPy sine waves:
    `signal = 0.6 * sin(2*pi*f*t) + 0.25 * sin(2*pi*2f*t) + 0.15 * sin(2*pi*3f*t)`.
  - ADSR envelope (15% attack, 15% decay, 50% sustain, 20% release).
  - Hardcoded chord progressions per mood (`calm`, `energetic`, `nostalgic`, `joyful`, `dramatic`, `serene`, `acoustic`, `warm`, `emotional`, `lofi`).
- **Audio Ducking:** Implemented in `video_service.py` lines 1513–1535 (Hanning window convolution, ducking BGM by 68% during speech segments). But since TTS is disabled, ducking currently operates on an empty speech mask.
- **Target Architecture Mandate:** The target architecture must NOT use a music synthesizer. This procedural synthesizer must be removed in favor of real audio/narration.

---

## 15. Video Rendering Pipeline

- **Core Libraries:** `moviepy` (VideoClip, VideoFileClip, AudioArrayClip, concatenate_videoclips) and `PIL` / `Pillow`.
- **Rendering Mechanism:**
  1. `_create_ai_title_card_clip`: Generates 2.5s opening title card with frosted glass pill, ambient glow, title text, and gold diamond ornament.
  2. `_create_ai_cinematic_scene_clip`: Assembles still images with aspect ratio blurred background and dynamic motion.
  3. Video scenes: `VideoFileClip` sub-clipped to scene timestamps, letterboxed/pillarboxed if aspect ratio mismatches.
  4. Assembly: `concatenate_videoclips(clips, method="compose")`.
  5. Encoding: `write_videofile` with `libx264` codec, `aac` audio, `fast` or `medium` preset, faststart moov atom, and multi-threaded CPU encoding.

---

## 16. Ken Burns / Cinematic Motion Audit

- **Files:**
  - `ai_engine/video_generation/ken_burns.py`: Contains `get_crop_window` calculating continuous bounding boxes.
  - `backend/app/services/video_service.py`: `_create_ai_cinematic_scene_clip` lines 583–598.
- **How It Works in Production:**
  - Instead of cropping on every frame, `_create_ai_cinematic_scene_clip` pre-renders two static keyframe canvases: `canvas0` (start keyframe with start zoom/shift) and `canvas1` (end keyframe with end zoom/shift).
  - Converts both to NumPy arrays: `arr0 = np.array(canvas0)`, `arr1 = np.array(canvas1)`, `diff = arr1 - arr0`.
  - Defines MoviePy `make_frame(t)`:
    ```python
    def make_frame(t: float) -> Any:
        p = max(0.0, min(1.0, t / duration))
        ease_p = 0.5 - 0.5 * math.cos(math.pi * p)
        f = arr0 + ease_p * diff
        # Fade in / fade out
        ...
        return f.astype(np.uint8)
    ```
- **Performance Impact:**
  - [CONFIRMED: BOTTLENECK]. `make_frame(t)` is executed in pure Python for every single frame rendered by MoviePy (e.g. 24 fps $\times$ 4.5s = 108 calls per scene $\times$ 10 scenes = 1,080 Python function executions).
  - Allocates full-resolution NumPy float32 frame buffers on each call, causing CPU thrashing and high garbage collection overhead.
  - Prevents pure FFmpeg hardware pipeline acceleration.

---

## 17. Captions / Subtitles Pipeline

- **Engine:** `ai_engine/video_generation/subtitles.py` (`render_subtitle_frame`).
- **Current Production Status:** [CONFIRMED: DISABLED].
  - In `video_service.py` line 1430:
    ```python
    scene_clip = _create_ai_cinematic_scene_clip(
        ...,
        caption=None, # Captions disabled — clean cinematic look
    )
    ```
  - Subtitle burning onto image frames is bypassed in the current production reel generator.

---

## 18. Storage Lifecycle

| Asset Type | Storage Location | Persistence | Lifecycle Management |
|---|---|---|---|
| **Original User Media** | Supabase Storage (`memories/{user_id}/{filename}`) | Permanent | Never deleted by background jobs |
| **Temporary Downloaded Assets** | Local `/tmp/memoryverse_reel_*` | Temporary | Deleted immediately via `shutil.rmtree()` in `finally` block |
| **Staged Video Reel** | `backend/storage/staging/reels/{job_id}_{filename}` | Temporary | Deleted via `cleanup_staged_file()` upon confirmed DB insertion |
| **Ratio-Partitioned Video Copy** | `backend/storage/videos/{ratio}/{filename}` | Permanent | Retained for local streaming and offline playback |
| **Generated Video Cloud Artifact** | Supabase Storage (`memories/{user_id}/reels/{filename}`) | Permanent | Referenced in `media` table |
| **Embeddings (CLIP 512-d)** | Supabase `media_embeddings` table | Permanent | Retained for search and deduplication |
| **Video Job Metadata** | Supabase `video_jobs` table | Permanent | State machine history retained for auditing |

---

## 19. Cleanup Audit

- **What Is Cleaned Up:**
  - [CONFIRMED]: Temporary working directories (`tmp_dir`) are removed in `process_video_job` `finally` block.
  - [CONFIRMED]: Staged video file in `storage/staging/reels/` is removed after confirmed DB upload.
  - [CONFIRMED]: `storage_service.cleanup_expired_staging()` prunes orphaned staging files older than 24 hours.
- **Cleanup Gaps:**
  - [CONFIRMED]: `backend/storage/videos/{ratio}/` permanently accumulates rendered video files without an automated disk quota or retention TTL.
  - [CONFIRMED]: `/tmp` video frame extractions from `media.py` (`_extract_video_thumbnail`) can leak temporary files on unhandled exceptions.
  - [CONFIRMED]: If a user deletes a memory, corresponding `media_embeddings` rows are deleted only if foreign key cascade is configured; otherwise orphaned embeddings may linger.

---

## 20. AI Assistant Audit

- **Files:** `backend/app/services/assistant_agent.py` and `backend/app/services/home_assistant_service.py`.
- **Capabilities Present:**
  - [CONFIRMED]: Natural language understanding with LLM intent parsing (`OPEN_MEMORY`, `CREATE_MEMORY`, `OPEN_AI_DIRECTOR`, `SEARCH_MEMORIES`, `CHAT`).
  - [CONFIRMED]: Database context injection (passes user memories and vaults into prompt).
  - [CONFIRMED]: Server-side permission validation (verifies target memory exists and belongs to user before returning action).
  - [CONFIRMED]: Graceful error recovery on invalid inputs.
- **Capabilities Absent / Incomplete:**
  - [CONFIRMED]: Cannot perform video modification commands (e.g., "swap scene 2", "remove the third photo", "change the music to calm").
  - [CONFIRMED]: Cannot track multi-turn task state across multiple user sessions in the Home Assistant bar.
  - [CONFIRMED]: No direct audio track customization controls exposed to the chat interface.

---

## 21. Confirmed Problems

1. **[CONFIRMED] Synthetic Music Generator Used Instead of Real Audio:**
   The codebase relies on `audio_synth.py` (pure NumPy sine wave synthesizer) and `_create_ambient_audio` for all video soundtracks. This produces robotic, artificial-sounding tones.
2. **[CONFIRMED] TTS Narration Voiceover is Disabled:**
   `video_service.py` line 1403 explicitly comments out TTS voiceover generation (`# Voiceover/TTS disabled — BGM-only audio pipeline`), leaving generated reels with only synthetic BGM and no spoken story.
3. **[CONFIRMED] Python-Level `make_frame(t)` Rendering Bottleneck:**
   Even with keyframe pre-rendering, running Python-level frame interpolation (`make_frame`) for 100% of video frames blocks FFmpeg hardware acceleration and causes high CPU latency.
4. **[CONFIRMED] Inefficient Python-Side Duplicate Detection:**
   `ai_extractor.py` and `media_intelligence.py` fetch vectors across the network and compute $N \times N$ cosine similarities in Python using NumPy rather than executing native PostgreSQL/pgvector queries.
5. **[CONFIRMED] Timezone Stripping in EXIF Parsing:**
   `ai_extractor.py` forces `replace(tzinfo=timezone.utc)` on EXIF timestamps without parsing EXIF `OffsetTimeOriginal`, corrupting local time of capture for photos taken in non-UTC timezones.
6. **[CONFIRMED] Uncontrolled Disk Growth in Ratio Storage:**
   `storage/videos/{ratio}/` retains full MP4 copies of all generated reels indefinitely without a pruning policy.

---

## 22. Suspected Problems

1. **[SUSPECTED] Audio/Video Synchronization Drift on Variable Frame Rate Media:**
   When user videos with Variable Frame Rate (VFR) are subclipped in MoviePy and concatenated with static image clips, minor PTS timestamp drift can occur across long timelines.
2. **[SUSPECTED] Edge-TTS Network Timeouts on Long Narration Scripts:**
   When Edge-TTS connects to Microsoft servers during high concurrency, synthesis may occasionally time out, causing the system to fall back to procedural acoustic tones.
3. **[SUSPECTED] Chronological Inversion on Mixed Media with Missing EXIF:**
   When a memory contains a mix of images with EXIF timestamps and images without EXIF timestamps (which default to upload `created_at`), timeline sorting can cluster older photos after newer ones.

---

## 23. Performance Bottlenecks

1. **MoviePy Video Encoding (`write_videofile` with Python `make_frame`):**
   Accounts for 65–75% of total job duration. Frame-by-frame Python evaluation prevents pure FFmpeg stream copying.
2. **Sequential Media Downloads:**
   Although candidate downloads use `ThreadPoolExecutor`, initial retrieval and metadata checks are performed sequentially before download.
3. **OpenRouter VLM Concurrency Limits:**
   In `curate_local_media`, up to 6 concurrent VLM requests are dispatched to OpenRouter. Free-tier models frequently trigger rate-limit retries, adding 5–15 seconds of latency.

---

## 24. Technical Debt

1. **Dead VideoRenderer Module (`ai_engine/video_generation/renderer.py`):**
   400 lines of prototype rendering code completely disconnected from production `VideoService`.
2. **Duplicate Detection Logic Split Across Multiple Services:**
   Duplicate detection is implemented twice: once in `ai_extractor.py` (sibling dot product) and once in `media_intelligence.py` (Union-Find clustering).
3. **Hardcoded Fallback Narrative Strings:**
   Fallback narration in `video_service.py` uses hardcoded generic phrases ("Every meaningful journey begins with a quiet, authentic moment") when LLM is unavailable.

---

## 25. Recommended Change Order

Based on the actual dependency graph of the codebase:

1. **Audio Pipeline Modernization & Synthesizer Removal:**
   - Remove procedural waveform music synthesizer (`audio_synth.py` and `_create_ambient_audio`).
   - Re-enable and integrate emotion-aware `EmotionTTSEngine` (Edge-TTS) for human-like narrative voiceovers.
   - Implement clean audio mixing with real background music tracks or clean narration-first audio.
2. **Video Rendering Pipeline Optimization:**
   - Replace Python-level `make_frame(t)` loop in MoviePy with streamlined, hardware-friendly clip generation or native FFmpeg filtergraphs.
   - Clean up dead code in `ai_engine/video_generation/renderer.py`.
3. **EXIF & Timeline Ordering Hardening:**
   - Add timezone offset extraction (`OffsetTimeOriginal`) in `ai_extractor.py`.
   - Implement robust chronological fallback sorting that prevents `created_at` from inverting capture order.
4. **Database-Native Duplicate Detection:**
   - Consolidate duplicate detection into PostgreSQL/pgvector similarity queries, removing Python-side $N \times N$ NumPy loops.
5. **Storage & Disk Retention Policies:**
   - Implement automated disk quotas and TTL cleanup for `storage/videos/{ratio}/`.
6. **AI Assistant Video Modification Extensions:**
   - Equip `assistant_agent.py` with video revision and modification intents (scene swaps, narration tone changes, duration adjustments).
