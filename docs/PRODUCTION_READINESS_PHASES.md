# MemoryVerse Production Readiness Roadmap

This document breaks the requested improvements into nine small, reviewable phases for the Flutter Android/iOS app, FastAPI backend, AI engine, and Supabase storage. It is a plan only: it does not implement code or create commits.

The phases preserve the previously discussed sequence: first establish a baseline, then handle emotion-aware narration, latency, the in-app assistant, private local-photo retrieval, video quality, broader RAG quality, multi-agent reliability, and production readiness. Before each phase starts, audit the active branch and current files; deliver the phase as focused changes with its checks.

## Branch comparison snapshot

Compared the committed current checkout (`final` and `research` both point to `97d4ab569bd0484cf6a662e91d1cb54c5e8aa994`) with the available `origin/main` reference (`c4b7c6fc00b25c18d6ceac0448e9fb1dfdd394bf`). There is no local `main` branch in this checkout, so `origin/main` is the comparison point. The working tree already has unrelated modified, deleted, and untracked files; the commit-to-commit comparison does not include those local edits. Preserve and review them before any implementation.

The branch-level diff is large (about 1,830 tracked paths), but much of it is RAG datasets, benchmark results/scripts, paper artifacts, and research documentation. The product-code differences most relevant to this roadmap are:

- **Frontend media/RAG:** the current research checkout changes Android/iOS media discovery to use `photo_manager`, adds local candidate metadata and batched curation in `rag_retrieval_service.dart`, and expands the create/review flow. This is meaningful progress toward local discovery, but readable local image files are still sent to the backend for visual curation, and selected files are uploaded when the user creates a memory. That is not a fully on-device image RAG path.
- **Assistant:** the assistant is present in both branches: `HomeAiAssistantBar`, its Riverpod controller, and `HomeAssistantService` are not unique to `main`. In the current checkout, the branch-specific addition recognizes a “create journal/memory about …” command, opens the create sheet with the prompt, and can auto-start discovery. It is currently a compact command/action bar, not a full conversational assistant experience.
- **Backend media:** current-branch changes add location inference and candidate metadata handling, and tighten video generation to honor explicit media selections. These are capabilities to retain and verify, not reasons to rewrite the whole backend.
- **RAG/research:** current research code adds an `alpha=0.7` hybrid stage and a confidence-triggered cross-encoder cascade with a `0.065` margin, alongside extensive evaluation assets. `origin/main` has the product multimodal retrieval baseline. Treat the research RAMP-RAG work and measured results as experimental evidence until the deployed call path, data, tests, and latency are verified; do not assume a paper benchmark is already the production behavior.
- **Video:** the current render paths still import Ken Burns cropping and procedural soundtrack synthesis (`ai_engine/video_generation/renderer.py`); `backend/app/services/video_service.py` also has an ambient-sound fallback. The video phase must remove these from every production call path, not just replace one renderer function.
- **Frontend visual design:** `home_screen.dart`, `home_ai_assistant_bar.dart`, the core theme/token files, and routing are effectively shared across the two branch tips. The branch comparison therefore does not show a professional visual redesign on either side. The current UI already has useful feature sections and dark styling, but the home flow is vertically long, the assistant is a small command field, and the assistant mixes a second design-token palette with literal purple colors while the broader widgets use other tokens. Phase 3 below addresses visual hierarchy and consistency without discarding working flows.

This is a focused product-code comparison, not a claim that every one of the research dataset/artifact paths was manually reviewed line by line. Re-run the comparison against the intended `main` commit if `origin/main` moves before implementation.

**Priority meanings:** Critical means a blocker for safe or credible production use; High means a major product-quality improvement; Normal means valuable but not a release blocker by itself. These labels express roadmap importance, not a claim that a check has already passed or failed.

## Phase overview

| Phase | Priority | Main outcome |
| --- | --- | --- |
| 1. Baseline and acceptance criteria | Critical | Measure the current app and agree how each phase will be judged. |
| 2. Emotion-aware, research-oriented narration and TTS | Critical | Narration explains supported emotional meaning and speech delivery fits that meaning. |
| 3. Reduce frontend/backend latency | Critical | Faster, measured interactions and resilient Android/iOS behavior. |
| 4. Helpful in-app assistant | High | A grounded assistant helps people understand and operate the app safely. |
| 5. Private local-photo retrieval | Critical | Find permitted local photos without requiring a bulk upload. |
| 6. Professionally composed video and emotion-matched music | High | Videos feel intentionally edited; no Ken Burns or procedural NumPy music in production. |
| 7. RAG quality, evidence, and evaluation | Critical | Retrieval is measurable, traceable, and does not invent evidence when there is no match. |
| 8. Multi-agent reliability and coordination | High | Agents have bounded responsibilities, validated handoffs, and recoverable failures. |
| 9. Production hardening and release readiness | Critical | Configuration, security, operations, and mobile releases are dependable. |

## Phase 1 — Baseline and acceptance criteria

**Priority: Critical**

### Fix

Replace assumptions about what is slow or broken with a reproducible current-state picture. Define observable acceptance criteria before implementation so improvement can be verified instead of guessed.

### How to fix it

- Map the active branch’s app, backend, RAG, agent, storage, and video flows. Record the current branch and commit in the phase notes.
- Measure representative journeys: app launch, memory browsing, image retrieval, assistant response, TTS preview, and video generation. Record p50/p95 latency, error rate, model/DB/queue time, and device/network conditions where available.
- Capture baseline test/analyzer/build results without changing dependencies or treating missing tools as passing checks.
- Define agreed per-phase acceptance criteria, privacy boundaries, supported Android/iOS versions, and video review criteria. Set latency budgets only after measuring the baseline.
- Record known gaps and distinguish verified behavior from a hypothesis or an untested provider integration.

### Main files/modules likely affected

- Read-only inventory across `frontend/`, `backend/`, `ai_engine/`, and `supabase/`
- Existing test locations: `frontend/test/`, `backend/tests/`, `ai_engine/rag/tests/`, and other `ai_engine/` tests
- Add phase notes to documentation only if the implementation phase needs a durable record; no production code should change in this baseline phase.

### What this gives the user

A clear starting point and measurable “done” conditions, so later work can be compared against current behavior.

### Completion checks

- The active branch and relevant flow are identified, and the baseline commands/results are recorded accurately.
- Latency budgets and device matrix are agreed from evidence rather than invented.
- Every planned acceptance criterion identifies how it will be checked and what evidence will be retained.

## Phase 2 — Emotion-aware, research-oriented narration and TTS

**Priority: Critical**

### Fix

Move narration beyond literal captions such as “friends are standing together and are happy.” Build a truthful story from available evidence: what happened, how emotional tone changes, and what the memory may mean. TTS should express the intended tone through supported voice delivery, pacing, pauses, and emphasis.

### How to fix it

- Create a structured narration plan from retrieved memories, event ordering, and available emotion signals. Keep source references and confidence alongside each claim so the system distinguishes observed details from interpretation.
- Update narrative prompts and agent rules to explain feelings and emotional progression when evidence supports it. Avoid diagnosing people or stating uncertain interpretations as facts.
- Produce sentence- or segment-level delivery instructions and map them only to controls the selected TTS provider actually supports. Fall back clearly to neutral delivery when unsupported.
- Let the user preview and regenerate narration before video generation. Show an honest failure/unavailable state instead of silently substituting unrelated speech or audio.
- Cache speech using versioned inputs such as normalized text, voice, delivery settings, provider/model, and schema version.
- Add deterministic tests for grounding, labels, fallback, cache keys, and malformed output. Keep live provider checks separate from offline tests.

### Main files/modules likely affected

- `backend/app/services/tts_service.py`, `memory_narrative_service.py`, and `ai_director_service.py`
- `ai_engine/emotion/emotion_analyzer.py`
- `ai_engine/story_generation/event_aggregator.py` and `arc_generator.py`
- `ai_engine/agents/narrator/agent.py` and evidence-manifest/prompt modules
- `ai_engine/video_generation/tts_engine.py`
- `ai_engine/prompts/story_prompts.py` and related narration prompts
- `frontend/lib/features/memories/presentation/video_creator_sheet.dart`
- `frontend/lib/core/providers/video_job_controller.dart`
- Focused tests in `backend/tests/`, `ai_engine/`, and `frontend/test/`

### What this gives the user

Personal narration with meaningful emotional progression, voice delivery that fits the story, and a review step before rendering.

### Completion checks

- Claims in generated samples trace to retrieved evidence; unsupported emotion/event claims are absent or explicitly qualified.
- Provider controls and neutral fallback are verified, not assumed.
- Preview, regeneration, and failure states work in the app; offline tests pass and provider tests are reported separately.

## Phase 3 — Professional mobile experience and latency

**Priority: Critical**

### Fix

Improve the app’s visual quality and interaction clarity while reducing delays in startup, browsing memories, retrieval, assistant responses, and video creation. The current branch comparison shows no meaningful home-screen redesign over `main`; the shared home screen has a long vertical flow, and the assistant uses a visually separate palette from the app’s main tokens.

### How to fix it

- **Visual upgrade:** start with wireframes for Home, Memories/Timeline, Create Memory, RAG review, assistant, and video preview. Use a photo-first editorial hierarchy: one clear primary action, a featured memory/story, scannable recent memories, a compact timeline preview, and collaboration areas lower in the page. Reduce competing cards and repeated large gaps; preserve the app’s dark/glass style only where it improves media focus and readability.
- **Design-system cleanup:** select one source of truth for colors, type, spacing, radii, elevation, and motion. Replace literal colors and ad-hoc text styles in feature widgets with tokens. Ensure light/dark themes, contrast, dynamic text size, touch-target size, screen-reader labels, loading/empty/error states, and small-device layouts are intentional.
- **Interaction polish:** make the create-memory path guided and previewable; use skeletons for media loading, clear permission explanations, visible upload/index progress, undo/cancel where appropriate, and confirmations for destructive or share actions. Avoid placing the assistant as a tall blocker at the top of the main content.
- **Latency:** use Phase 1’s same journeys to profile Flutter work, image decode/render, list growth, repeated requests, backend routes, database queries, model calls, and queue wait time.
- Apply pagination, thumbnails, lazy loading, request coalescing, bounded caches, or database indexes only where measurements show a benefit.
- Remove duplicate model work; fetch only needed fields; set explicit timeouts and bounded retry/backoff behavior.
- Keep long video jobs asynchronous with stable IDs, progress reporting, idempotent retries, and worker recovery.
- Improve loading, empty, offline, timeout, and retry states; verify app lifecycle and network changes on Android and iOS.
- Add redacted diagnostics and truthful health/readiness checks.

### Main files/modules likely affected

- `frontend/lib/features/home/presentation/home_screen.dart`
- `frontend/lib/features/home/presentation/home_ai_assistant_bar.dart`
- `frontend/lib/core/design/theme.dart`, `tokens.dart`, and `frontend/lib/core/theme/app_design_tokens.dart` (consolidate only after auditing all usages)
- `frontend/lib/features/memories/presentation/create_memory_sheet.dart`
- `frontend/lib/features/memories/presentation/widgets/memory_rag_review_view.dart`
- Other shared widgets in `frontend/lib/core/widgets/` and representative screens under `frontend/lib/features/`
- `frontend/lib/core/api/api_client.dart` and `frontend/lib/config/app_config.dart`
- `frontend/lib/core/providers/video_job_controller.dart` and `home_assistant_controller.dart`
- `frontend/lib/core/services/rag/rag_retrieval_service.dart` and `media_indexing_service.dart`
- `backend/app/config/settings.py` and `backend/app/core/db.py`
- `backend/app/api/media.py`
- `backend/app/services/media_service.py`, `video_service.py`, and `video_queue.py`
- Relevant schema/query files under `supabase/`
- Journey and service tests in `frontend/test/` and `backend/tests/`

### What this gives the user

An app that feels deliberately designed for personal memories, is easier to scan and operate, and responds faster with visible progress during longer work.

### Completion checks

- Before/after timings use the same conditions and meet budgets agreed after Phase 1.
- Android and iOS journeys cover representative devices and at least one slow/offline case.
- Wireframes and visual acceptance criteria are reviewed before significant UI implementation; updated screens are checked at compact and large text sizes and on both platforms.
- Feature screens use the chosen token system consistently, with no newly introduced one-off palette or inaccessible small controls.
- Duplicate requests, retries, and worker restart behavior are covered by deterministic tests.

## Phase 4 — Helpful in-app assistant

**Priority: High**

### Fix

Turn the existing assistant shared by both branches into a polished, genuinely useful in-app helper. Do not build a duplicate assistant. The current command/action bar and intent service are a useful starting point, but they are not yet a full chat experience with rich task guidance, grounded context, and clear progress.

### How to fix it

- Reuse the existing home assistant route/service, Riverpod controller, structured actions, and research-branch create-journal discovery path. Compare `origin/main` before merging any assistant changes and resolve duplicate or divergent intent handling deliberately.
- Agree which tasks it may perform, such as explaining features, finding a memory, starting a draft, guiding permissions, or checking a generation job. Offer a dedicated accessible conversation surface or a clearly expandable assistant panel instead of relying only on the compact command bar.
- Use an allowlist of app actions and typed inputs. Ground answers in the signed-in user’s data and current app state; say when an answer cannot be found.
- Require user confirmation before consequential actions such as deletion, account changes, or sharing/export. Generated text must not call arbitrary routes or bypass authorization.
- Add concise conversation history, cancellation, progress, error recovery, accessibility labels, and a clear return to normal navigation.
- Reuse authentication, repositories, and assistant services; keep execution logic out of widget lifecycle code.

### Main files/modules likely affected

- `frontend/lib/features/home/presentation/home_ai_assistant_bar.dart`
- `frontend/lib/core/providers/home_assistant_controller.dart`
- `frontend/lib/features/home/presentation/home_screen.dart` and shared conversation/input widgets
- `frontend/lib/core/api/api_client.dart`
- Existing assistant routes under `backend/app/api/`
- `backend/app/services/ai_director_service.py`
- `backend/app/services/home_assistant_service.py`
- Relevant definitions under `ai_engine/agents/` and `ai_engine/prompts/`
- Tests in `backend/tests/`, `ai_engine/`, and `frontend/test/`

### What this gives the user

A simpler way to find features, understand memories, and start supported tasks without navigating through every screen.

### Completion checks

- Every tool action is authorized for the active user and restricted to the documented allowlist.
- Unsupported or consequential actions are refused or confirmed explicitly.
- Tests cover arguments, tenant isolation, no-evidence cases, cancellation, and errors; mobile UI supports screen readers and small screens.

## Phase 5 — Private local-photo retrieval

**Priority: Critical**

### Fix

Finish the local-photo discovery work already started in the current branch so retrieval can find relevant permitted images without uploading the user’s entire library. Today local files are forwarded to backend visual curation, and selected assets are uploaded on memory creation; make that transfer boundary explicit and minimize it while protecting images, indexes, thumbnails, and search results.

### How to fix it

- Map the active branch’s retrieval design first. Preserve its established strategy—such as main’s multimodal RAG or research’s RAMP-RAG—unless a measured design review justifies a change. Do not merge branch-specific approaches by assumption.
- Index on-device media through the established platform services and least-privilege photo permissions. Keep stable app media identifiers and only the metadata/embeddings needed for retrieval.
- Keep discovery/indexing incremental and bounded; do not scan the full library on every prompt. Track permission scope and index state so limited access and revocation are reflected correctly.
- Make the boundary explicit: local retrieval can search media available on that device. If server-side generation needs bytes, request access for the selected images and transfer only those for that task; do not silently upload the library.
- Keep remote objects private, enforce authenticated per-user access and row-level/storage policies, and use short-lived access links when remote delivery is needed. Encrypt sensitive local caches where supported and define deletion/retention behavior.
- Exclude raw image bytes and unnecessary personal metadata from logs and analytics. Support index removal/rebuild after permission changes.
- Evaluate retrieval and account isolation using synthetic or consented fixtures; report quality and cold/warm latency separately.
- Compare Android and iOS asset identifiers, file availability, metadata behavior, and local caching separately; do not assume a filesystem path exists on both platforms.

### Main files/modules likely affected

- `frontend/lib/core/services/rag/media_indexing_service.dart` and `rag_retrieval_service.dart`
- `frontend/lib/core/services/media/media_service.dart`
- `frontend/lib/core/services/media/android_media_service.dart` and `ios_media_service.dart`
- `frontend/lib/features/memories/presentation/multi_media_picker_screen.dart`
- `frontend/lib/features/memories/presentation/widgets/media_permission_view.dart`
- `backend/app/api/media.py` and `backend/app/services/storage_service.py`
- `ai_engine/rag/image_media.py` and relevant retrieval/vector-store modules
- Existing policies/schema and forward-only migrations under `supabase/`
- Privacy, retrieval, and tenant-isolation tests in `frontend/test/`, `backend/tests/`, and `ai_engine/rag/tests/`

### What this gives the user

Search over their own permitted photos with a clear privacy boundary and without uploading their entire photo library.

### Completion checks

- Android/iOS denial, limited access, revocation, and re-grant flows behave correctly.
- No image is transferred before the user starts a feature that needs it and the app explains the transfer.
- Tests prove one account cannot retrieve another account’s media or thumbnails.
- No match returns no fabricated image; every result identifies its actual source.

## Phase 6 — Professionally composed video and emotion-matched music

**Priority: High**

### Fix

Create a deliberate edit with a coherent story, well-framed images, readable captions, natural pacing, emotionally grounded narration, and music that supports rather than competes with speech. Do not use Ken Burns motion or the NumPy procedural synthesizer in the production path.

### How to fix it

- Build an editable storyboard/timeline from the approved narrative and selected images. Each scene has a reason to appear, duration, crop/layout, transition, narration segment, and emotional intent.
- Remove Ken Burns from the default renderer and do not replace it with automatic camera movement that recreates the same effect. Prefer clean still composition and subject-aware crop/letterbox choices that preserve faces and avoid stretching or narrow centered imagery.
- Replace NumPy-generated tones with a curated, properly licensed catalog or approved music provider. Record license/source metadata; tag tracks by mood, energy, tempo, and instrumentation; select and mix against the story arc.
- Audit both rendering stacks: `ai_engine/video_generation/renderer.py` currently imports Ken Burns and the NumPy soundtrack module, while `backend/app/services/video_service.py` contains an ambient-sound fallback and uses the procedural soundtrack helper. Route all production generation through one verified composition/mix policy so a fallback cannot reintroduce the rejected output.
- Mix narration clearly above music with measured loudness, ducking, fades, and clipping checks. Align edits to meaningful beats only when it supports the story.
- Let users preview and change images, narration, and music before final export.
- Check aspect ratio, black frames, missing media, clipping, A/V sync, caption overflow, and render completion. Require full playback review; sampled frames cannot prove pacing or sound quality.

### Main files/modules likely affected

- `ai_engine/video_generation/renderer.py`
- `ai_engine/video_generation/ken_burns.py` (retire from the production route after call-site audit)
- `ai_engine/video_generation/audio_synth.py` (remove procedural soundtrack use from the production route)
- `ai_engine/video_generation/aspect_ratio_composer.py`, `transitions.py`, and `quality_validator.py`
- `ai_engine/agents/video/agent.py` and `motion_planner.py`
- `backend/app/services/video_service.py` and `video_queue.py`
- Flutter review/export screens including `video_creator_sheet.dart` and `video_player_screen.dart`
- Video fixtures/tests under `ai_engine/`, `backend/tests/`, and `frontend/test/`

### What this gives the user

Videos that feel intentionally assembled, with licensed emotion-matched music and narration that stays understandable.

### Completion checks

- Production output contains no Ken Burns animation and no NumPy-generated soundtrack.
- Track source/license is recorded, dialogue remains intelligible, and technical media checks pass.
- Full-length samples are reviewed on device in target aspect ratios, including landscape and portrait where supported.
- Human review covers pacing, composition, captions, transitions, and emotional accuracy.

## Phase 7 — RAG quality, evidence, and evaluation

**Priority: Critical**

### Fix

Strengthen retrieval beyond local media discovery. The current research query pipeline includes tuned hybrid fusion and a confidence-triggered reranker, but it also creates synthetic “matching” items when retrieval is empty. The Flutter RAG service falls back to filename/tag heuristics on errors. Those behaviors must be explicit development-only fallbacks or be replaced in production with honest no-match and recoverable error results.

### How to fix it

- Audit the active branch’s ingestion, chunking, metadata filters, modality routing, fusion, reranking, and evidence handoff before proposing changes.
- Preserve dataset provenance and use held-out, deterministic evaluation queries. Measure retrieval quality by modality and user intent; measure latency alongside relevance so a ranking gain does not silently make the app too slow.
- Verify which retrieval path the app actually calls in each branch and environment. Keep RAMP-RAG’s adaptive routing/reliability ideas distinct from the broader multimodal RAG modality coverage; select changes with held-out evaluation and a latency/cost comparison.
- Improve metadata filters and ranking only where ablations identify a repeatable failure. Reuse existing embeddings when possible and version caches/indexes when model or schema changes require rebuilds.
- Pass an evidence manifest from retrieval to the narrator/planner. Validate that claims cite retrieved media and that confidence/uncertainty is preserved through each stage.
- Change empty retrieval behavior to an honest “no matching memories found” response. Never create synthetic memories and present them as the user’s real data.
- Do not show the client heuristic fallback as AI-verified visual relevance. If it remains useful offline, label it as a local filename/metadata search and preserve uncertainty in the UI.
- Keep offline evaluation fixtures and expensive model-backed benchmarks separate; label real, mocked, synthetic, and proxy results accurately.

### Main files/modules likely affected

- `ai_engine/retrieval/hybrid_retriever.py` and `reranker.py`
- `ai_engine/pipelines/query_pipeline.py`
- `ai_engine/rag/image_media.py` and modality-specific ingestion/retrieval modules
- `ai_engine/agents/narrator/evidence_manifest.py`
- Planner and narrator agent implementations under `ai_engine/agents/`
- Retrieval and prompt configuration under `ai_engine/prompts/` and AI configuration modules
- `RAG_PAPER_REPRODUCIBILITY.md` and relevant evaluation scripts/docs
- Tests and benchmark fixtures under `ai_engine/rag/tests/` and `ai_engine/`
- `frontend/lib/core/services/rag/rag_retrieval_service.dart` and its no-network/empty-result UI states

### What this gives the user

More relevant memories, clear separation between visual evidence and metadata-only guesses, and honest behavior when nothing relevant is found or a service fails.

### Completion checks

- Evaluation uses the documented dataset/provenance and held-out queries; quality and latency are both reported.
- Empty retrieval cannot surface fabricated memories as user evidence.
- A backend error cannot silently become an apparently AI-verified match; any offline metadata-only mode is labeled and tested.
- Narration/planner output can be traced to evidence; tests cover filtering, ranking, and missing evidence.
- Benchmark documents distinguish measured results from proxy or synthetic results.

## Phase 8 — Multi-agent reliability and coordination

**Priority: High**

### Fix

Make multi-agent workflows predictable: each agent has a bounded role, validated inputs/outputs, and controlled handoffs. Recover cleanly when a model, retrieval step, or rendering step fails.

### How to fix it

- Draw the active workflow and define agent responsibilities, state fields, allowed tools, retry boundaries, and terminal outcomes. Avoid adding agents where a deterministic service is clearer.
- Validate every agent handoff with typed schemas; reject malformed, oversized, or unsupported output before it reaches the next agent.
- Carry user identity, evidence references, cancellation, deadlines, and job identifiers through the workflow. Enforce authorization at service/tool boundaries rather than trusting agent instructions.
- Add bounded retries, idempotency, timeouts, cancellation, checkpoint/recovery, and clear partial-failure states. Do not retry non-idempotent actions blindly.
- Record structured, redacted traces with stage timing and failure reason so a support engineer can explain what happened without exposing private media or prompts unnecessarily.
- Test orchestration with deterministic fake agents and injected failures; keep live model tests separate.

### Main files/modules likely affected

- `ai_engine/agents/` (planner, narrator, video, and shared state/schema modules)
- `ai_engine/agents/video/agent.py` and narrator/evidence-manifest modules
- Backend orchestration in `backend/app/services/ai_director_service.py`, `video_service.py`, and `video_queue.py`
- API request/response schemas under `backend/app/`
- Agent/orchestration tests under `ai_engine/` and `backend/tests/`

### What this gives the user

Fewer confusing partial results, safer retries, and more consistent outcomes when a workflow uses several AI steps.

### Completion checks

- Every handoff validates against a schema and has a defined timeout/error outcome.
- Tests cover cancellation, malformed output, tool authorization, retries, and worker/process recovery.
- Traces identify failing stages without leaking tokens, secrets, or unnecessary user content.

## Phase 9 — Production hardening and release readiness

**Priority: Critical**

### Fix

Remove unsafe hardcoded environment assumptions and close the operational gaps that prevent reliable production deployment across backend, AI services, Android, and iOS.

### How to fix it

- Inventory hardcoded environment-specific URLs, model names, timeouts, bucket names, and magic defaults. Move legitimate settings to typed, validated configuration with development/staging/production separation.
- Keep secrets in an approved secret store or ignored local environment files. Tracked examples contain placeholders only; logs and screenshots must not expose secrets or user data.
- Replace broad exception swallowing and “empty success” fallbacks with typed user-safe errors, actionable redacted logs, request IDs, and operation-specific retry rules.
- Audit authentication, authorization, tenant scoping, CORS, file type/size/path validation, private storage, rate limits, and log redaction; fix and test findings narrowly.
- Make queues durable and observable, add idempotency and interrupted-job recovery, and add truthful health/readiness checks.
- Review Android/iOS release permissions, identifiers, signing setup, network policy, privacy disclosures, and environment selection. Keep machine-specific SDK and signing files out of version control.
- Add CI checks for formatting/analyzers, unit tests, dependency/security checks, and mobile build artifacts. Document release, monitoring, backup, rollback, and incident procedures.

### Main files/modules likely affected

- `frontend/lib/config/app_config.dart` and `frontend/lib/core/api/api_client.dart`
- `backend/app/config/settings.py`, `backend/app/core/db.py`, and affected API/service modules
- AI provider/configuration modules under `ai_engine/`
- `.env.example` files (placeholders only)
- Supabase policies/schema and forward-only migrations under `supabase/`
- Android configuration under `frontend/android/` and iOS configuration under `frontend/ios/`
- CI/deployment configuration, documentation, and focused component tests

### What this gives the user

A system that can be safely configured and operated in development, staging, and production, with failures that are visible and recoverable instead of silently turning into missing or misleading content.

### Completion checks

- A clean environment can be configured from documented placeholders and the approved secret store; no real secret is tracked or logged.
- Invalid required configuration fails early with clear messages.
- Security and tenant-isolation checks pass, and background work recovers safely from interruption.
- Android and iOS release builds are produced and exercised through a documented staging checklist.

## Cross-phase release gates

Do not call the app production-ready based only on a successful local launch. Before production release, require:

- Component and relevant integration tests pass; ordinary offline tests mock external services and controlled staging checks are reported separately.
- Android and iOS flows are verified on representative device sizes, OS versions, permission states, and network conditions.
- Measured latency meets budgets agreed after Phase 1 baselining.
- Photo privacy, account isolation, deletion, logging, and storage policies pass review.
- Narration is evidence-grounded, voice delivery is reviewed, and complete video outputs receive full playback review.
- Every soundtrack has an approved license/source, and generated media passes technical checks.
- Release configuration, secrets, signing, monitoring, backup/recovery, and rollback procedures are documented and checked.

“No errors” cannot be guaranteed in every environment. The practical release bar is that known failure modes are tested, sensitive data stays protected, failures are surfaced clearly, and recovery paths are verified.

## Delivery approach

- Complete one phase at a time. Split implementation into focused, reviewable commits after that phase’s checks are complete; this document creates no commits.
- Before starting a phase, inspect the active branch and current versions of the listed files. The paths are primary touchpoints, not a promise that every file will change.
- Keep measured results, provider capabilities, and branch-specific RAG behavior tied to evidence. Do not treat benchmark results from one branch as proof about another.
- The main branch’s multimodal RAG and the research branch’s RAMP-RAG are branch-specific context. Phase 5 and Phase 7 should preserve the active branch’s design unless a separate measured decision approves a change.
- Report exactly which tests, device checks, and integrations were run for each phase.
