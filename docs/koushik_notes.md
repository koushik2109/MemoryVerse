# Koushik's MemoryVerse project notes

This is the working handoff document for the **tagged/final MemoryVerse folder**. It records the project map, user requirements, implementation boundaries, and a numbered history of fixes. Append a new `Fix #NNNN` entry after each completed fix; do not rewrite older entries or report a check as passed unless it was run after the relevant change.

## 1. Project summary

MemoryVerse is a personal memory application for organizing photos and videos, finding moments through timelines and search, collaborating through vaults, and turning selected memories into narrated video stories. The product is split into a Flutter client, a FastAPI backend, an AI engine, and Supabase persistence. A separate research code and artifact set evaluates RAMP-RAG and related retrieval methods.

This description is a repository guide, not a claim that every feature is complete or production-ready. Use source code, focused tests, and dated reports to establish current behavior.

## 2. Which folder is authoritative

- **Final project to update:** `/media/koushik_2109/Windows1/Users/gumma/Downloads/MemoryVerse`.
- **Separate comparison/source copy:** `/home/koushik_2109/MemoryVerse`.
- The home copy may contain newer product work, but it is not the destination project. Review and transfer only specific approved features; never replace or synchronize the folders wholesale.
- The final folder already contained extensive local changes before this documentation pass. Preserve them, inspect the relevant diff before each fix, and avoid broad resets, cleanups, or generated-file churn.
- Koushik's current delivery instruction is to stage all changes, make a short, clear commit on `final`, and push after every two newly completed fixes. Inspect the full status and diff before each batch. Do not change remotes or apply remote database changes.

## 3. Product requirements and boundaries

1. Keep the current UI styling and visual design unchanged during the requested backend, AI engine, and feature reliability work unless Koushik later asks for a UI change.
2. Keep the final folder's RAG implementation unchanged. RAG comparisons and reporting are allowed; changes to retrieval code, ranking, configuration, datasets, or RAG behavior require a separate explicit request.
3. Remove Ken Burns motion from every production video path, including alternate renderers and fallbacks.
4. Remove procedural synthesized music from production video paths. Use an approved, properly licensed music source; if none is configured or suitable, surface that state honestly instead of silently substituting unrelated audio.
5. Support both assistant experiences: the Home assistant and the assistant/AI Director available within a memory. Audit all entry points so they use the correct user, memory, and app context and handle progress, cancellation, empty answers, and failures consistently.
6. Keep backend routes thin, put shared behavior in established services/providers, respect account ownership, and preserve the existing Flutter design system without changing its styling.
7. Treat benchmark numbers and test reports as scoped evidence. Mark whether evidence is measured, mocked, synthetic, proxy-based, or historical.

## 4. Repository map

| Path | Responsibility |
|---|---|
| `frontend/` | Flutter application: authentication, Home, memories, memory detail, search, timeline, vaults, profile, settings, assistant surfaces, upload and video-job flows. |
| `frontend/lib/core/` | API client, configuration, navigation, Riverpod providers, repositories, media services, RAG client services, and shared design tokens/widgets. |
| `frontend/lib/features/` | Feature screens and presentation flows. Keep feature-specific UI in its existing area. |
| `backend/app/api/` | FastAPI route handlers and request wiring. |
| `backend/app/services/` | Domain operations for auth, memory, media, search, assistant, AI Director, narrative/TTS, storage, and video jobs. |
| `backend/app/core/` | Shared authorization, database, security, cache, and runtime configuration. |
| `backend/tests/` | Backend unit and integration-style tests; external services should be mocked unless a safe integration run is explicitly requested. |
| `ai_engine/agents/` | Planner, scorer, storyteller, narrator, and video-agent logic. |
| `ai_engine/embeddings/`, `emotion/`, `multimodal/`, `pipelines/` | Model-backed media/text processing, emotion analysis, fusion, and orchestration. |
| `ai_engine/rag/` | Standalone RAG service, modality-specific indexing/retrieval, scripts, fixtures, and research evaluation. Frozen for code changes under the current user constraint. |
| `ai_engine/video_generation/` | Rendering, audio, transitions, captions, TTS, composition, and output validation. Trace all production call sites before modifying video behavior. |
| `supabase/` | Database schema, seeds, and forward-only migrations. Do not apply migrations to a remote project without explicit approval. |
| `docs/` | Product, engineering, setup, audit, and research documentation. `docs/README.md` is the reading index. |
| `results/`, `research/`, `research_paper/`, `notebooks/`, `latex_fragments/`, `latex_fragments_v6/` | Research protocols, outputs, paper sources, notebooks, and generated research artifacts. Preserve provenance and distinguish auxiliary build output from evidence. |

## 5. Important user flows

### Memory and media flow

The Flutter app selects or discovers media, gathers the user's memory context, and calls the backend through its API/repository boundaries. The backend validates identity and ownership, stores metadata/media through configured persistence services, and can invoke AI services for analysis and search. Permission denial, empty selections, uploads, retries, and partial failures must be visible to the user.

### Assistant flows

- **Home assistant:** starts from Home and can help with app actions such as finding or creating a memory.
- **Memory assistant / AI Director:** works with one selected memory, its evidence and media, and its creative direction. It must not accidentally answer using another memory or lose the memory context during navigation.
- Audit all assistant entry points, providers/controllers, backend routes/services, and prompts together. Prefer shared domain operations with context-specific presentation rather than duplicate business logic.
- Validate structured model output before acting on it. Never allow free-form model output to select arbitrary endpoints or bypass authorization.

### Video flow

The user selects a memory and media, reviews a story direction, and starts asynchronous generation. Backend job state and frontend progress should stay aligned through completion, cancellation, failure, and retry. The final output should preserve source proportions, use deliberate composition, include only approved audio, pass technical validation, and remain reviewable in the app.

Current code includes Ken Burns and procedural audio components in rendering-related paths. Their removal is a future implementation requirement, not a completed change. Audit backend and AI-engine render routes and fallbacks before declaring the work finished.

## 6. RAG boundary and evaluation notes

The tagged/final folder contains the RAMP-RAG research implementation. The home copy has a different retrieval path. Keep these separate. Under the current request:

- Do not port the home copy's retrieval pipeline into the final folder.
- Do not edit RAG scoring, routing, tenant filters, empty-result behavior, datasets, tests, or configuration.
- A comparison may describe concrete code differences and artifact limitations, but must not imply that a research benchmark proves production user benefit.
- Cite the artifact and its evaluation scope when discussing a metric. The repository's research documents identify proxy datasets and held-out subsets; verify their current provenance before quoting results externally.

## 7. Documentation guide

Start with [`README.md`](README.md). It groups the material into product/engineering, setup, audit, and research sections. Older copies in [`legacy_documentation/`](legacy_documentation/) are preserved for comparison. Root-level research artifacts remain at their existing paths where reproduction scripts expect them.

The project explainer and architecture documents contain historical descriptions. They now carry notices directing readers to verify active code paths and distinguish current implementation from target behavior. Historical test reports remain historical; they are not a current green build.

## 8. Ignore files and their locations

All project files below are inside the tagged/final folder `/media/koushik_2109/Windows1/Users/gumma/Downloads/MemoryVerse`. Dotfiles may be hidden by the file browser; enable “show hidden files” to see them.

- [Root `.gitignore`](../.gitignore) — project-wide rules for local credentials, caches, generated builds, staged media, and research build output. This is the file updated in the documentation/hygiene pass.
- `backend/.gitignore` — Python/backend-local rules.
- `frontend/.gitignore` — Flutter/Dart build and plugin rules.
- `frontend/android/.gitignore`, `frontend/ios/.gitignore`, and `frontend/windows/.gitignore` — platform-generated files and signing/local configuration.
- `.idea/.gitignore`, `.pytest_cache/.gitignore`, and `backend/.pytest_cache/.gitignore` — editor or test-tool state.

The root and component/platform rules complement one another. The backend, frontend, and platform ignore files already contain their respective standard local/build rules; I inspected them and left their existing local edits intact. The project `.gitignore` files themselves are not ignored by Git.

## 9. Checks and reporting

Use the configured runtime and run the smallest relevant check after each implementation fix. Standard repository commands are listed in `AGENTS.md`. Do not install dependencies or run expensive/provider-backed benchmarks by default. For each fix, report:

- the exact check and working directory;
- the observed result, including warnings or failures;
- any platform/provider/database flow that was not exercised;
- the files changed and any remaining gap.

Never claim that everything is “perfect” or production-ready from static inspection alone. State what the evidence establishes and what remains unverified.

## 10. Numbered fix log

Use sequential IDs (`Fix #0001`, `Fix #0002`, …). Add one entry after each completed fix. If one request contains several independently reviewable fixes, give each a separate number. Preserve prior entries.

### Fix #0001 — Documentation entry point and ignore rules

- **Date:** 2026-10-30
- **Scope:** Documentation navigation, current-vs-historical labels, project handoff notes, and generated/local-file ignore rules.
- **Files:** `.gitignore`, `docs/README.md`, `docs/ARCHITECTURE_SPECIFICATION.md`, `docs/PROJECT_EXPLAINER.txt`, `docs/koushik_notes.md`.
- **What changed:** Added a categorized docs index; added this project map and durable fix-log format; clarified that historical test claims are not current verification; documented the requirement to remove Ken Burns and procedural music from production paths; refined ignores for local secrets, tool caches, builds, and staging output while allowing normal image assets and research results to remain trackable.
- **Verification:** The docs index links resolve. Ignore-rule checks confirmed that local credentials, Python environments, Flutter builds, staging output, and LaTeX auxiliaries are ignored, while image assets, benchmark JSON, docs, environment templates, and the paper PDF remain trackable. All 18 paths removed from tracking now match ignore rules. No application tests were run because this fix changes documentation and ignore rules only.
- **Index cleanup:** Removed 18 already-tracked generated paths (local staged videos and Flutter ephemeral outputs) from the Git index with `git rm --cached`; all working copies were retained. This stages their removal from the repository while keeping them available locally. An already-staged MoviePy temporary-file deletion was left unchanged and its filename pattern is now ignored too. `.gitignore` will prevent these generated paths from being added again.
- **Not included:** No backend, AI engine, Flutter behavior, UI styling, RAG code, database, or external service was changed.

### Fix #0002 — Confirm ignore-file locations in the tagged folder

- **Date:** 2026-10-30
- **Scope:** Locate and inspect every `.gitignore` in the final project folder and document its purpose.
- **Files:** `docs/koushik_notes.md`.
- **What changed:** Recorded the exact project-wide `.gitignore` path, each backend/frontend/platform ignore file, and the generated IDE/test-cache ignore files. Confirmed the hidden-file visibility detail for file browsers.
- **Verification:** All nine ignore files were found under the tagged project folder. Git does not ignore the `.gitignore` files themselves. The nested component/platform files were left unchanged because their rules are already scoped to those components and their current diffs include existing local edits.
- **Not included:** No nested `.gitignore` was rewritten; no application code or UI was changed.

### Fix #0003 — Diagnose the oversized Git change list and define commit hygiene

- **Date:** 2026-10-30
- **Scope:** Explain the reported 1k+ file change list, prevent line-ending churn on future checkouts, and distinguish generated/local files from project source and documentation.
- **Files:** `.gitattributes`, `.gitignore`, `docs/koushik_notes.md`.
- **Cause:** The Windows-mounted checkout had CRLF line endings for many text files while Git's index stored LF. Git therefore reported 1,858 tracked paths modified even though most had no content change. After line-ending normalization, exact object-ID comparison found that 1,840 worktree files matched the index; these were refreshed without staging content edits. The current checkout reports 18 content-modified tracked files, five unstaged deletions, 19 already-staged deletions, and 53 untracked files (many presented as grouped directories by Git). This explains the inflated count; it was not 1,000 new source files.
- **What changed:** Added `.gitattributes` to standardize text checkouts to LF while identifying common media/model/document formats as binary. Expanded root `.gitignore` for generated backend media, staging output, Flutter/build and Python caches, local credentials, and LaTeX build auxiliaries. The new generated MP4 under `backend/storage/videos/` is now ignored. Existing tracked files remain tracked until explicitly removed from the index; ignore rules do not erase or hide tracked content.
- **Index cleanup:** Eighteen tracked generated files (staged video artifacts and Flutter ephemeral outputs) were removed from the Git index only; their working copies were preserved. A separate staged deletion of a MoviePy temporary MP4 predated this cleanup and was left untouched. These index changes are visible in the current staged set; nothing was committed or pushed.
- **Commit review guidance:** The ignore policy, LF policy, curated documentation under `docs/`, and source research paper (`research_paper/README.md`, `.tex`, and PDF) are plausible project changes after reviewing the complete diff and confirming the paper/PDF provenance. The Windows local-properties example is suitable only if it contains placeholders and no machine-specific paths or credentials. `skills-lock.json` can be committed if the team intends to pin agent tooling; `opencode.jsonc` is currently only a schema declaration and has no demonstrated project configuration value. `devl.ps1` is a local Windows launcher and should be reviewed for portability and relevance before deciding to track it. Keep caches, virtual environments, build output, generated/staged media, local environment files, and LaTeX auxiliaries out of commits. Review the current app/source diffs separately; under the user's RAG constraint, do not include the three modified `ai_engine/rag/tests/` files in an approved change. Do not commit any of this until the user chooses the next implementation step.
- **Current change categories:** 19 staged deletions (18 generated-file index cleanups plus the previously staged MoviePy temp deletion); 18 content-modified tracked files; five unstaged deletions of old `documentation/` copies corresponding to documentation organization; 53 untracked files, including 45 documentation entries, three paper files, `.gitattributes`, `devl.ps1`, `opencode.jsonc`, `skills-lock.json`, and the Windows example file. The `docs/` directory display is grouped in ordinary `git status` output, so its untracked files are not all shown as separate lines.
- **Verification:** `git status --porcelain` reports 19 staged records, 23 unstaged records, and seven grouped untracked directory/file records representing 53 untracked files (49 status records total). `git diff --stat` reports 18 modified files and five removed old documentation files; `git diff --cached --stat` reports 19 deletions. `git diff --check` produced no whitespace errors. `git check-ignore -v` confirmed that backend generated videos/staging files, Flutter build output, and `.venv` paths match ignore rules. No application tests were run because this fix concerns repository hygiene and documentation only.
- **Not included:** No backend/AI behavior, RAG code or tests, Flutter behavior or styling, database, or external service was changed as part of this fix.

### Fix #0004 — Keep AI Director sessions isolated to the selected memory

- **Date:** 2026-10-31
- **Scope:** First implementation fix after the documentation and repository-hygiene work. Prevent the memory-scoped assistant from displaying or restoring another memory's conversation during session switches.
- **Files:** `frontend/lib/core/providers/ai_director_controller.dart`, `frontend/lib/features/memories/presentation/ai_director_screen.dart`, `frontend/test/ai_director_controller_test.dart`, `docs/koushik_notes.md`.
- **What changed:** The controller now clears its old session immediately when initializing a different memory, ignores stale responses from earlier session loads and in-flight operations, and restores the last confirmed message list if sending fails. The screen only renders a session whose `memory_id` matches the memory currently open, including the frame before the replacement session request starts.
- **Verification:** From `frontend/`, `flutter test test/ai_director_controller_test.dart` passed (2 tests). `flutter analyze lib/core/providers/ai_director_controller.dart lib/features/memories/presentation/ai_director_screen.dart test/ai_director_controller_test.dart` reported no issues. These checks cover local state and race handling; live backend, provider, and device flows were not exercised.
- **Not included:** No assistant styling, Home assistant behavior, backend service, RAG implementation/tests, database, or external service was changed. This is not a full audit or completion of the broader assistant feature.

### Fix #0005 — Clear stale Home assistant actions and replies

- **Date:** 2026-10-31
- **Scope:** Prevent an earlier Home assistant action or slower reply from remaining active after a newer query, including when the assistant panel is cleared.
- **Files:** `frontend/lib/core/providers/home_assistant_controller.dart`, `frontend/test/home_assistant_controller_test.dart`, `docs/koushik_notes.md`.
- **What changed:** Starting a query now resets the prior response and structured action while retaining suggested prompts. A request-generation guard discards stale success and error results; clearing the panel invalidates any pending request. A `NONE` action therefore cannot leave an older navigation/action button available.
- **Verification:** From `frontend/`, `flutter test test/home_assistant_controller_test.dart` passed (3 tests). `flutter analyze lib/core/providers/home_assistant_controller.dart test/home_assistant_controller_test.dart` reported no issues. No live backend/provider call was made.
- **Not included:** No Home assistant styling, backend logic, RAG code, data, or configuration was changed.

### Fix #0006 — Repair development scripts and add continuous checks

- **Date:** 2026-10-31
- **Scope:** Audit and harden the Linux and Windows development entry points, make research runners portable and non-installing, and add GitHub Actions checks for Bash, PowerShell, backend, and Flutter.
- **Files:** `.gitignore`, `.github/workflows/ci.yml`, `dev.sh`, `devl.sh`, `dev.ps1`, `devl.ps1`, `scripts/dev1.sh`, `reproduce.sh`, `reproduce_v6.sh`, `docs/koushik_notes.md`.
- **What changed:** Removed the ignore rule that hid the documented `devl.sh` launcher and added `.venv-linux/` to local environment ignores. `dev.sh` and `scripts/dev1.sh` now forward to the maintained Linux launcher; `dev.ps1` forwards to the maintained Windows launcher. Linux launch no longer kills processes on ports 8000/8001, checks health before reporting successful startup, accepts any selected Flutter target, fails clearly for unknown options or unavailable dependencies/devices, and stops masking Flutter failures with unrelated target fallbacks. It recognizes a separate `.venv-linux` when the normal `.venv` contains only a Windows environment. Windows device selection now accepts any Flutter target, and requested Flutter analysis failures stop launch. The reproduction scripts use a caller-selected or project-local Python path, no longer include a hard-coded macOS package directory, and do not install packages automatically. Added CI jobs for Bash syntax, PowerShell parsing, backend tests, and Flutter analysis/tests on pushes, pull requests, and manual runs.
- **Verification:** Bash syntax validation passed for all eight project Bash scripts present in this checkout (excluding generated Flutter environment output and local ignored agent utilities). `devl.sh --help` and invalid-option exit behavior passed. The workflow YAML parsed locally and contains all four expected jobs and push, pull-request, and manual triggers. Both reproduction runners fail cleanly with setup instructions because this host has no Linux project virtual environment. PowerShell is unavailable locally; the Windows CI job will parse both PowerShell launchers when GitHub Actions runs. `flutter analyze` could not start because the read-only mount prevents Flutter from writing its SDK cache; backend tests could not run because this host's Python lacks pytest and the project ML dependencies. No services, devices, model downloads, or benchmark reproductions were started.
- **GitHub note:** GitHub Actions checks commits after they are pushed. To prevent unreviewed changes from merging, require the four CI jobs as branch-protection checks on `final` (or the selected protected branch) in GitHub settings.
- **Not included:** No RAG application implementation, RAG tests, retrieval configuration, assistant behavior, UI styling, or generated benchmark artifact was changed. The five pre-existing RAG working-tree changes remain untouched.

### Fix #0007 — Rebuild Flutter metadata for Linux and support physical devices

- **Date:** 2026-10-31
- **Scope:** Resolve the Android build failure caused by Windows-generated paths, verify the Flutter client, and make Android emulator, iOS simulator, and physical-phone API addresses configurable.
- **Files:** `.gitignore`, `devl.sh`, `devl.ps1`, `frontend/.env.example`, `frontend/android/app/src/main/AndroidManifest.xml`, `frontend/ios/Runner/Info.plist`, `frontend/lib/core/api/api_client.dart`, `frontend/test/api_client_base_url_test.dart`, `docs/MOBILE_DEVELOPMENT.md`, `docs/README.md`, `docs/koushik_notes.md`.
- **What changed:** Removed generated Flutter build/package/platform metadata with `flutter clean` and recreated it using Linux `flutter pub get`; the regenerated Flutter application path now points to the mounted Linux project path. Added per-platform API overrides so Android emulator defaults to `10.0.2.2`, iOS simulator and desktop use loopback, and physical devices can use the development machine's LAN IP. Added URL selection tests. Android cleartext traffic is limited to the existing debug manifest; iOS ATS permits local-network development connections and declares the local-network usage purpose. Both maintained dev launchers bind the backend to loopback by default and accept `MEMORYVERSE_BIND_HOST=0.0.0.0` for explicit physical-device testing. Added the generated Android `.kotlin/` directory to ignores.
- **Verification:** Linux `flutter pub get` succeeded; `flutter analyze` reported no issues; all 20 Flutter tests passed, including URL selection for emulator, simulator, desktop/web, and physical-device overrides. Android manifests and iOS plist parse successfully. An Android 14 emulator is connected. Regenerated Flutter config and build files contain no old Windows project path. The APK build passed the original stale-path failure point, then requested the missing Android SDK CMake 3.22.1 package; it was stopped under the repository rule against installing tools automatically. The Android APK and emulator launch therefore remain unverified until CMake is installed. An iOS build cannot run on Linux because Xcode/macOS is required.
- **Setup review:** The automatic reviewer rejected changing the backend's default bind to `0.0.0.0` because it exposes the dev server on every interface. Kept loopback as the default and added an explicit opt-in for physical devices. The CMake SDK installation was not completed; its SDK folder currently contains only installer metadata.
- **Not included:** No UI styling, RAG implementation, RAG tests, or backend business logic was changed. The five pre-existing RAG working-tree changes remain untouched.

### Fix #0008 — Add an iOS simulator build to CI

- **Date:** 2026-10-31
- **Scope:** Ensure iOS platform code is compiled on a macOS/Xcode runner even when local development is on Linux.
- **Files:** `.github/workflows/ci.yml`, `docs/koushik_notes.md`.
- **What changed:** Added a macOS GitHub Actions job that installs the locked Flutter packages and builds the app for the iOS simulator with code signing disabled. This checks CocoaPods/Xcode integration without requiring a signing identity or device provisioning.
- **Verification:** Workflow YAML parsing and `git diff --check` passed. The job cannot run until the workflow is pushed to GitHub; no iOS build was run locally because this host is Linux and has no Xcode.
- **Not included:** No iOS UI styling, RAG code, backend behavior, or platform permissions were changed.

### Fix #0009 — Save downloaded media through the system media library

- **Date:** 2026-10-31
- **Scope:** Replace direct shared-storage file copies in media and video download flows with the platform media-library API.
- **Files:** `frontend/lib/core/services/media/media_gallery_export_service.dart`, `frontend/lib/features/memories/presentation/memory_media_viewer.dart`, `frontend/lib/features/memories/presentation/video_player_screen.dart`, `frontend/lib/features/memories/presentation/ai_director_screen.dart`, `frontend/lib/features/memories/presentation/widgets/movie_card.dart`, `frontend/android/app/src/main/AndroidManifest.xml`, `docs/koushik_notes.md`.
- **What changed:** Added one export service using `photo_manager` to save photos into the Pictures library and videos into Movies, under a MemoryVerse album/folder where supported. Replaced direct writes to `/storage/emulated/0/...` with the platform media-library API, and iOS Downloads-directory copies with Photos library saves. Desktop Downloads behavior and app-private offline copies remain in place. Added legacy external-write permission for Android 9 and below only; newer Android versions use scoped media storage. This avoids relying on Android shared-storage paths that scoped storage blocks.
- **Verification:** `flutter analyze` reported no issues and all 20 Flutter tests passed. The Android emulator is available, but its app build and live gallery-save operation remain unverified until Android SDK CMake 3.22.1 is installed. The iOS gallery path is statically checked only; iOS execution requires macOS/Xcode.
- **Not included:** No UI styling, RAG code, backend behavior, or media-discovery behavior was changed.
