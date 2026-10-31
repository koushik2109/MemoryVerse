# MemoryVerse project discovery

Audit date: 2026-10-09. This document records repository evidence, not the
intended architecture described by older project notes.

## Repository and runtime

- Frontend: Flutter/Dart under `frontend/`, using Riverpod, GoRouter, Dio,
  Supabase Flutter, and platform media services.
- Backend: FastAPI/Python under `backend/`, mounted at `/api/v1`.
- AI engine: a Python package under `ai_engine/` plus a separate FastAPI RAG
  process at port 8001.
- Persistence: Supabase PostgreSQL, Storage, and intended pgvector indexes.
- Jobs: a backend video worker with Redis when configured and a local journal
  fallback.
- Rendering: MoviePy/FFmpeg code in `backend/app/services/video_service.py`
  and `ai_engine/video_generation/renderer.py`.

## Entry points and checks

- Backend: `backend/app/main.py`, `uvicorn app.main:app` from `backend/`.
- RAG service: `ai_engine/rag/server.py`.
- Flutter: `frontend/lib/main.dart`.
- Existing Python tests are under `backend/tests/` and `ai_engine/test/`.
- Expected Flutter checks are `flutter analyze` and `flutter test`.

## Baseline evidence

- `python3 -m compileall -q backend ai_engine`: completed with warnings from a
  vendored RAG virtual environment, no compile error in project code.
- Root-system `python3 -m pytest -q` could not run because that interpreter has
  no pytest module; the repository `.venv` test command ran successfully.
- Flutter checks ran successfully using the available SDK at
  `/home/koushik_2109/develop/flutter/bin/flutter`.
- The working tree already contained user changes before this audit; those
  changes were preserved.

## Important facts established

1. The app is Supabase/Postgres based, not Astra DB based.
2. The frontend already has substantial screens and repositories, but API
   contract and deep-link coverage are limited.
3. The checked-in schema did not cover all tables used by the code. A versioned
   RAG migration was added separately rather than silently rewriting the
   existing schema.
4. Before this audit, direct media streaming did not authenticate the caller,
   and the standalone RAG service accepted unauthenticated caller-supplied
   user IDs.
5. Documentation contains stronger claims than the code verifies, especially
   around TTS, beat synchronization, durable jobs, and measured evaluation.
