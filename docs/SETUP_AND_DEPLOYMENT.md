# Setup and deployment

## Backend and AI services

1. Use Python 3.12/3.14 and create the repository's `.venv`.
2. Install `backend/requirements.txt` and the RAG requirements in the
   environment used to run the AI service.
3. Copy `backend/.env.example` to a private environment file. Set Supabase
   URL/keys, a long random `AI_ENGINE_SHARED_SECRET`, and explicit
   `CORS_ORIGINS`. Keep `ALLOW_LOCAL_JWT=false` outside an isolated local test.
4. Apply all migrations in `supabase/migrations/` to an isolated Supabase
   project before using RAG or jobs.
5. Start the AI service on 8001 and backend on 8000. The backend's
   `AI_ENGINE_URL` must reach the AI service and both processes must share the
   internal secret.

```bash
PYTHONPATH=backend:. .venv/bin/uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
PYTHONPATH=. .venv/bin/uvicorn ai_engine.rag.server:app --host 0.0.0.0 --port 8001
```

## Flutter

Install the supported Flutter SDK, then from `frontend/` run:

```bash
flutter pub get
flutter analyze
flutter test
flutter run
```

Configure Supabase and `API_BASE_URL` through the frontend environment rather
than committing credentials. The current workspace has a Flutter SDK at
`/home/koushik_2109/develop/flutter/bin/flutter`; other machines should use
their own SDK path.

## Production gates

- Apply database migrations and run Supabase RLS policy tests.
- Run `bash scripts/validate_supabase.sh` before applying migrations to a
  real Supabase project.
- Run cross-user authorization tests against isolated users.
- Use private storage and short-lived signed URLs.
- Persist model caches and use a real durable queue for multiple workers.
- Pin dependency versions and record model checkpoint licenses/versions.
- Configure structured logs, request IDs, metrics, backups, and deletion jobs.
