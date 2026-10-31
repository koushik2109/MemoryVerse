# Audit implementation follow-up

## Implemented

- `frontend/lib/core/config/app_config.dart`: centralized environment configuration;
  constant Dart defines, missing-dotenv handling, URL normalization, no hardcoded
  Supabase account fallback.
- `frontend/lib/main.dart`: configuration/initialization failure screen instead of
  entering an app without initialized authentication; removed forced dark override.
- `frontend/lib/core/api/api_client.dart`: uses centralized API configuration.
- `frontend/test/app_config_test.dart`: configuration tests; executed successfully.
- `backend/app/api/media.py`: downloads fail closed if job ownership is absent;
  removed fallback that returned the first MP4 in a shared directory regardless
  of the requested job.
- `backend/tests/test_security_regressions.py`: eight isolated security cases.
- `supabase/migrations/20261009_rag_tenant_security.sql`: qualified vector types and
  operator classes; invoker RPCs and service-role-only execute grants.
- `supabase/migrations/20261010_application_tables.sql`: draft additive application
  table migration. Not applied; do not deploy until database compatibility tests pass.

## Verification

`PYTHONPATH=backend:. .venv/bin/python -m pytest -q backend/tests ai_engine/test`
returned **114 passed, 13 warnings**. Warnings include library deprecations
and unavailable NVML initialization. These are not live tenant-isolation or
production render tests.

Dart formatting succeeded. Flutter analysis reported no issues and all six
Flutter tests passed.

## Correction to earlier audit

The older `backend/app/supabase/schema.sql` omits several tables, but the canonical
`supabase/schema.sql` already defines memories, video jobs, embeddings, conversations,
and messages. Earlier documents incorrectly generalized the older schema's gaps
to the whole repository. Use the canonical schema as the starting point, not the
older backend copy. Existing tables are not retrofitted by CREATE TABLE IF NOT EXISTS;
the new migration needs explicit upgrade-path tests before deployment. Historical
documentation claiming missing pytest is superseded: the root `.venv` runs pytest.

## Remaining release blockers

- No production database migration/RLS execution has been performed; the
  disposable local PostgreSQL/pgvector migration validation passed.
- Existing RPC overloads, existing storage policies, and upgrade constraints still
  need review; new RPC grants alone do not revoke older overloads.
- Device-specific Flutter behavior and release builds remain unverified.
- Full upload/search/follow-up/generation/deletion workflows and ML quality
  measurements remain unverified. No production-readiness claim is made.
