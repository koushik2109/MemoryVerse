# MemoryVerse repository instructions

This is the checked-in operating guide for agents working in this repository. It
applies from the repository root unless a more-specific `AGENTS.md` is added in
a child directory.

## Instruction precedence and local context

- Follow system/developer instructions and the user's current request first.
- Before non-trivial work, read `.agents/rules/rules.md` and
  `.agents/memory.md` when they exist. They are local, ignored context and may
  contain recent decisions or preferences that are not suitable for commits.
- Keep this file as the shareable baseline. Do not copy private credentials,
  tokens, personal notes, or raw local-memory payloads into it.
- If guidance conflicts, call out the conflict briefly and choose the smallest
  safe change consistent with the user's request.

## Repository map

- `frontend/` — Flutter client. UI, Riverpod state, repositories, media
  permissions, and platform integrations live here.
- `backend/` — FastAPI service and its Python tests.
- `ai_engine/` — LangGraph/AI and RAG code, evaluation scripts, and tests.
- `supabase/` — database schema, seed data, and forward-only migrations.
- `IEEE_Paper_MemoryVerse.tex`, `RAG_PAPER_REPRODUCIBILITY.md`, and related
  artifacts — research and benchmark documentation. Treat measured results and
  their provenance as part of the implementation contract.

## Safe change discipline

- Inspect `git status` and the relevant diff before editing. The working tree
  may contain user work; preserve it and do not reformat unrelated files.
- Make focused, reversible, surgical changes. Prefer a small patch over a
  rewrite. Do not use `git reset`, `git clean`, force checkout, or broad delete
  operations to resolve a problem.
- Do not create commits, push branches, change remotes, or alter global tool
  configuration unless the user explicitly asks.
- Never put secrets in source, tests, logs, screenshots, benchmark artifacts,
  or documentation. Keep API keys and Supabase credentials in ignored `.env`
  files or the user's configured secret store; use placeholders in examples.
- Do not run destructive database, storage, deployment, or external-service
  actions without explicit user approval. Review SQL and migration effects
  before applying them.
- Do not silently install dependencies or change the selected runtime/toolchain.
  If a required tool is missing, report the setup gap and continue with checks
  that are available.

## General implementation rules

- Preserve the existing architecture and naming conventions. Use descriptive
  names, `snake_case` file names, and normal `PascalCase` class names.
- Keep business logic out of presentation code where a repository/service or
  provider is the established boundary.
- Add or update a focused test for behavior changes and run the smallest
  meaningful project check. Never describe a check as passing unless it was
  actually run in this workspace.
- Treat generated files and lockfiles as tool-owned. Change them only when a
  dependency or platform configuration change requires regeneration, then
  review the resulting diff.
- Keep expensive or network-backed model benchmarks separate from ordinary
  unit tests. Do not rerun them by default; record whether results are measured,
  mocked, synthetic, or proxy-based.

## Flutter client rules (`frontend/`)

- Use Riverpod providers/notifiers for shared or asynchronous state. Do not put
  complex business logic in `StatefulWidget` lifecycle methods.
- Route Supabase and other persistence calls through the established repository
  layer; widgets should depend on providers/repositories rather than calling
  the database directly.
- Keep feature code under the existing feature-based layout and use the
  project's existing navigation and service boundaries.
- In UI feature code, use the design system in
  `frontend/lib/core/theme/app_design_tokens.dart`: `AppColors`,
  `AppGradients`, `AppTextStyles`, `AppSpacing`, `AppRadius`, and `AppTheme`.
  Do not add new hard-coded hex colors, `Colors.white`/`Colors.black`,
  ad-hoc font definitions, or direct `GoogleFonts` usage when a token exists.
  The token-definition file itself is the exception because it defines the
  palette and styles.
- Prefer `const` constructors and `final` locals where they are valid. Remove
  `const` only when a value is evaluated dynamically by a token or runtime
  state.
- Keep platform-specific permissions, media discovery, and generated plugin
  registrants inside their established platform/service boundaries. Do not hand
  edit generated registrants.
- For significant UI changes, align with the existing dark/glassmorphic visual
  language and use a wireframe/mockup before implementation when the user has
  not already supplied a visual direction.

## Python services and RAG rules (`backend/`, `ai_engine/`)

- Keep FastAPI route handlers thin; put domain behavior in the existing
  service/repository modules and preserve the current dependency-injection and
  configuration patterns.
- Keep tests deterministic and offline by default. Mock Supabase, model APIs,
  queues, and other external services unless an integration check is explicitly
  requested and safe to run.
- When changing retrieval or evaluation code, preserve dataset provenance,
  deterministic paths/ranking, and the distinction between real media
  validation, synthetic fixtures, and proxy consistency benchmarks.
- Do not replace measured research results with invented targets or silently
  rewrite benchmark artifacts. Update the reproducibility documentation when a
  command, dataset, metric, or interpretation changes.

## Supabase and migrations (`supabase/`)

- Treat checked-in schema files and migrations as an API contract. Migrations
  must be forward-only, narrowly scoped, and safe to review before execution.
- Do not apply SQL to a remote project, delete data, reset a database, or change
  auth/storage configuration without explicit user approval.
- Never place project URLs, service-role keys, OAuth secrets, or user data in
  migrations, fixtures, screenshots, or documentation.

## Checks and useful commands

Run checks from the component's working directory and use an already configured
environment. These are the normal local checks; a missing runtime or dependency
is a setup gap, not a reason to install silently:

```text
# FastAPI tests, using the repository virtualenv when it exists
cd backend && ../.venv/bin/python -m pytest tests/ -v

# RAG tests from the repository root
.venv/bin/python -m pytest ai_engine/rag/tests -q

# Flutter static analysis and tests
cd frontend && flutter analyze
cd frontend && flutter test
```

- Prefer a focused test target first, then expand to the relevant component
  suite. Run `flutter build` only when the change or requested verification is
  platform/build specific.
- UI or device verification is additional evidence, not a substitute for
  analyzer/tests. Report exactly which checks ran, their working directory,
  and any skipped or blocked checks.

## Completion and handoff

- Before reporting completion, inspect the final diff and confirm that only the
  intended files changed.
- Summarize what changed, the checks actually run, and any remaining setup or
  verification gap.
- Update `.agents/memory.md` only for durable project knowledge: a user
  preference, architectural decision, recurring bug/fix, or major milestone.
  Append concise entries without destroying history, and never copy secrets or
  raw private notes into the repository.
