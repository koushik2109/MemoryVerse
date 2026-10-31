# Technical debt and risk register

## Critical / high

1. Supabase service-role access means application filters remain security
   critical; live RLS tests are missing.
2. The original schema does not document every table referenced by the code.
   Jobs, memories, embeddings, conversation/session, and some RAG objects need
   complete versioned migrations.
3. The RAG service requires `AI_ENGINE_SHARED_SECRET` in both services. Until
   deployment configuration is updated, remote RAG calls should be considered
   unavailable rather than exposed without auth.
4. Dependency versions are not fully locked and Flutter is not available in the
   current environment.
5. The job fallback journal is not a substitute for a multi-process durable
   queue.

## Medium

- Several API contracts use unchecked response casts in Dart.
- Schema-error handling can turn missing tables into empty successful responses.
- Some authentication endpoints expose raw exception text.
- Model downloads happen at runtime and need a persistent cache volume.
- Existing project docs claim empirical scores that were not reproducible from
  the checked-in environment; those claims are treated as unverified.

## Deferred intentionally

- Replacing the working Supabase architecture with Astra DB.
- Introducing a new orchestration framework.
- Training or fine-tuning models without an inspected, licensed dataset and a
  runnable evaluation harness.
