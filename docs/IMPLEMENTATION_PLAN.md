# Implementation plan

## Completed in this audit slice

1. Protect media bytes with authenticated owner/member checks.
2. Make local JWT fallback explicit and remove the default secret.
3. Restrict backend CORS to configured origins.
4. Require a shared internal credential for standalone RAG and agent routes.
5. Pass tenant IDs to video/audio vector RPCs; add a migration and RLS.
6. Abstain with no results instead of returning invented media paths.
7. Add a deep-link adapter for `/memories/:id`.

## Next dependency-aware work

1. Apply and verify all missing Supabase migrations in an isolated project.
2. Add pytest/HTTP tests for media authorization, RAG auth, and job ownership.
3. Install/use the project Flutter SDK and add router/upload/video-job tests.
4. Run a small licensed fixture set through ingestion and retrieval; record
   Recall@K, MRR, nDCG@K, latency, and unsupported-answer rate.
5. Decide whether the RAG engine remains a private service or is in-process;
   remove the unused path after measuring startup and operational cost.
6. Verify FFmpeg render output and reconcile docs with actual narration/audio
   capability.
