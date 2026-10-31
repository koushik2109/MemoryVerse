# Feature update matrix

| Area | Current state | Update needed | Priority |
|---|---|---|---|
| Auth/config | Supabase PKCE and centralized frontend config | Run live provider flow and enforce production secret checks | High |
| Memory CRUD | Create/list/update/delete paths exist | Complete owner/member checks on every read and association | High |
| Uploads | Private storage, thumbnails, retry queue | Add content sniffing, quotas, and live storage tests | High |
| Media indexing | EXIF/CLIP/tag extraction exists | Add observable processing status and reprocessing versions | High |
| Search | Metadata/lexical/CLIP plus optional RAG | Add calibrated evaluation and per-result evidence references | High |
| Assistant | Chat/home assistant paths exist | Persist structured context and test no-result abstention live | High |
| Video jobs | Worker, Redis/local fallback, SSE/polling | Verify cancellation, retry-resume, durable recovery, and progress | High |
| Video playback | Signed stream URL support | Test mobile/web range playback and expiry handling | High |
| AI Director | Interactive session path exists | Persist sessions and authorize memory/vault access | High |
| Invites | Backend and router paths exist | Standardize one deep-link scheme across Android/iOS/web | Medium |
| Audio RAG | Separate service implemented | Add real audio fixtures, timestamps, and tenant integration tests | Medium |
| Research benchmark | Prototype scripts/results exist | Remove leakage/duplication and regenerate metrics from manifests | Critical |
| Paper | LaTeX source and stale PDF exist | Rewrite claims to match executable artifacts and rebuild cleanly | Critical |
| Observability | Logging and job state exist | Add request IDs, structured metrics, and alertable failure counters | Medium |

## Recommended implementation order

1. Live Supabase authorization and migration tests.
2. Upload/indexing status and video-job lifecycle tests.
3. Search/assistant evaluation with a clean, licensed fixture set.
4. Durable AI Director sessions and canonical invite links.
5. Research benchmark repair and paper regeneration.

## Completed in the current audit pass

- Upload-dependent auto-video generation now waits for the selected upload
  batch to finish and logs the returned job ID.
- Memory reads with a `vault_id` verify membership; direct memory reads verify
  the owner.
- Authenticated chat-to-AI-engine calls use the engine's required header.
- Narrative metadata is represented in both schema sources and has an additive
  migration.
- Video playback requests an authorized signed stream URL instead of opening a
  protected stream endpoint without credentials.
- Local generated-video URLs are corrected after the generated `media` row is
  created, and staged downloads recognize the canonical `staged_file_path`.
- Research benchmark corpus duplication/query leakage and metric definitions
  were corrected. Existing result JSON was intentionally not relabeled as a
  newly valid experiment.
- Ablation output no longer reports hard-coded baseline percentages as measured
  results; unsupported measurements are explicit `null` diagnostics.
