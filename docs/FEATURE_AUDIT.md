# Feature audit

| Capability | Evidence | Status | Gap / verification |
|---|---|---|---|
| Supabase auth | `frontend/lib/core/providers/auth_provider.dart`, `backend/app/api/auth.py` | Partial | Local tests pass; live provider flow remains unverified. |
| Vaults and membership | `backend/app/api/vaults.py`, `vault_service.py` | Partial | Needs live RLS/integration tests. |
| Upload and thumbnails | `backend/app/api/media.py` | Partial | File-content limits and live storage need verification. |
| Timeline | `backend/app/api/timeline.py`, Flutter timeline screen | Partial | Depends on complete production schema. |
| Semantic search | `search_service.py`, AI embeddings/RAG | Partial | No reproducible dataset run in this environment. |
| Conversational assistant | `ai_service.py`, `home_assistant_service.py` | Partial | Grounding and no-result behavior need live provider tests. |
| Video jobs | `video_queue.py`, `job_state_manager.py`, Flutter job controller | Partial | Durable behavior depends on Redis/DB and FFmpeg. |
| Video rendering | `video_service.py`, `renderer.py` | Partial | Renderer path exists; current docs overstate narration/beat sync. |
| Audio/video RAG | `ai_engine/rag/{audio,video}` | Partial | Migration and tenant filters added; live Supabase/model verification pending. |
| Cross-user isolation | media stream and RAG boundary | Improved, not proven | Security tests require isolated Supabase fixtures. |
| Mobile/web media | platform services in Flutter | Partial | Analyzer/tests pass; device and browser behavior requires platform testing. |
| Astra DB | no implementation found | Not applicable | Actual project uses Supabase/Postgres, not Astra DB. |

## Verified improvements in this audit

- Added authenticated owner/member authorization to direct media streaming.
- Added explicit private-service authentication to the standalone RAG routes.
- Added tenant parameters to visible vector RPC calls and a migration defining
  tenant-scoped RPCs.
- Removed synthetic media results from the generic query pipeline.
- Added a memory deep-link adapter to GoRouter.
