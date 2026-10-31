# Research bibliography and decision record

Sources were checked on 2026-10-09. No benchmark score is claimed here unless
it can be reproduced from this repository and a named dataset.

## Retrieval and multimodal representation

- Radford et al., *Learning Transferable Visual Models From Natural Language
  Supervision*, arXiv:2103.00020. https://arxiv.org/abs/2103.00020
  Supports the repository's CLIP-style shared image/text retrieval approach;
  it does not prove performance on private memory data.
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks*, NeurIPS 2020, arXiv:2005.11401. https://arxiv.org/abs/2005.11401
  Supports retrieving evidence before generation and exposing provenance.
- pgvector project documentation and source, current repository README:
  https://github.com/pgvector/pgvector
  Documents cosine distance, HNSW/IVFFlat tradeoffs, filtered search, and
  tenant-filtering implications.

## Security and data layer

- Supabase, *Row Level Security*: 
  https://supabase.com/docs/guides/database/postgres/row-level-security
  Requires RLS plus grants, policies, indexes, and policy tests; service-role
  keys bypass RLS and must remain server-side.
- OWASP, *Top 10 Web Application Security Risks (2025)*:
  https://owasp.org/www-project-top-ten/
  Used as the security review baseline for authorization, misconfiguration,
  injection, and supply-chain risks.

## Selection decisions

- Keep Supabase/Postgres because it is already integrated and provides auth,
  object storage, SQL metadata, and pgvector-compatible retrieval.
- Keep CLIP/SigLIP/BGE/Whisper as optional model components, but version and
  evaluate them per deployment. Shared embedding spaces are only used where the
  model actually supports them; unrelated vectors are not compared directly.
- Prefer exact or filtered vector retrieval plus reranking before adding a
  second database. This reduces operational complexity for a personal-memory
  workload.
- Use asynchronous jobs for rendering and bounded ingestion; do not make local
  fallback queues look durable in production.
