# RAG and evaluation operating design

## Recommended retrieval path

1. Ingest each source into a tenant-owned record with stable provenance:
   source ID, media ID, page/slide/time span, extractor, and confidence.
2. Store modality-specific vectors at their declared dimensions:
   BGE-M3 text vectors at 1024 dimensions, SigLIP image vectors at 768, and
   CLIP image vectors at 512 in the dedicated `clip_embedding` column.
3. Require `user_id` in every retrieval RPC and enforce it inside SQL before
   vector distance ordering.
4. Retrieve independently by modality, apply metadata/time filters, then merge
   with reciprocal-rank fusion or a weighted normalized score. Never merge
   results from different tenants.
5. Deduplicate by stable source/media ID and keep the highest-supported result.
6. Abstain with `no_sources` or `insufficient_evidence` when the score threshold
   or evidence coverage requirement is not met.
7. Generate answers/storyboards only from retrieved evidence references. Every
   sentence or scene must retain its evidence IDs and source location.

## Evaluation order

- Retrieval: Recall@K, Precision@K, HitRate@K, MRR, and nDCG@K.
- Answer support: exact evidence coverage and human support labels.
- Abstention: no-source precision/recall on deliberately unanswerable queries.
- Agreement: independent annotation followed by Cohen's kappa; target range is
  0.75–0.80, measured rather than manufactured.
- Video: MP4 decode success, duration/aspect correctness, audio presence,
  subtitle readability, source identity, and human review of ghosting/sync.

The offline scripts in `research/` are deterministic and do not call models or
databases. Live RAG and video metrics require real fixtures and should not be
reported from empty or synthetic inputs.
