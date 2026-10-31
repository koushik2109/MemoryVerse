# ML and RAG evaluation plan

## Retrieval

Use a versioned JSONL fixture containing `{query, user_id, relevant_ids}` and
run the same model/configuration against an isolated database snapshot.

- Recall@K = relevant results in top K / all relevant IDs.
- Precision@K = relevant results in top K / K.
- MRR = reciprocal rank of the first relevant result, averaged over queries.
- nDCG@K = discounted graded relevance normalized by the ideal ranking.
- Latency = p50/p95 from request receipt through ranked candidates.

Record model names, checkpoint revisions, embedding dimensions, database
indexes, filter selectivity, hardware, and random seeds. Evaluate no-result
queries separately and count unsupported claims in generated answers.

## Generation

Record successful render rate, output duration error, media-selection precision,
temporal ordering violations, render latency, and peak memory. Human ratings
are separate from deterministic render checks.

## Baselines

1. Metadata/lexical search only.
2. Dense retrieval only.
3. Dense + metadata filtering.
4. Dense + metadata + reranking + grounded answer validation.

The repository currently has no verified fixture results, so this plan does not
invent scores.
