# RAG ground-truth form guide

The friend-created form is useful if it records human judgments as a versioned
gold set. It does not train the model automatically. It lets us measure whether
retrieval returns the right media and whether generated answers are supported.

## How it works

1. A researcher writes a query such as “photos from the beach at sunset”.
2. The annotator identifies every relevant corpus item, including relevant
   items the system failed to retrieve.
3. Each item receives a grade from 0 to 3.
4. The annotation is exported as JSONL using `research/ground_truth_schema.json`.
5. Validate it:

```bash
python research/validate_ground_truth.py judgments.jsonl
```

6. The evaluator calculates Recall@K, Precision@K, MRR, and graded nDCG against
   these judgments.

The existing consumer curation screen is not sufficient as a ground-truth form:
it only shows returned candidates and captures user preference, not false
negatives or independently judged relevance.

## Example

```json
{"case_id":"case_0001","query":"photos from the beach at sunset","scope":"vault-a","modality":"image","relevant_items":[{"item_id":"media-001","grade":3},{"item_id":"media-007","grade":2}],"annotator_id":"researcher-01","dataset_version":"memoryverse-fixture-v1","source":"friend_form"}
```

The form is therefore useful for:

- Measuring retrieval quality before and after changes.
- Identifying missing-result failures.
- Comparing production retrieval with the offline benchmark.
- Testing no-result and abstention behavior.
- Supporting a defensible research paper.

It should not be used to upload private media or bypass per-user authorization.
