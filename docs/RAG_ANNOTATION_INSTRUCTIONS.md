# RAG ground-truth annotation instructions

## Should you use the form?

Yes—if the 100 questions are intended to evaluate MemoryVerse search,
chatbot grounding, or cross-modal retrieval. The form is useful only when the
annotator labels the correct media independently of what the current AI returns.
It should not be used as a simple “did I like the suggestions?” survey.

The friend’s original 100 questions were not present in this workspace, so the
provided CSV contains 100 numbered rows ready for those questions. Do not
invent question text or media IDs. Copy the exact 100 questions into the
`question` column and use stable media/corpus IDs.

## Files

- Form: `research/ground_truth_annotation_form.csv`
- Machine-readable schema: `research/ground_truth_schema.json`
- JSONL validator: `research/validate_ground_truth.py`
- Existing evaluation guidance: `docs/RAG_GROUND_TRUTH_GUIDE.md`

## How to complete each row

### Required fields

- `case_id`: Keep `case_0001` through `case_0100` unchanged.
- `question`: Paste the exact question, without rewriting it.
- `scope`: Dataset, album, vault, or fixture scope used to judge the question.
- `modality`: `image`, `video`, `audio`, `mixed`, or `text`.
- `expected_intent`: For example `SEARCH`, `FOLLOW_UP`, `FACTUAL_QA`,
  `GENERATE_VIDEO`, `NAVIGATION`, or `NO_RESULT`.
- `annotator_id`: A non-secret researcher identifier, not an email or token.
- `dataset_version`: Name/version of the corpus snapshot used for annotation.

### Relevance fields

- `relevant_item_ids`: All correct media IDs, separated by `|`.
- `graded_relevance`: Matching grades in the same order, separated by `|`.

Use this scale:

| Grade | Meaning |
|---|---|
| 3 | Directly answers the question; highly relevant |
| 2 | Clearly relevant but incomplete or less central |
| 1 | Related context but not a correct answer |
| 0 | Irrelevant; normally omit from relevant IDs |

Include all known relevant items, not only the top result returned by the
system. This is what makes Recall@K meaningful.

- `missing_relevant_item_ids`: Relevant IDs that the system failed to return.
  This is essential for false-negative analysis.
- `no_relevant_items`: Use `true` only when the question genuinely has no
  answer in the scoped corpus. Otherwise use `false`.

### Answer-grounding fields

Complete these after reviewing the chatbot answer for this question:

- `answer_evidence_ids`: IDs that genuinely support the answer, separated by
  `|`.
- `answer_is_supported`: `true`, `false`, or `not_applicable`.
- `notes`: Brief explanation of ambiguity or judgment.
- `error_categories`: One or more of:
  `wrong_date`, `wrong_place`, `wrong_person`, `wrong_modality`,
  `irrelevant_result`, `missing_result`, `duplicate_result`,
  `unsupported_claim`, `should_abstain`, `other`.
- `created_at`: ISO-8601 timestamp, for example
  `2026-10-13T15:30:00+05:30`.

## Annotation procedure for each question

1. Read the exact question without looking at the AI answer first.
2. Open the complete scoped corpus or album, not just the returned candidates.
3. Identify every media item that answers the question.
4. Assign grades using the 0–3 scale.
5. Mark `no_relevant_items=true` only after checking the complete scope.
6. Run the current MemoryVerse search/chat flow separately.
7. Record the returned IDs and whether any relevant item was missing.
8. Read the chatbot answer sentence by sentence.
9. Record only evidence IDs that support actual claims.
10. Mark unsupported dates, locations, people, emotions, or events as
    `unsupported_claim`.
11. Save one completed row, then continue to the next question.

## Important rules

- Never mark an item relevant merely because the model selected it.
- Never use filenames as IDs if stable media IDs are available.
- Never expose tokens, private URLs, or private personal information in notes.
- Do not infer a person’s identity, emotion, health, religion, or other
  sensitive trait from appearance alone.
- If the question is ambiguous, record the ambiguity in `notes` and use the
  narrowest defensible relevance judgment.
- For “these photos” follow-ups, annotate the conversation context and the
  selected result IDs used to resolve the reference.
- For video-generation questions, label the source media that should be
  selected; do not grade the final video aesthetic unless the form explicitly
  includes a generation-evaluation case.

## Quality control

After completing the CSV:

1. Convert it to JSONL using a controlled script or spreadsheet export.
2. Ensure exactly 100 non-empty case IDs exist.
3. Check that no case has duplicate item IDs.
4. Check that every relevance grade is 0–3.
5. Have a second annotator independently review at least 20% of cases.
6. Resolve disagreements before publishing metrics.
7. Run:

```bash
python research/validate_ground_truth.py judgments.jsonl
```

Do not use the 100 questions as a scientific benchmark until the corpus
version, relevant IDs, annotation rules, and disagreement process are recorded.
