# Human RAG answer-quality audit

Use the query, target keyword, retrieved context, and answer in the packet. Do not
infer or repair facts from outside the retrieved context. Reviewers must work
independently and must not see model scores or one another's labels.

For a simpler workflow, open `reviewer_1_form.html` or `reviewer_2_form.html` in a
browser. Review one case at a time, save the draft locally, and use “Download
completed CSV” when all cases are finished. Send only the downloaded CSV to the
project owner. The form exports the exact columns required by the agreement script.

## Packet metadata

- Packet version: `l10-album2-v1`

## Quick decision rule

1. **Faithfulness:** judge whether the answer's own claims are supported by the
   retrieved context. A truthful answer that says the context is insufficient is
   usually `supported`, even if it did not solve the user's question.
2. **Relevance:** judge whether the answer actually addresses the question and its
   target. A truthful refusal is usually `partial`, not `off_topic`.
3. **Correctness:** mark `uncertain` when the target cannot be established from
   the context. Mark `incorrect` only when the answer gives a wrong target or
   denies a target that the context clearly contains.

The target keyword is a checking hint, not evidence. Use the retrieved context as
the only evidence. If the answer says a detail is absent, check whether that
detail is actually present before accepting the statement.

Examples:

- Context says only `dog`; answer says it cannot verify a barking recording:
  `supported`, `partial`, `uncertain`.
- Context contains an exact district tag; answer says the tag is absent:
  `mixed`, `partial`, `incorrect`.
- Context contains an exact event detail; answer says that detail is absent:
  `mixed`, `partial`, `incorrect`.

## Faithfulness

- `supported`: every material factual claim in the answer is entailed by the retrieved context.
- `mixed`: at least one material claim is supported, but another is overstated, weakly supported, or unverifiable.
- `unsupported`: the answer's material claims are absent from or contradicted by the retrieved context.

## Relevance

- `direct`: answers the requested question and addresses the target detail.
- `partial`: addresses part of the question, is vague, or omits the target detail.
- `off_topic`: does not answer the question or answers a different task.

## Correctness

- `correct`: the answer gives the target detail supported by the context.
- `uncertain`: the context is insufficient or ambiguous; do not guess.
- `incorrect`: the answer gives a target detail contradicted by the context or a wrong target.

Record a short note for every `mixed`, `unsupported`, `partial`, `off_topic`,
`uncertain`, or `incorrect` label.
