# Research paper artifact status

`IEEE_Paper_MemoryVerse.tex` is a draft research manuscript, not a release
claim. The current implementation has multiple execution paths, while the
manuscript describes an experimental MM-RAG design that is not fully identical
to the production backend.

## Build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error IEEE_Paper_MemoryVerse.tex
pdfinfo IEEE_Paper_MemoryVerse.pdf
```

The source now compiles without requiring absent figure files; placeholder boxes
are emitted when figures have not been generated.

## Publication gate

Before submitting this paper, regenerate all tables from committed datasets and
per-query outputs. The current repository does not support treating the paper's
historical 500-query/50-query tables, ablation percentages, or `mh_12` case
study as reproduced evidence. See:

- `../docs/RESEARCH_TRUTH_MATRIX.md`
- `../docs/ML_EVALUATION_PLAN.md`
- `../docs/AUTHORITATIVE_WORKFLOW.md`

The paper should not be called production validation until live Supabase,
device, and deployment checks have been completed.
