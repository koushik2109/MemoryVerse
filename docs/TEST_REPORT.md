# Test report

Date: 2026-10-09

## Executed

| Command | Result |
|---|---|
| `PYTHONPATH=backend:. .venv/bin/python -m pytest -q backend/tests ai_engine/test` | **114 passed**, 13 warnings, 0 failures |
| `python3 -m py_compile` on changed Python modules | Passed |
| `PYTHONPATH=backend:. .venv/bin/pyright backend/app ai_engine/rag` | 0 errors, 0 warnings, 0 informations |
| `python3 research/evaluate_retrieval.py /tmp/omnirush-retrieval.jsonl` | Passed smoke fixture: Recall 1.0, Precision 0.5, MRR 1.0 |
| `docker compose config --quiet` | Passed with no Compose warnings |
| `bash scripts/validate_supabase.sh` | Passed disposable PostgreSQL migration validation |
| `/home/koushik_2109/develop/flutter/bin/flutter analyze` from `frontend/` | No issues found |
| `/home/koushik_2109/develop/flutter/bin/flutter test` from `frontend/` | **6 tests passed** |

## Could not run

- Live production Supabase/RLS/vector tests: only the disposable local
  PostgreSQL/pgvector validation was run; production was not contacted.
- Full production-scale media/video workload: only isolated synthetic media
  render tests were run.

The smoke metric output is only a script validation, not an application
benchmark.

The research paper source also compiles with `latexmk -pdf -interaction=nonstopmode
-halt-on-error` and its regenerated PDF passes `pdfinfo`. Missing figures are
handled with explicit placeholders; unresolved historical claims remain marked
as draft/unverified in `docs/RESEARCH_TRUTH_MATRIX.md`.
