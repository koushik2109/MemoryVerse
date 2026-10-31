# MemoryVerse documentation

This page is the reading guide for product, engineering, operations, and research documentation. Read the project-status notes before treating an implementation claim or test result as current.

## Recommended reading order

1. [Project handoff and fix log](koushik_notes.md) — system map, user requirements, current boundaries, and the numbered record of completed fixes.
2. [Project discovery](PROJECT_DISCOVERY.md) — repository observations from the dated audit; its findings are historical until rechecked.
3. [Existing architecture](EXISTING_ARCHITECTURE.md) — code-oriented architecture notes.
4. [Architecture specification](ARCHITECTURE_SPECIFICATION.md) — architecture diagram and implementation/target distinctions.
5. [Setup and deployment](SETUP_AND_DEPLOYMENT.md) — local setup guidance and production gates.
6. [Feature audit](FEATURE_AUDIT.md), [feature update matrix](FEATURE_UPDATE_MATRIX.md), and [technical debt](TECHNICAL_DEBT.md) — feature status, gaps, and risks.
7. [Production-readiness proposal](PRODUCTION_READINESS_PHASES.md) — proposed improvements. It is a plan, not evidence that the work is complete.

## Product and engineering references

| Document | Use it for |
|---|---|
| [Authoritative workflow](AUTHORITATIVE_WORKFLOW.md) | Source-of-truth and change workflow guidance. |
| [Backend and AI engine](backend_and_ai_engine.md) | Backend and AI service overview. |
| [Backend/AI audit](backend_ai_audit.md) | Findings from a prior audit; recheck before acting on them. |
| [Database validation](database-validation.md) | Database and migration validation notes. |
| [Implementation plan](IMPLEMENTATION_PLAN.md) | Previously proposed or completed implementation slices; check the fix log and code for current status. |
| [Project explainer](PROJECT_EXPLAINER.txt) | Long-form historical overview; see its status notice before relying on feature or quality claims. |
| [Project status and changes](PROJECT_STATUS_AND_CHANGES.md) | Status snapshot and change notes. |
| [Project review summary](PROJECT_REVIEW_SUMMARY.md) | Review findings and recommendations. |
| [MemoryVerse audit report](MEMORYVERSE_AUDIT_REPORT.md) and [audit follow-up](CHANGELOG_AUDIT_FOLLOWUP.md) | Prior audit findings and their follow-up notes. |
| [Test report](TEST_REPORT.md) | Historical check results with the environment/date recorded in the report. |

## Research and evaluation

| Document | Use it for |
|---|---|
| [RAG and evaluation](RAG_AND_EVALUATION.md) | Retrieval evaluation scope and methodology. |
| [RAG paper reproducibility](../RAG_PAPER_REPRODUCIBILITY.md) | Commands, datasets, metric definitions, and provenance for the research artifacts. |
| [Research truth matrix](RESEARCH_TRUTH_MATRIX.md) | Claim status and supporting evidence. |
| [Experiments](EXPERIMENTS.md) and [ML evaluation plan](ML_EVALUATION_PLAN.md) | Experimental protocols and planned evaluation. |
| [Research bibliography](RESEARCH_BIBLIOGRAPHY.md) and [novelty assessment](RESEARCH_NOVELTY_ASSESSMENT.md) | Literature references and research positioning. |
| [Model cards](MODEL_CARDS.md), [fine-tuning strategy](FINE_TUNING_STRATEGY.md), and [Qwen model summary](QWEN_MODELS_SUMMARY.md) | Model selection and training notes. |
| [RAG annotation instructions](RAG_ANNOTATION_INSTRUCTIONS.md) and [ground-truth guide](RAG_GROUND_TRUTH_GUIDE.md) | Annotation and evaluation data guidance. |
| [Metrics](metrics.md), [metrics report](metrics_report.html), and [experiments](EXPERIMENTS.md) | Results and supporting summaries. Verify each result’s provenance before reuse. |

## Paper and reviewer materials

- [IEEE paper notes](ieee.md), [paper enhancements](PAPER_ENHANCEMENTS_SUMMARY.md), and [presentation notes](ppt.md).
- The paper source and reproducibility artifacts are also kept at repository level and in `research_paper/`, `results/`, `latex_fragments/`, and `latex_fragments_v6/` to preserve their working paths.
- The [metrics report PDF](MemoryVerse_Metrics_Report.pdf) and [HTML report](metrics_report.html) are generated report artifacts; consult the accompanying methodology and provenance before quoting values.
- [SRS](srs.md) contains the requirements specification.

## Repository-level research records

These files remain at the repository root because existing paper and reproducibility workflows refer to those paths:

- `IEEE_Paper_MemoryVerse.tex` — manuscript source.
- `AUDIT.md`, `AUDIT_V6.md`, `MEMORYVERSE_AUDIT_REPORT.md`, `CHECKPOINT1.md`, and `CHECKPOINT2.md` — audit and checkpoint records.
- `RESULTS.md`, `RESULTS_SUMMARY.md`, `RESULTS_SUMMARY_V6.md`, `RUNLOG.md`, `CLAIMS_LEDGER.csv`, and `MANIFEST.json` — results, execution history, claims, and artifact manifest.
- `PREREG_*.md`, `PREREG_ADDENDUM.md`, `PAPER_FIXLIST*.md`, `RESPONSE_TO_REVIEWER_2.md`, and `Journal_Recommendations.md` — preregistration, review response, and manuscript follow-up.
- `CONFIG_LOG.md`, `MECHANISM.md`, `DECISION_GUIDE.md`, and `RESEARCH_NOVELTY_ASSESSMENT.md` — research method and positioning notes.

## Preserved older documents

`legacy_documentation/` contains the five former `documentation/` files preserved without merging them into the current versions. Compare both versions before deciding which content is authoritative.

## Repository map

- `frontend/` — Flutter application, providers, repositories, media services, and platform integrations.
- `backend/` — FastAPI routes, domain services, job processing, persistence adapters, and tests.
- `ai_engine/` — AI agents, embeddings, multimodal pipelines, RAG service, rendering, and tests.
- `supabase/` — schema, seed data, and forward-only migrations.
- `research/`, `research_paper/`, `results/`, `notebooks/`, and `latex_fragments*/` — research code and evidence artifacts.

## How to interpret status and results

- A document’s date is part of its meaning. Historical test reports do not certify the current working tree.
- A proposed feature is not an implemented feature. Confirm behavior in code and run the relevant check before marking it complete.
- Separate measured results from proxy, synthetic, mocked, or manually reviewed evidence. Keep benchmark provenance with every published metric.
- The current implementation and the desired product direction can differ. The project handoff notes identify user constraints; verify implementation status in code.
