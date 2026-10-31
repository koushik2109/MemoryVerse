# Research and paper truth matrix

## Evidence classification

- **Code-verified:** present in the current implementation and covered by a
  local check.
- **Measured/reproducible:** produced by a script from a versioned dataset and
  configuration, with per-query outputs.
- **Historical/unverified:** appears in older documents or result files but
  cannot be reproduced from the current repository.
- **Proposed:** design or roadmap material, not an implemented result.

## Current classification

| Claim | Classification | Reason |
|---|---|---|
| Flutter/FastAPI/Supabase application exists | Code-verified | Source and tests exist. |
| Backend CLIP metadata/indexing path | Code-verified | `AIExtractor` and backend search code. |
| Standalone SigLIP/BGE/Whisper RAG path | Code-verified | RAG routers and stores exist. |
| Dataset has 3,356 rows | Code-verified | Current JSONL contains 1,356 images and 2,000 audio rows. |
| 500-query retrieval benchmark | Historical/unverified | Current script discovers a different query count and prior corpus was duplicated. |
| 50-query E2E benchmark | Historical/unverified | Current script contains 12 cases, not 50. |
| Published Recall/MRR tables | Historical/unverified | Tables do not match tracked results and lack raw per-query artifacts. |
| Four-model ablation results | Historical/unverified | Several values are hard-coded estimates and use shared graph routing. |
| 0% hallucination | Unsupported as a general claim | Current auditor checks a narrow proxy, not atomic factuality. |
| BLIP/OCR/Milvus/Whisper-large-v3 production path | Not current | Current implementation uses Qwen captioning, Supabase/pgvector, and Whisper base. |
| LoRA/DPO/Qwen five-agent deployment | Proposed | No training or adapter implementation is present. |
| `mh_12` case study | Historical/unverified | No supporting trace artifact was found. |

## Benchmark repair completed

The benchmark code now removes the duplicated corpus append and uses standard
Recall@K and ideal nDCG construction. Dataset generation no longer places the
evaluation query text into the indexed text representation. Existing JSON result
files are not relabeled as new results; they must be regenerated before use.

The ablation runner now writes `research/ablations/graph_diagnostics.json`
instead of presenting hard-coded baseline percentages as measured scientific
results. It reports missing factuality/retention measurements as `null` and
labels its internal auditor/alignment values as diagnostics.

## Publication gate

Do not claim paper metrics until the repository contains:

- Dataset and query manifests with hashes.
- Per-query outputs and retrieved IDs.
- Exact model/checkpoint/runtime metadata.
- Clean train/dev/test separation.
- Actual ablation implementations and raw logs.
- A valid PDF with resolved citations, figures, and cross-references.
