# MemoryVerse — Complete Asset Inventory for Rapid Experimental Strengthening

**Date:** November 2026  
**Repository:** MemoryVerse (`IJCDS` submission branch)  
**Status:** VALIDATED & INDEPENDENTLY VERIFIED  

---

## 1. Asset Inventory Matrix

| Asset | Available? | Location | Validated? | Can reuse? | Notes / Provenance |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Final human query set** | **YES** | `data/human_queries_adjudicated.csv`, `data/human_queries_200.csv` | **YES** | **YES** | 400 instances (200 verbatim, 200 paraphrase) authored by researcher and friend; includes string deduplication and audit tags. |
| **Intended target receipt IDs** | **YES** | `data/human_queries_adjudicated.csv` (`original_single_target`), `data/human_queries_200.csv` (`receipt_id`) | **YES** | **YES** | Authoritative ground truth mapping directly to SROIE receipt corpus IDs (`sroie_000` to `sroie_625`). |
| **Merchant metadata** | **YES** | `ai_engine/rag/data/v2/public_benchmark_v2.json` (`key_entities.company`), `ai_engine/rag/data/sroie/key/*.json` | **YES** | **YES** | 626 structured JSON records with ground-truth vendor names (e.g., Unihakka, Gin Kee, Gardenia, 99 Speed Mart). |
| **Transaction dates** | **YES** | `ai_engine/rag/data/v2/public_benchmark_v2.json` (`key_entities.date`), `ai_engine/rag/data/sroie/key/*.json` | **YES** | **YES** | Exact transaction dates (e.g., `25/12/2018`, `05 MAR 2018`) verified against scanned receipt images and OCR. |
| **Word BM25 rankings** | **YES** | `results/v10/colqwen_per_query_v10.csv`, `results/v4/G-01/per_query_rankings.csv`, `data/human_queries_relevance_summary.csv` | **YES** | **YES** | Standard Robertson-Spärck Jones BM25 ($k_1=1.5, b=0.75$) evaluated across 813 test receipts. |
| **Character BM25 rankings** | **YES** | `results/v10/colqwen_per_query_v10.csv`, `results/v4/G-01/per_query_rankings.csv`, `data/human_queries_relevance_summary.csv` | **YES** | **YES** | 3–5 character $n$-gram BM25 with whitespace collapse, evaluated across 813 test receipts. |
| **Dense retrieval rankings** | **YES** | `results/v13_colqwen_robustness/colqwen_robustness_summary.csv`, `results/v6/dense_query_formulation_v6.csv`, `results/v3/G-01/sroie_modern_dense_summary.csv` | **YES** | **YES** | BGE-base (MRR 0.4909 on human queries), BGE-M3, and MiniLM evaluated across SROIE corpus. |
| **Receipt corpus / index** | **YES** | `ai_engine/rag/data/v2/public_benchmark_v2.json` | **YES** | **YES** | 1,626 total records (813 test corpus) with Apple Vision OCR, clean OCR, and degraded text variants. |
| **Cached embeddings** | **YES** | `~/.cache/huggingface/hub/` (`BAAI/bge-base-en-v1.5`, `all-MiniLM-L6-v2`, `bge-m3`, `siglip-base-patch16-224`, `clip-vit-base-patch32`) | **YES** | **YES** | Fully offline cached weights in HuggingFace cache; local Apple Silicon MPS and CPU inference ready. |
| **Cascade rankings** | **YES** | `results/gapfill_v1/T2-04/cascade_per_k_sweep.csv`, `results/v3/G-07/cascade_per_k_sweep.csv` | **YES** | **YES** | Complete $K \in \{5, 10, 20, 30, 50, 75, 100, 150, 200\}$ candidate sweep across Word, Char, and Hybrid BM25. |
| **Latency outputs** | **YES** | `FINAL_BLOCKER_RESOLUTION/cascade_measured_latency_cpu.csv`, `FINAL_BLOCKER_RESOLUTION/cascade_measured_latency.csv` | **YES** | **YES** | Rigorously partitioned live M2 CPU (e.g., $12.18$ ms at $K=50$) vs. Apple MPS ($1.26$ ms at $K=50$) measurements. |
| **Existing evaluation scripts** | **YES** | `scripts/gapfill_t3_02_generator_robustness.py`, `scripts/gapfill_t2_04_fusion_cascade.py`, `scripts/gapfill_g03_conversational_queries.py` | **YES** | **YES** | Self-contained, deterministic rankers with paired permutation testing and 5,000-iteration bootstrap engines. |

---

## 2. Analysis of Proposed Experiments

### Experiment 1 — Transaction-Date Ablation
- **Eligibility:** Exactly **56 human-authored paraphrase queries** contain explicit transaction-date expressions (e.g., `on 5th March`, `dated 28/12/2017`, `on 28 Dec 2017`, `on Christmas`, `in January`).
- **Control Group:** Exactly **36 queries** target merchants with multiple receipts in the corpus (where date ablation directly tests same-merchant transaction confusion), and **20 queries** target single-receipt merchants.
- **Feasibility:** 100% executable immediately using frozen BM25 and cached dense models. Requires zero external API calls or GPU cloud costs.

### Experiment 2 — Same-Merchant Hard-Negative Evaluation
- **Eligibility:** Across the entire 1,626-document pool (626 SROIE receipts), 276 query instances target merchants with multiple receipts. Under the frozen, leak-free 50/50 Dev/Test corpus split ($N=813$ test index), exactly 26 merchants have duplicate receipts partitioned into the Dev split ($26 \times 2 = 52$ queries). Consequently, exactly **224 query instances** (112 verbatim, 112 paraphrase) target receipts with active same-merchant distractors concurrently present in the 813-item test corpus. The released evaluations report the actual 224 queries with zero invented or missing observations.
- **Hard Distractor Pool:** Structured company metadata (`company`) in `public_benchmark_v2.json` provides deterministic same-merchant distractor sets without manual heuristic guessing.
- **Feasibility:** 100% executed and verified across Word BM25, Char BM25, and Dense BGE-Base against random distractors vs. same-merchant hard distractors.

### Experiment 3 — Cascade Candidate-Depth Sensitivity
- **Existing Evidence:** Already fully executed and preserved in `results/gapfill_v1/T2-04/cascade_per_k_sweep.csv` ($K \in \{5, 10, 20, 30, 50, 75, 100, 150, 200\}$) and partitioned hardware latency files (`cascade_measured_latency_cpu.csv` and `cascade_measured_latency.csv`).
- **Recommendation:** **Zero compute required**. We can synthesize and tabulate these existing frozen outputs directly into `CASCADE_DEPTH_RESULTS.csv` and the manuscript without expensive recalculation.

