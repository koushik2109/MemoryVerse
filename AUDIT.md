# Codebase and Evaluation Audit (Phase 0.1)

## 1. Datasets Used
- **Personal Photographic Corpus (`ocr_extracted_corpus.json` / `dataset/album{1,2,3}`)**:
  - 191 physical photos across three personal albums + 2 research documents (Patel et al. 2025 base paper and IEEE paper draft).
  - Apple Vision OCR (`VNRecognizeTextRequest`) extracted text: 54 photos contain readable scene text; total 193 indexed items (191 photos + 2 document PDFs).
  - Pre-registered split (`ramp_rag_probes_split.json`): 96 items in Dev, 97 items in Test.
  - Scene-text probes: 85 total probes (Set 1: 30 original probes; Set 2: 55 held-out test probes targeting genuine scene text).
- **MS-COCO Karpathy Split (`coco_karpathy_test.json`, `coco_blip_captions.json`, `coco_1k_images`)**:
  - 1,000 selected index images from MS-COCO test split.
  - Machine proxies: BLIP-generated captions.
  - 500 total queries: partitioned strictly into 250 Dev queries and 250 Test queries.
- **Unified RAG Dataset (`unified_rag_dataset.jsonl`)**:
  - 3,356 precomputed records: 2,000 ESC-50 category rows, 1,292 PhotoBench annotation rows, 64 L10 rows.
  - Used for Tier-1 proxy consistency benchmarking (334 deterministic queries).
- **Physical Audio (`ESC-50` WAVs)**:
  - 52 authentic 5-second 44.1 kHz WAV audio clips across 13 diagnostic classes.
  - Evaluated via LAION-CLAP embeddings and free-form audio queries.

## 2. Split Logic
- **Dev/Test Partitioning**:
  - Partitioned strictly by Image / Document ID (seeded with Python `random.seed(42)`):
    - Dev split: 96 items (`dev_record_ids`), 30 probes (`dev_probes`).
    - Test split: 97 items (`test_record_ids`), 55 probes (`test_probes`).
  - For COCO: First 250 queries assigned to Dev (`coco_dev_queries`), second 250 queries assigned to Test (`coco_test_queries`).
  - Zero test-split data is ever exposed during parameter tuning or router fitting.

## 3. Parameter Tuning Locus
- **$\alpha$ (Dense vs. Sparse Balance in Hybrid Fusion)**:
  - In baseline hybrid: Fixed statically at $\alpha = 0.70$ ($0.7 \times \text{Dense} + 0.3 \times \text{BM25}$).
  - In RAMP-RAG Router: Dynamically predicted as $\alpha_q \in \{0.25, 0.85\}$ using a logistic regression classifier fitted strictly on Dev queries (COCO Dev + Dev scene-text probes).
- **$\beta$ (Candidate Reranking Balance)**:
  - Fixed configuration setting: $\beta = 0.65$ in $S_{\text{rerank}} = \beta S_{\text{dense}} + (1-\beta)\text{Overlap}$.
  - Not post-hoc tuned on test data.
- **$\gamma$ (Clean Stream Fusion Weight)**:
  - Swept over $\gamma \in [0.0, 0.8]$ strictly on the Dev split under 20% noise to identify $\gamma^*$.
  - Frozen before any evaluation on the held-out Test split.
- **$\tau_{\text{margin}}$ (Cascade Escalation Threshold)**:
  - Heuristic threshold: $\tau = 0.065$ selected on Dev.
  - Conformal calibration: $\tau_{\text{conf}} = 0.135$ determined via split-conformal quantile calibration on Dev with target miss rate $\epsilon \le 0.05$.

## 4. Private vs. Public Probes
- **Private Probes**:
  - All 85 scene-text probes (30 Set 1, 55 Set 2) are derived from the authors' personal photo albums (`dataset/album1`, `album2`, `album3`).
  - Images represent private photographic archives and cannot be distributed publicly in benchmark repositories.
- **Public Probes**:
  - MS-COCO Karpathy split (500 queries over 1,000 public MS-COCO images) is public.
  - ESC-50 acoustic recordings are public.
- **Action Plan**:
  - Build a new, fully public, larger scene-text and document benchmark in Phase 1 (e.g., ICDAR, TextVQA/TextCaps, SROIE) with $\ge 300$ probes and $\ge 1,000$ index items to serve as the primary public benchmark, relegating personal album results to a secondary appendix validation.
