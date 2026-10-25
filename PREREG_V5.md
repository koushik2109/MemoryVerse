# Pre-Registration Document (v5): Study Extensions and Generalization

**Study Title:** When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise  
**Pre-Registration Date:** 2026-10-25T16:45:00+05:30  
**Repository Branch:** `research`  
**Execution Target:** Kaggle GPU Environment (Nvidia T4 x2 / P100, 16GB VRAM) & Local Smoke Harness  

---

## 1. Research Objectives & Motivation

The core thesis of this study is that in text-proxy multimodal retrieval, semantic query mismatch is a far more catastrophic failure mode than proxy OCR corruption. To provide definitive academic rigor that withstands reviewer scrutiny, this extension addresses six critical dimensions:
1. **Document-Native Visual Baseline (Task 1):** Directly comparing vision-language multi-vector retrieval (ColQwen2) against OCR text-proxy pipelines on receipt page images to determine whether bypassing text proxies with visual tokens resolves lexical and dense proxy failure modes.
2. **Proxy Quality Axis on Natural Images (Task 2):** Evaluating multiple vision-language caption generators (BLIP, BLIP-2, Florence-2) against gold human captions and native visual models (CLIP, SigLIP) to test whether improving caption generator capacity closes the proxy-native gap on MS-COCO.
3. **Query-Side Typo Noise vs. Document OCR Noise (Task 3):** Stress-testing retrieval under query-side character corruptions (5%, 10%, 20% typos) to determine whether user typing errors or document scanner noise dominate performance loss.
4. **Paraphrase Robustness across Multi-LLM Generators (Task 4):** Validating light-paraphrase query generation across multiple independent LLMs to prove that query mismatch penalties are not artifacts of a specific LLM prompt or generator.
5. **Empirical Decision Guide (Task 5):** Synthesizing an automated decision matrix identifying optimal retrieval architectures and statistically equivalent choices across all evaluated regimes.
6. **Supporting Engineering Analyses (Task 6):** Measuring real hardware costs (latency, peak VRAM, index size), qualitative failure case attribution, statistical power curves, and practical cross-encoder reranker stacks.

---

## 2. Pre-Registered Hypotheses & Decision Rules

### Hypothesis 1: Document-Native Visual Retrieval on Receipts (Confirmatory)
- **Hypothesis:** Document-native multi-vector retrieval (ColQwen2-v1.0 via page-image late-interaction MaxSim) eliminates OCR proxy failure on receipts by directly attending to visual tokens.
- **Evaluated Models:**
  - `vidore/colqwen2-v1.0` (2B parameter Qwen2-VL architecture, late-interaction MaxSim).
  - Baselines: Tuned Word BM25 ($k_1=0.9, b=0.5$), Char 3–5gram BM25, Tuned Hybrid ($\alpha^*$), Chunked Dense BGE-base, and BM25 + BGE-reranker cross-encoder cascade.
- **Corpus & Split:** 626 SROIE receipt page images (clean and degraded optical blur) evaluated on the frozen held-out Test split ($N=306$ probes, 813-item distractor pool including 500 COCO images).
- **Pre-Registered Decision Rule:** We will claim that "document-native visual retrieval outperforms the best proxy pipeline" **only if** the paired difference in MRR is positive with a 95% cluster bootstrap confidence interval strictly excluding zero ($\Delta > 0, p_{\text{Holm}} < 0.05$). If the CI includes zero or is negative, we will report that OCR text proxies remain competitive with or superior to native visual document retrieval.

### Hypothesis 2: Proxy-Quality Axis on Natural Images (Confirmatory)
- **Hypothesis:** Generating higher-capacity machine captions with advanced vision-language models closes the retrieval gap between text proxies and direct visual retrieval (CLIP / SigLIP) on MS-COCO.
- **Evaluated Caption Sources:**
  - Machine Proxy 1: BLIP (`Salesforce/blip-image-captioning-large`, frozen baseline).
  - Machine Proxy 2: BLIP-2 (`Salesforce/blip2-opt-2.7b`, fp16).
  - Machine Proxy 3: Florence-2 (`microsoft/Florence-2-base` or `microsoft/Florence-2-large`, detailed captioning).
  - Gold Standard: Human consensus ground truth captions (Karpathy test).
  - Native Visual Encoders: CLIP ViT-B/32 and SigLIP Base Patch16.
- **Corpus & Split:** 1,000 MS-COCO Karpathy-test images and held-out queries.
- **Pre-Registered Decision Rule:** We will claim that "advanced caption proxies close the gap to native visual retrieval" **only if** the paired difference between native visual MRR (CLIP or SigLIP) and the best caption proxy MRR has a 95% confidence interval including zero ($p \ge 0.05$). Otherwise, we will report the exact remaining empirical gap.

### Hypothesis 3: Query-Side Typo Noise vs. OCR Noise (Exploratory)
- **Hypothesis:** Sparse and dense text retrievers exhibit distinct vulnerability profiles to query-side character typos versus document-side OCR noise.
- **Typo Generation:** Synthetic character typos introduced at 5%, 10%, and 20% of characters (equal 1/3 mixture of character substitution, deletion, and adjacent transposition; seeded with 42, 43, 44; frozen to disk).
- **Evaluated Regimes:** Verbatim and light-paraphrase queries evaluated on Ground Truth text, Clean Apple Vision OCR, and Degraded Optical Blur OCR across Tuned BM25, Char 3–5gram BM25, Word+Char BM25 fusion, Chunked Dense BGE-base, and Tuned Hybrid.

### Hypothesis 4: Paraphrase Robustness Across Multi-LLM Generators (Confirmatory)
- **Hypothesis:** The large performance drop observed under light paraphrasing is an invariant property of lexical mismatch rather than an artifact of `gpt-4o-mini`.
- **Evaluated Generators:**
  - LLM 1: `gpt-4o-mini` (frozen v2/v3 benchmark queries, mean token overlap 41.2%).
  - LLM 2: Open-weights local model `Qwen/Qwen2.5-1.5B-Instruct` (or `Qwen/Qwen2.5-3B-Instruct`, $T=0.0$, strictly enforced token overlap band $35\% \le \text{overlap} \le 45\%$).
- **Audit & Invariance Criteria:**
  - Sentence-embedding cosine similarity distribution between original queries and paraphrases.
  - Fact-bearing token check ensuring key receipt entities (amounts, vendors, dates) are preserved.
  - Ranking stability: System ranking concordance across LLM 1 and LLM 2 paraphrases.

### Hypothesis 5: Automated Empirical Decision Guide (Synthesis)
- **Rule:** The decision guide matrix is synthesized automatically from empirical score tables.
- **Grid:** {Verbatim, Light Paraphrase, Answerable Question} $\times$ {Ground Truth, Clean OCR, Degraded Blur, Synthetic Noise} $\times$ {Document, Natural Image}.
- **Definition of "Best" and "Tied":** "Best" system is the highest point MRR; "Tied" systems include all candidates whose paired difference from the best system has a 95% bootstrap CI containing zero.

### Hypothesis 6: Supporting Analyses (Exploratory)
- **Cost Profiling:** Standardized per-query latency (mean and 95th percentile), peak GPU memory allocation (`torch.cuda.max_memory_allocated`), and on-disk index size.
- **Qualitative Failure Cases:** 6 stratified failure instances inspected and categorized by field type (`vendor`, `date`, `total`, `line_item`).
- **Statistical Power Analysis:** Computing minimum detectable effect size ($\Delta \text{MRR}$) at $80\%$ power and $\alpha=0.05$ across the 306-probe cluster partition.
- **Practical Reranker Stack Ablation:** Comparing BM25 alone, BM25 + MiniLM cross-encoder, BM25 + BGE reranker, and Tuned Hybrid + BGE reranker.

---

## 3. Dataset Integrity & Zero Test-Split Tuning Contract

1. **Strict Frozen Split Boundary:**
   - Corpus: 1,626 items (626 SROIE receipts, 1,000 COCO natural images).
   - Partition: Exact 50/50 Item ID split (813 Dev items, 813 Test items).
   - Probes: 306 Dev probes, 306 Test probes.
2. **Zero Test-Split Tuning Assertion:**
   - All hyperparameter values ($k_1^*=0.9, b^*=0.5$, hybrid $\alpha^*$) are loaded strictly from `results/v3/dev_tuned_parameters_v3.json`.
   - Every script contains programmatically enforced assertions verifying that Test item IDs never enter parameter fitting, vocabulary generation, or threshold calibration.
3. **Reproducibility & Statistical Standards:**
   - Random seeds: 42, 43, 44.
   - Cluster bootstrap by item ID ($B=2,000$, 95% percentile CIs).
   - Paired permutation tests ($N_{\text{perm}}=100{,}000$, sign flips).
   - Holm-Bonferroni correction within each hypothesis family.
   - Uniform random baseline reported analytically ($H_N/N$).

---

## 4. Kaggle GPU Execution & Packaging Architecture

- **Read-Only Input Dataset:** Mounted at `/kaggle/input/memoryverse-frozen-inputs-v5/` (or auto-discovered).
- **Working Output:** Written strictly to `/kaggle/working/` (hard limit 20 GB, target < 2 GB).
- **Cache Isolation:** `HF_HOME=/tmp/hf_cache`, `TRANSFORMERS_CACHE=/tmp/hf_cache`, `TORCH_HOME=/tmp/hf_cache` ensuring no model weights pollute `/kaggle/working`.
- **Resilient Multi-Stage Runner:**
  - Each stage produces a `.done` marker and updates `results_v5_partial.zip`.
  - Every stage is wrapped in `try...except` logging tracebacks to `RUNLOG.md` and continuing.
  - Automatic CUDA OOM handling: catch `OutOfMemoryError`, clear cache, halve batch size, retry.
- **Dual-Checkpoint Execution Flow:**
  - **Run A:** Stages 3 (Query Noise), 4 (Paraphrase Robustness), 2 (COCO Proxy Quality), 1 (ColQwen2 Visual Document Retriever). Emits `checkpoint1_native_visual.zip` and `CHECKPOINT1.md`.
  - **Run B:** Stages 6 (Supporting Analyses), 5 (Decision Guide Generator). Emits final `results_v5_final.zip` (< 2 GB, includes `SHA256SUMS.txt`).
  - `--run all`: Executes A followed by B in a single session.
