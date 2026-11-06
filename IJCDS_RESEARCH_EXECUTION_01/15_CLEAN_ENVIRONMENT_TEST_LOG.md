# CLEAN ENVIRONMENT TEST LOG (PHASE 15)
- Extraction Test: Unzipped FINAL_IJCDS_REPRODUCIBILITY_PACKAGE.zip into isolated scratch directory.
- SHA-256 Check: All 31 files verified against SHA256_MANIFEST.txt (100% matched).
- Compilation Test: Ran tectonic main.tex inside isolated source directory -> Compiled main.pdf cleanly (15 pages).
- Execution Test: Re-executed scripts/gapfill_t3_02_generator_robustness.py on CPU -> Output matched results/v6/char_vs_word_bm25_v6.csv.
- Reproducibility Classification:
  - Category A (100% runnable from package on CPU): 14 suites.
  - Category B (GPU inference via Kaggle notebook): ColQwen2 visual embeddings.
  - Category C (Frozen third-party API): Historical LLM bridging.
  - Category D (Not reproducible): 0 experiments.
