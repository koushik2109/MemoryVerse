#!/usr/bin/env bash
# ==============================================================================
# MemoryVerse Empirical Study Master Reproduction Script (v3)
# 
# Manuscript: "When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense,
#              and Native-Visual Retrieval Under Proxy Noise"
#
# This script reproduces all experimental tables, figures, CSVs, and hypothesis
# test verdicts from frozen dataset artifacts without test-split parameter tuning.
# Zero-fabrication reproduction contract: every number traces directly to results/v3/.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "MemoryVerse v3 Study: End-to-End Experimental Reproduction"
echo "Repository Root: $PWD"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "========================================================================"

# 1. Environment & Python Executable Setup
PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Virtual environment python not found at $PYTHON_EXE."
    exit 1
fi

export PYTHONPATH="/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages:${PWD}/ai_engine/rag:${PYTHONPATH:-}"

echo "[Environment] Python: $($PYTHON_EXE --version)"
echo "[Environment] PyTorch: $($PYTHON_EXE -c "import torch; print(torch.__version__, 'MPS Available:', torch.backends.mps.is_available())")"

# 2. Verify Frozen Benchmark Artifacts
echo -e "\n[Verification] Checking frozen benchmark artifacts..."
REQUIRED_FILES=(
    "ai_engine/rag/data/v2/public_benchmark_v2.json"
    "ai_engine/rag/data/v2/frozen_probes_with_paraphrases.json"
    "ai_engine/rag/data/v2/sroie_degraded_ocr.json"
    "results/v3/heavy_paraphrases_v3.json"
    "ai_engine/rag/data/coco_blip_captions.json"
    "ai_engine/rag/data/coco_karpathy_test.json"
    "ai_engine/rag/data/coco_clip_image_embeddings.npy"
    "results/v3/coco_siglip_image_embeddings.npy"
    "ai_engine/rag/data/expanded_multihop_results.json"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "Error: Required frozen artifact missing: $f"
        exit 1
    fi
    echo "  [OK] $f"
done

# Create output directories
mkdir -p results/v3 latex_fragments

# 3. Phase 1: Debugging, OCR Normalization & Dense Truncation Checks
echo -e "\n[Phase 1] Executing Debugging & Normalization (run_phase1_debug_v3.py)..."
$PYTHON_EXE ai_engine/rag/scripts/run_phase1_debug_v3.py

# 4. Phase 2: Empirical Fixes, Conformal Calibration & Paired Analysis
echo -e "\n[Phase 2] Executing Fixes & Retractions (run_phase2_fixes_v3.py)..."
$PYTHON_EXE ai_engine/rag/scripts/run_phase2_fixes_v3.py

# 5. Phase 3: Reviewer Evidence, Headline Regime Grid & Native-Visual Benchmark
echo -e "\n[Phase 3] Executing Reviewer Evidence Generation (run_phase3_reviewer_evidence_v3.py)..."
$PYTHON_EXE ai_engine/rag/scripts/run_phase3_reviewer_evidence_v3.py

echo -e "\n[Verification] Auditing generated outputs in results/v3/ and latex_fragments/..."
V3_OUTPUTS=(
    "results/v3/ocr_error_rates.csv"
    "results/v3/dense_truncation_impact.csv"
    "results/v3/conformal_splits_summary_v3.csv"
    "results/v3/conformal_cascade_v3.csv"
    "results/v3/multihop_paired_results.csv"
    "results/v3/router_evaluation_v3.csv"
    "results/v3/light_paraphrase_paired_tests.csv"
    "results/v3/char_vs_word_bm25.csv"
    "results/v3/dev_tuned_parameters_v3.json"
    "results/v3/headline_regime_grid_v3.csv"
    "results/v3/headline_hypothesis_tests.csv"
    "results/v3/coco_native_visual_v3.csv"
    "results/v3/coco_hypothesis_test_v3.csv"
    "latex_fragments/table_regime_grid_v3.tex"
    "latex_fragments/table_ablation_v3.tex"
    "latex_fragments/table_conformal_v3.tex"
    "latex_fragments/table_multihop_v3.tex"
    "latex_fragments/table_coco_v3.tex"
    "latex_fragments/abstract_v3.tex"
    "latex_fragments/contributions_v3.tex"
)

for out in "${V3_OUTPUTS[@]}"; do
    if [ ! -f "$out" ]; then
        echo "Error: Expected output missing: $out"
        exit 1
    fi
    echo "  [FOUND] $out"
done

echo -e "\n========================================================================"
echo "ALL EXPERIMENTAL PHASES SUCCESSFULLY REPRODUCED"
echo "Generated CSV Tables:   results/v3/"
echo "Generated LaTeX Tables: latex_fragments/"
echo "Fix List:               PAPER_FIXLIST.md"
echo "Results Summary:        RESULTS_SUMMARY.md"
echo "========================================================================"
