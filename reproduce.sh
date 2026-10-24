#!/usr/bin/env bash
# ==============================================================================
# MemoryVerse Empirical Study Reproduction Script (v2 Addendum)
# 
# Manuscript: "When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense,
#              and Native-Visual Retrieval Under Proxy Noise"
# Target Journal: International Journal of Computing and Digital Systems (IJCDS)
#
# This script reproduces all experimental tables, figures, CSVs, and hypothesis
# test verdicts from frozen dataset artifacts without any test-split parameter tuning.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "MemoryVerse v2 Study: End-to-End Experimental Reproduction"
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
    "ai_engine/rag/data/v2/dev_tuned_params_cache.json"
    "ai_engine/rag/data/coco_blip_captions.json"
    "ai_engine/rag/data/coco_karpathy_test.json"
    "ai_engine/rag/data/coco_clip_image_embeddings.npy"
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
mkdir -p results/v2 latex_fragments

# 3. Phase 2: Regime Analysis Grid (3 Query Types x 6 Noise Regimes x 12 Retrieval Systems)
echo -e "\n[Phase 2] Executing Regime Analysis Grid (run_regime_grid_v2.py)..."
$PYTHON_EXE ai_engine/rag/scripts/run_regime_grid_v2.py

# 4. Phase 4: Component Ablation & Rerankers (run_ablation_v2.py)
echo -e "\n[Phase 4] Executing Component Ablation & Rerankers (run_ablation_v2.py)..."
$PYTHON_EXE ai_engine/rag/scripts/run_ablation_v2.py

# 5. Phase 3: Mechanism Analysis (run_mechanism_analysis_v2.py)
echo -e "\n[Phase 3] Executing Mechanism Analysis (run_mechanism_analysis_v2.py)..."
$PYTHON_EXE ai_engine/rag/scripts/run_mechanism_analysis_v2.py

# 6. Phase 5: Conformal Cascade Calibration across 200 Random Splits (run_conformal_cascade_v2.py)
echo -e "\n[Phase 5] Executing Conformal Cascade Calibration (run_conformal_cascade_v2.py)..."
$PYTHON_EXE ai_engine/rag/scripts/run_conformal_cascade_v2.py

# 7. Phase 7: COCO Karpathy-Test Native-Visual Claim Hardening (run_coco_native_visual_v2.py)
echo -e "\n[Phase 7] Executing COCO Native-Visual Benchmark (run_coco_native_visual_v2.py)..."
$PYTHON_EXE ai_engine/rag/scripts/run_coco_native_visual_v2.py

# 8. Phase 6: Multi-Hop TOST Equivalence Testing (run_multihop_equivalence_v2.py)
echo -e "\n[Phase 6] Executing Multi-Hop TOST Equivalence Testing (run_multihop_equivalence_v2.py)..."
$PYTHON_EXE ai_engine/rag/scripts/run_multihop_equivalence_v2.py

echo -e "\n========================================================================"
echo "ALL EXPERIMENTAL PHASES SUCCESSFULLY REPRODUCED"
echo "Generated CSV Tables:   results/v2/"
echo "Generated LaTeX Tables: latex_fragments/"
echo "Generated Line & Bar Plots: results/v2/*.png"
echo "========================================================================"
