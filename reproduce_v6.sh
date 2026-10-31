#!/usr/bin/env bash
# ==============================================================================
# reproduce_v6.sh: Complete Reproduction of All Phase 0-7 Tables and Figures
# ==============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_EXE="${PYTHON_EXE:-$SCRIPT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_EXE" ] && [ -x "$SCRIPT_DIR/.venv-linux/bin/python" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv-linux/bin/python"
elif [ ! -x "$PYTHON_EXE" ] && [ -x "$HOME/MemoryVerse/.venv/bin/python" ]; then
    PYTHON_EXE="$HOME/MemoryVerse/.venv/bin/python"
fi

echo "=== Step 1: Checking Python Virtual Environment ==="
if [ ! -x "$PYTHON_EXE" ]; then
    echo "Python environment not found at $PYTHON_EXE." >&2
    echo "Create .venv and install kaggle/requirements_kaggle.txt, or set PYTHON_EXE to an environment with the research dependencies." >&2
    exit 1
fi

"$PYTHON_EXE" - <<'PY'
import importlib.util
required = ('rank_bm25', 'sentence_transformers', 'jiwer', 'rapidfuzz', 'pandas', 'matplotlib', 'torch', 'torchvision')
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit('Missing research packages: ' + ', '.join(missing) + '. Install kaggle/requirements_kaggle.txt in this environment.')
print('Python research dependencies are available.')
PY

echo "=== Step 2: Phase 1 Forensic Resolvers (Bugs a-i) ==="
"$PYTHON_EXE" ai_engine/rag/scripts/run_phase1_fixes_v6.py

echo "=== Step 3: Phase 2 Dose-Response Analysis & Dual Plot ==="
"$PYTHON_EXE" ai_engine/rag/scripts/run_phase2_dose_response_v6.py

echo "=== Step 4: Phase 5 Constructive Recipe Tuning & Evaluation ==="
"$PYTHON_EXE" ai_engine/rag/scripts/run_phase5_recipe_v6.py

echo "=== Step 5: Phase 6 Query-Side Remedies Evaluation ==="
"$PYTHON_EXE" ai_engine/rag/scripts/run_phase6_remedies_v6.py

echo "=== Step 6: Phase 7 Claims Ledger and Table Regeneration ==="
"$PYTHON_EXE" ai_engine/rag/scripts/run_phase7_claims_v6.py

echo "=== Reproduction Complete! All outputs generated in results/v6/ and latex_fragments_v6/ ==="
