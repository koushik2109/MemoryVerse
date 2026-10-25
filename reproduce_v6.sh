#!/usr/bin/env bash
# ==============================================================================
# reproduce_v6.sh: Complete Reproduction of All Phase 0-7 Tables and Figures
# ==============================================================================
set -e

echo "=== Step 1: Checking Python Virtual Environment ==="
if [ ! -d ".venv" ]; then
    echo "Creating virtualenv .venv..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
fi
.venv/bin/pip install -q rank_bm25 sentence_transformers jiwer rapidfuzz pandas matplotlib torch torchvision

echo "=== Step 2: Phase 1 Forensic Resolvers (Bugs a-i) ==="
.venv/bin/python ai_engine/rag/scripts/run_phase1_fixes_v6.py

echo "=== Step 3: Phase 2 Dose-Response Analysis & Dual Plot ==="
.venv/bin/python ai_engine/rag/scripts/run_phase2_dose_response_v6.py

echo "=== Step 4: Phase 5 Constructive Recipe Tuning & Evaluation ==="
.venv/bin/python ai_engine/rag/scripts/run_phase5_recipe_v6.py

echo "=== Step 5: Phase 6 Query-Side Remedies Evaluation ==="
.venv/bin/python ai_engine/rag/scripts/run_phase6_remedies_v6.py

echo "=== Step 6: Phase 7 Claims Ledger and Table Regeneration ==="
.venv/bin/python ai_engine/rag/scripts/run_phase7_claims_v6.py

echo "=== Reproduction Complete! All outputs generated in results/v6/ and latex_fragments_v6/ ==="
