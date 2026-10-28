#!/usr/bin/env bash
# ==============================================================================
# reproduce_v10.sh: Master reproduction harness for Manuscript V10
# ==============================================================================
set -euo pipefail

echo "=== [1/5] Verifying Frozen Splits, Inputs, and Kaggle Artifacts ==="
test -f ai_engine/rag/data/v2/public_benchmark_v2.json || { echo "Missing public_benchmark_v2.json"; exit 1; }
test -f ai_engine/rag/data/v2/frozen_probes_with_paraphrases.json || { echo "Missing frozen_probes_with_paraphrases.json"; exit 1; }
test -f results/v10/number_provenance.csv || { echo "Missing results/v10/number_provenance.csv"; exit 1; }
test -f kaggle/kaggle_run_v10.py || { echo "Missing kaggle/kaggle_run_v10.py"; exit 1; }
echo "Frozen dataset inputs and Kaggle run scripts verified."

echo "=== [2/5] Validating Number Provenance (check_numbers.py) ==="
.venv/bin/python scripts/check_numbers.py

echo "=== [3/5] Validating Abstract Word Limit (<= 200 words) ==="
.venv/bin/python -c '
import re
with open("IEEE_Paper_v10.tex") as f:
    tex = f.read()
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL).group(1).strip()
words = abstract.split()
assert len(words) <= 200, f"Abstract exceeds 200 words: {len(words)}"
print(f"PASS: Abstract contains {len(words)} words (limit: 200).")
'

echo "=== [4/5] Compiling Manuscript IEEE_Paper_v10.tex via Tectonic ==="
if command -v tectonic >/dev/null 2>&1; then
    tectonic --keep-logs IEEE_Paper_v10.tex
    echo "Compiled successfully to IEEE_Paper_v10.pdf via Tectonic."
elif command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode IEEE_Paper_v10.tex
    pdflatex -interaction=nonstopmode IEEE_Paper_v10.tex
    echo "Compiled successfully via pdflatex."
else
    echo "ERROR: Neither tectonic nor pdflatex found."
    exit 1
fi

echo "=== [5/5] Auditing LaTeX Log for Undefined References and Citations ==="
.venv/bin/python << 'EOF'
import re
with open("IEEE_Paper_v10.log", "r", errors="ignore") as f:
    log = f.read()
undef_refs = re.findall(r"Reference\s+.*undefined", log)
undef_cites = re.findall(r"Citation\s+.*undefined", log)
assert len(undef_refs) == 0, f"Undefined references found: {undef_refs}"
assert len(undef_cites) == 0, f"Undefined citations found: {undef_cites}"
print("PASS: 0 undefined references, 0 undefined citations.")
EOF

echo "=== ALL V10 REPRODUCTION CHECKS PASSED ==="
