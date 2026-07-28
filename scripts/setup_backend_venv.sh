#!/usr/bin/env bash
# =============================================================================
# setup_backend_venv.sh
# Creates and activates a Python virtual environment for the MemoryVerse backend.
#
# Usage:
#   chmod +x scripts/setup_backend_venv.sh
#   source scripts/setup_backend_venv.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
VENV_DIR="$BACKEND_DIR/.venv"
PYTHON_MIN="3.11"

echo "╔══════════════════════════════════════════╗"
echo "║   MemoryVerse — Backend Environment      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Check Python version ──────────────────────────────────────────────────────
PYTHON_BIN=$(command -v python3 || command -v python)
PYTHON_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍  Using Python $PYTHON_VERSION  ($PYTHON_BIN)"

if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
  echo "✅  Python version OK (>= $PYTHON_MIN)"
else
  echo "❌  Python $PYTHON_MIN+ required. Please upgrade Python."
  exit 1
fi

# ── Create venv ───────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
  echo "ℹ️   Virtual environment already exists at: $VENV_DIR"
else
  echo "📦  Creating virtual environment at: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  echo "✅  Virtual environment created."
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "✅  Virtual environment activated."

# ── Upgrade pip ───────────────────────────────────────────────────────────────
echo ""
echo "📦  Upgrading pip..."
pip install --quiet --upgrade pip

# ── Install dependencies ──────────────────────────────────────────────────────
echo "📦  Installing backend dependencies from requirements.txt..."
pip install --quiet -r "$BACKEND_DIR/requirements.txt"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✅  Backend environment ready!          ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  To activate manually:"
echo "    source backend/.venv/bin/activate"
echo ""
echo "  To start the dev server:"
echo "    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
