#!/usr/bin/env bash

# MemoryVerse Dev Launcher for macOS & Linux

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$SCRIPT_DIR/.venv"

echo ""
echo "========================================"
echo "       MemoryVerse Dev Launcher (macOS) "
echo "========================================"
echo ""

# 1. Kill existing backend process on port 8000
echo "[!]   Checking for existing backend processes on port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# 2. Check/create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "[!]   Virtual environment not found -- creating one..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q --upgrade pip
    "$VENV_DIR/bin/pip" install -q -r "$BACKEND_DIR/requirements.txt"
    echo "[OK]  Virtual environment created and dependencies installed."
else
    echo "[OK]  Virtual environment found."
fi

# 3. Check Flutter
if ! command -v flutter &> /dev/null; then
    echo "[ERR] Flutter is not in PATH. Please install Flutter for macOS."
    exit 1
else
    FLUTTER_VER=$(flutter --version | head -n 1)
    echo "[OK]  Flutter found: $FLUTTER_VER"
fi

# 4. Launch Backend
echo "[OK]  Launching Backend server..."
(cd "$BACKEND_DIR" && PYTHONPATH=. "$VENV_DIR/bin/python" -m uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

sleep 2

# 5. Launch Frontend
echo "[OK]  Launching Frontend Flutter app..."
(cd "$FRONTEND_DIR" && flutter run)

# Cleanup on exit
kill $BACKEND_PID 2>/dev/null || true
