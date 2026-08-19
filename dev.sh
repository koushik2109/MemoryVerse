#!/bin/bash

# =================================================================
#       MemoryVerse Dev Launcher (macOS / Linux)
# =================================================================

RUN_BACKEND=true
RUN_FRONTEND=true

# Parse flags
for arg in "$@"; do
  case $arg in
    --backend-only|-b)
      RUN_FRONTEND=false
      shift
      ;;
    --frontend-only|-f)
      RUN_BACKEND=false
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$SCRIPT_DIR/.venv"

# Helper print functions
print_ok() { echo -e "\033[0;32m[OK]  $1\033[0m"; }
print_warn() { echo -e "\033[0;33m[!]   $1\033[0m"; }
print_err() { echo -e "\033[0;31m[ERR] $1\033[0m"; exit 1; }

echo ""
echo "========================================"
echo "       MemoryVerse Dev Launcher         "
echo "========================================"
echo ""

# Backend Check
if [ "$RUN_BACKEND" = true ]; then
  if [ ! -d "$BACKEND_DIR" ]; then
    print_err "Backend directory not found: $BACKEND_DIR"
  fi

  # Port 8000 check
  print_warn "Checking for existing backend processes on port 8000..."
  PIDS=$(lsof -t -i:8000 2>/dev/null)
  if [ -n "$PIDS" ]; then
    print_warn "Killing existing process on port 8000 (PID: $PIDS)..."
    echo "$PIDS" | xargs kill -9 2>/dev/null
  fi

  # Python virtual environment check
  if [ ! -d "$VENV_DIR" ]; then
    print_warn "Virtual environment not found at $VENV_DIR -- creating one..."
    python3 -m venv "$VENV_DIR" || print_err "Failed to create virtual environment. Ensure python3 is installed."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$BACKEND_DIR/requirements.txt"
    print_ok "Virtual environment created and dependencies installed."
  else
    print_ok "Virtual environment found."
  fi
fi

# Frontend Check
if [ "$RUN_FRONTEND" = true ]; then
  if ! command -v flutter &> /dev/null; then
    print_err "Flutter is not in PATH. Install from https://flutter.dev"
  fi
  FLUTTER_VER=$(flutter --version | head -n 1)
  print_ok "Flutter found: $FLUTTER_VER"
fi

# Launching terminals (macOS specific terminal launcher, falls back to background jobs on Linux)
IS_MAC=false
if [[ "$OSTYPE" == "darwin"* ]]; then
  IS_MAC=true
fi

if [ "$RUN_BACKEND" = true ]; then
  print_ok "Starting Backend..."
  if [ "$IS_MAC" = true ]; then
    # Open in new macOS Terminal window
    osascript -e "tell application \"Terminal\" to do script \"cd '$BACKEND_DIR' && source '$VENV_DIR/bin/activate' && PYTHONPATH=. python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000\""
  else
    # Fallback/Linux: run in background and output to log
    cd "$BACKEND_DIR" && source "$VENV_DIR/bin/activate" && PYTHONPATH=. python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > "$SCRIPT_DIR/backend.log" 2>&1 &
    print_warn "Backend running in background. Logs: backend.log"
  fi
fi

if [ "$RUN_BACKEND" = true ] && [ "$RUN_FRONTEND" = true ]; then
  sleep 1
fi

if [ "$RUN_FRONTEND" = true ]; then
  print_ok "Starting Frontend..."
  if [ "$IS_MAC" = true ]; then
    # Open in new macOS Terminal window
    osascript -e "tell application \"Terminal\" to do script \"cd '$FRONTEND_DIR' && flutter run\""
  else
    # Fallback/Linux: run in background and output to log
    cd "$FRONTEND_DIR" && flutter run > "$SCRIPT_DIR/frontend.log" 2>&1 &
    print_warn "Frontend running in background. Logs: frontend.log"
  fi
fi

echo ""
if [ "$RUN_BACKEND" = true ] && [ "$RUN_FRONTEND" = true ]; then
  print_ok "Both services launching!"
  echo ""
  echo "  API Backend -> http://localhost:8000"
  echo "  API Docs    -> http://localhost:8000/docs"
  echo "  Frontend    -> check the Terminal window for device selection (supporting iOS/Simulator)"
elif [ "$RUN_BACKEND" = true ]; then
  echo "  API Backend -> http://localhost:8000"
elif [ "$RUN_FRONTEND" = true ]; then
  echo "  Frontend    -> check the Terminal window"
fi
echo ""
