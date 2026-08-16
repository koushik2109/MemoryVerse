#!/usr/bin/env bash
# ============================================================
#  MemoryVerse — Unified Development Launcher
#  Opens two separate terminal windows:
#    • Terminal 1 → Backend  (FastAPI / uvicorn)
#    • Terminal 2 → Frontend (Flutter)
#
#  Usage:  ./dev.sh
#          ./dev.sh --backend-only
#          ./dev.sh --frontend-only
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$SCRIPT_DIR/.venv"

# ── Colour helpers ────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
ok()   { echo -e "${GREEN}✔${NC}  $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "${RED}✘${NC}  $*"; exit 1; }

# ── Argument parsing ──────────────────────────────────────
RUN_BACKEND=true
RUN_FRONTEND=true
for arg in "$@"; do
  case "$arg" in
    --backend-only)  RUN_FRONTEND=false ;;
    --frontend-only) RUN_BACKEND=false  ;;
    --help|-h)
      echo "Usage: $0 [--backend-only | --frontend-only]"
      exit 0
      ;;
  esac
done

# ── Pre-flight checks ─────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║      MemoryVerse Dev Launcher        ║"
echo "╚══════════════════════════════════════╝"
echo ""

if $RUN_BACKEND; then
  [[ -d "$BACKEND_DIR" ]] || err "Backend directory not found: $BACKEND_DIR"
  
  warn "Checking for existing backend processes..."
  if command -v lsof >/dev/null 2>&1; then
    PIDS=$(lsof -t -i:8000 || true)
    if [ -n "$PIDS" ]; then
      warn "Killing existing process on port 8000 (PID: $PIDS)..."
      kill -9 $PIDS 2>/dev/null || true
    fi
  elif command -v fuser >/dev/null 2>&1; then
    if fuser 8000/tcp >/dev/null 2>&1; then
      warn "Killing existing process on port 8000..."
      fuser -k -9 8000/tcp >/dev/null 2>&1 || true
    fi
  else
    pkill -f "uvicorn app.main:app" >/dev/null 2>&1 || true
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    warn "Virtual environment not found at $VENV_DIR — creating one..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m ensurepip --quiet
    "$VENV_DIR/bin/python" -m pip install -q --upgrade pip
    "$VENV_DIR/bin/python" -m pip install -q -r "$BACKEND_DIR/requirements.txt"
    ok "Virtual environment created and dependencies installed."
  else
    ok "Virtual environment found."
  fi
fi

if $RUN_FRONTEND; then
  command -v flutter >/dev/null 2>&1 || err "Flutter is not in PATH. Install from https://flutter.dev"
  ok "Flutter found: $(flutter --version --machine 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get(\"frameworkVersion","?"))' 2>/dev/null || flutter --version | head -1)"
fi

# ── Backend command ───────────────────────────────────────
BACKEND_CMD="cd '$BACKEND_DIR' && source '$VENV_DIR/bin/activate' && echo '' && echo '  🚀  MemoryVerse Backend starting...' && echo '' && PYTHONPATH=. python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

# ── Frontend command ──────────────────────────────────────
FRONTEND_CMD="cd '$FRONTEND_DIR' && echo '' && echo '  📱  MemoryVerse Flutter starting...' && echo '' && flutter run"

# ── Terminal emulator detection + launch ─────────────────
launch_in_terminal() {
  local title="$1"
  local cmd="$2"

  # Try common terminal emulators in order of preference
  if command -v gnome-terminal &>/dev/null; then
    gnome-terminal --title="$title" -- bash -c "$cmd; exec bash" &
  elif command -v konsole &>/dev/null; then
    konsole --new-tab -p "tabtitle=$title" -e bash -c "$cmd; exec bash" &
  elif command -v xterm &>/dev/null; then
    xterm -title "$title" -e bash -c "$cmd; exec bash" &
  elif command -v tilix &>/dev/null; then
    tilix -a session-add-right --title="$title" -e bash -c "$cmd; exec bash" &
  elif command -v x-terminal-emulator &>/dev/null; then
    x-terminal-emulator -title "$title" -e bash -c "$cmd; exec bash" &
  else
    # Fallback: run in background processes in same terminal
    warn "No graphical terminal found. Running both in background."
    echo "  Backend log → /tmp/mv_backend.log"
    echo "  Frontend log → /tmp/mv_frontend.log"
    bash -c "$cmd" > /tmp/mv_backend.log 2>&1 &
    return
  fi
}

# ── Launch ────────────────────────────────────────────────
if $RUN_BACKEND && $RUN_FRONTEND; then
  ok "Opening Backend terminal..."
  launch_in_terminal "MemoryVerse — Backend" "$BACKEND_CMD"
  sleep 1  # small delay so windows don't stack perfectly
  ok "Opening Frontend terminal..."
  launch_in_terminal "MemoryVerse — Frontend" "$FRONTEND_CMD"
  echo ""
  ok "Both services launching!"
  echo ""
  echo "  📡  Backend  → http://localhost:8000"
  echo "  📡  API docs → http://localhost:8000/docs"
  echo "  📱  Frontend → check the Flutter terminal for device selection"
  echo ""
elif $RUN_BACKEND; then
  ok "Opening Backend terminal..."
  launch_in_terminal "MemoryVerse — Backend" "$BACKEND_CMD"
  echo "  📡  Backend  → http://localhost:8000"
elif $RUN_FRONTEND; then
  ok "Opening Frontend terminal..."
  launch_in_terminal "MemoryVerse — Frontend" "$FRONTEND_CMD"
  echo "  📱  Frontend → check the Flutter terminal"
fi

echo ""
echo "  Press Ctrl+C in each terminal window to stop."
echo ""
