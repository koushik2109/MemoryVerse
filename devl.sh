#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$SCRIPT_DIR/.venv"
AI_DIR="$SCRIPT_DIR/ai_engine/rag"
AI_VENV="$AI_DIR/.venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  if [[ -x "$SCRIPT_DIR/.venv-linux/bin/python" ]]; then
    VENV_DIR="$SCRIPT_DIR/.venv-linux"
  elif [[ -x "$HOME/MemoryVerse/.venv/bin/python" ]]; then
    VENV_DIR="$HOME/MemoryVerse/.venv"
  fi
fi

RUN_BACKEND=true
RUN_FRONTEND=true
RUN_AI=true
PREPARE_FLUTTER=false
ANALYZE=false
DEVICE="${FLUTTER_DEVICE:-}"
HEALTH_TIMEOUT=180
BIND_HOST="${MEMORYVERSE_BIND_HOST:-127.0.0.1}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]
  --backend-only       Start only the FastAPI backend
  --frontend-only      Start only the Flutter app
  --skip-ai            Do not start the RAG microservice
  --prepare-flutter    Run flutter pub get before launching the app
  --analyze            Run flutter analyze before launching the app
  --device ID          Select a Flutter device (defaults to Flutter's selection)
  --health-timeout SEC Timeout for backend/AI health checks (default: 180)
  Set MEMORYVERSE_BIND_HOST=0.0.0.0 to allow physical devices on your LAN to reach the backend.
  -h, --help           Show this help
USAGE
}

while (($#)); do
  case "$1" in
    --backend-only) RUN_FRONTEND=false ;;
    --frontend-only) RUN_BACKEND=false; RUN_AI=false ;;
    --skip-ai) RUN_AI=false ;;
    --prepare-flutter) PREPARE_FLUTTER=true ;;
    --analyze) ANALYZE=true ;;
    --device) shift; (($#)) || { echo "--device requires an ID" >&2; exit 2; }; DEVICE="$1" ;;
    --health-timeout) shift; (($#)) || { echo "--health-timeout requires seconds" >&2; exit 2; }; HEALTH_TIMEOUT="$1" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "$HEALTH_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "Health timeout must be a positive integer." >&2; exit 2; }

for required_dir in "$BACKEND_DIR" "$FRONTEND_DIR"; do
  [[ -d "$required_dir" ]] || { echo "Required project directory not found: $required_dir" >&2; exit 1; }
done

for tool in python3 lsof curl; do
  command -v "$tool" >/dev/null 2>&1 || { echo "Required command is missing: $tool" >&2; exit 1; }
done

for port in 8000 8001; do
  if [[ "$port" == 8000 ]] && ! $RUN_BACKEND; then continue; fi
  if [[ "$port" == 8001 ]] && ! $RUN_AI; then continue; fi
  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use. Stop that service yourself or configure a different port; this launcher will not kill it." >&2
    exit 1
  fi
done

require_python_env() {
  local python_path="$1" label="$2" probe="$3"
  [[ -x "$python_path" ]] || { echo "$label Python environment missing: $python_path. Create it and install the documented requirements first." >&2; exit 1; }
  "$python_path" -c "$probe" || { echo "$label Python environment is missing required packages. Install its requirements first." >&2; exit 1; }
}

if $RUN_BACKEND; then
  require_python_env "$VENV_DIR/bin/python" "Backend" 'import fastapi, uvicorn'
fi
if $RUN_AI; then
  [[ -x "$AI_VENV/bin/python" ]] || AI_VENV="$VENV_DIR"
  require_python_env "$AI_VENV/bin/python" "AI RAG engine" 'import fastapi, uvicorn'
fi
if $RUN_FRONTEND; then
  command -v flutter >/dev/null 2>&1 || { echo "Flutter is not in PATH." >&2; exit 1; }
  if ! $PREPARE_FLUTTER && [[ ! -f "$FRONTEND_DIR/.dart_tool/package_config.json" ]]; then
    echo "Flutter dependencies are not prepared. Run 'flutter pub get' in frontend or pass --prepare-flutter." >&2
    exit 1
  fi
  if [[ -n "$DEVICE" ]] && ! flutter devices --machine | python3 -c 'import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if any(x.get("id")==sys.argv[1] for x in d) else 1)' "$DEVICE"; then
    echo "Flutter device '$DEVICE' is not available. Check 'flutter devices'." >&2
    exit 1
  fi
fi

TERMINAL=""
for candidate in gnome-terminal konsole xfce4-terminal x-terminal-emulator xterm; do
  if command -v "$candidate" >/dev/null 2>&1; then TERMINAL="$candidate"; break; fi
done
if [[ -z "$TERMINAL" ]]; then
  echo "No supported graphical terminal found. Install gnome-terminal, konsole, xfce4-terminal, x-terminal-emulator, or xterm." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/memoryverse.XXXXXX")"
PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

launch() {
  local title="$1" command_text="$2"
  local service_script="$TMP_DIR/$(printf '%s' "$title" | tr -cs '[:alnum:]' '_').sh"
  {
    printf '%s\n' '#!/usr/bin/env bash' 'set +e'
    printf 'trap %q EXIT\n' "rm -f -- '$service_script'; rmdir '$TMP_DIR' 2>/dev/null || true"
    printf '%s\n' "$command_text"
    printf '%s\n' 'status=$?' "echo" "echo 'Service exited with status' \$status" 'bash'
  } > "$service_script"
  case "$TERMINAL" in
    gnome-terminal) gnome-terminal --title="$title" -- bash "$service_script" ;;
    konsole) konsole --new-tab -p "tabtitle=$title" -e bash "$service_script" ;;
    xfce4-terminal) xfce4-terminal --title="$title" --command="bash '$service_script'" ;;
    x-terminal-emulator) x-terminal-emulator -e bash "$service_script" ;;
    xterm) xterm -T "$title" -e bash "$service_script" ;;
  esac
}

wait_healthy() {
  local name="$1" url="$2" deadline=$((SECONDS + HEALTH_TIMEOUT))
  until curl --fail --silent --show-error "$url" 2>/dev/null | python3 -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("status")=="healthy" else 1)' 2>/dev/null; do
    if ((SECONDS >= deadline)); then echo "$name did not become healthy within ${HEALTH_TIMEOUT}s." >&2; return 1; fi
    sleep 1
  done
  echo "$name is healthy at $url"
}

if $RUN_AI; then
  launch "MemoryVerse AI RAG Engine" "cd $(printf '%q' "$SCRIPT_DIR") && PYTHONPATH=$(printf '%q' "$SCRIPT_DIR") $(printf '%q' "$AI_VENV/bin/python") -m uvicorn ai_engine.rag.server:app --reload --reload-dir ai_engine --host 127.0.0.1 --port 8001" || { echo "Could not open terminal for AI engine." >&2; exit 1; }
fi
if $RUN_BACKEND; then
  launch "MemoryVerse Backend" "cd $(printf '%q' "$BACKEND_DIR") && PYTHONPATH=. $(printf '%q' "$VENV_DIR/bin/python") -m uvicorn app.main:app --reload --reload-dir app --host $(printf '%q' "$BIND_HOST") --port 8000" || { echo "Could not open terminal for backend." >&2; exit 1; }
fi
if $RUN_BACKEND; then wait_healthy Backend http://127.0.0.1:8000/health; fi
if $RUN_AI; then wait_healthy "AI RAG engine" http://127.0.0.1:8001/health; fi

if $RUN_FRONTEND; then
  flutter_args=(run)
  [[ -z "$DEVICE" ]] || flutter_args+=(-d "$DEVICE")
  frontend_code="cd $(printf '%q' "$FRONTEND_DIR")"
  if $PREPARE_FLUTTER; then frontend_code+=" && flutter pub get"; fi
  if $ANALYZE; then frontend_code+=" && flutter analyze"; fi
  printf -v args_q '%q ' "${flutter_args[@]}"
  frontend_code+=" && flutter ${args_q% }"
  launch "MemoryVerse Flutter Frontend" "$frontend_code" || { echo "Could not open terminal for Flutter." >&2; exit 1; }
fi

echo "MemoryVerse services launched. Backend: http://127.0.0.1:8000; AI RAG: http://127.0.0.1:8001"
