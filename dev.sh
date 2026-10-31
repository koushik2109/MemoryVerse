#!/usr/bin/env bash
# Compatibility entry point. The maintained Linux launcher is devl.sh.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/devl.sh" "$@"
