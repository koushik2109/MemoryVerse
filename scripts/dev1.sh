#!/usr/bin/env bash
# Compatibility entry point for the root Linux development launcher.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$SCRIPT_DIR/devl.sh" "$@"
