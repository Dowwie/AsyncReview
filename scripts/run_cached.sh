#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_cached.sh 1.2.3 review --url ... -q "..."
#
VERSION="${1:?version required}"
shift || true

PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  arm64|aarch64) ARCH="arm64" ;;
  x86_64) ARCH="x64" ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac
PLATFORM_KEY="${PLATFORM}-${ARCH}"

if [ "$PLATFORM" = "darwin" ]; then
  BASE_CACHE="${ASYNCREVIEW_RUNTIME_DIR:-$HOME/Library/Caches/asyncreview}"
else
  BASE_CACHE="${ASYNCREVIEW_RUNTIME_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/asyncreview}"
fi

INSTALL_DIR="$BASE_CACHE/runtimes/$VERSION/$PLATFORM_KEY"
ENTRY="$INSTALL_DIR/bin/asyncreview"

if [ ! -x "$ENTRY" ]; then
  echo "Runtime not installed at: $ENTRY" >&2
  echo "Run: ./scripts/install_runtime_local.sh <artifact> $VERSION" >&2
  exit 1
fi

exec "$ENTRY" "$@"
