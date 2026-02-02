#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/install_runtime_local.sh dist/asyncreview-runtime-v1.2.3-darwin-arm64.tar.gz 1.2.3
#
# Installs into:
#   macOS: ~/Library/Caches/asyncreview/runtimes/<version>/<platform>/
#   Linux: ~/.cache/asyncreview/runtimes/<version>/<platform>/

ARTIFACT="${1:?artifact path required}"
VERSION="${2:?version required}"

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
TMP_DIR="$INSTALL_DIR.tmp-$$"

echo "==> Installing runtime"
echo "    Artifact: $ARTIFACT"
echo "    Target:   $INSTALL_DIR"

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

tar -C "$TMP_DIR" -xzf "$ARTIFACT"

# Atomic replace
rm -rf "$INSTALL_DIR"
mkdir -p "$(dirname "$INSTALL_DIR")"
mv "$TMP_DIR" "$INSTALL_DIR"

echo "==> Installed. Entrypoint:"
echo "    $INSTALL_DIR/bin/asyncreview"
