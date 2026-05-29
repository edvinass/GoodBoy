#!/usr/bin/env bash
# Build neo.tar.gz (and sync install.sh) for the Railway web install bundle.
#
# Usage:
#   ./scripts/release-web-tar.sh
#   ./scripts/release-web-tar.sh --output /tmp/neo.tar.gz
#   ./scripts/release-web-tar.sh --version 0.1.0
#
# Defaults:
#   Tarball:   web/public/neo.tar.gz
#   Installer: web/install.sh (copied from repo root; Railway only deploys web/)

set -euo pipefail

_update_version() {
  local ver="$1"
  sed -i '' "s/^version = \"[^\"]*\"/version = \"$ver\"/" "$NEO_DIR/pyproject.toml"
  echo "Updated version in pyproject.toml to $ver"
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEO_DIR="$ROOT/neo"
INSTALL_SH="$ROOT/install.sh"
DEFAULT_OUTPUT="$ROOT/web/public/neo.tar.gz"
WEB_INSTALL_SH="$ROOT/web/install.sh"

OUTPUT="$DEFAULT_OUTPUT"
VERSION=""

usage() {
  sed -n '2,12p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  # Auto-bump patch version from pyproject.toml
  current=$(grep -E '^version = ' "$NEO_DIR/pyproject.toml" | sed -E 's/version = "([^"]+)"/\1/')
  major=$(echo "$current" | cut -d. -f1)
  minor=$(echo "$current" | cut -d. -f2)
  patch=$(echo "$current" | cut -d. -f3)
  # Strip any pre-release suffix
  patch=$(echo "$patch" | sed 's/[^0-9].*//')
  new_patch=$((patch + 1))
  VERSION="${major}.${minor}.${new_patch}"
fi

echo "Setting version to $VERSION"
_update_version "$VERSION"

if [[ -d "$NEO_DIR/__pycache__" ]]; then
  echo "Cleaning __pycache__ before tar"
  rm -rf "$NEO_DIR/__pycache__"
fi

if [[ ! -d "$NEO_DIR" ]]; then
  echo "error: missing directory: $NEO_DIR" >&2
  exit 1
fi

if [[ ! -f "$NEO_DIR/pyproject.toml" ]]; then
  echo "error: missing $NEO_DIR/pyproject.toml" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "Creating $(basename "$OUTPUT") from neo/ …"
tar -czf "$OUTPUT" -C "$ROOT" \
  --exclude='__pycache__' \
  --exclude='*.py[cod]' \
  --exclude='.pytest_cache' \
  --exclude='*.egg-info' \
  --exclude='.venv' \
  neo

if [[ -n "$VERSION" ]]; then
  versioned="$ROOT/web/public/neo-${VERSION}.tar.gz"
  cp -f "$OUTPUT" "$versioned"
  echo "Also wrote $(basename "$versioned")"
fi

if [[ ! -f "$INSTALL_SH" ]]; then
  echo "error: missing $INSTALL_SH" >&2
  exit 1
fi
cp -f "$INSTALL_SH" "$WEB_INSTALL_SH"
echo "Synced install.sh -> web/install.sh"

bytes="$(wc -c <"$OUTPUT" | tr -d ' ')"
echo ""
echo "Commit and redeploy web/ (or push to trigger Railway)."
echo "Install:"
echo "  curl -fsSL https://YOUR-DOMAIN/install.sh | bash"
