#!/usr/bin/env bash
# Build goodboy.tar.gz (and sync install.sh) for the Railway web install bundle.
#
# Usage:
#   ./scripts/release-web-tar.sh
#   ./scripts/release-web-tar.sh --output /tmp/goodboy.tar.gz
#   ./scripts/release-web-tar.sh --version 0.1.0
#
# Defaults:
#   Tarball:  web/public/goodboy.tar.gz
#   Installer: web/public/install.sh (copied from repo root)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOODBOY_DIR="$ROOT/goodboy"
INSTALL_SH="$ROOT/install.sh"
DEFAULT_OUTPUT="$ROOT/web/public/goodboy.tar.gz"
DEFAULT_INSTALL_COPY="$ROOT/web/public/install.sh"

OUTPUT="$DEFAULT_OUTPUT"
VERSION=""
SKIP_INSTALL_SH=0

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
    --skip-install-sh)
      SKIP_INSTALL_SH=1
      shift
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$GOODBOY_DIR" ]]; then
  echo "error: missing directory: $GOODBOY_DIR" >&2
  exit 1
fi

if [[ ! -f "$GOODBOY_DIR/pyproject.toml" ]]; then
  echo "error: missing $GOODBOY_DIR/pyproject.toml" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "Creating $(basename "$OUTPUT") from goodboy/ …"
tar -czf "$OUTPUT" -C "$ROOT" \
  --exclude='__pycache__' \
  --exclude='*.py[cod]' \
  --exclude='.pytest_cache' \
  --exclude='*.egg-info' \
  --exclude='.venv' \
  goodboy

if [[ -n "$VERSION" ]]; then
  versioned="$ROOT/web/public/goodboy-${VERSION}.tar.gz"
  cp -f "$OUTPUT" "$versioned"
  echo "Also wrote $(basename "$versioned")"
fi

if [[ "$SKIP_INSTALL_SH" -eq 0 ]]; then
  if [[ ! -f "$INSTALL_SH" ]]; then
    echo "error: missing $INSTALL_SH" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$DEFAULT_INSTALL_COPY")"
  cp -f "$INSTALL_SH" "$DEFAULT_INSTALL_COPY"
  echo "Synced install.sh -> web/public/install.sh"
fi

bytes="$(wc -c <"$OUTPUT" | tr -d ' ')"
echo "Done: $OUTPUT ($(numfmt --to=iec-i --suffix=B "$bytes" 2>/dev/null || echo "${bytes} bytes"))"
echo ""
echo "Commit and redeploy web/ (or push to trigger Railway)."
echo "Install:"
echo "  GOODBOY_INSTALL_BASE_URL=https://YOUR-DOMAIN bash -c \"\$(curl -fsSL https://YOUR-DOMAIN/install.sh)\""
