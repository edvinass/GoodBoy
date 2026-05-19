#!/usr/bin/env bash
# GoodBoy installer
#
#   GOODBOY_INSTALL_BASE_URL=https://YOUR-DOMAIN bash -c "$(curl -fsSL https://YOUR-DOMAIN/install.sh)"
#   curl -fsSL https://raw.githubusercontent.com/edvinass/GoodBoy/main/install.sh | bash
#
# Env vars before curl do not reach the piped bash — set them on bash instead.
#
# Environment (optional):
#   GOODBOY_INSTALL_DIR        install location (default: ~/.local/share/goodboy)
#   GOODBOY_INSTALL_BASE_URL   download goodboy.tar.gz from here (no git/GitHub)
#   GOODBOY_REPO_URL           git remote when not using a tarball (default: GitHub)
#   GOODBOY_REPO_REF           branch or tag to install (default: main)
#   GOODBOY_SOURCE_DIR         use an existing checkout instead of fetching
#   GOODBOY_LOCAL=1            also install optional local GGUF dependencies
#   GOODBOY_NO_PATH=1          do not append the venv bin dir to shell startup files

set -euo pipefail

INSTALL_DIR="${GOODBOY_INSTALL_DIR:-$HOME/.local/share/goodboy}"
REPO_URL="${GOODBOY_REPO_URL:-https://github.com/edvinass/GoodBoy.git}"
REPO_REF="${GOODBOY_REPO_REF:-main}"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

info() { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

version_ge() {
  # usage: version_ge 3 10 3 11  -> true if 3.10 >= 3.11 is false; compares major.minor
  local a_maj=$1 a_min=$2 b_maj=$3 b_min=$4
  if (( a_maj > b_maj )); then return 0; fi
  if (( a_maj < b_maj )); then return 1; fi
  (( a_min >= b_min ))
}

find_python() {
  local candidates=(python3.13 python3.12 python3.11 python3.10 python3)
  local py maj min
  for py in "${candidates[@]}"; do
    if ! command -v "$py" >/dev/null 2>&1; then
      continue
    fi
    read -r maj min < <("$py" -c 'import sys; print(sys.version_info.major, sys.version_info.minor)')
    if version_ge "$maj" "$min" "$MIN_PYTHON_MAJOR" "$MIN_PYTHON_MINOR"; then
      printf '%s\n' "$py"
      return 0
    fi
  done
  return 1
}

venv_python() {
  if [[ -f "$INSTALL_DIR/.venv/Scripts/python.exe" ]]; then
    printf '%s\n' "$INSTALL_DIR/.venv/Scripts/python.exe"
  else
    printf '%s\n' "$INSTALL_DIR/.venv/bin/python"
  fi
}

venv_bin_dir() {
  if [[ -d "$INSTALL_DIR/.venv/Scripts" ]]; then
    printf '%s\n' "$INSTALL_DIR/.venv/Scripts"
  else
    printf '%s\n' "$INSTALL_DIR/.venv/bin"
  fi
}

package_dir() {
  if [[ -f "$INSTALL_DIR/goodboy/pyproject.toml" ]]; then
    printf '%s\n' "$INSTALL_DIR/goodboy"
  elif [[ -f "$INSTALL_DIR/python/pyproject.toml" ]]; then
    printf '%s\n' "$INSTALL_DIR/python"
  else
    die "Missing goodboy/ or python/ package under $INSTALL_DIR"
  fi
}

fetch_from_tarball() {
  local url="${GOODBOY_INSTALL_BASE_URL%/}/goodboy.tar.gz"
  local tmp archive
  need_cmd curl
  need_cmd tar
  tmp="$(mktemp -d)"
  archive="$tmp/goodboy.tar.gz"
  info "Downloading GoodBoy from $url"
  curl -fsSL "$url" -o "$archive"
  mkdir -p "$INSTALL_DIR"
  tar -xzf "$archive" -C "$INSTALL_DIR"
  rm -rf "$tmp"
}

ensure_source_tree() {
  if [[ -n "${GOODBOY_SOURCE_DIR:-}" ]]; then
    INSTALL_DIR="$(cd "$GOODBOY_SOURCE_DIR" && pwd)"
    info "Using existing source tree: $INSTALL_DIR"
    return 0
  fi

  if [[ -n "${GOODBOY_INSTALL_BASE_URL:-}" ]]; then
    fetch_from_tarball
    return 0
  fi

  need_cmd git
  if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Updating GoodBoy at $INSTALL_DIR"
    git -C "$INSTALL_DIR" fetch --depth 1 origin "$REPO_REF"
    git -C "$INSTALL_DIR" checkout -q FETCH_HEAD
  else
    info "Cloning GoodBoy into $INSTALL_DIR"
    mkdir -p "$(dirname "$INSTALL_DIR")"
    GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$INSTALL_DIR" 2>/dev/null \
      || GIT_TERMINAL_PROMPT=0 git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
  fi
}

create_venv() {
  local py="$1"
  if [[ -x "$(venv_python)" ]]; then
    info "Using existing venv: $INSTALL_DIR/.venv"
    return 0
  fi
  info "Creating venv: $INSTALL_DIR/.venv"
  "$py" -m venv "$INSTALL_DIR/.venv"
}

install_package() {
  local python pkg
  python="$(venv_python)"
  pkg="$(package_dir)"
  info "Installing GoodBoy (editable)..."
  "$python" -m pip install -q --upgrade pip
  "$python" -m pip install -q -e "$pkg"
  if [[ "${GOODBOY_LOCAL:-}" == "1" ]]; then
    info "Installing local model dependencies..."
    "$python" -m pip install -q -e "$pkg[local]"
  fi
}

path_line() {
  local bin_dir
  bin_dir="$(cd "$(venv_bin_dir)" && pwd)"
  printf 'export PATH="%s:$PATH"\n' "$bin_dir"
}

path_already_configured() {
  local bin_dir line
  bin_dir="$(cd "$(venv_bin_dir)" && pwd)"
  line="export PATH=\"$bin_dir:\$PATH\""
  local rc
  for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.profile"; do
    [[ -f "$rc" ]] || continue
    if grep -Fq "$line" "$rc" 2>/dev/null; then
      return 0
    fi
    if grep -Fq "$bin_dir" "$rc" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

configure_path() {
  if [[ "${GOODBOY_NO_PATH:-}" == "1" ]]; then
    return 0
  fi
  if path_already_configured; then
    info "PATH already configured for GoodBoy."
    return 0
  fi

  local line rc target=""
  line="$(path_line)"
  if [[ -n "${ZSH_VERSION:-}" ]] || [[ "${SHELL:-}" == *zsh* ]]; then
    target="$HOME/.zshrc"
  elif [[ -n "${BASH_VERSION:-}" ]] || [[ "${SHELL:-}" == *bash* ]]; then
    target="$HOME/.bashrc"
  fi
  if [[ -z "$target" ]]; then
    target="$HOME/.profile"
  fi

  info "Adding GoodBoy to PATH in $target"
  {
    printf '\n# GoodBoy CLI\n'
    printf '%s' "$line"
    printf '\n'
  } >>"$target"
}

print_next_steps() {
  local bin_dir
  bin_dir="$(cd "$(venv_bin_dir)" && pwd)"
  info ""
  info "GoodBoy is installed."
  info ""
  if [[ "${GOODBOY_NO_PATH:-}" == "1" ]] || ! path_already_configured; then
    info "Open a new terminal, or run:"
    info "  source \"$INSTALL_DIR/.venv/bin/activate\"   # omit on Windows"
    info ""
  else
    info "Open a new terminal (or: source ~/.zshrc), then:"
    info ""
  fi
  info "  goodboy setup    # once: API key and default model"
  info "  cd /path/to/your-repo && goodboy"
  info ""
  if [[ "${GOODBOY_NO_PATH:-}" == "1" ]]; then
    info "To add goodboy to your PATH manually:"
    info "  $(path_line | tr -d '\n')"
    info ""
  fi
  info "Install dir: $INSTALL_DIR"
  info "CLI binary:  $bin_dir/goodboy"
}

main() {
  local py
  py="$(find_python)" || die "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ is required (install python3 and try again)."

  ensure_source_tree
  package_dir >/dev/null

  create_venv "$py"
  install_package
  configure_path
  print_next_steps
}

main "$@"
