#!/usr/bin/env python3
"""Create .venv and install this project in editable mode."""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def create_venv() -> None:
    if VENV_DIR.exists():
        print(f"Using existing venv: {VENV_DIR}")
        return
    print(f"Creating venv: {VENV_DIR}")
    venv.create(VENV_DIR, with_pip=True)


def install_editable() -> None:
    python = venv_python()
    subprocess.run(
        [str(python), "-m", "pip", "install", "-e", "."],
        cwd=ROOT,
        check=True,
    )


def _bin_dir() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts"
    return VENV_DIR / "bin"


def main() -> None:
    create_venv()
    print("Installing package in editable mode...")
    install_editable()
    bin_dir = _bin_dir().resolve()
    print()
    print("Done.")
    print()
    print("To run goodboy from any directory, add this to your shell profile:")
    if sys.platform == "win32":
        print(f'  set PATH={bin_dir};%PATH%')
        print("  (User Environment Variables → Path in Windows Settings)")
    else:
        print(f'  export PATH="{bin_dir}:$PATH"')
        print("  Example (~/.zshrc on macOS):")
        print(f'    echo \'export PATH="{bin_dir}:$PATH"\' >> ~/.zshrc')
        print("    source ~/.zshrc")
    print()
    print("Then in a new terminal:")
    print("  goodboy setup   # once: API key and default model")
    print("  cd /path/to/any-repo && goodboy")
    print()
    print("Or activate only this shell session:")
    if sys.platform == "win32":
        print(rf"  {VENV_DIR}\Scripts\activate")
    else:
        print(f"  source {VENV_DIR}/bin/activate")


if __name__ == "__main__":
    main()
