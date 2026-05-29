"""Resolve the working directory for tool execution."""

from __future__ import annotations

import subprocess
from pathlib import Path


def resolve_workspace(start: Path | None = None) -> Path:
    """Use the git repository root when cwd is inside a repo; else resolved cwd."""
    base = (start or Path.cwd()).resolve()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=base,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return base
    if completed.returncode == 0:
        top = completed.stdout.strip()
        if top:
            return Path(top)
    return base
