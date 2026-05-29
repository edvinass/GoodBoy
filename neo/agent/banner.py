"""Startup banner: ASCII art, version, and tool description."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version

from settings import PACKAGE_DIR

_BRAND_STYLE = "rgb(0,255,65)"

MATRIX_ASCII = r"""
     .---.     .---.
    /     \   /     \
   |   o   | |   o   |
    \_____/   \_____/
"""


def _matrix_art() -> str:
    return "\n".join(
        f"[{_BRAND_STYLE}]{line}[/]" for line in MATRIX_ASCII.rstrip().splitlines()
    )


DESCRIPTION = "Autonomous coding agent"


def get_version() -> str:
    try:
        return version("neo")
    except PackageNotFoundError:
        text = (PACKAGE_DIR / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        return match.group(1) if match else "unknown"


def format_startup(
    *,
    model: str,
    reasoning_effort: str | None = None,
) -> str:
    ver = get_version()
    reasoning_line = ""
    if reasoning_effort:
        reasoning_line = (
            f"\n[dim]Reasoning[/] [{_BRAND_STYLE}]{reasoning_effort}[/]"
        )
    return (
        f"{_matrix_art()}\n"
        f"[bold {_BRAND_STYLE}]Neo[/] [dim]v{ver}[/]\n"
        f"[dim]Model[/] [{_BRAND_STYLE}]{model}[/]{reasoning_line}\n"
        f"{DESCRIPTION}"
    )
