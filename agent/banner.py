"""Startup banner: ASCII art, version, and tool description."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version

from settings import ROOT_DIR

_BRAND_STYLE = "rgb(139,69,19)"

DOG_ASCII = r"""
      / \__
     (    @\___
      /         O
     /   (_____/
    /_____/   U
"""


def _brown_dog_art() -> str:
    return "\n".join(f"[{_BRAND_STYLE}]{line}[/]" for line in DOG_ASCII.rstrip().splitlines())

DESCRIPTION = (
    "Autonomous coding agent for your machine. Describe a task — GoodBoy "
    "explores the repo, edits files, runs tests and builds, and reports back "
    "when done. Type @ to reference a file or folder in your message; type / "
    "for commands (/clear to reset, /model to change the LLM, /exit to quit)."
)


def get_version() -> str:
    try:
        return version("goodboy")
    except PackageNotFoundError:
        text = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        return match.group(1) if match else "unknown"


def format_startup(*, model: str) -> str:
    ver = get_version()
    return (
        f"{_brown_dog_art()}\n"
        f"[bold {_BRAND_STYLE}]GoodBoy[/] [dim]v{ver}[/]\n"
        f"[dim]Model[/] [{_BRAND_STYLE}]{model}[/]\n"
        f"{DESCRIPTION}"
    )
