"""Startup banner: ASCII art, version, and tool description."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version

from settings import ROOT_DIR

DOG_ASCII = r"""
      / \__
     (    @\___
      /         O
     /   (_____/
    /_____/   U
"""

DESCRIPTION = (
    "A local autonomous agent for your machine. Describe a task in plain "
    "English — GoodBoy plans steps, runs shell commands and Python, and "
    "reports back when done. Type exit or quit to leave."
)


def get_version() -> str:
    try:
        return version("goodboy")
    except PackageNotFoundError:
        text = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        return match.group(1) if match else "unknown"


def format_startup() -> str:
    ver = get_version()
    return (
        f"{DOG_ASCII.rstrip()}\n"
        f"[bold cyan]GoodBoy[/] [dim]v{ver}[/]\n"
        f"{DESCRIPTION}"
    )
