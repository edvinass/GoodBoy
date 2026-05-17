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
    "Autonomous coding agent for your machine. Describe what you want to do, "
    "and GoodBoy will explore the repo, edit files, run tests and builds, "
    "and report back when it is done. Use @ to reference a file or folder in "
    "your message, and type ? to see help for available commands and session settings."
)


def get_version() -> str:
    try:
        return version("goodboy")
    except PackageNotFoundError:
        text = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        return match.group(1) if match else "unknown"


def format_startup(
    *,
    model: str,
    auto_model_switch: bool = False,
    reasoning_effort: str | None = None,
) -> str:
    ver = get_version()
    routing = (
        "[dim]Routing[/] [green]agent[/] (/autoswitch on)"
        if auto_model_switch
        else "[dim]Routing[/] [yellow]session[/] (/model, /reasoning; /autoswitch off)"
    )
    reasoning_line = ""
    if reasoning_effort:
        reasoning_line = (
            f"\n[dim]Reasoning[/] [{_BRAND_STYLE}]{reasoning_effort}[/]"
        )
    return (
        f"{_brown_dog_art()}\n"
        f"[bold {_BRAND_STYLE}]GoodBoy[/] [dim]v{ver}[/]\n"
        f"[dim]Model[/] [{_BRAND_STYLE}]{model}[/]{reasoning_line}\n"
        f"{routing}\n"
        f"{DESCRIPTION}"
    )
