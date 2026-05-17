"""Heuristics for model routing and skipping the routing turn."""

from __future__ import annotations

import re

_COMPLEX_KEYWORDS = re.compile(
    r"\b("
    r"refactor|architecture|migrate|security|audit|debug|investigate|"
    r"performance|optimi[sz]e|redesign|multi[- ]file|entire|whole|"
    r"codebase|review|analy[sz]e|root cause|stuck|broken build"
    r")\b",
    re.IGNORECASE,
)

_SIMPLE_PATTERNS = re.compile(
    r"^(run|execute|fix|add|update|remove|delete|rename|format|lint|test|"
    r"pytest|npm|cargo|go test|git status|show|list|print|echo)\b",
    re.IGNORECASE,
)


def should_skip_routing_turn(task: str) -> bool:
    """Return True when turn-1 model routing is unlikely to help.

    Simple, short tasks can start on the session default model instead of
    spending a turn on the cheapest router model.
    """
    text = task.strip()
    if not text:
        return False
    if _COMPLEX_KEYWORDS.search(text):
        return False
    if len(text) <= 120 and _SIMPLE_PATTERNS.match(text):
        return True
    return False
