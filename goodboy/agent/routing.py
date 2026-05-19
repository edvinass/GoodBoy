"""Heuristics for model routing and skipping the routing turn."""

from __future__ import annotations

import re

from agent.task_policy import is_complex_task

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
    if is_complex_task(text):
        return False
    if len(text) <= 120 and _SIMPLE_PATTERNS.match(text):
        return True
    return False
