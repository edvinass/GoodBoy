"""Project memory files injected into the agent context."""

from __future__ import annotations

from pathlib import Path

_MEMORY_FILENAMES = (
    "AGENTS.md",
    ".goodboy/memory.md",
    "GOODBOY.md",
)
_MAX_MEMORY_CHARS = 8_000


def load_project_memory(workspace: Path) -> str | None:
    """Load the first present project memory file, truncated for context size."""
    root = workspace.resolve()
    for name in _MEMORY_FILENAMES:
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        if len(text) > _MAX_MEMORY_CHARS:
            text = text[:_MAX_MEMORY_CHARS] + "\n... [memory truncated]"
        return f"From `{name}`:\n\n{text}"
    return None
