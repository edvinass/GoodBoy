"""Extract user-facing status text from a partial streamed AgentStep JSON."""

from __future__ import annotations

import json
import re

_STATUS_COMPLETE = re.compile(r'"status"\s*:\s*"((?:[^"\\]|\\.)*)"')
_STATUS_PREFIX = re.compile(r'"status"\s*:\s*"((?:[^"\\]|\\.)*)')

# Generic field matcher used to surface live previews of any string-valued
# field. ``close`` is captured separately so we can tell completed values from
# the one the model is currently writing.
_FIELD_RE = re.compile(
    r'"(?P<key>\w+)"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)(?P<close>")?'
)

# String fields that make a useful one-line preview of "what the model is
# doing right now". ``action`` is intentionally excluded since it is an enum
# slug ("run_shell") rather than user-facing text.
_PREVIEW_KEYS = frozenset(
    {
        "status",
        "thought",
        "message",
        "command",
        "code",
        "path",
        "dest_path",
        "pattern",
        "glob",
        "git_op",
        "old_string",
        "new_string",
        "patch",
    }
)

# Avoid flashing a single character before any meaningful text has arrived.
_MIN_PREVIEW_CHARS = 3


def extract_streaming_status(partial_json: str) -> str | None:
    """Return decoded ``status`` when present in partial model output.

    Prefer a complete JSON string value; fall back to a sufficiently long
    in-progress value so the loading UI can update early while tokens stream.
    """
    if not partial_json:
        return None

    match = _STATUS_COMPLETE.search(partial_json)
    if match:
        try:
            return json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            return match.group(1)

    incomplete = _STATUS_PREFIX.search(partial_json)
    if incomplete is None:
        return None
    fragment = incomplete.group(1)
    if len(fragment) < 8:
        return None
    try:
        return json.loads(f'"{fragment}"')
    except json.JSONDecodeError:
        return fragment.replace('\\"', '"').replace("\\n", "\n")


def _decode_json_string(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")


def extract_streaming_preview(
    partial_json: str, *, max_chars: int = 100
) -> str | None:
    """Return a one-line preview of the model's most recent string field.

    Unlike :func:`extract_streaming_status`, this scans the whole partial
    JSON for any user-relevant string field (status, thought, message,
    command, code, ...) and returns the most recent one. That lets the
    loading UI show what the model is actually producing the moment any
    field starts streaming, even before the structured ``status`` is
    available.
    """
    if not partial_json:
        return None

    matches = list(_FIELD_RE.finditer(partial_json))
    if not matches:
        return None

    for match in reversed(matches):
        key = match.group("key")
        if key not in _PREVIEW_KEYS:
            continue
        raw_value = match.group("value")
        if not raw_value:
            continue
        decoded = _decode_json_string(raw_value)
        # Collapse to a single line so the spinner stays one row tall.
        first_line = decoded.split("\n", 1)[0].strip()
        if len(first_line) < _MIN_PREVIEW_CHARS:
            continue
        if len(first_line) > max_chars:
            first_line = first_line[: max_chars - 1].rstrip() + "…"
        return first_line

    return None
