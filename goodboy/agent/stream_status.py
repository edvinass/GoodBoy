"""Extract user-facing status text from a partial streamed AgentStep JSON."""

from __future__ import annotations

import json
import re

_STATUS_COMPLETE = re.compile(r'"status"\s*:\s*"((?:[^"\\]|\\.)*)"')
_STATUS_PREFIX = re.compile(r'"status"\s*:\s*"((?:[^"\\]|\\.)*)')


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
