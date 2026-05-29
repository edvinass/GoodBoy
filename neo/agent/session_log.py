"""Append-only session logs for debugging harness runs."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agent.types import AgentStep, ToolResult
from settings import get_settings


def default_log_dir() -> Path:
    return Path.home() / ".neo" / "logs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def instructions_digest(text: str) -> str:
    """Stable short id for a system-prompt body (session-log dedup)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return str(value)


class SessionLog:
    """JSONL log for one CLI session (may include multiple user tasks)."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._logged_instructions_id: str | None = None

    def event(self, kind: str, **fields: Any) -> None:
        record = {"ts": _utc_now(), "event": kind}
        for key, value in fields.items():
            record[key] = _to_jsonable(value)
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def log_llm_request(
        self,
        *,
        turn: int,
        model: str,
        reasoning_effort: str | None,
        instructions: str,
        input_text: str,
    ) -> None:
        digest = instructions_digest(instructions)
        fields: dict[str, Any] = {
            "turn": turn,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "instructions_id": digest,
            "input": input_text,
        }
        if digest != self._logged_instructions_id:
            self._logged_instructions_id = digest
            fields["instructions"] = instructions
        self.event("llm_request", **fields)

    def log_llm_response(self, *, turn: int, raw: str) -> None:
        self.event("llm_response", turn=turn, raw=raw)

    def log_agent_step(self, *, turn: int, step: AgentStep) -> None:
        self.event("agent_step", turn=turn, step=step)

    def log_tool_result(self, *, turn: int, result: ToolResult) -> None:
        self.event("tool_result", turn=turn, result=result)

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.close()


@contextmanager
def open_session_log(
    *,
    log_dir: Path | None = None,
    workspace: Path | None = None,
) -> Iterator[SessionLog | None]:
    """Create a session log file; yield None when logging is disabled."""
    cfg = get_settings()
    if not cfg.session_log_enabled:
        yield None
        return

    base = log_dir or cfg.session_log_dir or default_log_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = base / f"session-{stamp}.jsonl"
    log = SessionLog(path)
    log.event(
        "session_start",
        log_file=str(log.path),
        workspace=str(workspace) if workspace else None,
        cwd=str(Path.cwd()),
    )
    try:
        yield log
    finally:
        log.event("session_end")
        log.close()
