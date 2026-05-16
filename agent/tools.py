"""Shell and Python tool executors for the agent harness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agent.types import ToolResult

# Cap per-stream output sent back to the model (bytes before decode).
_MAX_OUTPUT_BYTES = 48 * 1024
_TRUNCATION_SUFFIX = "\n... [truncated]"


def _decode_stream(data: bytes | None) -> str:
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")


def _truncate_stream(text: str, *, max_bytes: int | None = None) -> str:
    limit = max_bytes if max_bytes is not None else _MAX_OUTPUT_BYTES
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text
    truncated = encoded[:limit].decode("utf-8", errors="ignore")
    return truncated + _TRUNCATION_SUFFIX


def run_shell(
    command: str,
    *,
    cwd: Path | str | None = None,
    timeout: float = 120.0,
) -> ToolResult:
    """Run a shell command with full user privileges (local dev harness)."""
    workdir = str(cwd) if cwd is not None else None
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            capture_output=True,
            timeout=timeout,
        )
        return ToolResult(
            executed=command,
            stdout=_truncate_stream(_decode_stream(completed.stdout)),
            stderr=_truncate_stream(_decode_stream(completed.stderr)),
            exit_code=completed.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            _truncate_stream(_decode_stream(exc.stdout))
            if exc.stdout is not None
            else ""
        )
        stderr = (
            _truncate_stream(_decode_stream(exc.stderr))
            if exc.stderr is not None
            else ""
        )
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        stderr += f"Command timed out after {timeout}s."
        return ToolResult(
            executed=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=None,
            timed_out=True,
        )
    except OSError as exc:
        return ToolResult(
            executed=command,
            stdout="",
            stderr=str(exc),
            exit_code=None,
            timed_out=False,
        )


def run_python(
    code: str,
    *,
    cwd: Path | str | None = None,
    timeout: float = 120.0,
) -> ToolResult:
    """Run Python code in a subprocess (same interpreter as the harness)."""
    workdir = str(cwd) if cwd is not None else None
    executed = f"{sys.executable} -c <code>"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=workdir,
            capture_output=True,
            timeout=timeout,
        )
        return ToolResult(
            executed=executed,
            stdout=_truncate_stream(_decode_stream(completed.stdout)),
            stderr=_truncate_stream(_decode_stream(completed.stderr)),
            exit_code=completed.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            _truncate_stream(_decode_stream(exc.stdout))
            if exc.stdout is not None
            else ""
        )
        stderr = (
            _truncate_stream(_decode_stream(exc.stderr))
            if exc.stderr is not None
            else ""
        )
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        stderr += f"Python execution timed out after {timeout}s."
        return ToolResult(
            executed=executed,
            stdout=stdout,
            stderr=stderr,
            exit_code=None,
            timed_out=True,
        )
    except OSError as exc:
        return ToolResult(
            executed=executed,
            stdout="",
            stderr=str(exc),
            exit_code=None,
            timed_out=False,
        )
