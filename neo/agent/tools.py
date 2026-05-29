"""Shell and Python tool executors for the agent harness."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from agent.types import ToolResult
from settings import DEFAULT_TOOL_TIMEOUT_SEC

AbortCheck = Callable[[], bool]
_ABORT_POLL_INTERVAL = 0.1
_ABORT_MESSAGE = "Aborted by user."

# Cap per-stream output sent back to the model (bytes before decode).
# Lowered from 48 KB so the *first* time output enters context it's already
# small; the head+tail split below preserves both exploratory output (signal at
# the top, e.g. ls/find/rg) and failure tails (signal at the bottom, e.g.
# pytest, build errors).
_MAX_OUTPUT_BYTES = 12 * 1024
_TRUNCATION_SUFFIX = "\n... [truncated]"


def _decode_stream(data: bytes | None) -> str:
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")


def _smart_truncate(
    text: str,
    *,
    head_bytes: int,
    tail_bytes: int,
) -> str:
    """Keep the first head_bytes and last tail_bytes, eliding the middle.

    When the encoded text fits within head_bytes + tail_bytes (plus the elision
    marker overhead) the original text is returned unchanged. Otherwise the
    output is `<head>\\n... [N bytes elided] ...\\n<tail>` where N is the byte
    count removed. Decoding uses errors="ignore" at the boundaries so we never
    emit a half-codepoint.
    """
    if head_bytes < 0 or tail_bytes < 0:
        raise ValueError("head_bytes and tail_bytes must be non-negative")
    encoded = text.encode("utf-8", errors="replace")
    budget = head_bytes + tail_bytes
    if len(encoded) <= budget or budget == 0:
        if budget == 0:
            return text
        return text
    elided = len(encoded) - budget
    head = encoded[:head_bytes].decode("utf-8", errors="ignore") if head_bytes else ""
    tail = encoded[-tail_bytes:].decode("utf-8", errors="ignore") if tail_bytes else ""
    marker = f"\n... [{elided} bytes elided] ...\n"
    return f"{head}{marker}{tail}"


def _truncate_stream(text: str, *, max_bytes: int | None = None) -> str:
    """Cap a tool stream using head+tail truncation.

    Splits the budget evenly between head and tail so both exploratory output
    (signal early) and failure output (signal late) survive truncation.
    """
    limit = max_bytes if max_bytes is not None else _MAX_OUTPUT_BYTES
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return _smart_truncate(text, head_bytes=head, tail_bytes=tail)


class _UserAborted(Exception):
    """Internal sentinel: subprocess was terminated because the user pressed Escape."""

    def __init__(self, stdout: bytes, stderr: bytes) -> None:
        self.stdout = stdout
        self.stderr = stderr


def _terminate_process(proc: subprocess.Popen) -> None:
    """Best-effort kill of a child process group started with ``start_new_session``."""
    try:
        if os.name == "posix" and proc.pid:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                proc.terminate()
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix" and proc.pid:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    proc.kill()
            else:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass


def _communicate_with_abort(
    proc: subprocess.Popen,
    *,
    timeout: float,
    abort_check: AbortCheck | None,
) -> tuple[bytes, bytes]:
    """Wait for ``proc`` to finish; honour ``abort_check`` and a hard ``timeout``.

    Raises :class:`subprocess.TimeoutExpired` on hard timeout and
    :class:`_UserAborted` when ``abort_check`` flips. In the abort case the
    child process is killed and any output already buffered is returned in
    the exception.
    """
    if abort_check is None:
        return proc.communicate(timeout=timeout)
    start = time.monotonic()
    while True:
        slice_timeout = min(_ABORT_POLL_INTERVAL, max(0.0, timeout - (time.monotonic() - start)))
        try:
            return proc.communicate(timeout=slice_timeout)
        except subprocess.TimeoutExpired:
            if abort_check():
                _terminate_process(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=1.0)
                except Exception:
                    stdout, stderr = b"", b""
                raise _UserAborted(stdout or b"", stderr or b"")
            if time.monotonic() - start >= timeout:
                raise subprocess.TimeoutExpired(proc.args, timeout)


def _spawn(command_or_args, *, shell: bool, cwd: str | None) -> subprocess.Popen:
    """Popen wrapper that puts the child in its own session on POSIX.

    A fresh session lets us kill the whole process group on abort, so things
    like ``bash -c "sleep 60"`` and pipelines die cleanly instead of leaking
    background children.
    """
    popen_kwargs: dict = {
        "shell": shell,
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    return subprocess.Popen(command_or_args, **popen_kwargs)


def run_shell(
    command: str,
    *,
    cwd: Path | str | None = None,
    timeout: float = DEFAULT_TOOL_TIMEOUT_SEC,
    abort_check: AbortCheck | None = None,
) -> ToolResult:
    """Run a shell command with full user privileges (local dev harness)."""
    workdir = str(cwd) if cwd is not None else None
    try:
        proc = _spawn(command, shell=True, cwd=workdir)
    except OSError as exc:
        return ToolResult(
            executed=command,
            stdout="",
            stderr=str(exc),
            exit_code=None,
            timed_out=False,
        )
    try:
        stdout_b, stderr_b = _communicate_with_abort(
            proc, timeout=timeout, abort_check=abort_check
        )
        return ToolResult(
            executed=command,
            stdout=_truncate_stream(_decode_stream(stdout_b)),
            stderr=_truncate_stream(_decode_stream(stderr_b)),
            exit_code=proc.returncode,
            timed_out=False,
        )
    except _UserAborted as aborted:
        stdout = _truncate_stream(_decode_stream(aborted.stdout))
        stderr = _truncate_stream(_decode_stream(aborted.stderr))
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        stderr += _ABORT_MESSAGE
        return ToolResult(
            executed=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=None,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        _terminate_process(proc)
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


def run_python(
    code: str,
    *,
    cwd: Path | str | None = None,
    timeout: float = DEFAULT_TOOL_TIMEOUT_SEC,
    abort_check: AbortCheck | None = None,
) -> ToolResult:
    """Run Python code in a subprocess (same interpreter as the harness)."""
    workdir = str(cwd) if cwd is not None else None
    executed = f"{sys.executable} -c <code>"
    try:
        proc = _spawn([sys.executable, "-c", code], shell=False, cwd=workdir)
    except OSError as exc:
        return ToolResult(
            executed=executed,
            stdout="",
            stderr=str(exc),
            exit_code=None,
            timed_out=False,
        )
    try:
        stdout_b, stderr_b = _communicate_with_abort(
            proc, timeout=timeout, abort_check=abort_check
        )
        return ToolResult(
            executed=executed,
            stdout=_truncate_stream(_decode_stream(stdout_b)),
            stderr=_truncate_stream(_decode_stream(stderr_b)),
            exit_code=proc.returncode,
            timed_out=False,
        )
    except _UserAborted as aborted:
        stdout = _truncate_stream(_decode_stream(aborted.stdout))
        stderr = _truncate_stream(_decode_stream(aborted.stderr))
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        stderr += _ABORT_MESSAGE
        return ToolResult(
            executed=executed,
            stdout=stdout,
            stderr=stderr,
            exit_code=None,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        _terminate_process(proc)
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
