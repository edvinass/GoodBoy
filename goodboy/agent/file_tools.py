"""Structured file tools: read, patch, and search-replace."""

from __future__ import annotations

import difflib
import subprocess
import tempfile
from pathlib import Path

from agent.tools import _truncate_stream
from agent.types import ToolResult


def _patch_success_stdout(
    path: str, before: str, after: str, *, prefix: str = "Patched"
) -> str:
    """Build stdout for a successful patch including a unified diff when possible."""
    stdout = f"{prefix} {path}\n"
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    if diff.strip():
        stdout += _truncate_stream(diff)
    return stdout


def _patch_strip_level(patch_text: str) -> int:
    """Infer ``patch -pN`` from unified-diff headers (``--- a/...`` → 1)."""
    for line in patch_text.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            name = line.split(maxsplit=1)[-1]
            if name.startswith("a/") or name.startswith("b/"):
                return 1
            break
    return 0


def _resolve_path(workspace: Path, path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        resolved = (workspace / raw).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {path}") from exc
    return resolved


def read_file(
    path: str,
    *,
    workspace: Path,
    start_line: int | None = None,
    end_line: int | None = None,
) -> ToolResult:
    """Read a file (optionally a line range) relative to the workspace."""
    executed = f"read_file {path}"
    try:
        target = _resolve_path(workspace, path)
    except ValueError as exc:
        return ToolResult(executed=executed, stderr=str(exc), exit_code=1)

    if not target.is_file():
        return ToolResult(
            executed=executed,
            stderr=f"Not a file: {path}",
            exit_code=1,
        )

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult(executed=executed, stderr=str(exc), exit_code=1)

    lines = text.splitlines(keepends=True)
    total = len(lines)
    start = 1 if start_line is None else max(1, start_line)
    end = total if end_line is None else min(total, end_line)
    if start > end:
        return ToolResult(
            executed=executed,
            stderr=f"Invalid line range: {start}-{end} (file has {total} lines)",
            exit_code=1,
        )

    slice_lines = lines[start - 1 : end]
    numbered = "".join(
        f"{start + i:6d}|{line}" for i, line in enumerate(slice_lines)
    )
    header = f"{path} (lines {start}-{end} of {total})\n"
    body = _truncate_stream(header + numbered)
    return ToolResult(executed=executed, stdout=body, exit_code=0)


def str_replace(
    path: str,
    old: str,
    new: str,
    *,
    workspace: Path,
) -> ToolResult:
    """Replace the first exact occurrence of ``old`` with ``new`` in a file."""
    executed = f"str_replace {path}"
    try:
        target = _resolve_path(workspace, path)
    except ValueError as exc:
        return ToolResult(executed=executed, stderr=str(exc), exit_code=1)

    if not target.is_file():
        return ToolResult(
            executed=executed,
            stderr=f"Not a file: {path}",
            exit_code=1,
        )

    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        return ToolResult(executed=executed, stderr=str(exc), exit_code=1)

    count = text.count(old)
    if count == 0:
        return ToolResult(
            executed=executed,
            stderr="old_string not found in file",
            exit_code=1,
        )
    if count > 1:
        return ToolResult(
            executed=executed,
            stderr=f"old_string is ambiguous ({count} matches); include more context",
            exit_code=1,
        )

    updated = text.replace(old, new, 1)
    try:
        target.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return ToolResult(executed=executed, stderr=str(exc), exit_code=1)

    diff = "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    stdout = f"Updated {path}\n"
    if diff.strip():
        stdout += _truncate_stream(diff)
    return ToolResult(executed=executed, stdout=stdout, exit_code=0)


def apply_patch(
    path: str,
    patch: str,
    *,
    workspace: Path,
) -> ToolResult:
    """Apply a unified diff to a file (``patch -pN`` from the workspace root)."""
    executed = f"apply_patch {path}"
    try:
        target = _resolve_path(workspace, path)
    except ValueError as exc:
        return ToolResult(executed=executed, stderr=str(exc), exit_code=1)

    if not target.is_file():
        return ToolResult(
            executed=executed,
            stderr=f"Not a file: {path}",
            exit_code=1,
        )

    patch_text = patch if patch.endswith("\n") else patch + "\n"
    try:
        before_text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult(executed=executed, stderr=str(exc), exit_code=1)

    workdir = workspace.resolve()
    patch_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".patch",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(patch_text)
            patch_path = tmp.name

        import shutil

        if shutil.which("patch") is None:
            return _apply_patch_python(
                target, patch_text, executed=executed, path=path, before_text=before_text
            )

        strip = _patch_strip_level(patch_text)
        completed = subprocess.run(
            [
                "patch",
                f"-p{strip}",
                "--forward",
                "-t",  # batch: never prompt (BSD patch otherwise hangs on TTY)
                "-i",
                patch_path,
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return _apply_patch_python(
            target, patch_text, executed=executed, path=path, before_text=before_text
        )
    except OSError as exc:
        return ToolResult(executed=executed, stderr=str(exc), exit_code=1)
    finally:
        if patch_path is not None:
            Path(patch_path).unlink(missing_ok=True)

    stdout = _truncate_stream(completed.stdout or "")
    stderr = _truncate_stream(completed.stderr or "")
    if completed.returncode != 0:
        fallback = _apply_patch_python(
            target, patch_text, executed=executed, path=path, before_text=before_text
        )
        if fallback.exit_code == 0:
            return fallback
        return ToolResult(
            executed=executed,
            stdout=stdout,
            stderr=stderr or "patch failed",
            exit_code=completed.returncode,
        )

    try:
        after_text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        after_text = before_text
    return ToolResult(
        executed=executed,
        stdout=_patch_success_stdout(path, before_text, after_text),
        stderr=stderr,
        exit_code=0,
    )


def _apply_patch_python(
    target: Path,
    patch_text: str,
    *,
    executed: str,
    path: str,
    before_text: str,
) -> ToolResult:
    """Best-effort single-hunk unified diff apply without patch(1)."""
    lines = before_text.splitlines(keepends=True)
    hunk_lines = [
        line
        for line in patch_text.splitlines(keepends=True)
        if line.startswith((" ", "+", "-")) and not line.startswith("+++")
        and not line.startswith("---")
    ]
    if not hunk_lines:
        return ToolResult(executed=executed, stderr="No hunk in patch", exit_code=1)

    old_lines = [line[1:] for line in hunk_lines if line.startswith((" ", "-"))]
    new_lines = [line[1:] for line in hunk_lines if line.startswith((" ", "+"))]
    old_block = "".join(old_lines)
    current = "".join(lines)
    if old_block not in current:
        return ToolResult(
            executed=executed,
            stderr="Patch context not found in file",
            exit_code=1,
        )
    updated = current.replace(old_block, "".join(new_lines), 1)
    target.write_text(updated, encoding="utf-8")
    return ToolResult(
        executed=executed,
        stdout=_patch_success_stdout(
            path,
            before_text,
            updated,
            prefix="Patched (python fallback)",
        ),
        exit_code=0,
    )
