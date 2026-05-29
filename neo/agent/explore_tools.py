"""Structured exploration tools: search, list, and git."""

from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

from agent.file_tools import _resolve_path
from agent.tools import _truncate_stream
from agent.types import ToolResult

DEFAULT_SEARCH_MAX = 80
DEFAULT_LIST_MAX = 200
DEFAULT_GIT_LOG_N = 20

_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "dist",
        "build",
        ".tox",
        "target",
        ".eggs",
        ".mypy_cache",
        ".ruff_cache",
    }
)

_GIT_OPS = frozenset({"status", "diff", "log"})


def _resolve_search_root(workspace: Path, path: str | None) -> Path:
    rel = (path or ".").strip() or "."
    return _resolve_path(workspace, rel)


def search_code(
    pattern: str,
    *,
    workspace: Path,
    path: str | None = None,
    glob: str | None = None,
    case_insensitive: bool = False,
    max_results: int = DEFAULT_SEARCH_MAX,
) -> ToolResult:
    """Search the workspace with ripgrep (or grep fallback)."""
    limit = max(1, min(max_results, 500))
    try:
        root = _resolve_search_root(workspace, path)
    except ValueError as exc:
        return ToolResult(
            executed=f"search_code {pattern!r}",
            stderr=str(exc),
            exit_code=1,
        )
    if not root.exists():
        return ToolResult(
            executed=f"search_code {pattern!r}",
            stderr=f"Path not found: {path or '.'}",
            exit_code=1,
        )

    rel_root = root.relative_to(workspace.resolve()) if root != workspace.resolve() else Path(".")
    search_path = str(rel_root) if str(rel_root) != "." else "."

    if shutil.which("rg"):
        cmd = [
            "rg",
            "--line-number",
            "--no-heading",
            "--color=never",
            f"--max-count={limit}",
        ]
        if case_insensitive:
            cmd.append("-i")
        if glob:
            cmd.extend(["-g", glob])
        cmd.extend([pattern, search_path])
        executed = " ".join(cmd)
        completed = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode == 0:
            lines = stdout.strip().splitlines()
            if len(lines) > limit:
                lines = lines[:limit]
                stdout = "\n".join(lines) + f"\n... [truncated to {limit} matches]"
            body = _truncate_stream("\n".join(lines) if lines else "(no matches)")
            return ToolResult(executed=executed, stdout=body, exit_code=0)
        if completed.returncode == 1 and not stdout.strip():
            return ToolResult(
                executed=executed,
                stdout="(no matches)",
                stderr=_truncate_stream(stderr) if stderr else "",
                exit_code=0,
            )
        return ToolResult(
            executed=executed,
            stdout=_truncate_stream(stdout),
            stderr=_truncate_stream(stderr),
            exit_code=completed.returncode,
        )

    return _search_code_grep(
        pattern,
        workspace=workspace,
        root=root,
        case_insensitive=case_insensitive,
        glob=glob,
        max_results=limit,
    )


def _search_code_grep(
    pattern: str,
    *,
    workspace: Path,
    root: Path,
    case_insensitive: bool,
    glob: str | None,
    max_results: int,
) -> ToolResult:
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return ToolResult(
            executed=f"search_code {pattern!r}",
            stderr=f"Invalid pattern: {exc}",
            exit_code=1,
        )

    matches: list[str] = []
    for file_path in sorted(root.rglob("*") if root.is_dir() else [root]):
        if not file_path.is_file():
            continue
        if _should_skip_path(file_path, workspace):
            continue
        if glob and not fnmatch.fnmatch(file_path.name, glob.lstrip("**/")):
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = file_path.relative_to(workspace.resolve())
        for line_no, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{rel}:{line_no}:{line.rstrip()}")
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    executed = f"search_code {pattern!r} (python fallback)"
    if not matches:
        return ToolResult(executed=executed, stdout="(no matches)", exit_code=0)
    body = "\n".join(matches)
    if len(matches) >= max_results:
        body += f"\n... [truncated to {max_results} matches]"
    return ToolResult(executed=executed, stdout=_truncate_stream(body), exit_code=0)


def _should_skip_path(path: Path, workspace: Path) -> bool:
    try:
        rel = path.relative_to(workspace.resolve())
    except ValueError:
        return True
    return any(part in _SKIP_DIR_NAMES for part in rel.parts)


def list_files(
    *,
    workspace: Path,
    path: str | None = None,
    glob: str | None = None,
    max_depth: int | None = None,
    max_results: int = DEFAULT_LIST_MAX,
) -> ToolResult:
    """List files under a workspace path with optional glob and depth limits."""
    limit = max(1, min(max_results, 1000))
    try:
        root = _resolve_search_root(workspace, path)
    except ValueError as exc:
        return ToolResult(executed=f"list_files {path or '.'}", stderr=str(exc), exit_code=1)
    executed = f"list_files {path or '.'}"
    if glob:
        executed += f" glob={glob!r}"
    if max_depth is not None:
        executed += f" max_depth={max_depth}"

    if not root.exists():
        return ToolResult(executed=executed, stderr=f"Path not found: {path or '.'}", exit_code=1)
    if not root.is_dir():
        rel = root.relative_to(workspace.resolve())
        return ToolResult(
            executed=executed,
            stdout=str(rel),
            exit_code=0,
        )

    base_depth = len(root.resolve().parts)
    found: list[str] = []
    glob_pattern = glob

    if glob_pattern and ("**" in glob_pattern or "/" in glob_pattern):
        iterator = root.glob(glob_pattern) if "**" not in glob_pattern else root.rglob(
            glob_pattern.replace("**/", "")
        )
        for item in sorted(iterator):
            if not item.is_file() or _should_skip_path(item, workspace):
                continue
            if max_depth is not None:
                depth = len(item.resolve().parts) - base_depth
                if depth > max_depth:
                    continue
            found.append(str(item.relative_to(workspace.resolve())))
            if len(found) >= limit:
                break
    else:
        for dirpath, dirnames, filenames in _walk_filtered(root, workspace):
            current = Path(dirpath)
            depth = len(current.resolve().parts) - base_depth
            if max_depth is not None and depth > max_depth:
                dirnames.clear()
                continue
            for name in sorted(filenames):
                if glob_pattern and not fnmatch.fnmatch(name, glob_pattern):
                    continue
                rel = Path(dirpath).joinpath(name).relative_to(workspace.resolve())
                found.append(str(rel))
                if len(found) >= limit:
                    break
            if len(found) >= limit:
                break

    if not found:
        return ToolResult(executed=executed, stdout="(no files)", exit_code=0)
    body = "\n".join(found)
    if len(found) >= limit:
        body += f"\n... [truncated to {limit} paths]"
    return ToolResult(executed=executed, stdout=_truncate_stream(body), exit_code=0)


def _walk_filtered(root: Path, workspace: Path):
    """os.walk-style traversal that prunes noisy directories."""
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        if _should_skip_path(current, workspace) and current != root.resolve():
            dirnames.clear()
            continue
        dirnames[:] = sorted(
            d for d in dirnames if d not in _SKIP_DIR_NAMES and not d.startswith(".")
        )
        yield dirpath, dirnames, filenames


def run_git(
    git_op: str,
    *,
    workspace: Path,
    path: str | None = None,
    staged: bool = False,
    max_results: int | None = None,
) -> ToolResult:
    """Run a bounded git read-only command in the workspace."""
    op = git_op.strip().lower()
    if op not in _GIT_OPS:
        return ToolResult(
            executed=f"git {git_op}",
            stderr=f"Unknown git_op {git_op!r}; use status, diff, or log",
            exit_code=1,
        )

    if shutil.which("git") is None:
        return ToolResult(executed=f"git {op}", stderr="git not found on PATH", exit_code=1)

    if not (workspace / ".git").exists() and not (workspace / ".git").is_file():
        try:
            probe = subprocess.run(
                ["git", "-C", str(workspace), "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe.returncode != 0:
                return ToolResult(
                    executed=f"git {op}",
                    stderr="Not a git repository",
                    exit_code=1,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(executed=f"git {op}", stderr=str(exc), exit_code=1)

    cmd = ["git", "-C", str(workspace)]
    if op == "status":
        cmd.extend(["status", "--short", "--branch"])
    elif op == "diff":
        cmd.append("diff")
        if staged:
            cmd.append("--staged")
        if path:
            try:
                target = _resolve_path(workspace, path)
                cmd.append(str(target.relative_to(workspace.resolve())))
            except ValueError as exc:
                return ToolResult(executed=f"git diff {path}", stderr=str(exc), exit_code=1)
    else:
        n = max(1, min(max_results or DEFAULT_GIT_LOG_N, 100))
        cmd.extend(["log", f"-{n}", "--oneline", "--decorate"])

    executed = " ".join(cmd)
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ToolResult(executed=executed, stderr=str(exc), exit_code=1)

    stdout = _truncate_stream(completed.stdout or "")
    stderr = _truncate_stream(completed.stderr or "")
    if completed.returncode != 0:
        return ToolResult(
            executed=executed,
            stdout=stdout,
            stderr=stderr or "git command failed",
            exit_code=completed.returncode,
        )
    if not stdout.strip() and op == "diff":
        stdout = "(no diff)"
    return ToolResult(executed=executed, stdout=stdout, stderr=stderr, exit_code=0)
