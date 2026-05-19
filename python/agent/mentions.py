"""@ file and directory mentions for user input."""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

_MENTION_PATTERN = re.compile(r"@([^\s@]+)")
_PATH_IGNORE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        "dist",
        "build",
    }
)
_MAX_SEARCH_RESULTS = 15


def active_mention_query(text_before_cursor: str) -> tuple[str, int] | None:
    """Return (query after @, start_position) when the cursor is in a mention."""
    match = re.search(r"@([^\s@]*)$", text_before_cursor)
    if match is None:
        return None
    return match.group(1), -len(match.group(1))


def search_workspace_paths(
    workspace: Path,
    query: str,
    *,
    limit: int = _MAX_SEARCH_RESULTS,
) -> list[str]:
    """Return workspace-relative paths matching query (fuzzy, case-insensitive)."""
    workspace = workspace.resolve()
    query_fold = query.casefold()
    scored: list[tuple[tuple[int, int, str], str]] = []

    for path in _list_workspace_paths(workspace):
        path_fold = path.casefold()
        if query_fold and not _fuzzy_match(path_fold, query_fold):
            continue
        scored.append((_rank_path(path_fold, query_fold), path))

    scored.sort(key=lambda item: item[0])
    return [path for _, path in scored[:limit]]


def expand_file_mentions(text: str, workspace: Path) -> str:
    """Replace @mentions with resolved workspace-relative paths in the prompt."""

    def replace_mention(match: re.Match[str]) -> str:
        resolved = resolve_mention_path(match.group(1), workspace)
        if resolved is None:
            return match.group(0)
        path = _display_path(resolved, workspace)
        if resolved.is_dir() and not path.endswith("/"):
            return f"{path}/"
        return path

    return _MENTION_PATTERN.sub(replace_mention, text)


def resolve_mention_path(ref: str, workspace: Path) -> Path | None:
    """Resolve a mention token to an existing path."""
    workspace = workspace.resolve()
    candidate = Path(ref).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (workspace / candidate).resolve()

    try:
        resolved.relative_to(workspace)
    except ValueError:
        if not candidate.is_absolute():
            return None

    if resolved.exists():
        return resolved
    return None


def _fuzzy_match(path_fold: str, query_fold: str) -> bool:
    if not query_fold:
        return True
    if query_fold in path_fold:
        return True
    query_index = 0
    for char in path_fold:
        if char == query_fold[query_index]:
            query_index += 1
            if query_index == len(query_fold):
                return True
    return False


def _rank_path(path_fold: str, query_fold: str) -> tuple[int, int, str]:
    if not query_fold:
        return (2, len(path_fold), path_fold)
    index = path_fold.find(query_fold)
    if index >= 0:
        basename = path_fold.rsplit("/", 1)[-1]
        basename_index = basename.find(query_fold)
        if basename_index == 0:
            tier = 0
        elif basename_index > 0:
            tier = 1
        else:
            tier = 2
        return (tier, index, path_fold)

    basename = path_fold.rsplit("/", 1)[-1]
    if _fuzzy_match(basename, query_fold):
        return (3, len(path_fold), path_fold)
    if _fuzzy_match(path_fold, query_fold):
        return (4, len(path_fold), path_fold)
    return (9, len(path_fold), path_fold)


def _list_workspace_paths(workspace: Path) -> list[str]:
    return list(_cached_workspace_paths(str(workspace), _workspace_cache_key(workspace)))


def _workspace_cache_key(workspace: Path) -> float:
    try:
        return workspace.stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=4)
def _cached_workspace_paths(workspace_str: str, _cache_key: float) -> tuple[str, ...]:
    workspace = Path(workspace_str)
    git_paths = _git_tracked_paths(workspace)
    if git_paths is not None:
        return tuple(git_paths)
    return tuple(_walk_workspace_paths(workspace))


def _git_tracked_paths(workspace: Path) -> list[str] | None:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(workspace),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None

    files = [part.decode("utf-8", errors="surrogateescape") for part in completed.stdout.split(b"\0") if part]
    paths: set[str] = set(files)
    for file_path in files:
        parent = Path(file_path).parent
        parts = parent.parts
        for index in range(1, len(parts) + 1):
            directory = str(Path(*parts[:index]))
            if directory != ".":
                paths.add(f"{directory}/")
    return sorted(paths, key=str.casefold)


def _walk_workspace_paths(workspace: Path) -> list[str]:
    paths: set[str] = set()
    for root, dirnames, filenames in workspace.walk(top_down=True):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in _PATH_IGNORE_DIRS and name != ".git"
        ]
        rel_root = root.relative_to(workspace)
        if rel_root.parts:
            paths.add(f"{rel_root.as_posix()}/")
        for name in filenames:
            rel = rel_root / name
            paths.add(rel.as_posix())
    return sorted(paths, key=str.casefold)


def _display_path(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
