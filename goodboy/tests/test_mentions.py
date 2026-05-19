"""Tests for @ file mention search and expansion."""

from pathlib import Path

from agent.mentions import (
    active_mention_query,
    expand_file_mentions,
    resolve_mention_path,
    search_workspace_paths,
)


def test_active_mention_query_detects_partial_path():
    assert active_mention_query("please read @agent/ui") == ("agent/ui", -len("agent/ui"))


def test_active_mention_query_ignores_completed_token():
    assert active_mention_query("see @agent/ui.py ") is None


def test_search_workspace_paths_matches_substring(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "ui.py").write_text("print('hi')\n", encoding="utf-8")
    (agent_dir / "registry.py").write_text("", encoding="utf-8")

    results = search_workspace_paths(tmp_path, "gi")
    assert ".gitignore" in results
    assert "agent/ui.py" in results


def test_expand_file_mentions_replaces_with_path(tmp_path: Path):
    target = tmp_path / "agent" / "ui.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n", encoding="utf-8")

    expanded = expand_file_mentions("fix @agent/ui.py please", tmp_path)
    assert expanded == "fix agent/ui.py please"


def test_expand_file_mentions_replaces_directory(tmp_path: Path):
    target = tmp_path / "agent"
    target.mkdir()
    (target / "ui.py").write_text("", encoding="utf-8")

    expanded = expand_file_mentions("look at @agent/", tmp_path)
    assert expanded == "look at agent/"


def test_expand_file_mentions_leaves_unknown_tokens(tmp_path: Path):
    assert expand_file_mentions("see @missing.py", tmp_path) == "see @missing.py"


def test_resolve_mention_path_supports_relative_and_absolute(tmp_path: Path):
    rel = tmp_path / "README.md"
    rel.write_text("hi", encoding="utf-8")

    assert resolve_mention_path("README.md", tmp_path) == rel.resolve()
    assert resolve_mention_path(str(rel.resolve()), tmp_path) == rel.resolve()
