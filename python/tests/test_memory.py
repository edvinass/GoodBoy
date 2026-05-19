"""Tests for project memory loading."""

from pathlib import Path

from agent.context import SessionContext
from agent.memory import load_project_memory


def test_load_agents_md(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("Run `pytest` before task_complete.", encoding="utf-8")
    text = load_project_memory(tmp_path)
    assert text is not None
    assert "pytest" in text


def test_memory_in_to_prompt(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("Use ruff for linting.", encoding="utf-8")
    ctx = SessionContext(user_task="lint", workspace=str(tmp_path))
    prompt = ctx.to_prompt()
    assert "Project memory" in prompt
    assert "ruff" in prompt
