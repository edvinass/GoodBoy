"""Tests for workspace resolution."""

from pathlib import Path

from agent.workspace import resolve_workspace

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_resolve_workspace_uses_git_root():
    nested = _REPO_ROOT / "neo"
    assert nested.is_dir()
    assert resolve_workspace(nested) == _REPO_ROOT.resolve()


def test_resolve_workspace_without_git_returns_cwd(tmp_path: Path):
    nested = tmp_path / "only" / "here"
    nested.mkdir(parents=True)
    assert resolve_workspace(nested) == nested.resolve()
