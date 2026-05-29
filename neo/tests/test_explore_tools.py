"""Tests for search_code, list_files, and git harness tools."""

import subprocess
from pathlib import Path

import pytest

from agent.explore_tools import list_files, run_git, search_code
from agent.file_tools import delete_file, move_file


def _init_git_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def sample_tree(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text("VALUE = 42\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    return tmp_path


def test_search_code_finds_match(sample_tree: Path):
    result = search_code("hello", workspace=sample_tree, path="src")
    assert result.exit_code == 0
    assert "main.py" in result.stdout
    assert "hello" in result.stdout


def test_search_code_respects_glob(sample_tree: Path):
    result = search_code("VALUE", workspace=sample_tree, glob="*.md")
    assert result.exit_code == 0
    assert result.stdout.strip() == "(no matches)" or "README" not in result.stdout


def test_search_code_rejects_escape(tmp_path: Path):
    result = search_code("x", workspace=tmp_path, path="/etc/passwd")
    assert result.exit_code == 1
    assert "escapes workspace" in result.stderr


def test_list_files_with_glob(sample_tree: Path):
    result = list_files(workspace=sample_tree, path="src", glob="*.py")
    assert result.exit_code == 0
    assert "src/main.py" in result.stdout
    assert "src/util.py" in result.stdout


def test_list_files_max_depth(sample_tree: Path):
    (sample_tree / "deep").mkdir()
    (sample_tree / "deep" / "nested.txt").write_text("x\n", encoding="utf-8")
    result = list_files(workspace=sample_tree, max_depth=0)
    assert result.exit_code == 0
    assert "nested.txt" not in result.stdout


def test_git_status_and_log(sample_tree: Path):
    if not __import__("shutil").which("git"):
        pytest.skip("git not installed")
    _init_git_repo(sample_tree)
    (sample_tree / "tracked.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=sample_tree, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=sample_tree,
        check=True,
        capture_output=True,
    )

    status = run_git("status", workspace=sample_tree)
    assert status.exit_code == 0

    log = run_git("log", workspace=sample_tree, max_results=5)
    assert log.exit_code == 0
    assert "init" in log.stdout


def test_delete_and_move_file(tmp_path: Path):
    src = tmp_path / "old.txt"
    src.write_text("data\n", encoding="utf-8")
    moved = tmp_path / "new.txt"

    move_result = move_file("old.txt", "new.txt", workspace=tmp_path)
    assert move_result.exit_code == 0
    assert not src.exists()
    assert moved.read_text(encoding="utf-8") == "data\n"

    delete_result = delete_file("new.txt", workspace=tmp_path)
    assert delete_result.exit_code == 0
    assert not moved.exists()


def test_move_file_rejects_existing_dest(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    result = move_file("a.txt", "b.txt", workspace=tmp_path)
    assert result.exit_code == 1
    assert "already exists" in result.stderr
