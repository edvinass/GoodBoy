"""Tests for structured file tools."""

import shutil
from pathlib import Path

from agent.file_tools import apply_patch, read_file, str_replace


def test_read_file_numbered(tmp_path: Path):
    target = tmp_path / "sample.py"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")
    result = read_file("sample.py", workspace=tmp_path, start_line=2, end_line=2)
    assert result.exit_code == 0
    assert "line2" in result.stdout
    assert "     2|" in result.stdout


def test_str_replace_unique_match(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("hello world\n", encoding="utf-8")
    result = str_replace("a.txt", "world", "there", workspace=tmp_path)
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == "hello there\n"


def test_str_replace_ambiguous_fails(tmp_path: Path):
    target = tmp_path / "b.txt"
    target.write_text("foo\nfoo\n", encoding="utf-8")
    result = str_replace("b.txt", "foo", "bar", workspace=tmp_path)
    assert result.exit_code == 1
    assert "ambiguous" in result.stderr


def test_apply_patch_unified_diff(tmp_path: Path):
    target = tmp_path / "c.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    patch = """--- a/c.txt
+++ b/c.txt
@@ -1,2 +1,2 @@
 alpha
-beta
+gamma
"""
    result = apply_patch("c.txt", patch, workspace=tmp_path)
    if result.exit_code != 0:
        # patch(1) may be unavailable in some CI images; skip gracefully.
        import shutil

        if shutil.which("patch") is None:
            return
    assert result.exit_code == 0
    assert "gamma" in target.read_text(encoding="utf-8")
    assert "--- a/c.txt" in result.stdout
    assert "+gamma" in result.stdout


def test_apply_patch_when_patch_unavailable(tmp_path: Path, monkeypatch):
    """Cover the Python fallback when patch(1) is not installed."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None if cmd == "patch" else shutil.which(cmd))
    target = tmp_path / "d.txt"
    target.write_text("hello\n", encoding="utf-8")
    patch = """--- a/d.txt
+++ b/d.txt
@@ -1 +1 @@
-hello
+world
"""
    result = apply_patch("d.txt", patch, workspace=tmp_path)
    # Fallback succeeds, exit_code should be 0 with Python fallback message
    assert result.exit_code == 0
    assert "Patched (python fallback)" in result.stdout
    assert target.read_text(encoding="utf-8") == "world\n"
