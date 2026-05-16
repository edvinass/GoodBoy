"""Tests for shell and Python tool executors."""

import sys
import time
from pathlib import Path

from agent.tools import run_python, run_shell


def test_run_shell_echo(tmp_path: Path):
    result = run_shell("echo hello", cwd=tmp_path, timeout=10.0)
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert not result.timed_out


def test_run_python_print(tmp_path: Path):
    result = run_python("print('hi')", cwd=tmp_path, timeout=10.0)
    assert result.exit_code == 0
    assert "hi" in result.stdout
    assert sys.executable in result.executed or "-c" in result.executed


def test_run_shell_timeout(tmp_path: Path):
    result = run_shell("sleep 5", cwd=tmp_path, timeout=0.2)
    assert result.timed_out
    assert "timed out" in result.stderr.lower()


def test_run_shell_non_utf8_stderr(tmp_path: Path):
    # Latin-1 byte 0xe1 is invalid as a standalone UTF-8 continuation byte.
    result = run_python(
        "import sys; sys.stderr.buffer.write(b'\\xe1\\n')",
        cwd=tmp_path,
        timeout=10.0,
    )
    assert result.exit_code == 0
    assert "\ufffd" in result.stderr


def test_truncation(tmp_path: Path, monkeypatch):
    import agent.tools as tools_mod

    monkeypatch.setattr(tools_mod, "_MAX_OUTPUT_BYTES", 8)
    result = run_python("print('x' * 100)", cwd=tmp_path, timeout=10.0)
    assert "[truncated]" in result.stdout
