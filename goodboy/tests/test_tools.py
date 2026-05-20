"""Tests for shell and Python tool executors."""

import sys
import time
from pathlib import Path

from agent.tools import _smart_truncate, _truncate_stream, run_python, run_shell


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

    monkeypatch.setattr(tools_mod, "_MAX_OUTPUT_BYTES", 32)
    result = run_python("print('x' * 500)", cwd=tmp_path, timeout=10.0)
    # Smart truncation keeps head + tail and elides the middle.
    assert "elided" in result.stdout
    assert result.stdout.startswith("x")
    assert result.stdout.rstrip().endswith("x")
    # We never bloat output past the cap by more than the elision marker.
    assert len(result.stdout.encode("utf-8")) < 200


def test_smart_truncate_under_budget_passthrough():
    text = "hello world"
    assert _smart_truncate(text, head_bytes=10, tail_bytes=10) == text


def test_smart_truncate_keeps_head_and_tail():
    text = "A" * 100 + "B" * 100 + "C" * 100
    out = _smart_truncate(text, head_bytes=20, tail_bytes=20)
    assert out.startswith("A" * 20)
    assert out.rstrip().endswith("C" * 20)
    assert "elided" in out
    # Middle B's are gone.
    assert "B" * 50 not in out


def test_smart_truncate_records_elided_byte_count():
    text = "X" * 1000
    out = _smart_truncate(text, head_bytes=10, tail_bytes=10)
    assert "[980 bytes elided]" in out


def test_truncate_stream_default_uses_head_tail_split():
    text = "head_marker_" + ("." * 50_000) + "_tail_marker"
    out = _truncate_stream(text)
    assert "head_marker_" in out
    assert "_tail_marker" in out
    assert "elided" in out


def test_truncate_stream_no_change_when_within_budget():
    text = "small output"
    assert _truncate_stream(text, max_bytes=1024) == text


def test_run_shell_abort_kills_subprocess_quickly(tmp_path: Path):
    """abort_check=True should kill a running shell command in well under the timeout."""
    aborted = {"flag": False}

    def _abort() -> bool:
        return aborted["flag"]

    import threading

    def _trip_abort_soon() -> None:
        time.sleep(0.2)
        aborted["flag"] = True

    threading.Thread(target=_trip_abort_soon, daemon=True).start()
    start = time.monotonic()
    result = run_shell("sleep 30", cwd=tmp_path, timeout=10.0, abort_check=_abort)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"abort_check should preempt long sleep, elapsed={elapsed:.2f}s"
    assert not result.timed_out
    assert "Aborted by user." in result.stderr


def test_run_python_abort_kills_subprocess_quickly(tmp_path: Path):
    aborted = {"flag": False}

    def _abort() -> bool:
        return aborted["flag"]

    import threading

    def _trip_abort_soon() -> None:
        time.sleep(0.2)
        aborted["flag"] = True

    threading.Thread(target=_trip_abort_soon, daemon=True).start()
    start = time.monotonic()
    result = run_python(
        "import time; time.sleep(30)",
        cwd=tmp_path,
        timeout=10.0,
        abort_check=_abort,
    )
    elapsed = time.monotonic() - start

    assert elapsed < 5.0
    assert "Aborted by user." in result.stderr


def test_run_shell_no_abort_callback_still_works(tmp_path: Path):
    result = run_shell("echo legacy", cwd=tmp_path, timeout=5.0, abort_check=None)
    assert result.exit_code == 0
    assert "legacy" in result.stdout
