"""Tests for streaming status extraction from partial JSON."""

from agent.stream_status import (
    extract_streaming_preview,
    extract_streaming_status,
)


def test_extract_complete_status():
    partial = '{"status": "Inspecting project structure", "action": "run_shell"'
    assert extract_streaming_status(partial) == "Inspecting project structure"


def test_extract_status_with_escapes():
    partial = '{"status": "Reading \\"main.py\\"", "action": "read_file"'
    assert extract_streaming_status(partial) == 'Reading "main.py"'


def test_extract_incomplete_status_when_long_enough():
    partial = '{"status": "Searching for relevan'
    assert extract_streaming_status(partial) == "Searching for relevan"


def test_extract_missing_status():
    assert extract_streaming_status('{"action": "run_shell"') is None


def test_extract_short_incomplete_status_ignored():
    assert extract_streaming_status('{"status": "Ru') is None


def test_preview_prefers_status_when_available():
    partial = '{"action": "run_shell", "status": "Inspecting project structure"'
    assert (
        extract_streaming_preview(partial) == "Inspecting project structure"
    )


def test_preview_returns_latest_field_being_written():
    partial = (
        '{"action": "run_shell", '
        '"status": "Inspecting", '
        '"thought": "Need to check the auth module before edit'
    )
    assert (
        extract_streaming_preview(partial)
        == "Need to check the auth module before edit"
    )


def test_preview_skips_action_enum():
    assert extract_streaming_preview('{"action": "run_shell"') is None


def test_preview_ignores_very_short_incomplete_value():
    assert extract_streaming_preview('{"action": "run_shell", "status": "I') is None


def test_preview_handles_command_field():
    partial = '{"action": "run_shell", "command": "pytest -q'
    assert extract_streaming_preview(partial) == "pytest -q"


def test_preview_collapses_code_to_first_line():
    partial = (
        '{"action": "run_python", '
        '"code": "import os\\nprint(os.getcwd())\\nfor i in range(5):'
    )
    assert extract_streaming_preview(partial) == "import os"


def test_preview_truncates_long_thought():
    long_thought = "A" * 250
    partial = f'{{"action": "run_shell", "thought": "{long_thought}"'
    preview = extract_streaming_preview(partial)
    assert preview is not None
    assert preview.endswith("…")
    assert len(preview) <= 100


def test_preview_decodes_escapes():
    partial = '{"action": "read_file", "status": "Reading \\"main.py\\""'
    assert extract_streaming_preview(partial) == 'Reading "main.py"'


def test_preview_returns_none_on_empty():
    assert extract_streaming_preview("") is None
    assert extract_streaming_preview("{") is None


def test_preview_skips_empty_value():
    partial = '{"action": "run_shell", "status": "", "thought": "Looking now'
    assert extract_streaming_preview(partial) == "Looking now"
