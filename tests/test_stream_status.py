"""Tests for streaming status extraction from partial JSON."""

from agent.stream_status import extract_streaming_status


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
