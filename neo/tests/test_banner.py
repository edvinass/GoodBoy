"""Tests for startup banner."""

import re

from agent.banner import DESCRIPTION, format_startup, get_version
from settings import PACKAGE_DIR


def test_get_version_matches_pyproject():
    text = (PACKAGE_DIR / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    expected = match.group(1) if match else "unknown"
    assert get_version() == expected


def test_format_startup_includes_version_model_and_description():
    text = format_startup(model="gpt-5.4-nano")
    assert "GoodBoy" in text
    assert get_version() in text
    assert "gpt-5.4-nano" in text
    assert "Model" in text
    assert "Routing" not in text
    assert "session" not in text.lower()
    assert DESCRIPTION == "Autonomous coding agent"
    assert DESCRIPTION in text


def test_format_startup_renders_reasoning_effort_when_provided():
    text = format_startup(model="gpt-5.4-nano", reasoning_effort="low")
    assert "Routing" not in text
    assert "/autoswitch" not in text
    assert "/model" not in text
    assert "/reasoning" not in text
    assert "low" in text
