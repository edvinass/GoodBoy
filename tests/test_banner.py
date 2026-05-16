"""Tests for startup banner."""

from agent.banner import DESCRIPTION, format_startup, get_version


def test_get_version_matches_pyproject():
    assert get_version() == "0.1.0"


def test_format_startup_includes_version_model_and_description():
    text = format_startup(model="gpt-4o-mini")
    assert "GoodBoy" in text
    assert get_version() in text
    assert "gpt-4o-mini" in text
    assert "Model" in text
    assert DESCRIPTION.split(".")[0] in text
