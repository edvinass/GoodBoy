"""Tests for startup banner."""

from agent.banner import DESCRIPTION, format_startup, get_version


def test_get_version_matches_pyproject():
    assert get_version() == "0.1.0"


def test_format_startup_includes_version_model_and_description():
    text = format_startup(model="gpt-5.4-nano")
    assert "GoodBoy" in text
    assert get_version() in text
    assert "gpt-5.4-nano" in text
    assert "Model" in text
    assert "session" in text.lower()
    assert DESCRIPTION.split(".")[0] in text


def test_format_startup_shows_agent_routing_when_autoswitch_on():
    text = format_startup(
        model="gpt-5.4-nano",
        auto_model_switch=True,
        reasoning_effort="low",
    )
    assert "agent" in text.lower()
    assert "low" in text
