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
    assert "Routing" not in text
    assert "session" not in text.lower()
    assert DESCRIPTION == "Autonomous coding agent"
    assert DESCRIPTION in text


def test_format_startup_omits_routing_when_autoswitch_on():
    text = format_startup(
        model="gpt-5.4-nano",
        auto_model_switch=True,
        reasoning_effort="low",
    )
    assert "Routing" not in text
    assert "/autoswitch" not in text
    assert "/model" not in text
    assert "/reasoning" not in text
    assert "low" in text
