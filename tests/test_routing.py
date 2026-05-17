"""Tests for routing heuristics."""

from agent.routing import should_skip_routing_turn


def test_skip_short_simple_task():
    assert should_skip_routing_turn("run pytest -q") is True


def test_no_skip_complex_task():
    assert should_skip_routing_turn("refactor the auth module architecture") is False
