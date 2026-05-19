"""Tests for routing heuristics."""

from agent.routing import should_skip_routing_turn
from agent.task_policy import is_complex_task


def test_skip_short_simple_task():
    assert should_skip_routing_turn("run pytest -q") is True


def test_no_skip_complex_task():
    assert should_skip_routing_turn("refactor the auth module architecture") is False


def test_is_complex_task_exported():
    assert is_complex_task("refactor main") is True
