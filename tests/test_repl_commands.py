"""Tests for harness slash command completion."""

from agent.repl_commands import (
    CLEAR_COMMAND_NAMES,
    EXIT_COMMAND_NAMES,
    active_slash_command_query,
    search_slash_commands,
)


def test_active_slash_command_query_at_line_start():
    assert active_slash_command_query("/cl") == ("cl", -2)
    assert active_slash_command_query("  /cl") is None


def test_active_slash_command_query_ignores_after_other_text():
    assert active_slash_command_query("fix /cl") is None


def test_search_slash_commands_filters_by_prefix():
    names = [command.name for command in search_slash_commands("cl")]
    assert names == ["clear"]

    names = [command.name for command in search_slash_commands("")]
    assert names == ["clear", "exit"]


def test_command_name_sets_match_harness():
    assert CLEAR_COMMAND_NAMES == frozenset({"clear", "new", "reset"})
    assert EXIT_COMMAND_NAMES == frozenset({"exit", "quit", "q"})
