"""Tests for harness slash command completion."""

from agent.repl_commands import (
    AUTOSWITCH_COMMAND_NAMES,
    CLEAR_COMMAND_NAMES,
    COMMANDS_COMMAND_NAMES,
    EXIT_COMMAND_NAMES,
    REPL_COMMANDS,
    active_slash_command_query,
    search_slash_commands,
    slash_command_display_meta,
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
    assert names == ["clear", "model", "reasoning", "commands", "autoswitch", "exit"]

    names = [command.name for command in search_slash_commands("model")]
    assert names == ["model"]


def test_slash_command_display_meta_shows_toggle_state():
    commands = {command.name: command for command in REPL_COMMANDS}
    assert "(off)" in slash_command_display_meta(commands["commands"])
    assert "(on)" in slash_command_display_meta(
        commands["commands"], show_commands=True
    )
    assert "(off)" in slash_command_display_meta(commands["autoswitch"])
    assert "(on)" in slash_command_display_meta(
        commands["autoswitch"], auto_model_switch=True
    )
    assert "(off)" not in slash_command_display_meta(commands["clear"])
    assert "(on)" not in slash_command_display_meta(commands["model"])
    assert "session: not set" in slash_command_display_meta(commands["reasoning"])
    assert "current: low" in slash_command_display_meta(
        commands["reasoning"],
        default_reasoning_effort="low",
        auto_model_switch=True,
    )
    assert "session" in slash_command_display_meta(
        commands["reasoning"],
        default_reasoning_effort="low",
        auto_model_switch=False,
    )
    assert "agent cannot change" in slash_command_display_meta(
        commands["model"], auto_model_switch=False
    )


def test_command_name_sets_match_harness():
    assert CLEAR_COMMAND_NAMES == frozenset({"clear", "new", "reset"})
    assert COMMANDS_COMMAND_NAMES == frozenset(
        {"commands", "cmds", "show-commands"}
    )
    assert AUTOSWITCH_COMMAND_NAMES == frozenset({"autoswitch", "auto"})
    assert EXIT_COMMAND_NAMES == frozenset({"exit", "quit", "q"})
