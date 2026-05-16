"""Tests for conversation UI formatting."""

import json

from rich.console import Console

from agent.types import AgentAction, AgentStep, ToolResult
from agent.ui import ConversationUI, _wrap_long_lines


def test_print_user_format(capsys):
    ui = ConversationUI()
    ui.print_user("hello\nworld")
    out = capsys.readouterr().out
    assert "You" in out
    assert "hello" in out
    assert "world" in out


def test_print_agent_with_subtitle(capsys):
    ui = ConversationUI()
    ui.print_agent("ls -la", subtitle="shell")
    out = capsys.readouterr().out
    assert "GoodBoy" in out
    assert "shell" in out
    assert "ls -la" in out


def test_print_agent_step_shows_model_with_flag(capsys):
    ui = ConversationUI(show_model=True)
    step = AgentStep(
        action=AgentAction.RUN_SHELL,
        command="ls",
    )
    ui.print_agent_step(step, model="gpt-4o-mini", reasoning="low")
    out = capsys.readouterr().out
    assert "model" in out
    assert "gpt-4o-mini" in out
    assert "reasoning" in out
    assert "low" in out


def test_print_agent_step_hides_model_without_flag(capsys):
    ui = ConversationUI(show_model=False)
    step = AgentStep(action=AgentAction.RUN_SHELL, command="ls")
    ui.print_agent_step(step, model="gpt-4o-mini")
    out = capsys.readouterr().out
    assert "gpt-4o-mini" not in out


def test_print_agent_step_need_user_input(capsys):
    ui = ConversationUI()
    step = AgentStep(
        action=AgentAction.NEED_USER_INPUT,
        thought="Need detail",
        message="Which file?",
    )
    ui.print_agent_step(step)
    out = capsys.readouterr().out
    assert "Need detail" in out
    assert "Which file?" in out


def test_print_tool_result_hidden_without_debug(capsys):
    ui = ConversationUI(debug=False)
    tool = ToolResult(executed="echo hi", stdout="hi\n", exit_code=0)
    ui.print_tool_result(tool)
    assert capsys.readouterr().out == ""


def test_print_tool_result_shown_in_debug(capsys):
    ui = ConversationUI(debug=True)
    tool = ToolResult(executed="echo hi", stdout="hi\n", exit_code=0)
    ui.print_tool_result(tool)
    out = capsys.readouterr().out
    assert "hi" in out
    assert "output" in out


def test_print_llm_request_with_debug_input(capsys):
    ui = ConversationUI(debug_input=True)
    ui.print_llm_request(
        turn=1,
        model="gpt-4o-mini",
        reasoning_effort=None,
        instructions="You are GoodBoy.",
        input_text="## User task\nfix bug",
    )
    out = capsys.readouterr().out
    assert "input" in out
    assert "turn" in out
    assert "instructions" in out
    assert "You are GoodBoy." in out
    assert "fix bug" in out
    assert "gpt-4o-mini" in out


def test_print_llm_request_hidden_without_flag(capsys):
    ui = ConversationUI(debug_input=False)
    ui.print_llm_request(
        turn=1,
        model="gpt-4o-mini",
        reasoning_effort=None,
        instructions="secret",
        input_text="task",
    )
    assert capsys.readouterr().out == ""


def test_print_llm_response_with_debug_output(capsys):
    ui = ConversationUI(debug_output=True)
    raw = '{"action":"task_complete","message":"done"}'
    ui.print_llm_response(turn=2, raw=raw)
    out = capsys.readouterr().out
    assert "output" in out
    assert "turn 2" in out
    assert "task_complete" in out
    assert "done" in out


def test_wrap_long_lines_splits_without_ellipsis():
    line = "a" * 100
    wrapped = _wrap_long_lines(line, width=40)
    assert "..." not in wrapped
    assert wrapped.replace("\n", "") == line


def test_print_llm_response_shows_full_long_json(capsys):
    long_command = "curl -s 'https://example.com/" + "a" * 80 + "'"
    raw = json.dumps({"action": "run_shell", "command": long_command})
    ui = ConversationUI(debug_output=True, console=Console(width=60))
    ui.print_llm_response(turn=1, raw=raw)
    out = capsys.readouterr().out
    assert "run_shell" in out
    assert out.count("a") >= 80
    assert "..." not in out


def test_print_agent_shell_command_wraps(capsys):
    long_command = "echo " + "x" * 120
    ui = ConversationUI(console=Console(width=60))
    ui.print_agent(long_command, subtitle="shell")
    out = capsys.readouterr().out
    assert out.count("x") >= 120
    assert "..." not in out


def test_print_llm_response_hidden_without_flag(capsys):
    ui = ConversationUI(debug_output=False)
    ui.print_llm_response(turn=1, raw='{"action":"failed"}')
    assert capsys.readouterr().out == ""


def test_directory_listing_renders_tree(capsys):
    ui = ConversationUI()
    listing = """__pycache__
cli

./cli:
app.py
"""
    ui.print_agent(f"Current directory structure:\n\n{listing}")
    out = capsys.readouterr().out
    assert "cli" in out
    assert "app.py" in out
