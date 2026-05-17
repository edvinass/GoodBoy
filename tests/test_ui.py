"""Tests for conversation UI formatting."""

import json
import os

import click
import pytest
from rich.console import Console

from agent.types import AgentAction, AgentStep, ToolResult
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document

import agent.ui as ui_module
from agent.ui import (
    ConversationUI,
    _AutoWidthConsole,
    _PasteState,
    _THEME,
    _accept_active_completion,
    _configure_prompt_toolkit,
    _wrap_long_lines,
    format_pasted_text_label,
)


def test_format_pasted_text_label():
    assert format_pasted_text_label(1, 53) == "[Pasted text #1 +52 lines]"


def test_accept_active_completion_applies_highlighted_choice():
    buffer = Buffer()
    buffer.text = "/cl"
    buffer.cursor_position = len(buffer.text)
    buffer.complete_state = CompletionState(
        original_document=buffer.document,
        completions=[Completion("clear", start_position=-2)],
    )
    buffer.go_to_completion(0)

    assert _accept_active_completion(buffer) is True
    assert buffer.text == "/clear"
    assert buffer.complete_state is None


def test_accept_active_completion_uses_first_when_none_highlighted():
    buffer = Buffer()
    buffer.text = "/"
    buffer.cursor_position = len(buffer.text)
    buffer.complete_state = CompletionState(
        original_document=buffer.document,
        completions=[
            Completion("clear", start_position=0),
            Completion("exit", start_position=0),
        ],
        complete_index=None,
    )

    assert _accept_active_completion(buffer) is True
    assert buffer.text == "/clear"
    assert buffer.complete_state is None


def test_paste_state_register_multiline():
    state = _PasteState()
    label = state.register_paste("a\nb\nc")
    assert label == "[Pasted text #1 +2 lines]"
    assert state.stored == "a\nb\nc"
    assert state.resolve("[Pasted text #1 +2 lines]") == "a\nb\nc"


def test_paste_state_single_line_returns_none():
    state = _PasteState()
    assert state.register_paste("hello") is None
    assert state.stored is None


def test_configure_prompt_toolkit_disables_cpr_in_vscode(monkeypatch):
    monkeypatch.delenv("PROMPT_TOOLKIT_NO_CPR", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    monkeypatch.setenv("TERM", "xterm-256color")
    ui_module._PROMPT_TOOLKIT_CONFIGURED = False
    _configure_prompt_toolkit()
    assert os.environ.get("PROMPT_TOOLKIT_NO_CPR") == "1"


def test_configure_prompt_toolkit_respects_existing_flag(monkeypatch):
    monkeypatch.setenv("PROMPT_TOOLKIT_NO_CPR", "1")
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    ui_module._PROMPT_TOOLKIT_CONFIGURED = False
    _configure_prompt_toolkit()
    assert os.environ.get("PROMPT_TOOLKIT_NO_CPR") == "1"


def test_prompt_user_multiline(monkeypatch):
    ui = ConversationUI()

    def _fake(paste_state: _PasteState, **_: object) -> str:
        return paste_state.register_paste("line one\nline two") or ""

    monkeypatch.setattr("agent.ui._prompt_user_line", _fake)
    assert ui.prompt_user() == "line one\nline two"


def test_prompt_user_shows_paste_label(capsys, monkeypatch):
    ui = ConversationUI()
    monkeypatch.setattr(
        "agent.ui._prompt_user_line",
        lambda paste_state, **_: (
            paste_state.register_paste("a\nb\n") or ""
        ),
    )
    ui.prompt_user()
    out = capsys.readouterr().out
    assert "[Pasted text #1 +1 lines]" in out


def test_prompt_user_abort_on_cancel(monkeypatch):
    ui = ConversationUI()
    monkeypatch.setattr("agent.ui._prompt_user_line", lambda paste_state, **_: None)

    with pytest.raises(click.Abort):
        ui.prompt_user()


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
    ui = ConversationUI(show_model=True, console=Console(width=72, theme=_THEME))
    step = AgentStep(
        action=AgentAction.RUN_SHELL,
        command="ls",
    )
    ui.print_agent_step(step, model="gpt-5.4-nano", reasoning="low")
    out = capsys.readouterr().out
    assert "model" in out
    assert "gpt-5.4-nano" in out
    assert "reasoning" in out
    assert "low" in out


def test_routing_table_width_matches_console(capsys):
    ui = ConversationUI(
        show_model=True,
        console=Console(width=72, height=25, theme=_THEME),
    )
    step = AgentStep(action=AgentAction.RUN_SHELL, command="ls")
    ui.print_agent_step(step, model="gpt-5.4-nano", reasoning="low")
    out = capsys.readouterr().out
    table_borders = [line for line in out.splitlines() if line.startswith("╭") or line.startswith("╰")]
    assert table_borders
    assert all(len(line) == ui._console.width for line in table_borders)


def test_auto_width_console_ignores_stale_columns_env(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")

    class _FakeTTY:
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return 1

    console = _AutoWidthConsole(file=_FakeTTY(), theme=_THEME)
    monkeypatch.setattr(
        os,
        "get_terminal_size",
        lambda _fd: os.terminal_size((80, 24)),
    )
    assert console.width == 80


def test_print_agent_step_hides_model_without_flag(capsys):
    ui = ConversationUI(show_model=False)
    step = AgentStep(action=AgentAction.RUN_SHELL, command="ls")
    ui.print_agent_step(step, model="gpt-5.4-nano")
    out = capsys.readouterr().out
    assert "gpt-5.4-nano" not in out


def test_print_agent_step_hides_shell_without_debug(capsys):
    ui = ConversationUI(debug=False)
    step = AgentStep(action=AgentAction.RUN_SHELL, command="curl -s example.com")
    ui.print_agent_step(step)
    out = capsys.readouterr().out
    assert "curl" not in out
    assert "shell" not in out


def test_print_agent_step_shows_shell_with_debug(capsys):
    ui = ConversationUI(debug=True)
    step = AgentStep(action=AgentAction.RUN_SHELL, command="curl -s example.com")
    ui.print_agent_step(step)
    out = capsys.readouterr().out
    assert "curl -s example.com" in out
    assert "shell" in out


def test_print_agent_step_shows_shell_with_show_commands(capsys):
    ui = ConversationUI(show_commands=True)
    step = AgentStep(action=AgentAction.RUN_SHELL, command="curl -s example.com")
    ui.print_agent_step(step)
    out = capsys.readouterr().out
    assert "curl -s example.com" in out
    assert "shell" in out


def test_print_agent_step_shows_full_python_with_show_commands(capsys):
    code = "import json\nimport urllib.request\nprint(json.load(urllib.request.urlopen('https://example.com')))"
    ui = ConversationUI(show_commands=True)
    step = AgentStep(action=AgentAction.RUN_PYTHON, code=code)
    ui.print_agent_step(step)
    out = capsys.readouterr().out
    assert "import json" in out
    assert "urllib.request" in out
    assert "print(json.load" in out
    assert "..." not in out
    assert "python" in out


def test_print_agent_python_code_wraps(capsys):
    long_line = "x = " + repr("a" * 120)
    ui = ConversationUI(show_commands=True, console=Console(width=60))
    ui.print_agent(long_line, subtitle="python")
    out = capsys.readouterr().out
    assert out.count("a") >= 120
    assert "..." not in out


def test_print_tool_result_hidden_with_show_commands_only(capsys):
    ui = ConversationUI(show_commands=True)
    tool = ToolResult(executed="echo hi", stdout="hi\n", exit_code=0)
    ui.print_tool_result(tool)
    assert capsys.readouterr().out == ""


def test_print_tool_result_shown_with_debug(capsys):
    ui = ConversationUI(debug=True)
    tool = ToolResult(executed="echo hi", stdout="hi\n", exit_code=0)
    ui.print_tool_result(tool)
    out = capsys.readouterr().out
    assert "hi" in out
    assert "stdout" in out


def test_default_hides_task_complete(capsys):
    ui = ConversationUI()
    ui.print_task_complete()
    assert capsys.readouterr().out == ""


def test_default_hides_failed_banner_shows_message(capsys):
    ui = ConversationUI()
    ui.print_failed("something broke")
    out = capsys.readouterr().out
    assert "Failed" not in out
    assert "something broke" in out
    assert "GoodBoy" in out


def test_default_print_agent_step_surfaces_switch_tools_notice(capsys):
    ui = ConversationUI()
    ui.print_agent_step(
        AgentStep(
            action=AgentAction.SWITCH_TOOLS,
            thought="Need search",
            tools=["web_search"],
        ),
        hosted_tools=["web_search"],
    )
    captured = capsys.readouterr()
    assert "web_search" in captured.err
    assert "Hosted tools enabled" in captured.err
    assert "Need search" not in captured.out + captured.err

    ui.print_agent_step(
        AgentStep(
            action=AgentAction.TASK_COMPLETE,
            thought="Done",
            message="London: cloudy",
        ),
        hosted_tools=["web_search"],
    )
    captured = capsys.readouterr()
    assert "London: cloudy" in captured.out
    assert "thought" not in captured.out
    assert "Enable hosted" not in captured.out


def test_default_print_agent_step_surfaces_switch_model_notice(capsys):
    ui = ConversationUI()
    ui.print_agent_step(
        AgentStep(
            action=AgentAction.SWITCH_MODEL,
            model="gpt-5.5",
        ),
    )
    captured = capsys.readouterr()
    assert "gpt-5.5" in captured.err
    assert "Model set to" in captured.err


def test_follow_print_agent_step_shows_thought(capsys):
    ui = ConversationUI(show_thoughts=True)
    ui.print_agent_step(
        AgentStep(
            action=AgentAction.RUN_SHELL,
            thought="Inspect the working tree",
            command="git status",
        ),
    )
    out = capsys.readouterr().out
    assert "Inspect the working tree" in out
    assert "thought" in out.lower()


def test_verbose_print_agent_step_shows_tool_switch(capsys):
    ui = ConversationUI(verbose=True)
    ui.print_agent_step(
        AgentStep(
            action=AgentAction.SWITCH_TOOLS,
            tools=["web_search"],
        ),
        hosted_tools=["web_search"],
    )
    out = capsys.readouterr().out
    assert "web_search" in out
    assert "Enable hosted" in out


def test_print_agent_step_need_user_input(capsys):
    ui = ConversationUI()
    step = AgentStep(
        action=AgentAction.NEED_USER_INPUT,
        thought="Need detail",
        message="Which file?",
    )
    ui.print_agent_step(step)
    out = capsys.readouterr().out
    assert "Need detail" not in out
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
        model="gpt-5.4-nano",
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
    assert "gpt-5.4-nano" in out


def test_print_llm_request_hidden_without_flag(capsys):
    ui = ConversationUI(debug_input=False)
    ui.print_llm_request(
        turn=1,
        model="gpt-5.4-nano",
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


def test_model_stream_writes_deltas_to_stderr(capsys):
    ui = ConversationUI(stream_output=True)
    ui.begin_model_stream(turn=1)
    ui.write_model_stream_delta('{"action":')
    ui.write_model_stream_delta('"task_complete"}')
    ui.end_model_stream()
    captured = capsys.readouterr()
    assert "turn 1" in captured.err
    assert '{"action":"task_complete"}' in captured.err


def test_model_stream_hidden_when_disabled(capsys):
    ui = ConversationUI(stream_output=False)
    ui.begin_model_stream(turn=1)
    ui.write_model_stream_delta("hidden")
    ui.end_model_stream()
    assert capsys.readouterr().err == ""


def test_print_llm_response_skipped_when_already_streamed(capsys):
    ui = ConversationUI(debug_output=True, stream_output=True)
    ui.print_llm_response(
        turn=1,
        raw='{"action":"task_complete"}',
        streamed=True,
    )
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


def test_ui_stop_request_consumed_once():
    ui = ConversationUI()
    assert ui.consume_stop_requested() is False
    ui.request_stop_after_current_step()
    assert ui.consume_stop_requested() is True
    assert ui.consume_stop_requested() is False


def test_clear_session_resets_history_and_startup():
    ui = ConversationUI()
    ui.print_agent("hello")
    ui.request_stop_after_current_step()
    ui.clear_session()
    assert ui.consume_stop_requested() is False
    assert len(ui._history) == 1
    assert ui._history[0][0] == "startup"
