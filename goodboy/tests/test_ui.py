"""Tests for conversation UI formatting."""

import json
import os
import sys

import click
import pytest
from rich.console import Console
from rich.text import Text

from agent.types import AgentAction, AgentStep, PlanItem, PlanItemStatus, ToolResult
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document

import agent.ui as ui_module
from agent.ui import (
    ConversationUI,
    _AutoWidthConsole,
    _PasteState,
    _THEME,
    _USER_INPUT_FOOTER,
    _USER_INPUT_MAX_LINES,
    _USER_INPUT_PLACEHOLDER,
    _accept_active_completion,
    _append_wrapped_content,
    _clear_user_input,
    _configure_prompt_toolkit,
    _diff_display_line_width,
    _extract_unified_diff,
    _format_diff_hunk_label,
    _format_diff_line_range,
    _format_unified_diff_for_display,
    _is_cmd_d_data,
    _is_shift_enter_data,
    _input_window_line_count,
    _truncate_diff_lines,
    _user_input_placeholder,
    _wrap_long_lines,
    activity_label,
    format_pasted_text_label,
)


def test_user_input_placeholder_and_footer_are_separate():
    assert _USER_INPUT_PLACEHOLDER == "Ask anything"
    assert _USER_INPUT_FOOTER == "@ files · / commands · ? help · Cmd+D clear"
    assert _user_input_placeholder() == [("class:placeholder", "Ask anything")]


def test_input_window_line_count_grows_with_buffer_lines():
    buffer = Buffer(multiline=False)
    assert _input_window_line_count(buffer) == 1
    buffer.text = "line one\nline two"
    assert _input_window_line_count(buffer) == 2
    buffer.text = "\n".join(f"line {i}" for i in range(_USER_INPUT_MAX_LINES + 5))
    assert _input_window_line_count(buffer) == _USER_INPUT_MAX_LINES


def test_input_frame_fragments_use_styled_classes():
    from agent.ui import (
        _input_top_fragments,
        _input_bottom_fragments,
        _input_footer_fragments,
    )

    top = _input_top_fragments()
    bottom = _input_bottom_fragments()
    assert top[0][0] == "class:input-border"
    assert top[0][1].startswith("╭") and top[0][1].endswith("╮")
    assert bottom[0][0] == "class:input-border"
    assert bottom[0][1].startswith("╰") and bottom[0][1].endswith("╯")
    footer = _input_footer_fragments()
    assert footer[0][0] == "class:input-footer"
    assert "@ files · / commands · ? help · Cmd+D clear" in footer[0][1]


def test_format_pasted_text_label():
    assert format_pasted_text_label(1, 53) == "[Pasted text #1 · +52 lines]"


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


def test_is_shift_enter_data_detects_xterm_shift_enter():
    assert _is_shift_enter_data("\x1b[27;2;13~") is True
    assert _is_shift_enter_data("\x1b[13;2u") is True
    assert _is_shift_enter_data("\x1b[13u") is False
    assert _is_shift_enter_data("\r") is False
    assert _is_shift_enter_data("\x0d") is False


def test_is_cmd_d_data_detects_cmd_d_encodings():
    assert _is_cmd_d_data("\x04") is True
    assert _is_cmd_d_data("\x1b[100;9u") is True
    assert _is_cmd_d_data("\x1b[27;9;100~") is True
    assert _is_cmd_d_data("\x1b[100u") is False


def test_clear_user_input_resets_buffer():
    buffer = Buffer()
    buffer.text = "hello"
    _clear_user_input(buffer)
    assert buffer.text == ""


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
    assert label == "[Pasted text #1 · +2 lines]"
    assert state.stored == "a\nb\nc"
    assert state.resolve("[Pasted text #1 · +2 lines]") == "a\nb\nc"


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
    assert "[Pasted text #1 · +1 lines]" in out


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


def test_print_session_log_path_only_in_debug(capsys):
    path = "/tmp/session-test.jsonl"
    ConversationUI().print_session_log_path(path)
    assert "Session log" not in capsys.readouterr().err

    ConversationUI(debug=True).print_session_log_path(path)
    err = capsys.readouterr().err
    assert "Session log" in err
    assert path in err


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


def test_activity_label_shell_hides_command():
    step = AgentStep(action=AgentAction.RUN_SHELL, command="curl -s secret.example")
    assert activity_label(step, phase="progress") == "running shell command"
    assert activity_label(step, phase="done") == "ran shell command"
    assert "curl" not in activity_label(step, phase="done")


def test_activity_label_read_file_uses_basename():
    step = AgentStep(action=AgentAction.READ_FILE, path="src/foo.py")
    assert activity_label(step, phase="progress") == "reading foo.py"
    assert activity_label(step, phase="done") == "read foo.py"


def test_print_harness_activity_shell_success_not_in_history(capsys):
    ui = ConversationUI()
    step = AgentStep(action=AgentAction.RUN_SHELL, command="curl -s example.com")
    result = ToolResult(executed="run_shell", exit_code=0)
    ui.print_harness_activity(step, result)
    out = capsys.readouterr().out
    assert "ran terminal command" not in out
    assert "curl" not in out


def test_print_harness_activity_shell_failure_still_shown(capsys):
    ui = ConversationUI()
    step = AgentStep(action=AgentAction.RUN_SHELL, command="false")
    result = ToolResult(executed="run_shell", exit_code=1, stderr="command failed")
    ui.print_harness_activity(step, result)
    out = capsys.readouterr().out
    assert "ran shell command" in out
    assert "command failed" in out


def test_consecutive_activity_lines_no_extra_blank(capsys):
    ui = ConversationUI()
    step = AgentStep(action=AgentAction.READ_FILE, path="a.py")
    ui.print_harness_activity(step, ToolResult(executed="read_file a.py", exit_code=0))
    step = AgentStep(
        action=AgentAction.STR_REPLACE,
        path="b.py",
        old_string="x",
        new_string="y",
    )
    ui.print_harness_activity(
        step,
        ToolResult(executed="str_replace b.py", stdout="Updated b.py\n", exit_code=0),
    )
    out = capsys.readouterr().out
    assert "◦ read a.py\n◦ wrote b.py" in out
    assert "◦ read a.py\n\n◦" not in out


def test_format_unified_diff_for_display_hides_headers_and_shows_lines():
    diff = """--- a/foo.py
+++ b/foo.py
@@ -1,2 +1,2 @@
 alpha
-beta
+gamma
"""
    rendered = _format_unified_diff_for_display(diff, width=80)
    plain = rendered.plain
    assert "--- a/foo.py" not in plain
    assert "+++ b/foo.py" not in plain
    assert "@@" not in plain
    assert "lines 1–2" in plain
    assert "alpha" in plain
    assert "- beta" in plain
    assert "+ gamma" in plain


def test_format_unified_diff_for_display_shows_truncation_note():
    diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n... [3 more lines]"
    rendered = _format_unified_diff_for_display(diff, width=80)
    assert "... [3 more lines]" in rendered.plain


def test_print_harness_activity_shows_file_diff(capsys):
    ui = ConversationUI()
    step = AgentStep(
        action=AgentAction.STR_REPLACE,
        path="foo.py",
        old_string="a",
        new_string="b",
    )
    diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n"
    result = ToolResult(
        executed="str_replace foo.py",
        stdout=f"Updated foo.py\n{diff}",
        exit_code=0,
    )
    ui.print_harness_activity(step, result)
    out = capsys.readouterr().out
    assert "wrote foo.py" in out
    assert "changes · foo.py" in out
    assert "- a" in out
    assert "+ b" in out


def test_print_harness_activity_skips_file_diff_in_debug(capsys):
    ui = ConversationUI(debug=True)
    step = AgentStep(
        action=AgentAction.STR_REPLACE,
        path="foo.py",
        old_string="a",
        new_string="b",
    )
    diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n"
    result = ToolResult(
        executed="str_replace foo.py",
        stdout=f"Updated foo.py\n{diff}",
        exit_code=0,
    )
    ui.print_harness_activity(step, result)
    out = capsys.readouterr().out
    assert "wrote foo.py" in out
    assert "diff ·" not in out


def test_print_harness_activity_skipped_when_verbose(capsys):
    ui = ConversationUI(verbose=True)
    step = AgentStep(action=AgentAction.RUN_SHELL, command="ls")
    result = ToolResult(executed="run_shell", exit_code=0)
    ui.print_harness_activity(step, result)
    assert capsys.readouterr().out == ""


def test_progress_label_prefers_status():
    step = AgentStep(
        action=AgentAction.RUN_SHELL,
        status="Implementing authentication flow",
        command="pytest -q",
    )
    from agent.ui import progress_label_for_step

    assert progress_label_for_step(step) == "Implementing authentication flow"


def test_progress_label_falls_back_to_activity():
    step = AgentStep(action=AgentAction.READ_FILE, path="src/auth.py")
    from agent.ui import progress_label_for_step

    assert progress_label_for_step(step) == "reading auth.py"


def test_thinking_default_label_non_tty(capsys, monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    ui = ConversationUI()
    with ui.thinking():
        pass
    assert "working on your task" in capsys.readouterr().out


def test_thinking_shows_plan_step_under_goodboy(capsys, monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    ui = ConversationUI()
    with ui.thinking(
        label="writing ui.py",
        plan_items=[
            PlanItem(id="1", text="recon", status=PlanItemStatus.DONE),
            PlanItem(
                id="2",
                text="refactor loop",
                status=PlanItemStatus.IN_PROGRESS,
            ),
            PlanItem(id="3", text="verify", status=PlanItemStatus.PENDING),
        ],
    ):
        pass
    out = capsys.readouterr().out
    assert "GoodBoy" in out
    assert "writing ui.py" in out
    assert '2/3 step "refactor loop"' in out


def test_thinking_updater_render_includes_plan_step():
    from rich.console import Group

    from agent.ui import ThinkingUpdater

    updater = ThinkingUpdater(
        _label="reading files",
        _console=Console(),
        _plan_step='2/4 step "map modules"',
    )
    rendered = updater._render()
    assert isinstance(rendered, Group)
    assert "GoodBoy" in rendered.renderables[0].plain
    assert '2/4 step "map modules"' in rendered.renderables[1].plain


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
    assert "Need search" in captured.out + captured.err

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
    assert "thought" in captured.out.lower()
    assert "Done" in captured.out
    assert "Enable hosted" not in captured.out


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
    assert "Need detail" in out
    assert "Which file?" in out


def test_render_plan_items_colors():
    from agent.ui import _render_plan_items

    rendered = _render_plan_items(
        [
            PlanItem(id="1", text="recon", status=PlanItemStatus.DONE),
            PlanItem(id="2", text="edit file", status=PlanItemStatus.IN_PROGRESS),
            PlanItem(id="3", text="verify", status=PlanItemStatus.PENDING),
            PlanItem(id="4", text="skip", status=PlanItemStatus.CANCELLED),
        ]
    )
    assert rendered.plain.splitlines() == [
        "[✓] 1. recon",
        "[→] 2. edit file",
        "[ ] 3. verify",
        "[–] 4. skip",
    ]
    assert rendered.spans[0].style == "white"
    assert rendered.spans[1].style == "white"
    assert rendered.spans[2].style == "dim"
    assert rendered.spans[3].style == "dim"


def test_print_plan_shows_plan_items(capsys):
    ui = ConversationUI()
    ui.print_plan(
        [
            PlanItem(id="1", text="recon", status=PlanItemStatus.DONE),
            PlanItem(
                id="2",
                text="edit file",
                status=PlanItemStatus.IN_PROGRESS,
            ),
        ],
    )
    out = capsys.readouterr().out
    assert "[✓] 1." in out
    assert "[→] 2. edit file" in out
    assert "(plan)" in out


def test_print_agent_step_does_not_show_plan_items(capsys):
    ui = ConversationUI()
    ui.print_agent_step(
        AgentStep(
            action=AgentAction.UPDATE_PLAN,
            plan_items=[
                PlanItem(id="a", text="write tests", status=PlanItemStatus.PENDING),
            ],
        ),
    )
    out = capsys.readouterr().out
    assert "(plan)" not in out


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


def test_request_abort_sets_flag_and_fires_callbacks():
    ui = ConversationUI()
    fired = []

    with ui.on_abort(lambda: fired.append("closed")):
        assert ui.is_abort_requested() is False
        ui.request_abort()
        assert ui.is_abort_requested() is True
        # request_abort also marks the legacy stop flag so the loop's
        # consume_stop_requested still works after the abort path runs.
        assert ui.consume_stop_requested() is True

    assert fired == ["closed"]


def test_on_abort_fires_immediately_when_already_aborted():
    ui = ConversationUI()
    ui.request_abort()
    fired = []
    with ui.on_abort(lambda: fired.append("late")):
        pass
    assert fired == ["late"]


def test_clear_abort_request_resets_state():
    ui = ConversationUI()
    ui.request_abort()
    assert ui.is_abort_requested() is True
    ui.clear_abort_request()
    assert ui.is_abort_requested() is False
    assert ui.consume_stop_requested() is False


def test_truncate_diff_lines_short_diff():
    diff = "line1\nline2\nline3"
    assert _truncate_diff_lines(diff, max_lines=5) == diff


def test_truncate_diff_lines_long_diff():
    diff = "\n".join(f"line{i}" for i in range(10))
    result = _truncate_diff_lines(diff, max_lines=4)
    assert result.endswith("... [6 more lines]")
    assert result.count("\n") == 4


def test_format_diff_line_range_single():
    assert _format_diff_line_range(5, 1) == "5"


def test_format_diff_line_range_range():
    assert _format_diff_line_range(5, 3) == "5–7"


def test_format_diff_hunk_label_same_range():
    assert _format_diff_hunk_label(10, 3, 10, 3) == "lines 10–12"


def test_format_diff_hunk_label_different_range():
    assert _format_diff_hunk_label(10, 3, 15, 5) == "lines 10–12 → 15–19"


def test_append_wrapped_content_adds_text():
    text = Text()
    _append_wrapped_content(text, "hello\nworld", style="red", width=80, indent="  ")
    assert "hello" in text.plain
    assert "\n" in text.plain


def test_diff_display_line_width_returns_min_3():
    assert _diff_display_line_width("no hunks here") >= 3


def test_diff_display_line_width_with_hunk():
    diff = "@@ -5,10 +7,12 @@"
    width = _diff_display_line_width(diff)
    assert width >= 2


def test_extract_unified_diff_finds_marker():
    stdout = "some output\n--- a/foo\n+++ b/foo\n@@ -1,3 +1,4 @@"
    result = _extract_unified_diff(stdout)
    assert result is not None
    assert result.startswith("--- a/foo")


def test_extract_unified_diff_none():
    assert _extract_unified_diff("no diff here") is None


def test_format_unified_diff_for_display_removes_headers(tmp_path):
    diff = "--- a/foo\n+++ b/foo\n@@ -1,3 +1,4 @@\n a\n-b\n+c\n d"
    result = _format_unified_diff_for_display(diff, width=80)
    assert "--- a/foo" not in result.plain
    assert "+++ b/foo" not in result.plain
    assert "a" in result.plain
    assert "b" in result.plain
    assert "c" in result.plain


def test_format_unified_diff_for_display_with_truncation_marker():
    diff = "... [5 more lines]"
    result = _format_unified_diff_for_display(diff, width=80)
    assert "..." in result.plain
    assert "5 more lines" in result.plain


def test_format_unified_diff_with_no_newline():
    diff = "@@ -1,2 +1,2 @@\n text\n\\ No newline at end of file"
    result = _format_unified_diff_for_display(diff, width=80)
    assert "No newline" in result.plain

