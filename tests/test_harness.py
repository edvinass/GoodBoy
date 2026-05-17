"""Tests for interactive harness REPL."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from unittest.mock import Mock

import pytest

from agent.context import ConversationExchange, SessionContext
from agent.harness import AgentHarness
from agent.loop import LoopOutcome, LoopResult
from agent.types import AgentAction, AgentStep, TurnRecord


@pytest.fixture(autouse=True)
def _disable_session_log(monkeypatch):
    @contextmanager
    def _noop(**_kwargs):
        yield None

    monkeypatch.setattr("agent.harness.open_session_log", _noop)


@dataclass
class FakeUI:
    prompts: Iterator[str]

    def __init__(self, prompts: Iterator[str], *, show_commands: bool = False) -> None:
        self.prompts = prompts
        self.show_commands = show_commands
        self.auto_model_switch = False
        self.clear_session_calls = 0
        self.notices: list[str] = []
        self._session_model = None
        self._session_reasoning = None

    def prompt_user(self) -> str:
        return next(self.prompts)

    def print_startup(self) -> None:
        pass

    def clear_session(self) -> None:
        self.clear_session_calls += 1

    def print_task_complete(self) -> None:
        pass

    def print_failed(self, message: str | None = None) -> None:
        pass

    def print_stopped(self, message: str) -> None:
        pass

    def print_notice(self, message: str) -> None:
        self.notices.append(message)

    def set_session_reasoning(self, effort: str | None) -> None:
        self._session_reasoning = effort

    def newline(self) -> None:
        pass

    def set_session_model(self, model: str) -> None:
        self._session_model = model


def _harness(outcome: LoopOutcome, prompts: list[str]) -> AgentHarness:
    loop = Mock()
    loop.refresh_system_prompt = Mock()
    loop.session_model = "gpt-5.4-nano"
    loop.run.return_value = LoopResult(
        outcome=outcome,
        message="done",
        context=SessionContext(user_task="task"),
    )
    return AgentHarness(loop=loop, ui=FakeUI(prompts=iter(prompts)))


def test_harness_runs_multiple_tasks_before_exit():
    harness = _harness(LoopOutcome.TASK_COMPLETE, ["first task", "second task", "exit"])
    assert harness.run() == 0
    assert harness._loop.run.call_count == 2


def test_harness_empty_after_task_exits_gracefully():
    harness = _harness(LoopOutcome.TASK_COMPLETE, ["do work", ""])
    assert harness.run() == 0
    assert harness._loop.run.call_count == 1


def test_harness_empty_first_prompt_fails():
    harness = _harness(LoopOutcome.TASK_COMPLETE, [""])
    assert harness.run() == 1
    harness._loop.run.assert_not_called()


def test_harness_tracks_failed_task_exit_code():
    harness = _harness(LoopOutcome.FAILED, ["bad task", "exit"])
    assert harness.run() == 1
    assert harness._loop.run.call_count == 1


def test_harness_passes_conversation_history_on_follow_up():
    loop = Mock()
    loop.workspace = "/tmp"
    loop.run.side_effect = [
        LoopResult(
            outcome=LoopOutcome.TASK_COMPLETE,
            message="London: cloudy",
            context=SessionContext(
                user_task="weather in London",
                active_hosted_tools=["web_search"],
            ),
        ),
        LoopResult(
            outcome=LoopOutcome.TASK_COMPLETE,
            message="Sydney: sunny",
            context=SessionContext(user_task="and in Sydney"),
        ),
    ]
    harness = AgentHarness(
        loop=loop,
        ui=FakeUI(prompts=iter(["weather in London", "and in Sydney", "exit"])),
    )
    assert harness.run() == 0
    assert loop.run.call_count == 2

    first_kwargs = loop.run.call_args_list[0].kwargs
    assert first_kwargs.get("context") is None

    second_kwargs = loop.run.call_args_list[1].kwargs
    ctx = second_kwargs["context"]
    assert ctx is not None
    assert ctx.user_task == "and in Sydney"
    assert len(ctx.conversation_history) == 1
    assert ctx.conversation_history[0] == ConversationExchange(
        user="weather in London",
        assistant="London: cloudy",
    )
    assert ctx.active_hosted_tools == ["web_search"]
    assert len(harness._conversation_history) == 2


def test_harness_stopped_task_waits_for_next_user_prompt_to_resume():
    loop = Mock()
    loop.workspace = "/tmp"
    stopped_context = SessionContext(user_task="long task")
    stopped_context.add_turn(
        TurnRecord(
            turn=1,
            step=AgentStep(
                action=AgentAction.RUN_SHELL,
                command="echo first",
            ),
        )
    )
    loop.run.side_effect = [
        LoopResult(
            outcome=LoopOutcome.STOPPED,
            message="Stopped after the current step.",
            context=stopped_context,
        ),
        LoopResult(
            outcome=LoopOutcome.TASK_COMPLETE,
            message="done",
            context=SessionContext(user_task="long task"),
        ),
    ]

    harness = AgentHarness(
        loop=loop,
        ui=FakeUI(prompts=iter(["long task", "continue", "exit"])),
    )

    assert harness.run() == 0
    assert loop.run.call_count == 2
    first_kwargs = loop.run.call_args_list[0].kwargs
    assert first_kwargs.get("context") is None

    second_kwargs = loop.run.call_args_list[1].kwargs
    resumed_ctx = second_kwargs["context"]
    assert resumed_ctx is not stopped_context
    assert len(resumed_ctx.turns) == 1
    assert resumed_ctx.user_task == "long task"
    assert any("continue" in reply for reply in resumed_ctx.user_replies)


def test_harness_clear_resets_conversation_history():
    loop = Mock()
    loop.workspace = "/tmp"
    loop.run.side_effect = [
        LoopResult(
            outcome=LoopOutcome.TASK_COMPLETE,
            message="London: cloudy",
            context=SessionContext(
                user_task="weather in London",
                active_hosted_tools=["web_search"],
            ),
        ),
        LoopResult(
            outcome=LoopOutcome.TASK_COMPLETE,
            message="Paris: rainy",
            context=SessionContext(user_task="weather in Paris"),
        ),
    ]
    ui = FakeUI(prompts=iter(["weather in London", "clear", "weather in Paris", "exit"]))
    harness = AgentHarness(loop=loop, ui=ui)

    assert harness.run() == 0
    assert loop.run.call_count == 2
    assert ui.clear_session_calls == 1

    second_kwargs = loop.run.call_args_list[1].kwargs
    ctx = second_kwargs["context"]
    assert ctx is None
    assert harness._conversation_history == [
        ConversationExchange(user="weather in Paris", assistant="Paris: rainy")
    ]
    assert harness._last_active_hosted_tools == []


def test_harness_reasoning_command_changes_loop_and_persists(monkeypatch):
    loop = Mock()
    loop.workspace = "/tmp"
    loop.session_model = "gpt-5.4-nano"
    loop.session_reasoning = None
    loop.set_session_reasoning = Mock()
    loop.run.return_value = LoopResult(
        outcome=LoopOutcome.TASK_COMPLETE,
        message="done",
        context=SessionContext(user_task="task"),
    )
    ui = FakeUI(prompts=iter(["/reasoning", "exit"]))
    harness = AgentHarness(loop=loop, ui=ui)

    monkeypatch.setattr(
        "agent.harness.select_reasoning_interactive",
        lambda **_: "low",
    )
    saved: dict[str, str] = {}
    monkeypatch.setattr("agent.harness.save_env", saved.update)

    assert harness.run() == 0
    loop.set_session_reasoning.assert_called_once_with("low")
    assert ui._session_reasoning == "low"
    assert saved == {"GOODBOY_REASONING_EFFORT": "low"}
    loop.run.assert_not_called()


def test_harness_model_command_changes_loop_and_persists(monkeypatch, tmp_path):
    loop = Mock()
    loop.workspace = "/tmp"
    loop.session_model = "gpt-5.4-nano"
    loop._allowed_models = ["gpt-5.4-nano", "gpt-5.4-mini"]
    loop.run.return_value = LoopResult(
        outcome=LoopOutcome.TASK_COMPLETE,
        message="done",
        context=SessionContext(user_task="task"),
    )
    ui = FakeUI(prompts=iter(["/model", "exit"]))
    ui.set_session_model = Mock()
    harness = AgentHarness(loop=loop, ui=ui)

    monkeypatch.setattr(
        "agent.harness.select_model_interactive",
        lambda **_: "gpt-5.4-mini",
    )
    saved: dict[str, str] = {}
    monkeypatch.setattr("agent.harness.save_env", saved.update)

    assert harness.run() == 0
    loop.set_session_model.assert_called_once_with("gpt-5.4-mini")
    ui.set_session_model.assert_called_with("gpt-5.4-mini")
    assert saved == {"OPENAI_MODEL": "gpt-5.4-mini"}
    loop.run.assert_not_called()


def test_harness_slash_clear_command():
    harness = _harness(LoopOutcome.TASK_COMPLETE, ["/clear", "new task", "exit"])
    assert harness.run() == 0
    assert harness._ui.clear_session_calls == 1
    assert harness._loop.run.call_count == 1


def test_harness_slash_commands_toggles_visibility(monkeypatch):
    loop = Mock()
    loop.run.return_value = LoopResult(
        outcome=LoopOutcome.TASK_COMPLETE,
        message="done",
        context=SessionContext(user_task="task"),
    )
    loop.refresh_system_prompt = Mock()
    ui = FakeUI(prompts=iter(["/commands", "exit"]))
    harness = AgentHarness(loop=loop, ui=ui)
    saved: dict[str, str] = {}
    monkeypatch.setattr("agent.harness.save_env", saved.update)

    assert harness.run() == 0
    assert loop.run.call_count == 0
    assert ui.show_commands is True
    assert loop.refresh_system_prompt.call_count >= 2
    assert "on" in ui.notices[0].lower()
    assert saved == {"GOODBOY_SHOW_COMMANDS": "true"}


def test_harness_slash_autoswitch_toggles_and_persists(monkeypatch):
    loop = Mock()
    loop.run.return_value = LoopResult(
        outcome=LoopOutcome.TASK_COMPLETE,
        message="done",
        context=SessionContext(user_task="task"),
    )
    loop.refresh_system_prompt = Mock()
    loop.session_model = "gpt-5.4-nano"
    ui = FakeUI(prompts=iter(["/autoswitch", "exit"]))
    harness = AgentHarness(loop=loop, ui=ui)
    saved: dict[str, str] = {}
    monkeypatch.setattr("agent.harness.save_env", saved.update)

    assert harness.run() == 0
    assert loop.run.call_count == 0
    assert ui.auto_model_switch is True
    loop.refresh_system_prompt.assert_called()
    assert saved == {"GOODBOY_AUTO_MODEL_SWITCH": "true"}


def test_harness_clear_discards_paused_context():
    loop = Mock()
    loop.workspace = "/tmp"
    stopped_context = SessionContext(user_task="long task")
    stopped_context.add_turn(
        TurnRecord(
            turn=1,
            step=AgentStep(
                action=AgentAction.RUN_SHELL,
                command="echo first",
            ),
        )
    )
    loop.run.side_effect = [
        LoopResult(
            outcome=LoopOutcome.STOPPED,
            message="Stopped after the current step.",
            context=stopped_context,
        ),
        LoopResult(
            outcome=LoopOutcome.TASK_COMPLETE,
            message="done",
            context=SessionContext(user_task="fresh task"),
        ),
    ]
    ui = FakeUI(prompts=iter(["long task", "clear", "fresh task", "exit"]))
    harness = AgentHarness(loop=loop, ui=ui)

    assert harness.run() == 0
    assert loop.run.call_count == 2

    second_kwargs = loop.run.call_args_list[1].kwargs
    ctx = second_kwargs["context"]
    assert ctx is None
    assert harness._paused_context is None
