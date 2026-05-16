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

    def __init__(self, prompts: Iterator[str]) -> None:
        self.prompts = prompts
        self.clear_session_calls = 0

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
        pass

    def newline(self) -> None:
        pass


def _harness(outcome: LoopOutcome, prompts: list[str]) -> AgentHarness:
    loop = Mock()
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


def test_harness_slash_clear_command():
    harness = _harness(LoopOutcome.TASK_COMPLETE, ["/clear", "new task", "exit"])
    assert harness.run() == 0
    assert harness._ui.clear_session_calls == 1
    assert harness._loop.run.call_count == 1


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
