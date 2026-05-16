"""Tests for interactive harness REPL."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from unittest.mock import Mock

import pytest

from agent.context import ConversationExchange, SessionContext
from agent.harness import AgentHarness
from agent.loop import LoopOutcome, LoopResult


@pytest.fixture(autouse=True)
def _disable_session_log(monkeypatch):
    @contextmanager
    def _noop(**_kwargs):
        yield None

    monkeypatch.setattr("agent.harness.open_session_log", _noop)


@dataclass
class FakeUI:
    prompts: Iterator[str]

    def prompt_user(self) -> str:
        return next(self.prompts)

    def print_startup(self) -> None:
        pass

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
