"""Tests for interactive harness REPL."""

from dataclasses import dataclass
from typing import Iterator
from unittest.mock import Mock

from agent.context import SessionContext
from agent.harness import AgentHarness
from agent.loop import LoopOutcome, LoopResult


@dataclass
class FakeUI:
    prompts: Iterator[str]

    def prompt_user(self) -> str:
        return next(self.prompts)


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
