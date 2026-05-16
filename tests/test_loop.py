"""Tests for agent loop orchestration."""

import json
from pathlib import Path

from agent.context import SessionContext
from agent.loop import AgentLoop, LoopOutcome
from agent.types import AgentAction, AgentStep


def _llm_responses(responses: list[AgentStep]):
    payloads = [json.dumps(step.model_dump(mode="json")) for step in responses]
    index = {"i": 0}

    def fake_llm(**_kwargs):
        i = index["i"]
        index["i"] += 1
        return payloads[i]

    return fake_llm


def test_loop_task_complete(tmp_path: Path):
    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        llm_call=_llm_responses(
            [
                AgentStep(
                    action=AgentAction.TASK_COMPLETE,
                    message="All done.",
                )
            ]
        ),
    )
    result = loop.run("do something")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert result.message == "All done."
    assert len(result.context.turns) == 1


def test_loop_runs_shell_then_completes(tmp_path: Path):
    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        llm_call=_llm_responses(
            [
                AgentStep(
                    action=AgentAction.RUN_SHELL,
                    command="echo loop-test",
                ),
                AgentStep(
                    action=AgentAction.TASK_COMPLETE,
                    message="finished",
                ),
            ]
        ),
    )
    result = loop.run("echo test")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert len(result.context.turns) == 2
    assert result.context.turns[0].tool_result is not None
    assert "loop-test" in result.context.turns[0].tool_result.stdout


def test_loop_need_user_input(tmp_path: Path):
    loop = AgentLoop(
        workspace=tmp_path,
        llm_call=_llm_responses(
            [
                AgentStep(
                    action=AgentAction.NEED_USER_INPUT,
                    message="Which file?",
                )
            ]
        ),
    )
    result = loop.run("edit file")
    assert result.outcome == LoopOutcome.NEED_USER_INPUT
    assert result.message == "Which file?"


def test_loop_invalid_json_twice_fails(tmp_path: Path):
    calls = {"n": 0}

    def bad_llm(**_kwargs):
        calls["n"] += 1
        return "not json"

    loop = AgentLoop(workspace=tmp_path, max_turns=5, llm_call=bad_llm)
    result = loop.run("task")
    assert result.outcome == LoopOutcome.FAILED
    assert "invalid JSON" in result.message


def test_loop_resumes_after_user_reply(tmp_path: Path):
    llm = _llm_responses(
        [
            AgentStep(
                action=AgentAction.NEED_USER_INPUT,
                message="Which branch?",
            ),
            AgentStep(
                action=AgentAction.TASK_COMPLETE,
                message="merged",
            ),
        ]
    )
    loop = AgentLoop(workspace=tmp_path, llm_call=llm)

    first = loop.run("merge")
    assert first.outcome == LoopOutcome.NEED_USER_INPUT
    first.context.add_user_reply("main")

    second = loop.run("merge", context=first.context)
    assert second.outcome == LoopOutcome.TASK_COMPLETE
    assert "main" in second.context.to_prompt()
