"""Tests for agent loop orchestration."""

import json
from pathlib import Path

from agent.context import SessionContext
from agent.loop import AgentLoop, LoopOutcome
from agent.types import AgentAction, AgentStep

_ALLOWED = ["gpt-4o-mini", "gpt-5.4-mini", "gpt-5.5", "o4-mini"]


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
        allowed_models=_ALLOWED,
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
        allowed_models=_ALLOWED,
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
        allowed_models=_ALLOWED,
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

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        llm_call=bad_llm,
    )
    result = loop.run("task")
    assert result.outcome == LoopOutcome.FAILED
    assert "invalid JSON" in result.message


def test_loop_ask_user_continues_in_one_run(tmp_path: Path):
    replies = iter(["main"])

    def ask_user() -> str:
        return next(replies)

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
    loop = AgentLoop(
        workspace=tmp_path, allowed_models=_ALLOWED, llm_call=llm
    )
    result = loop.run("merge", ask_user=ask_user)
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert "main" in result.context.to_prompt()


def test_loop_repeat_question_after_reply_nudges_model(tmp_path: Path):
    """Repeated confirmation questions get a harness nudge instead of blocking."""
    replies = iter(["yes, commit"])

    def ask_user() -> str:
        return next(replies)

    same_q = "Please confirm if you'd like me to create a commit message"
    llm = _llm_responses(
        [
            AgentStep(action=AgentAction.NEED_USER_INPUT, message=same_q),
            AgentStep(action=AgentAction.NEED_USER_INPUT, message=same_q),
            AgentStep(action=AgentAction.TASK_COMPLETE, message="done"),
        ]
    )
    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=10,
        allowed_models=_ALLOWED,
        llm_call=llm,
    )
    result = loop.run("commit changes", ask_user=ask_user)
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert any("already asked" in e.lower() for e in result.context.parse_errors)


def test_loop_max_clarifications_fails(tmp_path: Path):
    replies = iter(["a", "b", "c", "d"])

    def ask_user() -> str:
        return next(replies)

    def always_ask(**_kwargs):
        return json.dumps(
            AgentStep(
                action=AgentAction.NEED_USER_INPUT,
                message="Still unclear?",
            ).model_dump(mode="json")
        )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=20,
        max_clarifications=3,
        allowed_models=_ALLOWED,
        llm_call=always_ask,
    )
    result = loop.run("task", ask_user=ask_user)
    assert result.outcome == LoopOutcome.FAILED
    assert "too many times" in result.message


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
    loop = AgentLoop(
        workspace=tmp_path, allowed_models=_ALLOWED, llm_call=llm
    )

    first = loop.run("merge")
    assert first.outcome == LoopOutcome.NEED_USER_INPUT
    first.context.add_user_reply("main")

    second = loop.run("merge", context=first.context)
    assert second.outcome == LoopOutcome.TASK_COMPLETE
    assert "main" in second.context.to_prompt()


def test_loop_pending_model_applied_on_next_call(tmp_path: Path):
    calls: list[dict] = []

    def tracking_llm(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return json.dumps(
                AgentStep(
                    action=AgentAction.RUN_SHELL,
                    command="echo one",
                    model="gpt-5.4-mini",
                ).model_dump(mode="json")
            )
        return json.dumps(
            AgentStep(
                action=AgentAction.TASK_COMPLETE,
                message="done",
            ).model_dump(mode="json")
        )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        model="gpt-4o-mini",
        llm_call=tracking_llm,
    )
    result = loop.run("task")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert len(calls) == 2
    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[1]["model"] == "gpt-5.4-mini"


def test_loop_stops_after_repeated_failed_shell(tmp_path: Path):
    cmd = "false"
    llm = _llm_responses(
        [
            AgentStep(action=AgentAction.RUN_SHELL, command=cmd),
            AgentStep(action=AgentAction.RUN_SHELL, command=cmd),
            AgentStep(action=AgentAction.RUN_SHELL, command=cmd),
        ]
    )
    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=10,
        allowed_models=_ALLOWED,
        llm_call=llm,
    )
    result = loop.run("run failing command")
    assert result.outcome == LoopOutcome.FAILED
    assert "repeatedly" in result.message.lower()
    assert len(result.context.turns) == 2


def test_loop_invalid_model_parse_error_then_recovery(tmp_path: Path):
    calls = {"n": 0}

    def llm(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "action": "run_shell",
                    "command": "echo x",
                    "model": "not-a-real-model",
                }
            )
        return json.dumps(
            AgentStep(
                action=AgentAction.TASK_COMPLETE,
                message="ok",
            ).model_dump(mode="json")
        )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        llm_call=llm,
    )
    result = loop.run("task")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert any("Unknown model" in e for e in result.context.parse_errors)


def test_loop_reasoning_effort_rejected_for_gpt4o_mini(tmp_path: Path):
    calls = {"n": 0}

    def llm(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "action": "task_complete",
                    "message": "ok",
                    "reasoning_effort": "high",
                }
            )
        return json.dumps(
            AgentStep(
                action=AgentAction.TASK_COMPLETE,
                message="ok",
            ).model_dump(mode="json")
        )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        model="gpt-4o-mini",
        llm_call=llm,
    )
    result = loop.run("task")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert any("does not support" in e for e in result.context.parse_errors)
