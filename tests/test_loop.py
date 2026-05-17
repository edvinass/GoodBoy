"""Tests for agent loop orchestration."""

import json
from pathlib import Path

from agent.context import SessionContext
from agent.loop import AgentLoop, LoopOutcome
from agent.types import AgentAction, AgentStep, TurnRecord
from agent.ui import ConversationUI

_ALLOWED = ["gpt-4.1-nano", "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.5"]


def _llm_responses(responses: list[AgentStep]):
    payloads = [json.dumps(step.model_dump(mode="json")) for step in responses]
    index = {"i": 0}

    def fake_llm(**_kwargs):
        i = index["i"]
        index["i"] += 1
        return payloads[i]

    return fake_llm


def test_loop_default_reasoning_effort_applied(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GOODBOY_REASONING_EFFORT=low\n", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    from settings import get_settings

    get_settings.cache_clear()

    captured: dict[str, object] = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
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
        model="gpt-5.4-nano",
        llm_call=fake_llm,
    )
    loop.run("task")
    assert captured.get("reasoning_effort") == "low"
    get_settings.cache_clear()


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


def test_loop_model_field_on_run_shell_applied_on_next_call(tmp_path: Path):
    calls: list[dict] = []

    def tracking_llm(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return json.dumps(
                AgentStep(
                    action=AgentAction.RUN_SHELL,
                    command="true",
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
        model="gpt-5.4-nano",
        ui=ConversationUI(auto_model_switch=True),
        llm_call=tracking_llm,
    )
    result = loop.run("task")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert len(calls) == 2
    assert calls[0]["model"] == "gpt-4.1-nano"
    assert calls[1]["model"] == "gpt-5.4-mini"
    assert loop.session_model == "gpt-5.4-mini"


def test_loop_autoswitch_routes_first_turn_on_cheapest_model(tmp_path: Path):
    calls: list[dict] = []

    def tracking_llm(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return json.dumps(
                AgentStep(
                    action=AgentAction.RUN_SHELL,
                    command="echo hi",
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
        model="gpt-5.5",
        llm_call=tracking_llm,
        ui=ConversationUI(auto_model_switch=True),
    )
    result = loop.run("task")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert len(calls) == 2
    assert calls[0]["model"] == "gpt-4.1-nano"
    assert calls[1]["model"] == "gpt-5.4-mini"
    assert loop.session_model == "gpt-5.4-mini"


def test_loop_autoswitch_routing_turn_requires_model(tmp_path: Path):
    calls: list[dict] = []

    def tracking_llm(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return json.dumps(
                AgentStep(
                    action=AgentAction.RUN_SHELL,
                    command="echo hi",
                ).model_dump(mode="json")
            )
        return json.dumps(
            AgentStep(
                action=AgentAction.RUN_SHELL,
                command="echo hi",
                model="gpt-5.4-mini",
            ).model_dump(mode="json")
        )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        model="gpt-5.5",
        llm_call=tracking_llm,
        ui=ConversationUI(auto_model_switch=True),
    )
    result = loop.run("task")
    assert any("Routing turn" in e for e in result.context.parse_errors)
    assert calls[0]["model"] == "gpt-4.1-nano"
    assert len(calls) >= 2


def test_loop_autoswitch_skips_router_on_resume(tmp_path: Path):
    calls: list[dict] = []

    def tracking_llm(**kwargs):
        calls.append(dict(kwargs))
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
        model="gpt-5.5",
        llm_call=tracking_llm,
        ui=ConversationUI(auto_model_switch=True),
    )
    paused = SessionContext(user_task="task", workspace=str(tmp_path))
    paused.add_turn(
        TurnRecord(
            turn=1,
            step=AgentStep(action=AgentAction.RUN_SHELL, command="echo x"),
        )
    )
    loop.run("task", context=paused)
    assert calls[0]["model"] == "gpt-5.5"


def test_loop_pending_model_applied_on_next_call(tmp_path: Path):
    calls: list[dict] = []

    def tracking_llm(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return json.dumps(
                AgentStep(
                    action=AgentAction.SWITCH_MODEL,
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
        model="gpt-5.4-nano",
        ui=ConversationUI(auto_model_switch=True),
        llm_call=tracking_llm,
    )
    result = loop.run("task")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert len(calls) == 2
    assert calls[0]["model"] == "gpt-4.1-nano"
    assert calls[1]["model"] == "gpt-5.4-mini"
    assert loop.session_model == "gpt-5.4-mini"


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
                    "action": "switch_model",
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
        ui=ConversationUI(auto_model_switch=True),
        llm_call=llm,
    )
    result = loop.run("task")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert any("Unknown model" in e for e in result.context.parse_errors)


def test_loop_switch_tools_enables_hosted_tools(tmp_path: Path):
    calls: list[dict] = []

    def tracking_llm(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return json.dumps(
                AgentStep(
                    action=AgentAction.SWITCH_TOOLS,
                    tools=["web_search"],
                ).model_dump(mode="json")
            )
        return json.dumps(
            AgentStep(
                action=AgentAction.TASK_COMPLETE,
                message="Sunny, 18°C in London.",
            ).model_dump(mode="json")
        )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        model="gpt-5.4-nano",
        llm_call=tracking_llm,
    )
    result = loop.run("what is the weather in London")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert result.message == "Sunny, 18°C in London."
    assert len(calls) == 2
    assert "tools" not in calls[0]
    assert calls[1]["tools"] == ["web_search"]
    assert result.context.active_hosted_tools == ["web_search"]


def test_loop_switch_tools_rejects_reasoning_on_same_turn(tmp_path: Path):
    calls = {"n": 0}

    def llm(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "action": "switch_tools",
                    "tools": ["web_search"],
                    "reasoning_effort": "high",
                }
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
        ui=ConversationUI(auto_model_switch=True),
        llm_call=llm,
    )
    result = loop.run("weather in London")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert any("reasoning_effort on switch_tools" in e for e in result.context.parse_errors)


def test_loop_switch_tools_rejects_model_on_same_turn(tmp_path: Path):
    calls = {"n": 0}

    def llm(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "action": "switch_tools",
                    "tools": ["web_search"],
                    "model": "gpt-5.4-mini",
                }
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
        llm_call=llm,
    )
    result = loop.run("weather in London")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert any("switch_tools" in e and "model" in e for e in result.context.parse_errors)


def test_loop_switch_tools_rejects_web_search_on_gpt41_nano(tmp_path: Path):
    llm_calls: list[dict] = []

    def llm(**kwargs):
        llm_calls.append(dict(kwargs))
        if len(llm_calls) == 1:
            return json.dumps(
                {
                    "action": "switch_tools",
                    "tools": ["web_search"],
                }
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
        model="gpt-4.1-nano",
        llm_call=llm,
    )
    result = loop.run("weather in London")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert any(
        "does not support hosted tool 'web_search'" in e
        and "gpt-4.1-nano" in e
        for e in result.context.parse_errors
    )
    assert len(llm_calls) == 2
    assert "tools" not in llm_calls[0]
    assert "tools" not in llm_calls[1]
    assert result.context.active_hosted_tools == []


def test_loop_reasoning_effort_rejected_for_non_reasoning_model(tmp_path: Path):
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
        model="gpt-4.1-nano",
        ui=ConversationUI(auto_model_switch=True),
        llm_call=llm,
    )
    result = loop.run("task")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert any("does not support" in e for e in result.context.parse_errors)


def test_loop_stop_requested_before_tool_still_runs_planned_command(tmp_path: Path):
    calls = {"n": 0}

    def llm(**_kwargs):
        calls["n"] += 1
        return json.dumps(
            AgentStep(
                action=AgentAction.RUN_SHELL,
                command="echo paused-step",
                thought="Echoed a line for the user.",
            ).model_dump(mode="json")
        )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        llm_call=llm,
    )
    result = loop.run("echo then pause", stop_requested=lambda: True)

    assert result.outcome == LoopOutcome.STOPPED
    assert calls["n"] == 1
    assert len(result.context.turns) == 1
    assert result.context.turns[0].tool_result is not None
    assert "paused-step" in result.context.turns[0].tool_result.stdout
    assert "Echoed a line for the user." in result.message
    assert "Paused — say continue when you're ready." in result.message


def test_loop_stop_requested_during_tool_pauses_after_tool_step(tmp_path: Path):
    calls = {"n": 0}
    stop_checks = iter([True])

    def llm(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                AgentStep(
                    action=AgentAction.RUN_SHELL,
                    command="echo current-step",
                ).model_dump(mode="json")
            )
        return json.dumps(
            AgentStep(
                action=AgentAction.TASK_COMPLETE,
                message="should not run yet",
            ).model_dump(mode="json")
        )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        llm_call=llm,
    )
    result = loop.run("echo then pause", stop_requested=lambda: next(stop_checks))

    assert result.outcome == LoopOutcome.STOPPED
    assert calls["n"] == 1
    assert len(result.context.turns) == 1
    assert result.context.turns[0].tool_result is not None
    assert "current-step" in result.context.turns[0].tool_result.stdout
    assert "Paused — say continue when you're ready." in result.message


def test_loop_resume_after_stop_continues_from_existing_context(tmp_path: Path):
    calls = {"n": 0}
    stop_checks = iter([True, False])

    def llm(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                AgentStep(
                    action=AgentAction.RUN_SHELL,
                    command="echo first",
                ).model_dump(mode="json")
            )
        return json.dumps(
            AgentStep(
                action=AgentAction.TASK_COMPLETE,
                message="resumed",
            ).model_dump(mode="json")
        )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        llm_call=llm,
    )

    first = loop.run("pause me", stop_requested=lambda: next(stop_checks))
    assert first.outcome == LoopOutcome.STOPPED

    second = loop.run("pause me", context=first.context, stop_requested=lambda: next(stop_checks))
    assert second.outcome == LoopOutcome.TASK_COMPLETE
    assert second.message == "resumed"
    assert calls["n"] == 2
    assert len(second.context.turns) == 2


def test_loop_rejects_model_when_autoswitch_off(tmp_path: Path):
    calls = {"n": 0}

    def llm(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "action": "run_shell",
                    "command": "echo hi",
                    "model": "gpt-5.4-mini",
                }
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
        ui=ConversationUI(auto_model_switch=False),
        llm_call=llm,
    )
    result = loop.run("task")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert any("automatic model switching is off" in e for e in result.context.parse_errors)


def test_loop_rejects_reasoning_effort_when_autoswitch_off(tmp_path: Path):
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
        model="gpt-5.4-mini",
        ui=ConversationUI(auto_model_switch=False),
        llm_call=llm,
    )
    result = loop.run("task")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert any("reasoning_effort" in e and "automatic model switching is off" in e for e in result.context.parse_errors)


# ---------------------------------------------------------------------------
# Phase 2: previous_response_id chaining
# ---------------------------------------------------------------------------


def test_response_chain_sends_delta_and_previous_id_after_first_call(
    tmp_path: Path, monkeypatch
):
    calls: list[dict] = []
    payloads = [
        json.dumps(
            AgentStep(
                action=AgentAction.RUN_SHELL,
                command="echo chain-test",
            ).model_dump(mode="json")
        ),
        json.dumps(
            AgentStep(
                action=AgentAction.TASK_COMPLETE,
                message="done",
            ).model_dump(mode="json")
        ),
    ]

    def fake_with_id(**kwargs):
        calls.append(dict(kwargs))
        idx = len(calls) - 1
        return payloads[idx], f"resp_{idx}", None

    monkeypatch.setattr(
        "agent.loop.complete_structured_with_id", fake_with_id
    )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        # llm_call=None enables chaining when response_chain_enabled is True.
        response_chain_enabled=True,
    )
    result = loop.run("explore")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert len(calls) == 2

    # First call: full prompt, no previous_response_id.
    first = calls[0]
    assert "previous_response_id" not in first
    assert "## User task" in first["input"]

    # Second call: delta input, chained.
    second = calls[1]
    assert second["previous_response_id"] == "resp_0"
    # Delta drops the user_task header and only includes the new tool result.
    assert "## User task" not in second["input"]
    assert "chain-test" in second["input"]


def test_response_chain_recovers_when_server_drops_chain(
    tmp_path: Path, monkeypatch
):
    from llm import ResponseChainBroken

    calls: list[dict] = []
    payloads = [
        (
            json.dumps(
                AgentStep(
                    action=AgentAction.RUN_SHELL,
                    command="echo first",
                ).model_dump(mode="json")
            ),
            "resp_0",
        ),
        # Second call raises chain-broken; loop should recover and retry once.
        ResponseChainBroken("Previous response with id resp_0 not found"),
        (
            json.dumps(
                AgentStep(
                    action=AgentAction.TASK_COMPLETE,
                    message="recovered",
                ).model_dump(mode="json")
            ),
            "resp_2",
        ),
    ]

    def fake_with_id(**kwargs):
        calls.append(dict(kwargs))
        outcome = payloads[len(calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        text, rid = outcome
        return text, rid, None

    monkeypatch.setattr(
        "agent.loop.complete_structured_with_id", fake_with_id
    )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        response_chain_enabled=True,
    )
    result = loop.run("explore")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert result.message == "recovered"

    # Three underlying calls: first ok, second chain-broken, third recovery.
    assert len(calls) == 3
    # The recovery call drops previous_response_id and re-sends a full prompt.
    recovery = calls[2]
    assert "previous_response_id" not in recovery
    assert "## User task" in recovery["input"]


def test_response_chain_disabled_uses_full_prompt_each_turn(
    tmp_path: Path, monkeypatch
):
    """With chaining off, every call gets the full prompt and no previous id."""
    calls: list[dict] = []
    payloads = [
        json.dumps(
            AgentStep(
                action=AgentAction.RUN_SHELL,
                command="echo a",
            ).model_dump(mode="json")
        ),
        json.dumps(
            AgentStep(
                action=AgentAction.TASK_COMPLETE,
                message="ok",
            ).model_dump(mode="json")
        ),
    ]

    def fake_with_id(**kwargs):
        calls.append(dict(kwargs))
        return payloads[len(calls) - 1], f"resp_{len(calls) - 1}", None

    monkeypatch.setattr(
        "agent.loop.complete_structured_with_id", fake_with_id
    )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        response_chain_enabled=False,
    )
    result = loop.run("explore")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert len(calls) == 2
    for call in calls:
        assert "previous_response_id" not in call
        assert "## User task" in call["input"]


def test_response_chain_resets_on_switch_model(tmp_path: Path, monkeypatch):
    """Switching the model invalidates the server-side chain."""
    calls: list[dict] = []
    payloads = [
        json.dumps(
            AgentStep(
                action=AgentAction.SWITCH_MODEL,
                model="gpt-5.4-mini",
            ).model_dump(mode="json")
        ),
        json.dumps(
            AgentStep(
                action=AgentAction.TASK_COMPLETE,
                message="done",
            ).model_dump(mode="json")
        ),
    ]

    def fake_with_id(**kwargs):
        calls.append(dict(kwargs))
        return payloads[len(calls) - 1], f"resp_{len(calls) - 1}", None

    monkeypatch.setattr(
        "agent.loop.complete_structured_with_id", fake_with_id
    )

    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        ui=ConversationUI(auto_model_switch=True),
        response_chain_enabled=True,
    )
    result = loop.run("plan")
    assert result.outcome == LoopOutcome.TASK_COMPLETE
    assert len(calls) == 2
    # Second call after switch_model must NOT carry previous_response_id and
    # must re-send the full prompt.
    second = calls[1]
    assert "previous_response_id" not in second
    assert "## User task" in second["input"]
