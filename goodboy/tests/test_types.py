"""Tests for agent protocol types."""

import json

import pytest
from pydantic import ValidationError

from agent.types import AgentAction, AgentStep, parse_agent_step


def test_parse_run_shell():
    raw = json.dumps(
        {
            "action": "run_shell",
            "status": "Listing project files",
            "thought": "list files",
            "command": "ls",
        }
    )
    step = parse_agent_step(raw)
    assert step.action == AgentAction.RUN_SHELL
    assert step.status == "Listing project files"
    assert step.command == "ls"


def test_parse_strips_markdown_fence():
    raw = """```json
{"action": "task_complete", "message": "done"}
```"""
    step = parse_agent_step(raw)
    assert step.action == AgentAction.TASK_COMPLETE
    assert step.message == "done"


def test_run_shell_requires_command():
    with pytest.raises(ValidationError):
        AgentStep.model_validate({"action": "run_shell"})


def test_need_user_input_requires_message():
    with pytest.raises(ValidationError):
        AgentStep.model_validate({"action": "need_user_input"})


def test_switch_model_requires_model():
    with pytest.raises(ValidationError):
        AgentStep.model_validate({"action": "switch_model"})


def test_switch_model_with_reasoning_effort():
    step = AgentStep.model_validate(
        {
            "action": "switch_model",
            "model": "gpt-5.4-nano",
            "reasoning_effort": "low",
        }
    )
    assert step.model == "gpt-5.4-nano"
    assert step.reasoning_effort == "low"


def test_model_on_run_shell():
    step = AgentStep.model_validate(
        {
            "action": "run_shell",
            "command": "pwd",
            "model": "gpt-5.4-nano",
        }
    )
    assert step.model == "gpt-5.4-nano"


def test_model_not_on_switch_tools():
    with pytest.raises(ValidationError, match="switch_tools"):
        AgentStep.model_validate(
            {
                "action": "switch_tools",
                "tools": ["web_search"],
                "model": "gpt-5.4-mini",
            }
        )


def test_invalid_reasoning_effort_rejected():
    with pytest.raises(ValidationError):
        AgentStep.model_validate(
            {
                "action": "task_complete",
                "message": "done",
                "reasoning_effort": "ultra",
            }
        )


def test_parse_switch_tools():
    raw = json.dumps({"action": "switch_tools", "tools": ["web_search"]})
    step = parse_agent_step(raw)
    assert step.action == AgentAction.SWITCH_TOOLS
    assert step.tools == ["web_search"]


def test_parse_switch_api_alias():
    raw = json.dumps({"action": "switch_api", "tools": ["web_search"]})
    step = parse_agent_step(raw)
    assert step.action == AgentAction.SWITCH_API
    assert step.tools == ["web_search"]


def test_switch_tools_requires_tools():
    with pytest.raises(ValidationError):
        AgentStep.model_validate({"action": "switch_tools"})


def test_switch_tools_rejects_invalid_tool():
    with pytest.raises(ValidationError):
        AgentStep.model_validate(
            {"action": "switch_tools", "tools": ["not_a_real_tool"]}
        )


def test_tools_only_on_switch_tools():
    with pytest.raises(ValidationError, match="switch_tools"):
        AgentStep.model_validate(
            {
                "action": "run_shell",
                "command": "pwd",
                "tools": ["web_search"],
            }
        )


def test_model_strips_whitespace():
    with pytest.raises(ValidationError):
        AgentStep.model_validate(
            {"action": "switch_model", "model": "  "}
        )


def test_parse_duplicate_json_objects_uses_first():
    """Models sometimes emit the same action twice in one response."""
    one = {
        "action": "run_shell",
        "thought": "list files",
        "command": "ls",
    }
    raw = json.dumps(one) + json.dumps(one)
    step = parse_agent_step(raw)
    assert step.action == AgentAction.RUN_SHELL
    assert step.command == "ls"


def test_update_plan_requires_plan_items():
    with pytest.raises(ValidationError):
        AgentStep.model_validate({"action": "update_plan"})


def test_update_plan_with_items():
    step = AgentStep.model_validate(
        {
            "action": "update_plan",
            "plan_items": [
                {"id": "1", "text": "recon", "status": "pending"},
            ],
        }
    )
    assert step.action == AgentAction.UPDATE_PLAN
    assert len(step.plan_items or []) == 1


def test_remember_requires_memory():
    with pytest.raises(ValidationError):
        AgentStep.model_validate({"action": "remember"})


def test_plan_items_only_on_update_plan():
    with pytest.raises(ValidationError, match="update_plan"):
        AgentStep.model_validate(
            {
                "action": "run_shell",
                "command": "pwd",
                "plan_items": [{"id": "1", "text": "x", "status": "pending"}],
            }
        )


def test_parse_concatenated_json_objects_uses_first_only():
    first = {"action": "run_shell", "command": "pwd"}
    second = {"action": "task_complete", "message": "done"}
    raw = json.dumps(first) + json.dumps(second)
    step = parse_agent_step(raw)
    assert step.action == AgentAction.RUN_SHELL
    assert step.command == "pwd"


def test_parse_all_agent_steps_returns_every_object():
    from agent.types import parse_all_agent_steps

    raw = json.dumps({"action": "run_shell", "command": "pwd"}) + json.dumps(
        {"action": "task_complete", "message": "done"}
    )
    steps = parse_all_agent_steps(raw)
    assert len(steps) == 2
    assert steps[0].action == AgentAction.RUN_SHELL
    assert steps[1].action == AgentAction.TASK_COMPLETE
