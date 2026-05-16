"""Tests for agent protocol types."""

import json

import pytest
from pydantic import ValidationError

from agent.types import AgentAction, AgentStep, parse_agent_step


def test_parse_run_shell():
    raw = json.dumps(
        {
            "action": "run_shell",
            "thought": "list files",
            "command": "ls",
        }
    )
    step = parse_agent_step(raw)
    assert step.action == AgentAction.RUN_SHELL
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


def test_optional_model_and_reasoning_effort():
    step = AgentStep.model_validate(
        {
            "action": "run_shell",
            "command": "pwd",
            "model": "gpt-4o-mini",
            "reasoning_effort": "low",
        }
    )
    assert step.model == "gpt-4o-mini"
    assert step.reasoning_effort == "low"


def test_invalid_reasoning_effort_rejected():
    with pytest.raises(ValidationError):
        AgentStep.model_validate(
            {
                "action": "task_complete",
                "message": "done",
                "reasoning_effort": "ultra",
            }
        )


def test_parse_switch_api():
    raw = json.dumps({"action": "switch_api", "tools": ["web_search"]})
    step = parse_agent_step(raw)
    assert step.action == AgentAction.SWITCH_API
    assert step.tools == ["web_search"]


def test_switch_api_requires_tools():
    with pytest.raises(ValidationError):
        AgentStep.model_validate({"action": "switch_api"})


def test_switch_api_rejects_invalid_tool():
    with pytest.raises(ValidationError):
        AgentStep.model_validate(
            {"action": "switch_api", "tools": ["not_a_real_tool"]}
        )


def test_model_strips_whitespace():
    step = AgentStep.model_validate(
        {"action": "task_complete", "message": "done", "model": "  "}
    )
    assert step.model is None
