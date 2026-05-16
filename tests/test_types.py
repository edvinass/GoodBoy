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


def test_switch_model_requires_model():
    with pytest.raises(ValidationError):
        AgentStep.model_validate({"action": "switch_model"})


def test_switch_model_with_reasoning_effort():
    step = AgentStep.model_validate(
        {
            "action": "switch_model",
            "model": "gpt-4o-mini",
            "reasoning_effort": "low",
        }
    )
    assert step.model == "gpt-4o-mini"
    assert step.reasoning_effort == "low"


def test_model_only_on_switch_model():
    with pytest.raises(ValidationError, match="switch_model"):
        AgentStep.model_validate(
            {
                "action": "run_shell",
                "command": "pwd",
                "model": "gpt-4o-mini",
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
