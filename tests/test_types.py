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
