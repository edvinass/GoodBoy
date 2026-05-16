"""Tests for harness tool registry."""

from agent.registry import (
    DEFAULT_TOOLS,
    format_tools_section,
    get_tool,
    is_harness_tool,
    is_valid_action,
)
from agent.types import AgentAction


def test_default_tools_include_shell_and_python():
    actions = {t.action for t in DEFAULT_TOOLS}
    assert AgentAction.RUN_SHELL in actions
    assert AgentAction.RUN_PYTHON in actions


def test_format_tools_section_lists_tools():
    text = format_tools_section()
    assert "run_shell" in text
    assert "run_python" in text
    assert "Best for" in text


def test_get_tool_and_harness_checks():
    assert get_tool(AgentAction.RUN_SHELL) is not None
    assert get_tool(AgentAction.TASK_COMPLETE) is None
    assert is_harness_tool(AgentAction.RUN_PYTHON)
    assert not is_harness_tool(AgentAction.FAILED)
    assert is_valid_action(AgentAction.NEED_USER_INPUT)
