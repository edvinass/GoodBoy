"""Harness tool registry for the GoodBoy agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.types import AgentAction

HARNESS_ACTIONS = frozenset({AgentAction.RUN_SHELL, AgentAction.RUN_PYTHON})
ROUTING_ACTIONS = frozenset(
    {
        AgentAction.SWITCH_MODEL,
        AgentAction.SWITCH_TOOLS,
        AgentAction.SWITCH_API,
    }
)
TERMINAL_ACTIONS = frozenset(
    {
        AgentAction.NEED_USER_INPUT,
        AgentAction.TASK_COMPLETE,
        AgentAction.FAILED,
    }
)


@dataclass(frozen=True)
class ToolSpec:
    """Local harness tool the agent can invoke via action."""

    action: AgentAction
    name: str
    description: str
    when_to_use: str
    avoid_when: str
    layer: Literal["harness", "openai"] = "harness"


DEFAULT_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        action=AgentAction.RUN_SHELL,
        name="run_shell",
        description=(
            "Execute a shell command in the project workspace (full privileges; not sandboxed). "
            "Primary tool for navigating, searching, reading, creating, and editing files."
        ),
        when_to_use=(
            "ls/find/tree; rg/grep; cat/head; git; heredocs/sed/tee for edits; npm/pytest/cargo/go "
            "test; builds, linters, formatters; installing deps; any CLI the project uses."
        ),
        avoid_when="Heavy structured data processing is clearer in run_python.",
    ),
    ToolSpec(
        action=AgentAction.RUN_PYTHON,
        name="run_python",
        description="Execute Python via the same interpreter as the harness (workspace cwd).",
        when_to_use=(
            "Multi-file refactors, AST transforms, parsing build output, generating patches, "
            "or logic that is awkward in shell."
        ),
        avoid_when="A focused shell command (rg, sed, pytest, git) is enough.",
    ),
)

_TOOLS_BY_ACTION = {t.action: t for t in DEFAULT_TOOLS}


def get_tool(action: AgentAction) -> ToolSpec | None:
    return _TOOLS_BY_ACTION.get(action)


def is_harness_tool(action: AgentAction) -> bool:
    return action in HARNESS_ACTIONS


def is_routing_action(action: AgentAction) -> bool:
    return action in ROUTING_ACTIONS


def is_valid_action(action: AgentAction) -> bool:
    return (
        action in HARNESS_ACTIONS
        or action in ROUTING_ACTIONS
        or action in TERMINAL_ACTIONS
    )


def format_tools_section(tools: tuple[ToolSpec, ...] | None = None) -> str:
    """Build prompt section describing harness tools."""
    specs = tools if tools is not None else DEFAULT_TOOLS
    lines = [
        "## Harness tools (local codebase work)",
        "Set `action` to a tool below, or a terminal action (need_user_input, task_complete, failed). No read_file/write_file — use run_shell.",
        "",
    ]
    for spec in specs:
        lines.append(
            f"- **{spec.name}**: {spec.description} Best for: {spec.when_to_use} Avoid: {spec.avoid_when}"
        )
    return "\n".join(lines)
