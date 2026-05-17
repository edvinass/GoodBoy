"""Harness tool registry for the GoodBoy agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.types import AgentAction

HARNESS_ACTIONS = frozenset(
    {
        AgentAction.RUN_SHELL,
        AgentAction.RUN_PYTHON,
        AgentAction.READ_FILE,
        AgentAction.APPLY_PATCH,
        AgentAction.STR_REPLACE,
    }
)
ROUTING_ACTIONS = frozenset(
    {
        AgentAction.SWITCH_MODEL,
        AgentAction.SWITCH_TOOLS,
        AgentAction.SWITCH_API,
    }
)
META_ACTIONS = frozenset(
    {
        AgentAction.UPDATE_PLAN,
        AgentAction.REMEMBER,
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
    ToolSpec(
        action=AgentAction.READ_FILE,
        name="read_file",
        description="Read a workspace file with optional line range (numbered output).",
        when_to_use="Inspect source before editing; read specific functions or config sections.",
        avoid_when="You only need a quick grep — use run_shell + rg.",
    ),
    ToolSpec(
        action=AgentAction.STR_REPLACE,
        name="str_replace",
        description="Replace one exact unique occurrence of old_string with new_string in a file.",
        when_to_use="Small, precise edits when the old text is unique in the file.",
        avoid_when="The match is ambiguous, spans many lines, or needs a multi-hunk diff — use apply_patch.",
    ),
    ToolSpec(
        action=AgentAction.APPLY_PATCH,
        name="apply_patch",
        description="Apply a unified diff patch to a workspace file via patch(1).",
        when_to_use="Multi-line edits, refactors, or changes best expressed as a unified diff.",
        avoid_when="A one-line str_replace is enough.",
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
        or action in META_ACTIONS
        or action in TERMINAL_ACTIONS
    )


def format_tools_section(tools: tuple[ToolSpec, ...] | None = None) -> str:
    """Build prompt section describing harness tools."""
    specs = tools if tools is not None else DEFAULT_TOOLS
    lines = [
        "## Harness tools (local codebase work)",
        "Set `action` to a tool below, meta action (update_plan, remember), or terminal action "
        "(need_user_input, task_complete, failed).",
        "",
    ]
    for spec in specs:
        lines.append(
            f"- **{spec.name}**: {spec.description} Best for: {spec.when_to_use} Avoid: {spec.avoid_when}"
        )
    return "\n".join(lines)
