"""Harness tool registry for the GoodBoy agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.types import AgentAction

HARNESS_ACTIONS = frozenset({AgentAction.RUN_SHELL, AgentAction.RUN_PYTHON})
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
        description="Execute a shell command on the user's machine (full privileges; not sandboxed).",
        when_to_use="shell CLIs, package managers, test runners, file ops, pipelines (&&, pipes).",
        avoid_when="Complex data transforms are easier in Python.",
    ),
    ToolSpec(
        action=AgentAction.RUN_PYTHON,
        name="run_python",
        description="Execute Python via the same interpreter as the harness.",
        when_to_use="Parsing, algorithms, one-off scripts, structured manipulation.",
        avoid_when="A simple single shell command suffices.",
    ),
)

_TOOLS_BY_ACTION = {t.action: t for t in DEFAULT_TOOLS}


def get_tool(action: AgentAction) -> ToolSpec | None:
    return _TOOLS_BY_ACTION.get(action)


def is_harness_tool(action: AgentAction) -> bool:
    return action in HARNESS_ACTIONS


def is_valid_action(action: AgentAction) -> bool:
    return action in HARNESS_ACTIONS or action in TERMINAL_ACTIONS


def format_tools_section(tools: tuple[ToolSpec, ...] | None = None) -> str:
    """Build prompt section describing harness tools."""
    specs = tools if tools is not None else DEFAULT_TOOLS
    lines = [
        "## Harness tools (pick action for this turn's work)",
        "Set action to one of the tool IDs below, or a terminal action "
        "(need_user_input, task_complete, failed).",
        "",
    ]
    for spec in specs:
        lines.append(f"- **{spec.name}**: {spec.description}")
        lines.append(f"  - Best for: {spec.when_to_use}")
        lines.append(f"  - Avoid when: {spec.avoid_when}")
    lines.append("")
    lines.append(
        "Terminal actions: need_user_input, task_complete, failed "
        "(not executed as tools)."
    )
    return "\n".join(lines)
