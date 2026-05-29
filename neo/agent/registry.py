"""Harness tool registry for the Neo agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.types import AgentAction

HARNESS_ACTIONS = frozenset(
    {
        AgentAction.RUN_SHELL,
        AgentAction.RUN_PYTHON,
        AgentAction.READ_FILE,
        AgentAction.SEARCH_CODE,
        AgentAction.LIST_FILES,
        AgentAction.GIT,
        AgentAction.APPLY_PATCH,
        AgentAction.STR_REPLACE,
        AgentAction.DELETE_FILE,
        AgentAction.MOVE_FILE,
    }
)
# Read-only, side-effect-free actions safe to dispatch concurrently when the
# model emits multiple JSON objects in a single response.
BATCHABLE_READ_ACTIONS = frozenset(
    {
        AgentAction.READ_FILE,
        AgentAction.SEARCH_CODE,
        AgentAction.LIST_FILES,
        AgentAction.GIT,
    }
)
ROUTING_ACTIONS = frozenset(
    {
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
        avoid_when="You only need match locations — use search_code.",
    ),
    ToolSpec(
        action=AgentAction.SEARCH_CODE,
        name="search_code",
        description=(
            "Search the codebase with ripgrep; returns path:line:content rows "
            "(capped, structured output)."
        ),
        when_to_use=(
            "Find symbols, strings, imports, or usages across the repo. "
            "Set pattern (required); optional path, glob, case_insensitive, max_results."
        ),
        avoid_when="You already know the file path — use read_file.",
    ),
    ToolSpec(
        action=AgentAction.LIST_FILES,
        name="list_files",
        description="List files under a workspace path with optional glob and depth limits.",
        when_to_use=(
            "Discover layout, find configs, or enumerate files matching a glob. "
            "Optional path (default .), glob, max_depth, max_results."
        ),
        avoid_when="You need file contents — use read_file.",
    ),
    ToolSpec(
        action=AgentAction.GIT,
        name="git",
        description="Read-only git: status, diff, or oneline log (bounded output).",
        when_to_use=(
            "Check branch/status, review diffs (git_op diff; staged true for index), "
            "or recent commits (git_op log; max_results for count)."
        ),
        avoid_when="Git needs write operations — use run_shell.",
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
    ToolSpec(
        action=AgentAction.DELETE_FILE,
        name="delete_file",
        description="Delete a single file under the workspace (not directories).",
        when_to_use="Remove a file the user asked to delete or that is obsolete after a refactor.",
        avoid_when="Removing a directory tree — use run_shell.",
    ),
    ToolSpec(
        action=AgentAction.MOVE_FILE,
        name="move_file",
        description="Move or rename a file within the workspace (path → dest_path).",
        when_to_use="Rename modules or relocate a file without rewriting contents.",
        avoid_when="Many files move — use run_shell git mv or a script.",
    ),
)

_TOOLS_BY_ACTION = {t.action: t for t in DEFAULT_TOOLS}


def get_tool(action: AgentAction) -> ToolSpec | None:
    return _TOOLS_BY_ACTION.get(action)


def is_harness_tool(action: AgentAction) -> bool:
    return action in HARNESS_ACTIONS


def is_batchable_read(action: AgentAction) -> bool:
    """Whether this action can run alongside other reads in one parallel batch."""
    return action in BATCHABLE_READ_ACTIONS


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
