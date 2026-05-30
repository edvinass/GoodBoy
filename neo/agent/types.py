"""Structured agent protocol types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator, model_validator


class PlanItemStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


class PlanItem(BaseModel):
    """One row in the durable task plan."""

    id: str
    text: str
    status: PlanItemStatus = PlanItemStatus.PENDING


class AgentAction(str, Enum):
    RUN_SHELL = "run_shell"
    RUN_PYTHON = "run_python"
    READ_FILE = "read_file"
    SEARCH_CODE = "search_code"
    LIST_FILES = "list_files"
    GIT = "git"
    APPLY_PATCH = "apply_patch"
    STR_REPLACE = "str_replace"
    DELETE_FILE = "delete_file"
    MOVE_FILE = "move_file"
    SWITCH_TOOLS = "switch_tools"
    SWITCH_API = "switch_api"  # deprecated alias for switch_tools
    UPDATE_PLAN = "update_plan"
    REMEMBER = "remember"
    NEED_USER_INPUT = "need_user_input"
    TASK_COMPLETE = "task_complete"
    FAILED = "failed"


_SWITCH_TOOLS_ACTIONS = frozenset(
    {AgentAction.SWITCH_TOOLS, AgentAction.SWITCH_API}
)


class AgentStep(BaseModel):
    """Single turn from the LLM; must be valid JSON matching this schema."""

    action: AgentAction
    status: str | None = None
    thought: str | None = None
    command: str | None = None
    code: str | None = None
    path: str | None = None
    dest_path: str | None = None
    pattern: str | None = None
    glob: str | None = None
    case_insensitive: bool | None = None
    max_results: int | None = None
    max_depth: int | None = None
    git_op: str | None = None
    staged: bool | None = None
    patch: str | None = None
    old_string: str | None = None
    new_string: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    message: str | None = None
    tools: list[str] | None = None
    plan_items: list[PlanItem] | None = None
    memory: list[str] | None = None

    @field_validator(
        "status",
        "command",
        "code",
        "path",
        "message",
        "pattern",
        "glob",
        "git_op",
        "dest_path",
        mode="before",
    )
    @classmethod
    def _strip_optional_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("old_string", "new_string", "patch", mode="before")
    @classmethod
    def _preserve_edit_strings(cls, value: Any) -> Any:
        # Edit payloads must keep leading/trailing whitespace (Python indentation).
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_tools(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return cleaned or None
        return value

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        from agent.models import OpenAITool

        for tool in value:
            OpenAITool(tool)
        return value

    @model_validator(mode="after")
    def _validate_action_fields(self) -> AgentStep:
        if self.action == AgentAction.RUN_SHELL:
            if not self.command:
                raise ValueError("run_shell requires non-empty 'command'")
        elif self.action == AgentAction.RUN_PYTHON:
            if not self.code:
                raise ValueError("run_python requires non-empty 'code'")
        elif self.action == AgentAction.READ_FILE:
            if not self.path:
                raise ValueError("read_file requires non-empty 'path'")
        elif self.action == AgentAction.SEARCH_CODE:
            if not self.pattern:
                raise ValueError("search_code requires non-empty 'pattern'")
        elif self.action == AgentAction.LIST_FILES:
            pass
        elif self.action == AgentAction.GIT:
            if not self.git_op:
                raise ValueError("git requires non-empty 'git_op' (status, diff, or log)")
            op = self.git_op.strip().lower()
            if op not in ("status", "diff", "log"):
                raise ValueError("git_op must be status, diff, or log")
            self.git_op = op
        elif self.action == AgentAction.DELETE_FILE:
            if not self.path:
                raise ValueError("delete_file requires non-empty 'path'")
        elif self.action == AgentAction.MOVE_FILE:
            if not self.path:
                raise ValueError("move_file requires non-empty 'path'")
            if not self.dest_path:
                raise ValueError("move_file requires non-empty 'dest_path'")
        elif self.action == AgentAction.APPLY_PATCH:
            if not self.path:
                raise ValueError("apply_patch requires non-empty 'path'")
            if not self.patch:
                raise ValueError("apply_patch requires non-empty 'patch'")
        elif self.action == AgentAction.STR_REPLACE:
            if not self.path:
                raise ValueError("str_replace requires non-empty 'path'")
            if self.old_string is None or self.old_string == "":
                raise ValueError("str_replace requires non-empty 'old_string'")
            if self.new_string is None:
                self.new_string = ""
        elif self.action in _SWITCH_TOOLS_ACTIONS:
            if not self.tools:
                raise ValueError("switch_tools requires non-empty 'tools' list")
        elif self.action == AgentAction.UPDATE_PLAN:
            if not self.plan_items:
                raise ValueError("update_plan requires non-empty 'plan_items' list")
        elif self.action == AgentAction.REMEMBER:
            if not self.memory:
                raise ValueError("remember requires non-empty 'memory' list")
            if len(self.memory) > 5:
                raise ValueError("remember allows at most 5 memory strings per turn")
            for item in self.memory:
                if len(item) > 500:
                    raise ValueError(
                        "each memory string must be at most 500 characters"
                    )
        elif self.action in (
            AgentAction.NEED_USER_INPUT,
            AgentAction.TASK_COMPLETE,
            AgentAction.FAILED,
        ):
            if not self.message:
                raise ValueError(f"{self.action.value} requires non-empty 'message'")
        if self.tools is not None and self.action not in _SWITCH_TOOLS_ACTIONS:
            raise ValueError(
                "tools is only allowed with switch_tools — use "
                '{"action": "switch_tools", "tools": ["web_search"]}'
            )
        if self.plan_items is not None and self.action != AgentAction.UPDATE_PLAN:
            raise ValueError("plan_items is only allowed with update_plan")
        if self.memory is not None and self.action != AgentAction.REMEMBER:
            raise ValueError("memory is only allowed with remember")
        if self.pattern is not None and self.action != AgentAction.SEARCH_CODE:
            raise ValueError("pattern is only allowed with search_code")
        if self.glob is not None and self.action not in (
            AgentAction.SEARCH_CODE,
            AgentAction.LIST_FILES,
        ):
            raise ValueError("glob is only allowed with search_code or list_files")
        if self.case_insensitive is not None and self.action != AgentAction.SEARCH_CODE:
            raise ValueError("case_insensitive is only allowed with search_code")
        if self.max_depth is not None and self.action != AgentAction.LIST_FILES:
            raise ValueError("max_depth is only allowed with list_files")
        if self.max_results is not None and self.action not in (
            AgentAction.SEARCH_CODE,
            AgentAction.LIST_FILES,
            AgentAction.GIT,
        ):
            raise ValueError(
                "max_results is only allowed with search_code, list_files, or git"
            )
        if self.git_op is not None and self.action != AgentAction.GIT:
            raise ValueError("git_op is only allowed with git")
        if self.staged is not None and self.action != AgentAction.GIT:
            raise ValueError("staged is only allowed with git")
        if self.dest_path is not None and self.action != AgentAction.MOVE_FILE:
            raise ValueError("dest_path is only allowed with move_file")
        return self


class ToolResult(BaseModel):
    """Captured output from a shell or Python tool execution."""

    executed: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False


class TurnRecord(BaseModel):
    """One harness turn: agent step plus optional tool result."""

    turn: int
    step: AgentStep
    tool_result: ToolResult | None = None
    parse_error: str | None = None
    call_model: str | None = None
    call_reasoning_effort: str | None = None


def _strip_markdown_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _load_first_json_object(text: str) -> Any:
    """Decode the first JSON value; ignore trailing concatenated objects."""
    import json

    decoder = json.JSONDecoder()
    try:
        data, _end = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    return data


def parse_all_agent_steps(raw: str) -> list[AgentStep]:
    """Parse every JSON action object in a (possibly concatenated or array) response."""
    import json

    text = _strip_markdown_json_fence(raw.strip())
    if not text:
        return []
    # Accept a top-level JSON array of step objects.
    if text.lstrip().startswith('['):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            pass  # Fall through to concatenated-object parser
        else:
            if isinstance(data, list):
                return [AgentStep.model_validate(item) for item in data]
    decoder = json.JSONDecoder()
    steps: list[AgentStep] = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        data, end = decoder.raw_decode(text, pos)
        steps.append(AgentStep.model_validate(data))
        pos = end
    return steps


def parse_agent_step(raw: str) -> AgentStep:
    """Parse and validate JSON text into an AgentStep."""
    steps = parse_all_agent_steps(raw)
    if not steps:
        raise ValueError("empty model response")
    return steps[0]
