"""Structured agent protocol types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from llm import REASONING_EFFORT

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentAction(str, Enum):
    RUN_SHELL = "run_shell"
    RUN_PYTHON = "run_python"
    SWITCH_MODEL = "switch_model"
    SWITCH_TOOLS = "switch_tools"
    SWITCH_API = "switch_api"  # deprecated alias for switch_tools
    NEED_USER_INPUT = "need_user_input"
    TASK_COMPLETE = "task_complete"
    FAILED = "failed"


_SWITCH_TOOLS_ACTIONS = frozenset(
    {AgentAction.SWITCH_TOOLS, AgentAction.SWITCH_API}
)


class AgentStep(BaseModel):
    """Single turn from the LLM; must be valid JSON matching this schema."""

    action: AgentAction
    thought: str | None = None
    command: str | None = None
    code: str | None = None
    message: str | None = None
    tools: list[str] | None = None
    model: str | None = None
    reasoning_effort: str | None = None

    @field_validator(
        "command", "code", "message", "model", "reasoning_effort", mode="before"
    )
    @classmethod
    def _strip_optional_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or None
        return value

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

    @field_validator("reasoning_effort")
    @classmethod
    def _validate_reasoning_effort_value(cls, value: str | None) -> str | None:
        if value is not None and value not in REASONING_EFFORT:
            raise ValueError(
                f"reasoning_effort must be one of: {', '.join(REASONING_EFFORT)}"
            )
        return value

    @model_validator(mode="after")
    def _validate_action_fields(self) -> AgentStep:
        if self.action == AgentAction.RUN_SHELL:
            if not self.command:
                raise ValueError("run_shell requires non-empty 'command'")
        elif self.action == AgentAction.RUN_PYTHON:
            if not self.code:
                raise ValueError("run_python requires non-empty 'code'")
        elif self.action == AgentAction.SWITCH_MODEL:
            if not self.model:
                raise ValueError("switch_model requires non-empty 'model'")
        elif self.action in _SWITCH_TOOLS_ACTIONS:
            if not self.tools:
                raise ValueError("switch_tools requires non-empty 'tools' list")
        elif self.action in (
            AgentAction.NEED_USER_INPUT,
            AgentAction.TASK_COMPLETE,
            AgentAction.FAILED,
        ):
            if not self.message:
                raise ValueError(f"{self.action.value} requires non-empty 'message'")
        if self.model is not None and self.action in _SWITCH_TOOLS_ACTIONS:
            raise ValueError(
                "model is not allowed on switch_tools — set model on the next "
                "run_shell, run_python, switch_model, or terminal action"
            )
        if self.tools is not None and self.action not in _SWITCH_TOOLS_ACTIONS:
            raise ValueError(
                "tools is only allowed with switch_tools — use "
                '{"action": "switch_tools", "tools": ["web_search"]}'
            )
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


def parse_agent_step(raw: str) -> AgentStep:
    """Parse and validate JSON text into an AgentStep."""
    text = _strip_markdown_json_fence(raw.strip())
    data = _load_first_json_object(text)
    return AgentStep.model_validate(data)
