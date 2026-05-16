"""Core agent orchestration loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from agent.context import SessionContext
from agent.prompt import SYSTEM_PROMPT
from agent.tools import run_python, run_shell
from agent.types import (
    AgentAction,
    AgentStep,
    TurnRecord,
    parse_agent_step,
)
from llm import complete_structured
from settings import get_settings

LLMCall = Callable[..., str]


class LoopOutcome(str, Enum):
    TASK_COMPLETE = "task_complete"
    NEED_USER_INPUT = "need_user_input"
    FAILED = "failed"
    MAX_TURNS = "max_turns"


@dataclass(frozen=True)
class LoopResult:
    outcome: LoopOutcome
    message: str
    context: SessionContext


class AgentLoop:
    """Run the turn-based agent loop until a terminal action or max turns."""

    def __init__(
        self,
        *,
        workspace: Path | None = None,
        max_turns: int | None = None,
        tool_timeout: float | None = None,
        model: str | None = None,
        llm_call: LLMCall | None = None,
    ) -> None:
        cfg = get_settings()
        self.workspace = workspace or Path.cwd()
        self.max_turns = max_turns if max_turns is not None else cfg.max_turns
        self.tool_timeout = (
            tool_timeout if tool_timeout is not None else cfg.tool_timeout_sec
        )
        self.model = model
        self._llm_call = llm_call or complete_structured

    def run(self, task: str, *, context: SessionContext | None = None) -> LoopResult:
        ctx = context or SessionContext(user_task=task)
        json_schema = AgentStep.model_json_schema()
        consecutive_parse_failures = 0

        for turn in range(1, self.max_turns + 1):
            raw = self._llm_call(
                input=ctx.to_prompt(),
                instructions=SYSTEM_PROMPT,
                json_schema=json_schema,
                model=self.model,
            )

            try:
                step = parse_agent_step(raw)
                consecutive_parse_failures = 0
            except Exception as exc:
                err = f"Turn {turn}: invalid JSON — {exc}"
                ctx.add_parse_error(err)
                consecutive_parse_failures += 1
                if consecutive_parse_failures >= 2:
                    return LoopResult(
                        outcome=LoopOutcome.FAILED,
                        message=f"Agent returned invalid JSON twice: {exc}",
                        context=ctx,
                    )
                continue

            if step.action == AgentAction.NEED_USER_INPUT:
                ctx.add_turn(TurnRecord(turn=turn, step=step))
                return LoopResult(
                    outcome=LoopOutcome.NEED_USER_INPUT,
                    message=step.message or "",
                    context=ctx,
                )

            if step.action == AgentAction.TASK_COMPLETE:
                ctx.add_turn(TurnRecord(turn=turn, step=step))
                return LoopResult(
                    outcome=LoopOutcome.TASK_COMPLETE,
                    message=step.message or "",
                    context=ctx,
                )

            if step.action == AgentAction.FAILED:
                ctx.add_turn(TurnRecord(turn=turn, step=step))
                return LoopResult(
                    outcome=LoopOutcome.FAILED,
                    message=step.message or "",
                    context=ctx,
                )

            tool_result = None
            if step.action == AgentAction.RUN_SHELL:
                tool_result = run_shell(
                    step.command or "",
                    cwd=self.workspace,
                    timeout=self.tool_timeout,
                )
            elif step.action == AgentAction.RUN_PYTHON:
                tool_result = run_python(
                    step.code or "",
                    cwd=self.workspace,
                    timeout=self.tool_timeout,
                )

            ctx.add_turn(
                TurnRecord(turn=turn, step=step, tool_result=tool_result)
            )

        return LoopResult(
            outcome=LoopOutcome.MAX_TURNS,
            message=f"Exceeded maximum turns ({self.max_turns}).",
            context=ctx,
        )
