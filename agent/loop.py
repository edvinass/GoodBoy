"""Core agent orchestration loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from agent.clarifications import (
    count_need_user_input_turns,
    is_affirmative_reply,
    is_repeat_question,
    last_need_user_input_message,
    proceed_directive,
    repeat_question_directive,
)
from agent.context import SessionContext
from agent.prompt import SYSTEM_PROMPT
from agent.tools import run_python, run_shell
from agent.ui import ConversationUI, tool_activity
from agent.types import (
    AgentAction,
    AgentStep,
    TurnRecord,
    parse_agent_step,
)
from llm import complete_structured
from settings import get_settings

LLMCall = Callable[..., str]
AskUser = Callable[[], str]


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
        ui: ConversationUI | None = None,
        max_clarifications: int | None = None,
    ) -> None:
        cfg = get_settings()
        self.workspace = workspace or Path.cwd()
        self.max_turns = max_turns if max_turns is not None else cfg.max_turns
        self.tool_timeout = (
            tool_timeout if tool_timeout is not None else cfg.tool_timeout_sec
        )
        self.max_clarifications = (
            max_clarifications
            if max_clarifications is not None
            else cfg.max_clarifications
        )
        self.model = model
        self._llm_call = llm_call or complete_structured
        self._ui = ui

    def run(
        self,
        task: str,
        *,
        context: SessionContext | None = None,
        ask_user: AskUser | None = None,
    ) -> LoopResult:
        ctx = context or SessionContext(user_task=task)
        json_schema = AgentStep.model_json_schema()
        consecutive_parse_failures = 0

        for turn in range(1, self.max_turns + 1):
            if self._ui is not None:
                with self._ui.thinking():
                    raw = self._llm_call(
                        input=ctx.to_prompt(),
                        instructions=SYSTEM_PROMPT,
                        json_schema=json_schema,
                        model=self.model,
                    )
            else:
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

            if self._ui is not None:
                self._ui.print_agent_step(step)

            if step.action == AgentAction.NEED_USER_INPUT:
                question = step.message or ""

                if count_need_user_input_turns(ctx.turns) >= self.max_clarifications:
                    return LoopResult(
                        outcome=LoopOutcome.FAILED,
                        message=(
                            f"Agent asked for input too many times "
                            f"({self.max_clarifications}). "
                            "Try a more specific task or run again."
                        ),
                        context=ctx,
                    )

                prior = last_need_user_input_message(ctx.turns)
                if prior and ctx.user_replies and is_repeat_question(prior, question):
                    ctx.add_parse_error(repeat_question_directive())
                    ctx.add_turn(TurnRecord(turn=turn, step=step))
                    continue

                ctx.add_turn(TurnRecord(turn=turn, step=step))

                if ask_user is None:
                    return LoopResult(
                        outcome=LoopOutcome.NEED_USER_INPUT,
                        message=question,
                        context=ctx,
                    )

                reply = ask_user().strip()
                if not reply:
                    return LoopResult(
                        outcome=LoopOutcome.FAILED,
                        message="No reply provided.",
                        context=ctx,
                    )
                ctx.add_user_reply(reply)
                if is_affirmative_reply(reply):
                    ctx.add_parse_error(proceed_directive(reply))
                continue

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
                if self._ui is not None:
                    with tool_activity(self._ui, "shell"):
                        tool_result = run_shell(
                            step.command or "",
                            cwd=self.workspace,
                            timeout=self.tool_timeout,
                        )
                else:
                    tool_result = run_shell(
                        step.command or "",
                        cwd=self.workspace,
                        timeout=self.tool_timeout,
                    )
            elif step.action == AgentAction.RUN_PYTHON:
                if self._ui is not None:
                    with tool_activity(self._ui, "python"):
                        tool_result = run_python(
                            step.code or "",
                            cwd=self.workspace,
                            timeout=self.tool_timeout,
                        )
                else:
                    tool_result = run_python(
                        step.code or "",
                        cwd=self.workspace,
                        timeout=self.tool_timeout,
                    )

            if self._ui is not None and tool_result is not None:
                self._ui.print_tool_result(tool_result)

            ctx.add_turn(
                TurnRecord(turn=turn, step=step, tool_result=tool_result)
            )

        return LoopResult(
            outcome=LoopOutcome.MAX_TURNS,
            message=f"Exceeded maximum turns ({self.max_turns}).",
            context=ctx,
        )
