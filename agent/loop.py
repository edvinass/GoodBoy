"""Core agent orchestration loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from agent.clarifications import (
    count_matching_shell_commands,
    count_need_user_input_turns,
    is_repeat_question,
    last_need_user_input_message,
    proceed_directive,
    repeat_question_directive,
    should_add_proceed_directive,
    stuck_shell_directive,
)
from agent.session_log import SessionLog
from agent.workspace import resolve_workspace
from agent.context import SessionContext
from agent.models import (
    cheapest_model_with_tools,
    model_supports_openai_tool,
    resolve_reasoning_effort,
    validate_reasoning_effort_for_model,
)
from agent.prompt import build_system_prompt
from agent.registry import get_tool, is_harness_tool, is_valid_action
from agent.tools import run_python, run_shell
from agent.ui import ConversationUI, tool_activity
from agent.types import (
    AgentAction,
    AgentStep,
    TurnRecord,
    _SWITCH_TOOLS_ACTIONS,
    parse_agent_step,
)
from openai import APIConnectionError

from llm import complete_structured, format_api_connection_error, get_curated_models
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
        allowed_models: list[str] | None = None,
        llm_call: LLMCall | None = None,
        ui: ConversationUI | None = None,
        max_clarifications: int | None = None,
    ) -> None:
        cfg = get_settings()
        self.workspace = resolve_workspace(workspace or Path.cwd())
        self.max_turns = max_turns if max_turns is not None else cfg.max_turns
        self.tool_timeout = (
            tool_timeout if tool_timeout is not None else cfg.tool_timeout_sec
        )
        self.max_clarifications = (
            max_clarifications
            if max_clarifications is not None
            else cfg.max_clarifications
        )
        self._default_model = model or cfg.default_model
        self._allowed_models = allowed_models or get_curated_models()
        self._pending_model: str | None = None
        self._pending_reasoning: str | None = None
        self._hosted_tools: tuple[str, ...] = ()
        self._llm_call = llm_call or complete_structured
        self._ui = ui
        debug_mode = (
            (ui.show_commands or ui.debug) if ui is not None else False
        )
        self._instructions = build_system_prompt(
            allowed_models=self._allowed_models,
            debug=debug_mode,
        )

    def run(
        self,
        task: str,
        *,
        context: SessionContext | None = None,
        ask_user: AskUser | None = None,
        session_log: SessionLog | None = None,
    ) -> LoopResult:
        ctx = context or SessionContext(
            user_task=task,
            workspace=str(self.workspace),
        )
        if ctx.workspace is None:
            ctx = ctx.model_copy(update={"workspace": str(self.workspace)})
        if context is not None and ctx.active_hosted_tools:
            self._hosted_tools = tuple(ctx.active_hosted_tools)
        if session_log is not None:
            session_log.event(
                "task_start",
                task=task,
                workspace=str(self.workspace),
            )
        json_schema = AgentStep.model_json_schema()
        consecutive_parse_failures = 0

        def finish(result: LoopResult) -> LoopResult:
            if session_log is not None:
                session_log.event(
                    "task_end",
                    outcome=result.outcome.value,
                    message=result.message,
                    turns=len(result.context.turns),
                )
            return result

        for turn in range(1, self.max_turns + 1):
            call_model = self._pending_model or self._default_model
            call_reasoning = resolve_reasoning_effort(
                call_model, self._pending_reasoning
            )
            self._pending_model = None
            self._pending_reasoning = None

            llm_kwargs = {
                "input": ctx.to_prompt(),
                "instructions": self._instructions,
                "json_schema": json_schema,
                "model": call_model,
                "reasoning_effort": call_reasoning,
            }
            if self._hosted_tools:
                llm_kwargs["tools"] = list(self._hosted_tools)

            if session_log is not None:
                session_log.log_llm_request(
                    turn=turn,
                    model=call_model,
                    reasoning_effort=call_reasoning,
                    instructions=self._instructions,
                    input_text=llm_kwargs["input"],
                )

            try:
                if self._ui is not None:
                    self._ui.print_llm_request(
                        turn=turn,
                        model=call_model,
                        reasoning_effort=call_reasoning,
                        instructions=self._instructions,
                        input_text=llm_kwargs["input"],
                    )
                    with self._ui.thinking():
                        raw = self._llm_call(**llm_kwargs)
                    self._ui.print_llm_response(turn=turn, raw=raw)
                else:
                    raw = self._llm_call(**llm_kwargs)
            except APIConnectionError as exc:
                return finish(
                    LoopResult(
                        outcome=LoopOutcome.FAILED,
                        message=format_api_connection_error(exc),
                        context=ctx,
                    )
                )

            if session_log is not None:
                session_log.log_llm_response(turn=turn, raw=raw)

            try:
                step = parse_agent_step(raw)
                consecutive_parse_failures = 0
            except Exception as exc:
                err = f"Turn {turn}: invalid JSON — {exc}"
                ctx.add_parse_error(err)
                if session_log is not None:
                    session_log.event("parse_error", turn=turn, error=err, raw=raw)
                consecutive_parse_failures += 1
                if consecutive_parse_failures >= 2:
                    return finish(
                        LoopResult(
                            outcome=LoopOutcome.FAILED,
                            message=f"Agent returned invalid JSON twice: {exc}",
                            context=ctx,
                        )
                    )
                continue

            routing_err = self._validate_routing(step, call_model)
            if routing_err:
                ctx.add_parse_error(routing_err)
                if session_log is not None:
                    session_log.event("parse_error", turn=turn, error=routing_err)
                consecutive_parse_failures += 1
                if consecutive_parse_failures >= 2:
                    return finish(
                        LoopResult(
                            outcome=LoopOutcome.FAILED,
                            message=routing_err,
                            context=ctx,
                        )
                    )
                continue

            consecutive_parse_failures = 0
            self._apply_pending_routing(step)

            if session_log is not None:
                session_log.log_agent_step(turn=turn, step=step)

            if self._ui is not None:
                self._ui.print_agent_step(
                    step,
                    model=call_model,
                    reasoning=call_reasoning,
                    next_model=self._pending_model,
                    next_reasoning=self._pending_reasoning,
                    hosted_tools=list(self._hosted_tools) or None,
                )

            if step.action == AgentAction.SWITCH_MODEL:
                ctx.add_turn(
                    TurnRecord(
                        turn=turn,
                        step=step,
                        call_model=call_model,
                        call_reasoning_effort=call_reasoning,
                    )
                )
                continue

            if step.action in _SWITCH_TOOLS_ACTIONS:
                tools_err = self._apply_switch_tools(step, call_model)
                if tools_err:
                    ctx.add_parse_error(tools_err)
                    if session_log is not None:
                        session_log.event("parse_error", turn=turn, error=tools_err)
                    consecutive_parse_failures += 1
                    if consecutive_parse_failures >= 2:
                        return finish(
                            LoopResult(
                                outcome=LoopOutcome.FAILED,
                                message=tools_err,
                                context=ctx,
                            )
                        )
                    ctx.add_turn(
                        TurnRecord(
                            turn=turn,
                            step=step,
                            call_model=call_model,
                            call_reasoning_effort=call_reasoning,
                        )
                    )
                    continue

                consecutive_parse_failures = 0
                ctx = ctx.model_copy(
                    update={"active_hosted_tools": list(self._hosted_tools)}
                )
                ctx.add_turn(
                    TurnRecord(
                        turn=turn,
                        step=step,
                        call_model=call_model,
                        call_reasoning_effort=call_reasoning,
                    )
                )
                continue

            if step.action == AgentAction.NEED_USER_INPUT:
                question = step.message or ""

                if count_need_user_input_turns(ctx.turns) >= self.max_clarifications:
                    return finish(
                        LoopResult(
                            outcome=LoopOutcome.FAILED,
                            message=(
                                f"Agent asked for input too many times "
                                f"({self.max_clarifications}). "
                                "Try a more specific task or run again."
                            ),
                            context=ctx,
                        )
                    )

                prior = last_need_user_input_message(ctx.turns)
                if prior and ctx.user_replies and is_repeat_question(prior, question):
                    ctx.add_parse_error(repeat_question_directive())
                    ctx.add_turn(
                        TurnRecord(
                            turn=turn,
                            step=step,
                            call_model=call_model,
                            call_reasoning_effort=call_reasoning,
                        )
                    )
                    continue

                ctx.add_turn(
                    TurnRecord(
                        turn=turn,
                        step=step,
                        call_model=call_model,
                        call_reasoning_effort=call_reasoning,
                    )
                )

                if ask_user is None:
                    return finish(
                        LoopResult(
                            outcome=LoopOutcome.NEED_USER_INPUT,
                            message=question,
                            context=ctx,
                        )
                    )

                reply = ask_user().strip()
                if session_log is not None:
                    session_log.event("user_reply", turn=turn, reply=reply)
                if not reply:
                    return finish(
                        LoopResult(
                            outcome=LoopOutcome.FAILED,
                            message="No reply provided.",
                            context=ctx,
                        )
                    )
                ctx.add_user_reply(reply)
                if should_add_proceed_directive(reply):
                    directive = proceed_directive(reply)
                    ctx.add_parse_error(directive)
                    if session_log is not None:
                        session_log.event("harness_directive", turn=turn, text=directive)
                continue

            if step.action == AgentAction.TASK_COMPLETE:
                ctx.add_turn(
                    TurnRecord(
                        turn=turn,
                        step=step,
                        call_model=call_model,
                        call_reasoning_effort=call_reasoning,
                    )
                )
                return finish(
                    LoopResult(
                        outcome=LoopOutcome.TASK_COMPLETE,
                        message=step.message or "",
                        context=ctx,
                    )
                )

            if step.action == AgentAction.FAILED:
                ctx.add_turn(
                    TurnRecord(
                        turn=turn,
                        step=step,
                        call_model=call_model,
                        call_reasoning_effort=call_reasoning,
                    )
                )
                return finish(
                    LoopResult(
                        outcome=LoopOutcome.FAILED,
                        message=step.message or "",
                        context=ctx,
                    )
                )

            tool_result = None
            if is_harness_tool(step.action):
                spec = get_tool(step.action)
                if spec is None:
                    ctx.add_parse_error(
                        f"Unknown harness tool action: {step.action.value}"
                    )
                    continue

                if step.action == AgentAction.RUN_SHELL:
                    runner = run_shell
                    payload = step.command or ""
                else:
                    runner = run_python
                    payload = step.code or ""

                if self._ui is not None:
                    with tool_activity(self._ui, spec.name.replace("run_", "")):
                        tool_result = runner(
                            payload,
                            cwd=self.workspace,
                            timeout=self.tool_timeout,
                        )
                else:
                    tool_result = runner(
                        payload,
                        cwd=self.workspace,
                        timeout=self.tool_timeout,
                    )

            if self._ui is not None and tool_result is not None:
                self._ui.print_tool_result(tool_result)

            if session_log is not None and tool_result is not None:
                session_log.log_tool_result(turn=turn, result=tool_result)

            if (
                step.action == AgentAction.RUN_SHELL
                and tool_result is not None
                and step.command
            ):
                prior_runs = count_matching_shell_commands(ctx.turns, step.command)
                failed = tool_result.exit_code not in (0, None)
                if failed and prior_runs >= 1:
                    ctx.add_parse_error(
                        stuck_shell_directive(step.command, tool_result.exit_code)
                    )
                if failed and prior_runs >= 2:
                    return finish(
                        LoopResult(
                            outcome=LoopOutcome.FAILED,
                            message=(
                                f"Shell command failed repeatedly (exit "
                                f"{tool_result.exit_code}): {step.command}"
                            ),
                            context=ctx,
                        )
                    )

            ctx.add_turn(
                TurnRecord(
                    turn=turn,
                    step=step,
                    tool_result=tool_result,
                    call_model=call_model,
                    call_reasoning_effort=call_reasoning,
                )
            )

        return finish(
            LoopResult(
                outcome=LoopOutcome.MAX_TURNS,
                message=f"Exceeded maximum turns ({self.max_turns}).",
                context=ctx,
            )
        )

    def _validate_routing(self, step: AgentStep, call_model: str) -> str | None:
        if not is_valid_action(step.action):
            return f"Invalid action '{step.action.value}'."

        if step.action in _SWITCH_TOOLS_ACTIONS:
            if step.reasoning_effort is not None:
                return (
                    "Do not set reasoning_effort on switch_tools. Use switch_model "
                    "or set reasoning_effort on a later turn."
                )

        if step.action == AgentAction.SWITCH_MODEL:
            if step.model not in self._allowed_models:
                allowed = ", ".join(self._allowed_models[:8])
                suffix = "..." if len(self._allowed_models) > 8 else ""
                return (
                    f"Unknown model '{step.model}'. "
                    f"Pick from allowlist: {allowed}{suffix}"
                )

        if step.reasoning_effort is not None:
            next_model = step.model or self._pending_model or call_model
            err = validate_reasoning_effort_for_model(
                next_model, step.reasoning_effort
            )
            if err:
                return err

        return None

    def _apply_pending_routing(self, step: AgentStep) -> None:
        if step.action == AgentAction.SWITCH_MODEL and step.model is not None:
            self._pending_model = step.model
        if step.reasoning_effort is not None:
            self._pending_reasoning = step.reasoning_effort

    def _apply_switch_tools(self, step: AgentStep, call_model: str) -> str | None:
        tools = step.tools or []
        for tool in tools:
            if not model_supports_openai_tool(call_model, tool):
                suggestion = cheapest_model_with_tools(self._allowed_models, tools)
                if suggestion:
                    return (
                        f"Model '{call_model}' does not support hosted tool '{tool}'. "
                        f"On your next turn, use switch_model to '{suggestion}' "
                        "(do not call switch_tools again) and continue the task."
                    )
                return (
                    f"Model '{call_model}' does not support hosted tool '{tool}' and "
                    "no allowlisted model supports it."
                )

        self._hosted_tools = tuple(tools)
        return None
