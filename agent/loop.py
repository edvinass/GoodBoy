"""Core agent orchestration loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from agent.clarifications import (
    build_stopped_message,
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
from agent.context import ContextWatermark, SessionContext
from agent.models import (
    cheapest_capable_model,
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

from llm import (
    REASONING_EFFORT,
    ResponseChainBroken,
    complete_structured,
    complete_structured_with_id,
    format_api_connection_error,
    get_curated_models,
)
from settings import get_settings

LLMCall = Callable[..., str]
AskUser = Callable[[], str]
StopRequested = Callable[[], bool]


class LoopOutcome(str, Enum):
    TASK_COMPLETE = "task_complete"
    NEED_USER_INPUT = "need_user_input"
    FAILED = "failed"
    MAX_TURNS = "max_turns"
    STOPPED = "stopped"


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
        recent_full_turns: int | None = None,
        response_chain_enabled: bool | None = None,
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
        self.recent_full_turns = (
            recent_full_turns
            if recent_full_turns is not None
            else cfg.context_recent_full_turns
        )
        self._default_model = model or cfg.default_model
        self._default_reasoning: str | None = cfg.default_reasoning_effort
        self._allowed_models = allowed_models or get_curated_models()
        self._pending_model: str | None = None
        self._pending_reasoning: str | None = None
        self._hosted_tools: tuple[str, ...] = ()
        # Custom llm_call hooks (tests, mocks) bypass response chaining; only
        # the production path through complete_structured_with_id can chain.
        self._llm_call = llm_call
        self._chain_enabled = bool(
            response_chain_enabled
            if response_chain_enabled is not None
            else cfg.response_chain_enabled
        ) and llm_call is None
        self._last_response_id: str | None = None
        self._chain_watermark: ContextWatermark | None = None
        self._ui = ui
        self._instructions = self._build_instructions()

    def _build_instructions(self) -> str:
        ui = self._ui
        return build_system_prompt(
            allowed_models=self._allowed_models,
            debug=ui.debug if ui is not None else False,
            show_thoughts=(
                (ui.show_thoughts or ui.verbose) if ui is not None else False
            ),
            show_commands=ui.show_commands if ui is not None else False,
            auto_model_switch=(
                ui.auto_model_switch if ui is not None else False
            ),
        )

    def refresh_system_prompt(self) -> None:
        """Rebuild the system prompt after runtime UI toggles."""
        self._instructions = self._build_instructions()

    @property
    def session_model(self) -> str:
        return self._default_model

    @property
    def session_reasoning(self) -> str | None:
        return self._default_reasoning

    def set_session_model(self, model: str) -> None:
        """Set the default model for subsequent tasks in this harness session."""
        if model not in self._allowed_models:
            raise ValueError(f"Model not in allowlist: {model}")
        self._default_model = model
        self._pending_model = None
        self._reset_response_chain()

    def set_session_reasoning(self, effort: str | None) -> None:
        """Set the default reasoning effort for subsequent LLM calls in this session."""
        if effort is not None and effort not in REASONING_EFFORT:
            raise ValueError(f"Unknown reasoning effort: {effort}")
        self._default_reasoning = effort
        self._pending_reasoning = None

    def run(
        self,
        task: str,
        *,
        context: SessionContext | None = None,
        ask_user: AskUser | None = None,
        session_log: SessionLog | None = None,
        stop_requested: StopRequested | None = None,
    ) -> LoopResult:
        ctx = context or SessionContext(
            user_task=task,
            workspace=str(self.workspace),
            recent_full_turns=self.recent_full_turns,
        )
        if ctx.workspace is None:
            ctx = ctx.model_copy(update={"workspace": str(self.workspace)})
        # Honour the loop's window size even when callers (harness resume,
        # tests) pass a pre-built context that still carries the model default.
        if ctx.recent_full_turns != self.recent_full_turns:
            ctx = ctx.model_copy(
                update={"recent_full_turns": self.recent_full_turns}
            )
        # Each task runs an independent server-side response chain.
        self._reset_response_chain()
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
        fresh_task = len(ctx.turns) == 0

        def finish(result: LoopResult) -> LoopResult:
            if session_log is not None:
                session_log.event(
                    "task_end",
                    outcome=result.outcome.value,
                    message=result.message,
                    turns=len(result.context.turns),
                )
            return result

        def stop_after_current_step() -> LoopResult | None:
            if stop_requested is None or not stop_requested():
                return None
            return finish(
                LoopResult(
                    outcome=LoopOutcome.STOPPED,
                    message=build_stopped_message(turns=ctx.turns),
                    context=ctx,
                )
            )

        for turn in range(1, self.max_turns + 1):
            routing_turn = (
                fresh_task
                and self._auto_model_switch_enabled()
                and turn == 1
                and self._pending_model is None
            )
            if routing_turn:
                call_model = self._router_model()
            else:
                call_model = self._pending_model or self._default_model
            effort = self._pending_reasoning
            if effort is None:
                effort = self._default_reasoning
            call_reasoning = resolve_reasoning_effort(call_model, effort)
            self._pending_model = None
            self._pending_reasoning = None

            input_text, prev_id = self._build_llm_input(ctx)
            llm_kwargs = {
                "input": input_text,
                "instructions": self._instructions,
                "json_schema": json_schema,
                "model": call_model,
                "reasoning_effort": call_reasoning,
            }
            if self._hosted_tools:
                llm_kwargs["tools"] = list(self._hosted_tools)
            if prev_id is not None:
                llm_kwargs["previous_response_id"] = prev_id

            if session_log is not None:
                session_log.log_llm_request(
                    turn=turn,
                    model=call_model,
                    reasoning_effort=call_reasoning,
                    instructions=self._instructions,
                    input_text=llm_kwargs["input"],
                )

            try:
                raw, response_id = self._invoke_llm(
                    llm_kwargs, ctx=ctx, turn=turn, session_log=session_log
                )
            except APIConnectionError as exc:
                return finish(
                    LoopResult(
                        outcome=LoopOutcome.FAILED,
                        message=format_api_connection_error(exc),
                        context=ctx,
                    )
                )

            self._record_chain_progress(ctx, response_id)

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

            routing_err = self._routing_turn_model_required(routing_turn, step)
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
                ctx.add_turn(
                    TurnRecord(
                        turn=turn,
                        step=step,
                        call_model=call_model,
                        call_reasoning_effort=call_reasoning,
                    )
                )
                continue

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
                # The chain is bound to the prior model; switching invalidates
                # the server-side conversation state.
                self._reset_response_chain()
                ctx.add_turn(
                    TurnRecord(
                        turn=turn,
                        step=step,
                        call_model=call_model,
                        call_reasoning_effort=call_reasoning,
                    )
                )
                stopped = stop_after_current_step()
                if stopped is not None:
                    return stopped
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
                # Hosted-tool set changed; restart the chain so the new tools
                # take effect on the next call.
                self._reset_response_chain()
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
                stopped = stop_after_current_step()
                if stopped is not None:
                    return stopped
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
            stopped = stop_after_current_step()
            if stopped is not None:
                return stopped

        return finish(
            LoopResult(
                outcome=LoopOutcome.MAX_TURNS,
                message=f"Exceeded maximum turns ({self.max_turns}).",
                context=ctx,
            )
        )

    def _reset_response_chain(self) -> None:
        self._last_response_id = None
        self._chain_watermark = None

    def _build_llm_input(
        self, ctx: SessionContext
    ) -> tuple[str, str | None]:
        """Pick between full and delta input based on chain state.

        Returns (input_text, previous_response_id_or_None). The id is only set
        when chaining is on AND we have a prior response to chain from AND we
        recorded a watermark to compute a delta against.
        """
        if (
            self._chain_enabled
            and self._last_response_id is not None
            and self._chain_watermark is not None
        ):
            return (
                ctx.to_prompt_delta(self._chain_watermark),
                self._last_response_id,
            )
        return ctx.to_prompt(), None

    def _record_chain_progress(
        self, ctx: SessionContext, response_id: str | None
    ) -> None:
        if not self._chain_enabled or response_id is None:
            return
        self._last_response_id = response_id
        # Snapshot what the model has now seen on the server side so the next
        # turn's delta only includes items appended after this point.
        self._chain_watermark = ctx.watermark()

    def _do_llm_call(
        self, llm_kwargs: dict, *, stream: bool, on_text_delta=None
    ) -> tuple[str, str | None]:
        """Dispatch to the id-aware client when available, else legacy str path.

        When a custom ``llm_call`` was injected (tests), we cannot capture a
        response id; chaining is disabled in __init__ for that case.
        """
        if self._llm_call is not None:
            extras: dict = {}
            if stream:
                extras["stream"] = True
                extras["on_text_delta"] = on_text_delta
            text = self._llm_call(**llm_kwargs, **extras)
            return text, None
        if stream:
            return complete_structured_with_id(
                **llm_kwargs,
                stream=True,
                on_text_delta=on_text_delta,
            )
        return complete_structured_with_id(**llm_kwargs)

    def _invoke_llm(
        self,
        llm_kwargs: dict,
        *,
        ctx: SessionContext,
        turn: int,
        session_log: SessionLog | None,
    ) -> tuple[str, str | None]:
        """Wrap the LLM call with UI plumbing and chain-broken recovery."""

        def _call_through_ui() -> tuple[str, str | None]:
            if self._ui is None:
                return self._do_llm_call(llm_kwargs, stream=False)
            self._ui.print_llm_request(
                turn=turn,
                model=llm_kwargs["model"],
                reasoning_effort=llm_kwargs.get("reasoning_effort"),
                instructions=llm_kwargs["instructions"],
                input_text=llm_kwargs["input"],
            )
            streamed = self._ui.stream_output
            if streamed:
                self._ui.begin_model_stream(turn=turn)
                try:
                    raw_text, rid = self._do_llm_call(
                        llm_kwargs,
                        stream=True,
                        on_text_delta=self._ui.write_model_stream_delta,
                    )
                finally:
                    self._ui.end_model_stream()
            else:
                with self._ui.thinking():
                    raw_text, rid = self._do_llm_call(llm_kwargs, stream=False)
            self._ui.print_llm_response(
                turn=turn, raw=raw_text, streamed=streamed
            )
            return raw_text, rid

        try:
            return _call_through_ui()
        except ResponseChainBroken as exc:
            # Server-side chain expired or was rejected. Drop the chain, fall
            # back to a stateless full prompt for this turn, and continue.
            if session_log is not None:
                session_log.event(
                    "response_chain_reset",
                    turn=turn,
                    reason=str(exc),
                )
            if self._ui is not None:
                self._ui.print_notice(
                    "Response chain expired — resending full context."
                )
            self._reset_response_chain()
            llm_kwargs = dict(llm_kwargs)
            llm_kwargs.pop("previous_response_id", None)
            llm_kwargs["input"] = ctx.to_prompt()
            return _call_through_ui()

    def _auto_model_switch_enabled(self) -> bool:
        return self._ui.auto_model_switch if self._ui is not None else False

    def _router_model(self) -> str:
        return (
            cheapest_capable_model(self._allowed_models) or self._default_model
        )

    def _routing_turn_model_required(
        self, routing_turn: bool, step: AgentStep
    ) -> str | None:
        if not routing_turn:
            return None
        if step.action in (AgentAction.TASK_COMPLETE, AgentAction.FAILED):
            return None
        if step.action == AgentAction.SWITCH_MODEL:
            return None
        if self._pending_model is not None:
            return None
        return (
            "Routing turn (automatic model switching): set **model** on this "
            "action (or use switch_model) so the next LLM call uses an "
            "appropriate model from the allowlist."
        )

    def _validate_routing(self, step: AgentStep, call_model: str) -> str | None:
        if not is_valid_action(step.action):
            return f"Invalid action '{step.action.value}'."

        if not self._auto_model_switch_enabled():
            if step.action == AgentAction.SWITCH_MODEL:
                return (
                    "switch_model is disabled while automatic model switching is off. "
                    "The user sets the session model with /model."
                )
            if step.model is not None:
                return (
                    "Do not set model in JSON while automatic model switching is off. "
                    "The user sets the session model with /model."
                )
            if step.reasoning_effort is not None:
                return (
                    "Do not set reasoning_effort in JSON while automatic model "
                    "switching is off. The user sets default reasoning with /reasoning."
                )

        if step.action in _SWITCH_TOOLS_ACTIONS:
            if step.model is not None:
                return (
                    "Do not set model on switch_tools. Set model on the next "
                    "run_shell, run_python, switch_model, or terminal action."
                )
            if step.reasoning_effort is not None:
                return (
                    "Do not set reasoning_effort on switch_tools. Set "
                    "reasoning_effort on a later turn (optionally with model)."
                )

        if step.model is not None:
            if step.model not in self._allowed_models:
                allowed = ", ".join(self._allowed_models[:8])
                suffix = "..." if len(self._allowed_models) > 8 else ""
                return (
                    f"Unknown model '{step.model}'. "
                    f"Pick from allowlist: {allowed}{suffix}"
                )
            for tool in self._hosted_tools:
                if not model_supports_openai_tool(step.model, tool):
                    return (
                        f"Model '{step.model}' does not support hosted tool "
                        f"'{tool}' required by Active API."
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
        if not self._auto_model_switch_enabled():
            return
        if step.model is not None:
            if step.model != self._default_model:
                # New model means a new chain on the server side.
                self._reset_response_chain()
            self._pending_model = step.model
            self._default_model = step.model
            if self._ui is not None:
                self._ui.set_session_model(step.model)
        if step.reasoning_effort is not None:
            self._pending_reasoning = step.reasoning_effort

    def _apply_switch_tools(self, step: AgentStep, call_model: str) -> str | None:
        tools = step.tools or []
        for tool in tools:
            if not model_supports_openai_tool(call_model, tool):
                suggestion = cheapest_model_with_tools(self._allowed_models, tools)
                if suggestion:
                    if self._auto_model_switch_enabled():
                        return (
                            f"Model '{call_model}' does not support hosted tool '{tool}'. "
                            f"On your next turn, set model to '{suggestion}' "
                            "(run_shell/switch_model; do not call switch_tools again) "
                            "and continue the task."
                        )
                    return (
                        f"Model '{call_model}' does not support hosted tool '{tool}'. "
                        f"Use need_user_input and ask the user to run /model and choose "
                        f"'{suggestion}', then continue the task."
                    )
                return (
                    f"Model '{call_model}' does not support hosted tool '{tool}' and "
                    "no allowlisted model supports it."
                )

        self._hosted_tools = tuple(tools)
        return None
