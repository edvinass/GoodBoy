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
from agent.context import (
    ContextWatermark,
    SessionContext,
    effective_plan_items,
    should_print_plan_progress,
)
from agent.models import (
    model_supports_openai_tool,
    resolve_reasoning_effort,
)
from agent.file_tools import apply_patch, read_file, str_replace
from agent.prompt import build_session_prompt_suffix, build_stable_system_prompt
from agent.registry import get_tool, is_harness_tool, is_valid_action
from agent.task_policy import (
    is_edit_action,
    is_verification_command,
    plan_blocks_edit,
    validate_plan_submission,
    verification_blocks_complete,
)
from agent.tools import run_python, run_shell
from llm import TokenUsage
from agent.stream_status import extract_streaming_status
from agent.ui import (
    ConversationUI,
    ThinkingUpdater,
    activity_label,
    progress_label_for_step,
    tool_activity,
)
from agent.types import (
    AgentAction,
    AgentStep,
    PlanItem,
    ToolResult,
    TurnRecord,
    _SWITCH_TOOLS_ACTIONS,
    parse_agent_step,
    parse_all_agent_steps,
)
from openai import APIConnectionError

from llm import (
    REASONING_EFFORT,
    ResponseChainBroken,
    UserAbort,
    complete_structured,
    complete_structured_with_id,
    format_api_connection_error,
    get_selectable_models,
    is_deepseek_model,
    is_local_model,
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
        self._explicit_recent_full_turns: int | None = recent_full_turns
        self._cloud_recent_full_turns = (
            recent_full_turns
            if recent_full_turns is not None
            else cfg.context_recent_full_turns
        )
        self._local_recent_full_turns = (
            recent_full_turns
            if recent_full_turns is not None
            else cfg.local_recent_full_turns
        )
        self._default_model = model or cfg.default_model
        self.recent_full_turns = self._window_for_model(self._default_model)
        self._default_reasoning: str | None = cfg.default_reasoning_effort
        self._allowed_models = allowed_models or get_selectable_models(
            api_key=cfg.openai_api_key
        )
        self._hosted_tools: tuple[str, ...] = ()
        # Custom llm_call hooks (tests, mocks) bypass response chaining; only
        # the production path through complete_structured_with_id can chain.
        self._llm_call = llm_call
        self._chain_enabled = (
            bool(
                response_chain_enabled
                if response_chain_enabled is not None
                else cfg.response_chain_enabled
            )
            and llm_call is None
            and not is_local_model(self._default_model)
            and not is_deepseek_model(self._default_model)
        )
        self._last_response_id: str | None = None
        self._chain_watermark: ContextWatermark | None = None
        self._plan_mode = cfg.plan_mode
        self._working_memory_max = cfg.working_memory_max
        self._verify_before_complete = cfg.verify_before_complete
        self._ui = ui
        self._stable_instructions = self._build_stable_instructions()
        self._session_suffix = self._build_session_suffix()
        self._instructions = self._compose_instructions()

    def _build_stable_instructions(self) -> str:
        return build_stable_system_prompt(allowed_models=self._allowed_models)

    def _build_session_suffix(self) -> str:
        ui = self._ui
        return build_session_prompt_suffix(
            debug=ui.debug if ui is not None else False,
            show_thoughts=(
                (ui.show_thoughts or ui.verbose) if ui is not None else False
            ),
            show_commands=ui.show_commands if ui is not None else False,
        )

    def _compose_instructions(self) -> str:
        return f"{self._stable_instructions}\n\n{self._session_suffix}"

    def refresh_system_prompt(self) -> None:
        """Rebuild instructions after runtime UI toggles."""
        self._session_suffix = self._build_session_suffix()
        self._instructions = self._compose_instructions()

    @property
    def session_model(self) -> str:
        return self._default_model

    @property
    def session_reasoning(self) -> str | None:
        return self._default_reasoning

    def _window_for_model(self, model: str) -> int:
        """Pick a transcript window that fits the model's context budget.

        Local GGUF models typically have small (8k) windows, so the default
        cloud-sized 15-turn window blows the prompt budget and forces the
        local truncator to throw away context every call. Use a tighter
        window when the active model is local.
        """
        if is_local_model(model):
            return self._local_recent_full_turns
        return self._cloud_recent_full_turns

    def set_session_model(self, model: str) -> None:
        """Set the default model for subsequent tasks in this harness session."""
        selectable = get_selectable_models(
            api_key=get_settings().openai_api_key
        )
        if model not in selectable:
            raise ValueError(f"Model not available: {model}")
        if is_local_model(model) or is_local_model(self._default_model):
            from agent.local_llm import clear_runner_cache

            clear_runner_cache()
        self._default_model = model
        self._allowed_models = self._models_for_session()
        self._hosted_tools = ()
        self._reset_response_chain()
        self.recent_full_turns = self._window_for_model(model)
        if is_local_model(model) or is_deepseek_model(model):
            self._chain_enabled = False
        elif self._llm_call is None:
            self._chain_enabled = get_settings().response_chain_enabled

    def _models_for_session(self) -> list[str]:
        """Allowlist for routing validation; locals only when session is local."""
        from agent.local_llm import list_installed_models

        if is_local_model(self._default_model):
            installed = list_installed_models()
            return installed if installed else [self._default_model]
        return get_selectable_models(api_key=get_settings().openai_api_key)

    def set_session_reasoning(self, effort: str | None) -> None:
        """Set the default reasoning effort for subsequent LLM calls in this session."""
        if effort is not None and effort not in REASONING_EFFORT:
            raise ValueError(f"Unknown reasoning effort: {effort}")
        self._default_reasoning = effort

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
        # Re-arm the local-model context-full warning so the next truncation
        # event in this task triggers exactly one notice (instead of one per
        # LLM call, which spammed the terminal).
        if is_local_model(self._default_model):
            from agent.local_llm import reset_context_full_notice

            reset_context_full_notice()
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

        def abort_now() -> LoopResult:
            return finish(
                LoopResult(
                    outcome=LoopOutcome.STOPPED,
                    message=build_stopped_message(turns=ctx.turns),
                    context=ctx,
                )
            )

        for turn in range(1, self.max_turns + 1):
            call_model = self._default_model
            effort = self._default_reasoning
            local_call = is_local_model(call_model)
            deepseek_call = is_deepseek_model(call_model)
            if local_call:
                call_reasoning = None
            else:
                call_reasoning = resolve_reasoning_effort(call_model, effort)

            input_text, prev_id = self._build_llm_input(ctx)
            llm_kwargs = {
                "input": input_text,
                "instructions": self._instructions,
                "json_schema": json_schema,
                "model": call_model,
                "reasoning_effort": call_reasoning,
            }
            if self._hosted_tools and not local_call and not deepseek_call:
                llm_kwargs["tools"] = list(self._hosted_tools)
            if prev_id is not None and not local_call and not deepseek_call:
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
                raw, response_id, usage = self._invoke_llm(
                    llm_kwargs, ctx=ctx, turn=turn, session_log=session_log
                )
            except UserAbort:
                if session_log is not None:
                    session_log.event("user_abort", turn=turn, phase="llm")
                return abort_now()
            except APIConnectionError as exc:
                return finish(
                    LoopResult(
                        outcome=LoopOutcome.FAILED,
                        message=format_api_connection_error(exc),
                        context=ctx,
                    )
                )

            self._record_chain_progress(ctx, response_id)
            if self._ui is not None and usage is not None:
                self._ui.print_turn_usage(
                    turn=turn,
                    model=call_model,
                    usage=usage,
                )

            if session_log is not None:
                session_log.log_llm_response(turn=turn, raw=raw)

            plan_print_baseline = list(ctx.plan_items)
            try:
                step = parse_agent_step(raw)
                for extra in parse_all_agent_steps(raw):
                    if (
                        extra.action == AgentAction.UPDATE_PLAN
                        and extra.plan_items
                    ):
                        ctx = ctx.model_copy(
                            update={"plan_items": list(extra.plan_items)}
                        )
                plan_print_baseline = self._maybe_print_plan_progress(
                    ctx, plan_print_baseline
                )
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

            action_err = self._validate_action(step, call_model)
            if action_err:
                ctx.add_parse_error(action_err)
                if session_log is not None:
                    session_log.event("parse_error", turn=turn, error=action_err)
                consecutive_parse_failures += 1
                if consecutive_parse_failures >= 2:
                    return finish(
                        LoopResult(
                            outcome=LoopOutcome.FAILED,
                            message=action_err,
                            context=ctx,
                        )
                    )
                continue

            consecutive_parse_failures = 0

            if session_log is not None:
                session_log.log_agent_step(turn=turn, step=step)

            if self._ui is not None:
                self._ui.print_agent_step(
                    step,
                    model=call_model,
                    reasoning=call_reasoning,
                    hosted_tools=list(self._hosted_tools) or None,
                )

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

            if step.action == AgentAction.UPDATE_PLAN:
                plan_err = validate_plan_submission(ctx, step.plan_items)
                if plan_err:
                    ctx.add_parse_error(plan_err)
                    if session_log is not None:
                        session_log.event(
                            "parse_error", turn=turn, error=plan_err
                        )
                    consecutive_parse_failures += 1
                    if consecutive_parse_failures >= 2:
                        return finish(
                            LoopResult(
                                outcome=LoopOutcome.FAILED,
                                message=plan_err,
                                context=ctx,
                            )
                        )
                    continue
                ctx = ctx.model_copy(update={"plan_items": list(step.plan_items or [])})
                plan_print_baseline = self._maybe_print_plan_progress(
                    ctx, plan_print_baseline
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

            if step.action == AgentAction.REMEMBER:
                ctx.append_working_memory(
                    step.memory or [],
                    max_items=self._working_memory_max,
                )
                ctx.add_turn(
                    TurnRecord(
                        turn=turn,
                        step=step,
                        call_model=call_model,
                        call_reasoning_effort=call_reasoning,
                    )
                )
                if self._ui is not None:
                    self._ui.print_agent_step(
                        step,
                        model=call_model,
                        reasoning=call_reasoning,
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
                verify_err = verification_blocks_complete(
                    ctx,
                    verify_before_complete=self._verify_before_complete,
                )
                if verify_err:
                    ctx.add_parse_error(verify_err)
                    if session_log is not None:
                        session_log.event(
                            "parse_error", turn=turn, error=verify_err
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

                edit_gate = plan_blocks_edit(
                    ctx, step.action, plan_mode=self._plan_mode
                )
                if edit_gate:
                    ctx.add_parse_error(edit_gate)
                    if session_log is not None:
                        session_log.event(
                            "parse_error", turn=turn, error=edit_gate
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

                progress_label = progress_label_for_step(step)
                spinner_plan = effective_plan_items(ctx, raw)
                if self._ui is not None:
                    with tool_activity(
                        self._ui, progress_label, plan_items=spinner_plan
                    ):
                        tool_result = self._run_harness_tool(step)
                else:
                    tool_result = self._run_harness_tool(step)

            if self._ui is not None and tool_result is not None:
                self._ui.print_harness_activity(step, tool_result)
                self._ui.print_tool_result(tool_result)

            if session_log is not None and tool_result is not None:
                session_log.log_tool_result(turn=turn, result=tool_result)

            if tool_result is not None and self._abort_requested():
                if session_log is not None:
                    session_log.event("user_abort", turn=turn, phase="tool")
                ctx.add_turn(
                    TurnRecord(
                        turn=turn,
                        step=step,
                        tool_result=tool_result,
                        call_model=call_model,
                        call_reasoning_effort=call_reasoning,
                    )
                )
                return abort_now()

            if tool_result is not None:
                succeeded = tool_result.exit_code == 0 and not tool_result.timed_out
                if succeeded and is_edit_action(step.action):
                    ctx = ctx.model_copy(update={"last_edit_turn": turn})
                if (
                    succeeded
                    and step.action == AgentAction.RUN_SHELL
                    and step.command
                    and is_verification_command(step.command)
                ):
                    ctx = ctx.model_copy(update={"last_verify_turn": turn})

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

    def _maybe_print_plan_progress(
        self,
        ctx: SessionContext,
        baseline: list[PlanItem],
    ) -> list[PlanItem]:
        """Show plan when created (0 done) or when a step is marked done."""
        if (
            self._ui is not None
            and should_print_plan_progress(baseline, ctx.plan_items)
        ):
            self._ui.print_plan(ctx.plan_items)
        return list(ctx.plan_items)

    def _record_chain_progress(
        self, ctx: SessionContext, response_id: str | None
    ) -> None:
        if not self._chain_enabled or response_id is None:
            return
        self._last_response_id = response_id
        # Snapshot what the model has now seen on the server side so the next
        # turn's delta only includes items appended after this point.
        self._chain_watermark = ctx.watermark()

    def _abort_requested(self) -> bool:
        if self._ui is None:
            return False
        check = getattr(self._ui, "is_abort_requested", None)
        if check is None:
            return False
        try:
            return bool(check())
        except Exception:
            return False

    def _abort_check(self) -> Callable[[], bool] | None:
        if self._ui is None:
            return None
        check = getattr(self._ui, "is_abort_requested", None)
        if check is None:
            return None
        return check  # type: ignore[return-value]

    def _run_harness_tool(self, step: AgentStep) -> ToolResult:
        ws = self.workspace
        timeout = self.tool_timeout
        abort_check = self._abort_check()
        if step.action == AgentAction.RUN_SHELL:
            return run_shell(
                step.command or "",
                cwd=ws,
                timeout=timeout,
                abort_check=abort_check,
            )
        if step.action == AgentAction.RUN_PYTHON:
            return run_python(
                step.code or "",
                cwd=ws,
                timeout=timeout,
                abort_check=abort_check,
            )
        if step.action == AgentAction.READ_FILE:
            return read_file(
                step.path or "",
                workspace=ws,
                start_line=step.start_line,
                end_line=step.end_line,
            )
        if step.action == AgentAction.STR_REPLACE:
            return str_replace(
                step.path or "",
                step.old_string or "",
                step.new_string or "",
                workspace=ws,
            )
        if step.action == AgentAction.APPLY_PATCH:
            return apply_patch(step.path or "", step.patch or "", workspace=ws)
        return ToolResult(
            executed=step.action.value,
            stderr=f"Unhandled tool: {step.action.value}",
            exit_code=1,
        )

    def _do_llm_call(
        self, llm_kwargs: dict, *, stream: bool, on_text_delta=None
    ) -> tuple[str, str | None, TokenUsage | None]:
        """Dispatch to the id-aware client when available, else legacy str path."""
        if self._llm_call is not None:
            extras: dict = {}
            if stream:
                extras["stream"] = True
                extras["on_text_delta"] = on_text_delta
            text = self._llm_call(**llm_kwargs, **extras)
            return text, None, None
        abort_check = self._abort_check()
        if stream:
            return complete_structured_with_id(
                **llm_kwargs,
                stream=True,
                on_text_delta=on_text_delta,
                abort_check=abort_check,
            )
        return complete_structured_with_id(**llm_kwargs, abort_check=abort_check)

    def _invoke_llm(
        self,
        llm_kwargs: dict,
        *,
        ctx: SessionContext,
        turn: int,
        session_log: SessionLog | None,
    ) -> tuple[str, str | None, TokenUsage | None]:
        """Wrap the LLM call with UI plumbing and chain-broken recovery."""

        def _call_through_ui() -> tuple[str, str | None, TokenUsage | None]:
            if self._ui is None:
                return self._do_llm_call(llm_kwargs, stream=False)
            self._ui.print_llm_request(
                turn=turn,
                model=llm_kwargs["model"],
                reasoning_effort=llm_kwargs.get("reasoning_effort"),
                instructions=llm_kwargs["instructions"],
                input_text=llm_kwargs["input"],
            )
            show_stream = self._ui.stream_output
            buffer: list[str] = []
            thinking_holder: list[ThinkingUpdater] = []

            def _on_text_delta(delta: str) -> None:
                if not delta:
                    return
                buffer.append(delta)
                if show_stream:
                    self._ui.write_model_stream_delta(delta)
                    return
                status = extract_streaming_status("".join(buffer))
                if status and thinking_holder:
                    thinking_holder[0].update(status)

            # Rich's Live spinner (used by `thinking`) and direct stderr writes
            # from streaming both move the shared terminal cursor. Running both
            # concurrently makes the spinner's "move up + erase" passes wipe
            # parts of the streamed text. When `stream_output` is on the
            # streamed text is the progress indicator, so skip the spinner.
            if show_stream:
                self._ui.begin_model_stream(turn=turn)
                try:
                    with self._ui.escape_stop_listener():
                        raw_text, rid, usage = self._do_llm_call(
                            llm_kwargs,
                            stream=True,
                            on_text_delta=_on_text_delta,
                        )
                finally:
                    self._ui.end_model_stream()
            else:
                with self._ui.thinking(
                    plan_items=effective_plan_items(ctx)
                ) as thinking_updater:
                    thinking_holder.append(thinking_updater)
                    raw_text, rid, usage = self._do_llm_call(
                        llm_kwargs,
                        stream=True,
                        on_text_delta=_on_text_delta,
                    )
            self._ui.print_llm_response(
                turn=turn, raw=raw_text, streamed=show_stream
            )
            return raw_text, rid, usage

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

    def _validate_action(self, step: AgentStep, call_model: str) -> str | None:
        if not is_valid_action(step.action):
            return f"Invalid action '{step.action.value}'."

        if step.action in _SWITCH_TOOLS_ACTIONS:
            if is_local_model(call_model):
                return (
                    "switch_tools is not available with local models. "
                    "Use run_shell with rg/grep for codebase search."
                )
            if is_deepseek_model(call_model):
                return (
                    "switch_tools is not available with DeepSeek models. "
                    "Use run_shell with rg/grep for codebase search."
                )

        return None

    def _apply_switch_tools(self, step: AgentStep, call_model: str) -> str | None:
        tools = step.tools or []
        for tool in tools:
            if not model_supports_openai_tool(call_model, tool):
                return (
                    f"Model '{call_model}' does not support hosted tool '{tool}'. "
                    f"Use need_user_input and ask the user to run /model and choose "
                    "a model that supports the tool, then continue the task."
                )

        self._hosted_tools = tuple(tools)
        return None
