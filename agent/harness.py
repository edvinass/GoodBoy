"""Interactive CLI harness for the GoodBoy agent."""

from __future__ import annotations

import click

from agent.context import ConversationExchange, SessionContext
from agent.loop import AgentLoop, LoopOutcome, LoopResult
from agent.session_log import open_session_log
from agent.repl_commands import (
    AUTOSWITCH_COMMAND_NAMES,
    CLEAR_COMMAND_NAMES,
    COMMANDS_COMMAND_NAMES,
    EXIT_COMMAND_NAMES,
    HELP_COMMAND_NAMES,
    MODEL_COMMAND_NAMES,
    PLAN_COMMAND_NAMES,
    REASONING_COMMAND_NAMES,
    RETRY_COMMAND_NAMES,
    STREAM_COMMAND_NAMES,
    format_help_text,
    is_repl_slash_command,
    next_plan_mode,
)
from agent.ui import ConversationUI
from agent.local_llm import clear_runner_cache
from llm import (
    MODEL_LABELS,
    get_selectable_models,
    is_local_model,
    local_model_label,
    select_model_interactive,
    select_reasoning_interactive,
)
from settings import (
    GOODBOY_AUTO_MODEL_SWITCH_VAR,
    GOODBOY_PLAN_MODE_VAR,
    GOODBOY_REASONING_EFFORT_VAR,
    GOODBOY_SHOW_COMMANDS_VAR,
    OPENAI_MODEL_VAR,
    get_settings,
    is_configured,
    save_env,
)


class AgentHarness:
    """Run the agent loop and handle clarifications."""

    def __init__(
        self,
        *,
        show_thoughts: bool = False,
        verbose: bool = False,
        show_model: bool = False,
        show_commands: bool = False,
        auto_model_switch: bool = False,
        debug: bool = False,
        debug_input: bool = False,
        debug_output: bool = False,
        stream_output: bool = False,
        loop: AgentLoop | None = None,
        ui: ConversationUI | None = None,
    ) -> None:
        self._ui = ui or ConversationUI(
            show_thoughts=show_thoughts,
            verbose=verbose,
            show_model=show_model,
            show_commands=show_commands,
            auto_model_switch=auto_model_switch,
            debug=debug,
            debug_input=debug_input,
            debug_output=debug_output,
            stream_output=stream_output,
            workspace=None,
            model=None,
        )
        self._loop = loop or AgentLoop(ui=self._ui)
        if self._loop._ui is None:
            self._loop._ui = self._ui
        if getattr(self._ui, "_workspace", None) is None:
            self._ui._workspace = self._loop.workspace
        if self._ui._session_model is None:
            self._ui.set_session_model(self._loop.session_model)
        if getattr(self._ui, "_session_reasoning", None) is None:
            setter = getattr(self._ui, "set_session_reasoning", None)
            if setter is not None:
                setter(self._loop.session_reasoning)
        self._loop.refresh_system_prompt()
        self._conversation_history: list[ConversationExchange] = []
        self._last_active_hosted_tools: list[str] = []
        self._paused_context: SessionContext | None = None
        self._last_task: str | None = None

    def run(self) -> int:
        """Run the interactive harness; return process exit code."""
        self._ui.print_startup()
        exit_code = 0
        first_prompt = True

        with open_session_log(workspace=self._loop.workspace) as session_log:
            if session_log is not None:
                self._ui.print_session_log_path(session_log.path)

            while True:
                try:
                    task = self._ui.prompt_user()
                except (click.Abort, EOFError, KeyboardInterrupt):
                    self._ui.newline()
                    if session_log is not None:
                        session_log.event("session_interrupted")
                    return exit_code

                if session_log is not None and not is_repl_slash_command(task):
                    session_log.event("user_input", text=task)

                if self._should_exit(task):
                    if first_prompt and not task:
                        self._ui.print_notice("No task provided.")
                        return 1
                    return exit_code

                if self._is_help_command(task):
                    self._print_help()
                    continue

                if self._is_retry_command(task):
                    if self._last_task is None:
                        self._ui.print_notice("No previous task to retry.")
                        continue
                    task = self._last_task
                    if session_log is not None:
                        session_log.event("task_retry", task=task)
                elif not is_repl_slash_command(task):
                    self._last_task = task

                if self._is_clear_command(task):
                    self._clear_conversation()
                    if session_log is not None:
                        session_log.event("conversation_cleared")
                    continue

                if self._is_model_command(task):
                    self._change_model()
                    if session_log is not None:
                        session_log.event("model_changed", model=self._loop.session_model)
                    continue

                if self._is_reasoning_command(task):
                    self._change_reasoning()
                    if session_log is not None:
                        session_log.event(
                            "reasoning_changed",
                            reasoning_effort=self._loop.session_reasoning,
                        )
                    continue

                if self._is_plan_command(task):
                    self._change_plan_mode()
                    if session_log is not None:
                        session_log.event(
                            "plan_mode_changed",
                            plan_mode=self._loop._plan_mode,
                        )
                    continue

                if self._is_commands_command(task):
                    self._toggle_commands()
                    if session_log is not None:
                        session_log.event(
                            "commands_visibility_changed",
                            show_commands=self._ui.show_commands,
                        )
                    continue

                if self._is_autoswitch_command(task):
                    self._toggle_autoswitch()
                    if session_log is not None:
                        session_log.event(
                            "auto_model_switch_changed",
                            auto_model_switch=self._ui.auto_model_switch,
                        )
                    continue

                if self._is_stream_command(task):
                    self._toggle_stream()
                    if session_log is not None:
                        session_log.event(
                            "stream_output_changed",
                            stream_output=self._ui.stream_output,
                        )
                    continue

                first_prompt = False
                ctx = self._build_session_context(task)
                result = self._loop.run(
                    task,
                    context=ctx,
                    ask_user=self._ui.prompt_user,
                    session_log=session_log,
                    stop_requested=getattr(self._ui, "consume_stop_requested", None),
                )
                exit_code = max(exit_code, self._report_outcome(result, task=task))

    @staticmethod
    def _normalize_command(task: str) -> str:
        normalized = task.strip().lower()
        if normalized.startswith("/"):
            return normalized[1:]
        return normalized

    @classmethod
    def _should_exit(cls, task: str) -> bool:
        normalized = cls._normalize_command(task)
        return not normalized or normalized in EXIT_COMMAND_NAMES

    @classmethod
    def _is_help_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in HELP_COMMAND_NAMES

    @classmethod
    def _is_retry_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in RETRY_COMMAND_NAMES

    @classmethod
    def _is_clear_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in CLEAR_COMMAND_NAMES

    def _print_help(self) -> None:
        text = format_help_text(
            show_commands=self._ui.show_commands,
            auto_model_switch=self._ui.auto_model_switch,
            stream_output=self._ui.stream_output,
            session_model=self._loop.session_model,
            default_reasoning_effort=self._loop.session_reasoning,
            plan_mode=self._loop._plan_mode,
        )
        self._ui.print_notice(text)

    @classmethod
    def _is_model_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in MODEL_COMMAND_NAMES

    @classmethod
    def _is_reasoning_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in REASONING_COMMAND_NAMES

    @classmethod
    def _is_plan_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in PLAN_COMMAND_NAMES

    @classmethod
    def _is_commands_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in COMMANDS_COMMAND_NAMES

    @classmethod
    def _is_autoswitch_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in AUTOSWITCH_COMMAND_NAMES

    @classmethod
    def _is_stream_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in STREAM_COMMAND_NAMES

    def _toggle_commands(self) -> None:
        self._ui.show_commands = not self._ui.show_commands
        save_env(
            {
                GOODBOY_SHOW_COMMANDS_VAR: (
                    "true" if self._ui.show_commands else "false"
                )
            }
        )
        self._loop.refresh_system_prompt()
        if self._ui.show_commands:
            self._ui.print_notice(
                "Command visibility on — shell and Python runs are shown; output is hidden."
            )
        else:
            self._ui.print_notice("Command visibility off.")

    def _toggle_stream(self) -> None:
        self._ui.stream_output = not self._ui.stream_output
        if self._ui.stream_output:
            self._ui.print_notice(
                "Stream output on — model responses print as they are generated."
            )
        else:
            self._ui.print_notice("Stream output off.")

    def _toggle_autoswitch(self) -> None:
        self._ui.auto_model_switch = not self._ui.auto_model_switch
        save_env(
            {
                GOODBOY_AUTO_MODEL_SWITCH_VAR: (
                    "true" if self._ui.auto_model_switch else "false"
                )
            }
        )
        self._loop.refresh_system_prompt(rebuild_stable=True)
        if self._ui.auto_model_switch:
            self._ui.print_notice(
                "Automatic model switching on — turn 1 uses the cheapest model "
                "to pick the next model; the agent may change models and "
                "reasoning effort between turns when needed."
            )
        else:
            self._ui.print_notice(
                "Automatic model switching off — model and reasoning are session-only "
                "(/model, /reasoning). The agent cannot change them per turn."
            )
        self._ui.refresh_startup_banner()

    def _change_model(self) -> None:
        cfg = get_settings()
        try:
            chosen = select_model_interactive(
                models=get_selectable_models(api_key=cfg.openai_api_key),
                default=self._loop.session_model,
                api_key=cfg.openai_api_key,
            )
        except click.ClickException as exc:
            self._ui.print_notice(str(exc))
            return

        if is_local_model(chosen) or is_local_model(self._loop.session_model):
            clear_runner_cache()
        self._loop.set_session_model(chosen)
        save_env({OPENAI_MODEL_VAR: chosen})
        self._ui.set_session_model(chosen)
        self._loop.refresh_system_prompt(rebuild_stable=True)
        if is_local_model(chosen):
            label = local_model_label(chosen)
        else:
            label = MODEL_LABELS.get(chosen, chosen)
        self._ui.print_notice(f"Model set to {label} ({chosen}).")

    def _change_reasoning(self) -> None:
        if is_local_model(self._loop.session_model):
            self._ui.print_notice(
                "Reasoning effort does not apply to local models."
            )
            return
        try:
            chosen = select_reasoning_interactive(
                default=self._loop.session_reasoning,
            )
        except click.ClickException as exc:
            self._ui.print_notice(str(exc))
            return

        self._loop.set_session_reasoning(chosen)
        self._ui.set_session_reasoning(chosen)
        self._loop.refresh_system_prompt()
        if chosen is None:
            save_env({GOODBOY_REASONING_EFFORT_VAR: ""})
            self._ui.print_notice(
                "Default reasoning effort cleared — reasoning models use the API default."
            )
        else:
            save_env({GOODBOY_REASONING_EFFORT_VAR: chosen})
            self._ui.print_notice(f"Default reasoning effort set to {chosen}.")

    def _change_plan_mode(self) -> None:
        chosen = next_plan_mode(self._loop._plan_mode)
        self._loop._plan_mode = chosen
        save_env({GOODBOY_PLAN_MODE_VAR: chosen})
        self._ui.set_plan_mode(chosen)
        self._loop.refresh_system_prompt()
        self._ui.print_notice(f"Plan mode set to {chosen}.")

    def _clear_conversation(self) -> None:
        """Drop cross-task model context and reset the on-screen transcript."""
        self._conversation_history.clear()
        self._paused_context = None
        self._last_active_hosted_tools = []
        self._ui.clear_session()
        self._ui.print_notice(
            "Conversation cleared. Prior messages will not be sent to the model."
        )

    def _build_session_context(self, task: str) -> SessionContext | None:
        if self._paused_context is not None:
            ctx = self._paused_context.model_copy(deep=True)
            self._paused_context = None
            ctx.add_user_reply(
                "User resumed after pressing Escape and said: " + task.strip()
            )
            if self._conversation_history:
                ctx.conversation_history = list(self._conversation_history)
            return ctx

        if not self._conversation_history:
            return None
        return SessionContext(
            user_task=task,
            workspace=str(self._loop.workspace),
            conversation_history=list(self._conversation_history),
            active_hosted_tools=list(self._last_active_hosted_tools),
        )

    def _record_completed_exchange(self, task: str, result: LoopResult) -> None:
        assistant = (result.message or "").strip()
        if not assistant:
            for record in reversed(result.context.turns):
                if record.step.action.value == "task_complete" and record.step.message:
                    assistant = record.step.message.strip()
                    break
        if assistant:
            self._conversation_history.append(
                ConversationExchange(user=task.strip(), assistant=assistant)
            )

    def _remember_session_state(self, result: LoopResult) -> None:
        if result.context.active_hosted_tools:
            self._last_active_hosted_tools = list(result.context.active_hosted_tools)

    def _report_outcome(self, result: LoopResult, *, task: str) -> int:
        self._remember_session_state(result)
        if result.outcome == LoopOutcome.TASK_COMPLETE:
            self._record_completed_exchange(task, result)
            self._ui.print_task_complete()
            return 0

        if result.outcome == LoopOutcome.FAILED:
            extra = None
            if not any(
                t.step.action.value == "failed"
                for t in result.context.turns
            ):
                extra = result.message
            self._ui.print_failed(extra)
            return 1

        if result.outcome == LoopOutcome.STOPPED:
            self._paused_context = result.context
            self._ui.print_stopped(result.message)
            return 0

        if result.outcome == LoopOutcome.MAX_TURNS:
            self._ui.print_stopped(result.message)
            return 1

        if result.outcome == LoopOutcome.NEED_USER_INPUT:
            self._ui.print_notice(result.message)
            return 1

        self._ui.print_notice(f"Unexpected outcome: {result.outcome}")
        return 1


def run_harness(
    *,
    show_thoughts: bool = False,
    verbose: bool = False,
    show_model: bool = False,
    show_commands: bool = False,
    auto_model_switch: bool = False,
    debug: bool = False,
    debug_input: bool = False,
    debug_output: bool = False,
    stream_output: bool = False,
) -> None:
    """Entry point for the GoodBoy harness."""
    cfg = get_settings()
    if not is_configured(cfg):
        click.echo("Not configured yet. Run: goodboy setup", err=True)
        raise SystemExit(1)
    use_autoswitch = (auto_model_switch or cfg.auto_model_switch) and not is_local_model(
        cfg.default_model
    )
    harness = AgentHarness(
        show_thoughts=show_thoughts,
        verbose=verbose,
        show_model=show_model,
        show_commands=show_commands or cfg.show_commands,
        auto_model_switch=use_autoswitch,
        debug=debug,
        debug_input=debug_input,
        debug_output=debug_output,
        stream_output=stream_output,
    )
    raise SystemExit(harness.run())
