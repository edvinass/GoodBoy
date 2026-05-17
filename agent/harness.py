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
    MODEL_COMMAND_NAMES,
)
from agent.ui import ConversationUI
from llm import MODEL_LABELS, select_model_interactive
from settings import (
    GOODBOY_AUTO_MODEL_SWITCH_VAR,
    GOODBOY_SHOW_COMMANDS_VAR,
    OPENAI_MODEL_VAR,
    get_settings,
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
        self._loop.refresh_system_prompt()
        self._conversation_history: list[ConversationExchange] = []
        self._last_active_hosted_tools: list[str] = []
        self._paused_context: SessionContext | None = None

    def run(self) -> int:
        """Run the interactive harness; return process exit code."""
        self._ui.print_startup()
        exit_code = 0
        first_prompt = True

        with open_session_log(workspace=self._loop.workspace) as session_log:
            if session_log is not None and self._ui.debug:
                self._ui.print_notice(f"Session log: {session_log.path}")

            while True:
                try:
                    task = self._ui.prompt_user()
                except (click.Abort, EOFError, KeyboardInterrupt):
                    self._ui.newline()
                    if session_log is not None:
                        session_log.event("session_interrupted")
                    return exit_code

                if session_log is not None:
                    session_log.event("user_input", text=task)

                if self._should_exit(task):
                    if first_prompt and not task:
                        self._ui.print_notice("No task provided.")
                        return 1
                    return exit_code

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
    def _is_clear_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in CLEAR_COMMAND_NAMES

    @classmethod
    def _is_model_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in MODEL_COMMAND_NAMES

    @classmethod
    def _is_commands_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in COMMANDS_COMMAND_NAMES

    @classmethod
    def _is_autoswitch_command(cls, task: str) -> bool:
        return cls._normalize_command(task) in AUTOSWITCH_COMMAND_NAMES

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

    def _toggle_autoswitch(self) -> None:
        self._ui.auto_model_switch = not self._ui.auto_model_switch
        save_env(
            {
                GOODBOY_AUTO_MODEL_SWITCH_VAR: (
                    "true" if self._ui.auto_model_switch else "false"
                )
            }
        )
        self._loop.refresh_system_prompt()
        if self._ui.auto_model_switch:
            self._ui.print_notice(
                "Automatic model switching on — the agent may change models "
                "between turns when needed."
            )
        else:
            self._ui.print_notice(
                "Automatic model switching off — strict cost policy applies."
            )

    def _change_model(self) -> None:
        try:
            chosen = select_model_interactive(
                models=list(self._loop._allowed_models),
                default=self._loop.session_model,
            )
        except click.ClickException as exc:
            self._ui.print_notice(str(exc))
            return

        self._loop.set_session_model(chosen)
        save_env({OPENAI_MODEL_VAR: chosen})
        self._ui.set_session_model(chosen)
        label = MODEL_LABELS.get(chosen, chosen)
        self._ui.print_notice(f"Model set to {label} ({chosen}).")

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
) -> None:
    """Entry point for the GoodBoy harness."""
    cfg = get_settings()
    if not cfg.openai_api_key:
        click.echo("Not configured yet. Run: goodboy setup", err=True)
        raise SystemExit(1)
    harness = AgentHarness(
        show_thoughts=show_thoughts,
        verbose=verbose,
        show_model=show_model,
        show_commands=show_commands or cfg.show_commands,
        auto_model_switch=auto_model_switch or cfg.auto_model_switch,
        debug=debug,
        debug_input=debug_input,
        debug_output=debug_output,
    )
    raise SystemExit(harness.run())
