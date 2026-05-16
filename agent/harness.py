"""Interactive CLI harness for the GoodBoy agent."""

from __future__ import annotations

import click

from agent.context import ConversationExchange, SessionContext
from agent.loop import AgentLoop, LoopOutcome, LoopResult
from agent.session_log import open_session_log
from agent.ui import ConversationUI
from settings import get_settings


_EXIT_COMMANDS = frozenset({"exit", "quit", "q"})


class AgentHarness:
    """Run the agent loop and handle clarifications."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        show_model: bool = False,
        show_commands: bool = False,
        debug: bool = False,
        debug_input: bool = False,
        debug_output: bool = False,
        loop: AgentLoop | None = None,
        ui: ConversationUI | None = None,
    ) -> None:
        self._ui = ui or ConversationUI(
            verbose=verbose,
            show_model=show_model,
            show_commands=show_commands,
            debug=debug,
            debug_input=debug_input,
            debug_output=debug_output,
        )
        self._loop = loop or AgentLoop(ui=self._ui)
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
    def _should_exit(task: str) -> bool:
        normalized = task.strip().lower()
        return not normalized or normalized in _EXIT_COMMANDS

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
    verbose: bool = False,
    show_model: bool = False,
    show_commands: bool = False,
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
        verbose=verbose,
        show_model=show_model,
        show_commands=show_commands,
        debug=debug,
        debug_input=debug_input,
        debug_output=debug_output,
    )
    raise SystemExit(harness.run())
