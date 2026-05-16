"""Interactive CLI harness for the GoodBoy agent."""

from __future__ import annotations

import click

from agent.loop import AgentLoop, LoopOutcome, LoopResult
from agent.session_log import open_session_log
from agent.ui import ConversationUI
from settings import get_settings


_EXIT_COMMANDS = frozenset({"exit", "quit", "q"})


class AgentHarness:
    """Greet the user, run the agent loop, and handle clarifications."""

    GREETING = "How can I help?"

    def __init__(
        self,
        *,
        debug: bool = False,
        debug_input: bool = False,
        debug_output: bool = False,
        loop: AgentLoop | None = None,
        ui: ConversationUI | None = None,
    ) -> None:
        self._ui = ui or ConversationUI(
            debug=debug,
            debug_input=debug_input,
            debug_output=debug_output,
        )
        self._loop = loop or AgentLoop(ui=self._ui)

    def run(self) -> int:
        """Run the interactive harness; return process exit code."""
        self._ui.print_startup()
        self._ui.print_greeting(self.GREETING)
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
                result = self._loop.run(
                    task,
                    ask_user=self._ui.prompt_user,
                    session_log=session_log,
                )
                exit_code = max(exit_code, self._report_outcome(result))

    @staticmethod
    def _should_exit(task: str) -> bool:
        normalized = task.strip().lower()
        return not normalized or normalized in _EXIT_COMMANDS

    def _report_outcome(self, result: LoopResult) -> int:
        if result.outcome == LoopOutcome.TASK_COMPLETE:
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
        debug=debug,
        debug_input=debug_input,
        debug_output=debug_output,
    )
    raise SystemExit(harness.run())
