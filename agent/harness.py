"""Interactive CLI harness for the GoodBoy agent."""

from __future__ import annotations

import click

from agent.loop import AgentLoop, LoopOutcome, LoopResult
from agent.ui import ConversationUI
from settings import get_settings


class AgentHarness:
    """Greet the user, run the agent loop, and handle clarifications."""

    GREETING = "GoodBoy: How can I help?"

    def __init__(
        self,
        *,
        loop: AgentLoop | None = None,
        ui: ConversationUI | None = None,
    ) -> None:
        self._ui = ui or ConversationUI()
        self._loop = loop or AgentLoop(ui=self._ui)

    def run(self) -> int:
        """Run the interactive harness; return process exit code."""
        click.echo(self.GREETING)
        task = self._ui.prompt_user()
        if not task:
            click.echo("No task provided.", err=True)
            return 1

        context = None
        while True:
            result = self._loop.run(task, context=context)
            code = self._handle_result(result)
            if code is not None:
                return code
            context = result.context

    def _handle_result(self, result: LoopResult) -> int | None:
        if result.outcome == LoopOutcome.TASK_COMPLETE:
            click.echo()
            click.echo(click.style("✓ Task complete", fg="green", bold=True))
            return 0

        if result.outcome == LoopOutcome.FAILED:
            click.echo()
            click.echo(click.style("✗ Failed", fg="red", bold=True))
            if not any(
                t.step.action.value == "failed"
                for t in result.context.turns
            ):
                click.echo(result.message, err=True)
            return 1

        if result.outcome == LoopOutcome.MAX_TURNS:
            click.echo()
            click.echo(click.style("⚠ Stopped", fg="yellow", bold=True))
            click.echo(result.message, err=True)
            return 1

        if result.outcome == LoopOutcome.NEED_USER_INPUT:
            reply = self._ui.prompt_user()
            if not reply:
                click.echo("No reply provided; ending session.", err=True)
                return 1
            result.context.add_user_reply(reply)
            return None

        click.echo(f"Unexpected outcome: {result.outcome}", err=True)
        return 1


def run_harness() -> None:
    """Entry point for the GoodBoy harness."""
    cfg = get_settings()
    if not cfg.openai_api_key:
        click.echo("Not configured yet. Run: goodboy setup", err=True)
        raise SystemExit(1)
    harness = AgentHarness()
    raise SystemExit(harness.run())
