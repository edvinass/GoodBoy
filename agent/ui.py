"""Terminal formatting and activity indicators for the harness."""

from __future__ import annotations

import itertools
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import click

from agent.types import AgentAction, AgentStep, ToolResult

_AGENT = click.style("GoodBoy", fg="cyan", bold=True)
_USER = click.style("You", fg="green", bold=True)
_DIM = {"dim": True}
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _indent_body(text: str, *, prefix: str = "  ") -> None:
    for line in text.splitlines() or [""]:
        click.echo(f"{prefix}{line}")


class ConversationUI:
    """Format user and agent messages distinctly in the terminal."""

    def __init__(self, *, debug: bool = False) -> None:
        self.debug = debug

    def prompt_user(self) -> str:
        """Read a user line under the You: label."""
        click.echo()
        click.echo(f"{_USER}:")
        return click.prompt(
            "",
            prompt_suffix=" ",
            show_default=False,
        ).strip()

    def print_user(self, text: str) -> None:
        click.echo()
        click.echo(f"{_USER}:")
        _indent_body(text)

    def print_agent(
        self,
        text: str,
        *,
        subtitle: str | None = None,
    ) -> None:
        click.echo()
        if subtitle:
            click.echo(f"{_AGENT} {click.style(f'({subtitle})', **_DIM)}:")
        else:
            click.echo(f"{_AGENT}:")
        _indent_body(text)

    def print_agent_step(self, step: AgentStep) -> None:
        """Show the agent's reasoning and intended action."""
        if step.thought:
            self.print_agent(step.thought)

        if step.action == AgentAction.RUN_SHELL and step.command:
            self.print_agent(step.command, subtitle="shell")
        elif step.action == AgentAction.RUN_PYTHON and step.code:
            preview = step.code.strip()
            if "\n" in preview:
                preview = preview.splitlines()[0] + " ..."
            self.print_agent(preview, subtitle="python")
        elif step.message and step.action in (
            AgentAction.NEED_USER_INPUT,
            AgentAction.TASK_COMPLETE,
            AgentAction.FAILED,
        ):
            self.print_agent(step.message)

    def print_tool_result(self, result: ToolResult) -> None:
        """Brief tool output summary after execution (debug mode only)."""
        if not self.debug:
            return

        lines: list[str] = []
        if result.timed_out:
            lines.append(click.style("Timed out.", fg="yellow"))
        elif result.exit_code is not None and result.exit_code != 0:
            lines.append(click.style(f"Exit code: {result.exit_code}", fg="yellow"))

        if result.stdout.strip():
            out = result.stdout.rstrip()
            if len(out) > 400:
                out = out[:400] + "\n... [output truncated for display]"
            lines.append(out)

        if result.stderr.strip():
            err = result.stderr.rstrip()
            if len(err) > 200:
                err = err[:200] + " ..."
            lines.append(click.style(err, fg="red"))

        if not lines:
            lines.append(click.style("(no output)", **_DIM))

        click.echo()
        click.echo(f"{_AGENT} {click.style('(output)', **_DIM)}:")
        _indent_body("\n".join(lines))

    @contextmanager
    def thinking(self, label: str = "is thinking") -> Iterator[None]:
        """Show a spinner on stderr while the agent waits on the LLM."""
        with _ThinkingSpinner(label) as spinner:
            yield spinner


class _ThinkingSpinner:
    def __init__(self, label: str) -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _ThinkingSpinner:
        if not sys.stderr.isatty():
            click.echo(f"{_AGENT} {self._label}...", err=True)
            return self

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if sys.stderr.isatty():
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()

    def _run(self) -> None:
        frames = itertools.cycle(_SPINNER_FRAMES)
        while not self._stop.is_set():
            frame = next(frames)
            sys.stderr.write(f"\r  {_AGENT} {frame} {self._label}...")
            sys.stderr.flush()
            if self._stop.wait(0.08):
                break


@contextmanager
def tool_activity(ui: ConversationUI, subtitle: str) -> Iterator[None]:
    """Spinner while a shell or Python tool runs (TTY only)."""
    if not sys.stderr.isatty():
        yield
        return
    with ui.thinking(label=f"is running {subtitle}"):
        yield
