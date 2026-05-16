"""Terminal formatting and activity indicators for the harness."""

from __future__ import annotations

import json
import re
import sys
from contextlib import contextmanager
from typing import Iterator

import click
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.document import PasteMode
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.styles import Style, merge_styles
from questionary.constants import DEFAULT_STYLE
from rich.box import ROUNDED
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.tree import Tree

from agent.banner import format_startup
from settings import get_settings
from agent.types import AgentAction, AgentStep, ToolResult

_THEME = Theme(
    {
        "agent": "bold cyan",
        "user": "bold green",
        "subtitle": "dim italic",
        "shell": "yellow",
        "python": "magenta",
        "success": "bold green",
        "error": "bold red",
        "warning": "bold yellow",
        "muted": "dim",
    }
)

_SUBTITLE_ICONS = {
    "shell": ("$", "shell"),
    "python": ("›", "python"),
    "output": ("↳", "output"),
}

_LS_SECTION = re.compile(r"^\./(.+):$")

_USER_INPUT_PLACEHOLDER = "Ask anything…"


def format_pasted_text_label(paste_id: int, line_count: int) -> str:
    """Summary shown when the user pastes multiline text (e.g. +52 lines = 53 total)."""
    if line_count < 2:
        raise ValueError("line_count must be at least 2 for a paste label")
    return f"[Pasted text #{paste_id} +{line_count - 1} lines]"


class _PasteState:
    """Tracks multiline clipboard paste for display vs. submitted text."""

    def __init__(self) -> None:
        self.counter = 0
        self.stored: str | None = None
        self.label: str | None = None

    def register_paste(self, text: str) -> str | None:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        if len(lines) < 2:
            return None
        self.counter += 1
        self.stored = normalized.rstrip("\n")
        self.label = format_pasted_text_label(self.counter, len(lines))
        return self.label

    def resolve(self, buffer_text: str) -> str:
        if self.stored is not None:
            return self.stored
        return buffer_text


def _attach_paste_handler(buffer: Buffer, paste_state: _PasteState) -> None:
    """Collapse multiline paste (Ctrl+V and bracketed paste) to a short label."""
    original_paste = buffer.paste_clipboard_data

    def paste_clipboard_data(
        data: ClipboardData,
        paste_mode: PasteMode = PasteMode.EMACS,
        count: int = 1,
    ) -> None:
        if count != 1:
            original_paste(data, paste_mode=paste_mode, count=count)
            return
        label = paste_state.register_paste(data.text)
        if label is None:
            original_paste(data, paste_mode=paste_mode, count=count)
            return
        buffer.text = label

    buffer.paste_clipboard_data = paste_clipboard_data  # type: ignore[method-assign]


def _prompt_user_line(paste_state: _PasteState) -> str | None:
    """Single-line prompt: Enter sends; multiline paste is collapsed to a label."""
    style = merge_styles(
        [
            DEFAULT_STYLE,
            Style.from_dict(
                {
                    "placeholder": "dim",
                    "question": "",
                }
            ),
        ]
    )

    def get_prompt_tokens() -> list[tuple[str, str]]:
        return [("class:question", " │ ")]

    session = PromptSession(
        get_prompt_tokens,
        style=style,
        multiline=False,
        placeholder=_USER_INPUT_PLACEHOLDER,
    )
    _attach_paste_handler(session.default_buffer, paste_state)
    try:
        return session.prompt()
    except (KeyboardInterrupt, EOFError):
        return None


def _wrap_long_lines(text: str, *, width: int) -> str:
    """Break lines longer than width so panels do not clip with an ellipsis."""
    if width < 20:
        width = 20
    wrapped: list[str] = []
    for line in text.splitlines():
        if len(line) <= width:
            wrapped.append(line)
            continue
        start = 0
        while start < len(line):
            wrapped.append(line[start : start + width])
            start += width
    return "\n".join(wrapped)


def _syntax(text: str, lexer: str, *, width: int) -> Syntax:
    """Syntax block with long lines folded to the available width."""
    return Syntax(
        _wrap_long_lines(text, width=width),
        lexer,
        theme="monokai",
        background_color="default",
        word_wrap=True,
    )


def _pretty_json(raw: str) -> str:
    """Format JSON for display; fall back to raw text if not valid JSON."""
    try:
        return json.dumps(json.loads(raw), indent=2)
    except (json.JSONDecodeError, TypeError):
        return raw


def _looks_like_markdown(text: str) -> bool:
    if "```" in text:
        return True
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return True
        if stripped.startswith(("- ", "* ", "1. ")):
            return True
    return False


def _looks_like_directory_listing(text: str) -> bool:
    return bool(_LS_SECTION.search(text))


def _directory_tree(text: str) -> Tree:
    root = Tree("[bold cyan]📂[/] [bold].[/]")
    current: Tree | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        section = _LS_SECTION.match(line)
        if section:
            node = root
            for part in section.group(1).split("/"):
                node = _tree_find_or_add(node, part)
            current = node
            continue
        if current is not None:
            current.add(f"[muted]{line}[/]")
        else:
            root.add(line)

    return root


def _tree_label_plain(node: Tree) -> str:
    label = node.label
    return label.plain if isinstance(label, Text) else str(label)


def _tree_find_or_add(parent: Tree, label: str) -> Tree:
    for child in parent.children:
        if _tree_label_plain(child) == label:
            return child
    return parent.add(f"[bold]{label}[/]")


def _render_body(text: str, *, subtitle: str | None = None, width: int = 100) -> RenderableType:
    body = text.rstrip() or ""
    panel_width = max(width - 6, 40)
    if subtitle == "shell":
        return Panel(
            _syntax(body, "bash", width=panel_width),
            title="[shell]command[/]",
            border_style="yellow",
            box=ROUNDED,
            padding=(0, 1),
        )
    if subtitle == "python":
        return Panel(
            _syntax(body, "python", width=panel_width),
            title="[python]code[/]",
            border_style="magenta",
            box=ROUNDED,
            padding=(0, 1),
        )
    if _looks_like_directory_listing(body):
        return Panel(
            _directory_tree(body),
            title="[muted]directory[/]",
            border_style="cyan",
            box=ROUNDED,
            padding=(0, 1),
        )
    if _looks_like_markdown(body):
        return Panel(
            Markdown(body),
            border_style="cyan",
            box=ROUNDED,
            padding=(0, 1),
        )
    return Panel(body, border_style="cyan", box=ROUNDED, padding=(0, 1))


class ConversationUI:
    """Format user and agent messages distinctly in the terminal."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        show_model: bool = False,
        show_commands: bool = False,
        debug: bool = False,
        debug_input: bool = False,
        debug_output: bool = False,
        console: Console | None = None,
    ) -> None:
        self.verbose = verbose
        self.show_model = show_model
        self.show_commands = show_commands
        self.debug = debug
        self.debug_input = debug_input
        self.debug_output = debug_output
        self._console = console or Console(theme=_THEME)
        self._err = Console(theme=_THEME, stderr=True)

    def _panel_text_width(self) -> int:
        return max(self._console.width - 6, 40)

    def print_startup(self) -> None:
        self._console.print()
        self._console.print(
            Panel(
                Text.from_markup(format_startup(model=get_settings().default_model)),
                border_style="cyan",
                box=ROUNDED,
                padding=(0, 2),
            )
        )

    def print_task_complete(self) -> None:
        if not self.verbose:
            return
        self._console.print()
        self._console.print("[success]✓[/] [success]Task complete[/]")

    def print_failed(self, message: str | None = None) -> None:
        if not self.verbose:
            if message:
                self.print_agent(message)
            return
        self._console.print()
        self._console.print("[error]✗[/] [error]Failed[/]")
        if message:
            self._err.print(Panel(message, border_style="red", box=ROUNDED))

    def print_stopped(self, message: str) -> None:
        if not self.verbose:
            if message:
                self.print_agent(message)
            return
        self._console.print()
        self._console.print("[warning]⚠[/] [warning]Stopped[/]")
        self._err.print(Panel(message, border_style="yellow", box=ROUNDED))

    def print_notice(self, message: str) -> None:
        self._err.print(f"[warning]{message}[/]")

    def newline(self) -> None:
        self._console.print()

    def _print_header(self, role: str, *, subtitle: str | None = None) -> None:
        self._console.print()
        if role == "user":
            title = Text.from_markup("[user]▸ You[/]")
        else:
            title = Text.from_markup("[agent]◆ GoodBoy[/]")
            if subtitle:
                icon, label = _SUBTITLE_ICONS.get(subtitle, ("·", subtitle))
                title.append(f"  [{icon}] ", style="subtitle")
                title.append(f"({label})", style="subtitle")
        self._console.print(title)

    def prompt_user(self) -> str:
        """Read user input; Enter sends; multiline paste shows a collapsed label."""
        self._print_header("user")
        paste_state = _PasteState()
        result = _prompt_user_line(paste_state)
        if result is None:
            raise click.Abort()
        text = paste_state.resolve(result)
        if paste_state.label:
            self._console.print(
                Panel(
                    paste_state.label,
                    border_style="green",
                    box=ROUNDED,
                    padding=(0, 1),
                )
            )
        return text.strip()

    def print_user(self, text: str) -> None:
        self._print_header("user")
        self._console.print(
            Panel(
                text.rstrip() or "",
                border_style="green",
                box=ROUNDED,
                padding=(0, 1),
            )
        )

    def print_agent(
        self,
        text: str,
        *,
        subtitle: str | None = None,
    ) -> None:
        self._print_header("agent", subtitle=subtitle)
        self._console.print(
            _render_body(text, subtitle=subtitle, width=self._console.width)
        )

    def print_llm_request(
        self,
        *,
        turn: int,
        model: str,
        reasoning_effort: str | None,
        instructions: str,
        input_text: str,
    ) -> None:
        """Print full prompt payload sent to the model (debug -i)."""
        if not self.debug_input:
            return

        meta = Table(show_header=False, box=None, padding=(0, 1))
        meta.add_column(style="muted")
        meta.add_column()
        meta.add_row("turn", str(turn))
        meta.add_row("model", model)
        if reasoning_effort:
            meta.add_row("reasoning", reasoning_effort)

        panel_width = self._panel_text_width()
        self._print_header("agent", subtitle="input")
        self._console.print(
            Group(
                Panel(meta, title="[muted]request[/]", border_style="dim", box=ROUNDED),
                Panel(
                    _syntax(instructions, "markdown", width=panel_width),
                    title="[muted]instructions[/]",
                    border_style="dim",
                    box=ROUNDED,
                    padding=(0, 1),
                ),
                Panel(
                    _syntax(input_text, "markdown", width=panel_width),
                    title="[muted]input[/]",
                    border_style="dim",
                    box=ROUNDED,
                    padding=(0, 1),
                ),
            )
        )

    def print_llm_response(self, *, turn: int, raw: str) -> None:
        """Print raw model response without truncation (debug -o)."""
        if not self.debug_output:
            return

        panel_width = self._panel_text_width()
        self._print_header("agent", subtitle="output")
        self._console.print(
            Panel(
                _syntax(_pretty_json(raw), "json", width=panel_width),
                title=f"[muted]turn {turn}[/]",
                border_style="dim",
                box=ROUNDED,
                padding=(0, 1),
            )
        )

    def print_agent_step(
        self,
        step: AgentStep,
        *,
        model: str | None = None,
        reasoning: str | None = None,
        next_model: str | None = None,
        next_reasoning: str | None = None,
        hosted_tools: list[str] | None = None,
    ) -> None:
        """Show the agent's reasoning and intended action."""
        if not self.verbose:
            if step.message and step.action in (
                AgentAction.NEED_USER_INPUT,
                AgentAction.TASK_COMPLETE,
                AgentAction.FAILED,
            ):
                self.print_agent(step.message)
            if self._show_tool_io and step.action == AgentAction.RUN_SHELL and step.command:
                self.print_agent(step.command, subtitle="shell")
            elif self._show_tool_io and step.action == AgentAction.RUN_PYTHON and step.code:
                preview = step.code.strip()
                if "\n" in preview:
                    preview = preview.splitlines()[0] + " ..."
                self.print_agent(preview, subtitle="python")
            elif self.show_model and (model or reasoning):
                routing = Table(
                    show_header=False, box=ROUNDED, border_style="dim", padding=(0, 1)
                )
                routing.add_column(style="muted")
                routing.add_column()
                if model:
                    routing.add_row("model", model)
                if reasoning:
                    routing.add_row("reasoning", reasoning)
                self._console.print()
                self._console.print(routing)
            return

        show_routing = (
            (self.show_model and model)
            or next_model
            or next_reasoning
            or hosted_tools
        )
        if show_routing:
            routing = Table(show_header=False, box=ROUNDED, border_style="dim", padding=(0, 1))
            routing.add_column(style="muted")
            routing.add_column()
            if self.show_model and model:
                routing.add_row("model", model)
            if self.show_model and reasoning:
                routing.add_row("reasoning", reasoning)
            if hosted_tools:
                routing.add_row("hosted tools", ", ".join(hosted_tools))
            if next_model:
                routing.add_row("next model", next_model)
            if next_reasoning:
                routing.add_row("next reasoning", next_reasoning)
            self._console.print()
            self._console.print(routing)

        if step.thought:
            self._print_header("agent", subtitle="thought")
            self._console.print(
                Panel(
                    step.thought,
                    border_style="cyan",
                    box=ROUNDED,
                    padding=(0, 1),
                )
            )

        if self._show_tool_io and step.action == AgentAction.RUN_SHELL and step.command:
            self.print_agent(step.command, subtitle="shell")
        elif self._show_tool_io and step.action == AgentAction.RUN_PYTHON and step.code:
            preview = step.code.strip()
            if "\n" in preview:
                preview = preview.splitlines()[0] + " ..."
            self.print_agent(preview, subtitle="python")
        elif step.action == AgentAction.SWITCH_MODEL and step.model:
            self.print_agent(step.model, subtitle="model")
        elif step.action in (
            AgentAction.SWITCH_TOOLS,
            AgentAction.SWITCH_API,
        ) and step.tools:
            self.print_agent(
                f"Enable hosted tools: {', '.join(step.tools)}",
                subtitle="tools",
            )
        elif step.message and step.action in (
            AgentAction.NEED_USER_INPUT,
            AgentAction.TASK_COMPLETE,
            AgentAction.FAILED,
        ):
            self.print_agent(step.message)

    @property
    def _show_tool_io(self) -> bool:
        return self.show_commands or self.debug

    def print_tool_result(self, result: ToolResult) -> None:
        """Brief tool output summary after execution (-c / -d)."""
        if not self._show_tool_io:
            return

        meta = Table(show_header=False, box=ROUNDED, border_style="dim", padding=(0, 1))
        meta.add_column(style="muted")
        meta.add_column()
        if result.timed_out:
            meta.add_row("status", "[warning]timed out[/]")
        elif result.exit_code is not None:
            style = "success" if result.exit_code == 0 else "warning"
            meta.add_row("exit code", f"[{style}]{result.exit_code}[/]")

        panel_width = self._panel_text_width()
        parts: list[RenderableType] = [Panel(meta, title="[muted]run[/]", border_style="dim", box=ROUNDED)]

        if result.stdout.strip():
            parts.append(
                Panel(
                    _syntax(result.stdout.rstrip(), "text", width=panel_width),
                    title="[muted]stdout[/]",
                    border_style="cyan",
                    box=ROUNDED,
                    padding=(0, 1),
                )
            )

        if result.stderr.strip():
            parts.append(
                Panel(
                    Text(
                        _wrap_long_lines(result.stderr.rstrip(), width=panel_width),
                        style="error",
                    ),
                    title="[muted]stderr[/]",
                    border_style="red",
                    box=ROUNDED,
                    padding=(0, 1),
                )
            )

        if not result.stdout.strip() and not result.stderr.strip() and not result.timed_out:
            parts.append(Panel("[muted](no output)[/]", border_style="dim", box=ROUNDED))

        self._print_header("agent", subtitle="output")
        self._console.print(Group(*parts))

    @contextmanager
    def thinking(self, label: str = "is thinking") -> Iterator[None]:
        """Show a spinner on stderr while the agent waits on the LLM."""
        if not sys.stderr.isatty():
            self._err.print(f"[agent]◆ GoodBoy[/] [muted]{label}…[/]")
            yield
            return

        with self._err.status(
            f"[agent]◆ GoodBoy[/] [muted]{label}…[/]",
            spinner="dots",
        ):
            yield


@contextmanager
def tool_activity(ui: ConversationUI, subtitle: str) -> Iterator[None]:
    """Spinner while a shell or Python tool runs (TTY only)."""
    if not sys.stderr.isatty():
        yield
        return
    with ui.thinking(label=f"is running {subtitle}"):
        yield
