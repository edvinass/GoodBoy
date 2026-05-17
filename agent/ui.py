"""Terminal formatting and activity indicators for the harness."""

from __future__ import annotations

import json
import os
import re
import select
import sys
import termios
import threading
import time
import tty
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import click
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document, PasteMode
from prompt_toolkit.shortcuts import CompleteStyle, PromptSession
from prompt_toolkit.styles import Style, merge_styles
from questionary.constants import DEFAULT_STYLE
from rich.box import ROUNDED
from rich.console import Console, ConsoleDimensions, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.tree import Tree

from agent.banner import format_startup
from agent.mentions import (
    active_mention_query,
    expand_file_mentions,
    search_workspace_paths,
)
from agent.repl_commands import (
    active_slash_command_query,
    search_slash_commands,
    slash_command_display_meta,
)
from agent.workspace import resolve_workspace
from settings import get_settings
from agent.types import AgentAction, AgentStep, ToolResult

_BRAND_STYLE = "rgb(139,69,19)"

_THEME = Theme(
    {
        "agent": f"bold {_BRAND_STYLE}",
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

_resize_poller_installed = False


def _install_resize_poller(ui: "ConversationUI") -> None:
    global _resize_poller_installed
    if _resize_poller_installed:
        return
    _resize_poller_installed = True
    ui._last_terminal_width: int | None = None

    def _poll() -> None:
        while True:
            time.sleep(0.25)
            current = ui._fresh_terminal_width()
            if ui._last_terminal_width is None:
                ui._last_terminal_width = current
                continue
            if current == ui._last_terminal_width:
                continue
            ui._last_terminal_width = current
            ui._request_redraw()

    threading.Thread(
        target=_poll,
        daemon=True,
        name="goodboy-resize-poller",
    ).start()


class _AutoWidthConsole(Console):
    """Console sized from the output stream, not a stale COLUMNS env var."""

    @property
    def size(self) -> ConsoleDimensions:
        file = self.file
        if hasattr(file, "isatty") and file.isatty() and hasattr(file, "fileno"):
            try:
                width, height = os.get_terminal_size(file.fileno())
                return ConsoleDimensions(
                    max(width - self.legacy_windows, 1),
                    max(height, 1),
                )
            except OSError:
                pass
        return super().size


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


class _UserInputCompleter(Completer):
    """Offer slash commands at line start or file paths after @."""

    def __init__(
        self,
        workspace: Path,
        *,
        show_commands: bool = False,
        auto_model_switch: bool = False,
        default_reasoning_effort: str | None = None,
    ) -> None:
        self._workspace = workspace.resolve()
        self._show_commands = show_commands
        self._auto_model_switch = auto_model_switch
        self._default_reasoning_effort = default_reasoning_effort

    def get_completions(self, document: Document, complete_event: object) -> Iterator[Completion]:
        del complete_event
        text_before = document.text_before_cursor

        slash = active_slash_command_query(text_before)
        if slash is not None:
            query, start_position = slash
            for command in search_slash_commands(query):
                yield Completion(
                    text=command.name,
                    start_position=start_position,
                    display=[("class:mention.choice", f"+ /{command.name}")],
                    display_meta=slash_command_display_meta(
                        command,
                        show_commands=self._show_commands,
                        auto_model_switch=self._auto_model_switch,
                        default_reasoning_effort=self._default_reasoning_effort,
                    ),
                )
            return

        mention = active_mention_query(text_before)
        if mention is None:
            return
        query, start_position = mention
        for path in search_workspace_paths(self._workspace, query):
            yield Completion(
                text=path,
                start_position=start_position,
                display=[("class:mention.choice", f"+ {path}")],
            )


def _prompt_user_line(
    paste_state: _PasteState,
    *,
    workspace: Path | None = None,
    show_commands: bool = False,
    auto_model_switch: bool = False,
    default_reasoning_effort: str | None = None,
) -> str | None:
    """Single-line prompt: Enter sends; multiline paste is collapsed to a label."""
    root = (workspace or resolve_workspace()).resolve()
    style = merge_styles(
        [
            DEFAULT_STYLE,
            Style.from_dict(
                {
                    "placeholder": "dim",
                    "question": "",
                    "mention.choice": "ansibrightblue",
                    "completion-menu": "bg:#1e1e1e",
                    "completion-menu.completion": "",
                    "completion-menu.completion.current": "bg:ansiblue",
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
        completer=_UserInputCompleter(
            root,
            show_commands=show_commands,
            auto_model_switch=auto_model_switch,
            default_reasoning_effort=default_reasoning_effort,
        ),
        complete_while_typing=True,
        complete_style=CompleteStyle.COLUMN,
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
    root = Tree(f"[bold {_BRAND_STYLE}]📂[/] [bold].[/]")
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


def _role_panel_title(role: str, *, subtitle: str | None = None) -> Text:
    if role == "user":
        return Text.from_markup("[user]👤 You[/]")
    title = Text.from_markup("[agent]🐶 GoodBoy[/]")
    if subtitle:
        icon, label = _SUBTITLE_ICONS.get(subtitle, ("·", subtitle))
        title.append(f"  [{icon}] ", style="subtitle")
        title.append(f"({label})", style="subtitle")
    return title


def _render_body(text: str, *, subtitle: str | None = None, width: int = 100) -> RenderableType:
    body = text.rstrip() or ""
    panel_width = max(width - 6, 40)
    agent_title = _role_panel_title("agent", subtitle=subtitle)
    panel_kwargs = {
        "border_style": _BRAND_STYLE,
        "box": ROUNDED,
        "padding": (0, 1),
        "width": width,
        "title": agent_title,
    }
    if subtitle == "shell":
        return Panel(
            _syntax(body, "bash", width=panel_width),
            title=agent_title,
            subtitle="[shell]command[/]",
            border_style="yellow",
            box=ROUNDED,
            padding=(0, 1),
            width=width,
        )
    if subtitle == "python":
        return Panel(
            _syntax(body, "python", width=panel_width),
            title=agent_title,
            subtitle="[python]code[/]",
            border_style="magenta",
            box=ROUNDED,
            padding=(0, 1),
            width=width,
        )
    if _looks_like_directory_listing(body):
        return Panel(
            _directory_tree(body),
            title=agent_title,
            subtitle="[muted]directory[/]",
            border_style=_BRAND_STYLE,
            box=ROUNDED,
            padding=(0, 1),
            width=width,
        )
    if _looks_like_markdown(body):
        return Panel(
            Markdown(body),
            **panel_kwargs,
        )
    return Panel(
        _wrap_long_lines(body, width=panel_width),
        **panel_kwargs,
    )


class ConversationUI:
    """Format user and agent messages distinctly in the terminal."""

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
        console: Console | None = None,
        workspace: Path | str | None = None,
        model: str | None = None,
    ) -> None:
        self.show_thoughts = show_thoughts
        self.verbose = verbose
        self.show_model = show_model
        self.show_commands = show_commands
        self.auto_model_switch = auto_model_switch
        self.debug = debug
        self.debug_input = debug_input
        self.debug_output = debug_output
        self._workspace = (
            Path(workspace).resolve() if workspace is not None else None
        )
        self._session_model = model
        self._session_reasoning: str | None = get_settings().default_reasoning_effort
        self._console = console or _AutoWidthConsole(theme=_THEME)
        self._err = _AutoWidthConsole(theme=_THEME, stderr=True)
        self._last_terminal_width: int | None = None
        self._history: list[tuple[str, dict[str, Any]]] = []
        self._display_lock = threading.Lock()
        self._at_prompt = False
        self._pending_redraw = False
        self._transient_ui = False
        self._stop_requested = False
        _install_resize_poller(self)


    def request_stop_after_current_step(self) -> None:
        """Request that the agent pause after the currently running step."""
        self._stop_requested = True

    def consume_stop_requested(self) -> bool:
        """Return and clear a pending stop-after-current-step request."""
        requested = self._stop_requested
        self._stop_requested = False
        return requested

    @contextmanager
    def _escape_stop_listener(self) -> Iterator[None]:
        """Listen for Escape while the agent is busy, without aborting the step."""
        if (
            not sys.stdin.isatty()
            or not sys.stdout.isatty()
            or threading.current_thread() is not threading.main_thread()
        ):
            yield
            return

        stop_event = threading.Event()
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except (OSError, termios.error):
            yield
            return

        def listen() -> None:
            try:
                while not stop_event.is_set():
                    readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not readable:
                        continue
                    ch = os.read(fd, 1)
                    if ch == b"\x1b":
                        self.request_stop_after_current_step()
                        stop_event.set()
                        return
            except OSError:
                return

        try:
            tty.setcbreak(fd)
            listener = threading.Thread(
                target=listen,
                daemon=True,
                name="goodboy-escape-listener",
            )
            listener.start()
            yield
        finally:
            stop_event.set()
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except (OSError, termios.error):
                pass

    def _is_interactive_tty(self) -> bool:
        return sys.stdout.isatty()

    def _fresh_terminal_width(self) -> int:
        """Current stdout columns; avoids stale Rich/COLUMNS sizing after resize."""
        try:
            if sys.stdout.isatty():
                columns = os.get_terminal_size(sys.stdout.fileno()).columns
                return max(columns - self._console.legacy_windows, 40)
        except OSError:
            pass
        return max(self._console.width, 40)

    def _panel(self, renderable: RenderableType, **kwargs) -> Panel:
        kwargs.setdefault("box", ROUNDED)
        kwargs.setdefault("width", self._fresh_terminal_width())
        return Panel(renderable, **kwargs)

    def _panel_text_width(self) -> int:
        return max(self._fresh_terminal_width() - 6, 40)

    def _header_title(self, role: str, *, subtitle: str | None = None) -> Text:
        return _role_panel_title(role, subtitle=subtitle)

    def _iter_history_block(self, kind: str, data: dict[str, Any]) -> Iterator[RenderableType | str]:
        if kind == "startup":
            model = data.get("model", self.session_model)
            yield self._panel(
                Text.from_markup(format_startup(model=model)),
                border_style=_BRAND_STYLE,
                padding=(0, 2),
            )
            return
        if kind == "header":
            yield ""
            yield self._header_title(data["role"], subtitle=data.get("subtitle"))
            return
        if kind == "user_message":
            yield ""
            user_title = _role_panel_title("user")
            if data.get("paste_label"):
                yield self._panel(
                    data["paste_label"],
                    title=user_title,
                    border_style="green",
                    padding=(0, 1),
                )
            panel_width = self._panel_text_width()
            yield self._panel(
                _wrap_long_lines(data["text"].rstrip() or "", width=panel_width),
                title=user_title,
                border_style="green",
                padding=(0, 1),
            )
            return
        if kind == "agent_message":
            yield ""
            yield _render_body(
                data["text"],
                subtitle=data.get("subtitle"),
                width=self._fresh_terminal_width(),
            )
            return
        if kind == "routing":
            yield ""
            yield self._kv_table(data["rows"])
            return
        if kind == "thought":
            yield ""
            panel_width = self._panel_text_width()
            yield self._panel(
                _wrap_long_lines(data["text"], width=panel_width),
                title=_role_panel_title("agent", subtitle="thought"),
                border_style=_BRAND_STYLE,
                padding=(0, 1),
            )
            return
        if kind == "status":
            yield ""
            yield data["markup"]
            return
        if kind == "tool_result":
            yield ""
            yield self._render_tool_result_group(data)
            return
        if kind == "llm_request":
            yield ""
            yield self._render_llm_request_group(data)
            return
        if kind == "llm_response":
            yield ""
            panel_width = self._panel_text_width()
            yield self._panel(
                _syntax(_pretty_json(data["raw"]), "json", width=panel_width),
                title=_role_panel_title("agent", subtitle="output"),
                subtitle=f"[muted]turn {data['turn']}[/]",
                border_style="dim",
                padding=(0, 1),
            )
            return

    def _render_tool_result_group(self, data: dict[str, Any]) -> Group:
        meta_rows: list[tuple[str, str]] = []
        if data.get("timed_out"):
            meta_rows.append(("status", "[warning]timed out[/]"))
        elif data.get("exit_code") is not None:
            style = "success" if data["exit_code"] == 0 else "warning"
            meta_rows.append(("exit code", f"[{style}]{data['exit_code']}[/]"))
        terminal_width = self._fresh_terminal_width()
        panel_width = self._panel_text_width()
        meta = self._kv_table(meta_rows, nested=True)
        parts: list[RenderableType] = [
            self._panel(
                meta,
                title="[muted]run[/]",
                border_style="dim",
                width=terminal_width,
            )
        ]
        stdout = data.get("stdout", "")
        stderr = data.get("stderr", "")
        if stdout.strip():
            parts.append(
                self._panel(
                    _syntax(stdout.rstrip(), "text", width=panel_width),
                    title="[muted]stdout[/]",
                    border_style=_BRAND_STYLE,
                    padding=(0, 1),
                    width=terminal_width,
                )
            )
        if stderr.strip():
            parts.append(
                self._panel(
                    Text(
                        _wrap_long_lines(stderr.rstrip(), width=panel_width),
                        style="error",
                    ),
                    title="[muted]stderr[/]",
                    border_style="red",
                    padding=(0, 1),
                    width=terminal_width,
                )
            )
        if not stdout.strip() and not stderr.strip() and not data.get("timed_out"):
            parts.append(
                self._panel("[muted](no output)[/]", border_style="dim", width=terminal_width)
            )
        return self._panel(
            Group(*parts),
            title=_role_panel_title("agent", subtitle="output"),
            border_style=_BRAND_STYLE,
            padding=(0, 0),
            width=terminal_width,
        )

    def _render_llm_request_group(self, data: dict[str, Any]) -> Group:
        meta_rows: list[tuple[str, str]] = [
            ("turn", str(data["turn"])),
            ("model", data["model"]),
        ]
        if data.get("reasoning_effort"):
            meta_rows.append(("reasoning", data["reasoning_effort"]))
        terminal_width = self._fresh_terminal_width()
        panel_width = self._panel_text_width()
        meta = self._kv_table(meta_rows, box=None, nested=True)
        return self._panel(
            Group(
                self._panel(
                    meta,
                    title="[muted]request[/]",
                    border_style="dim",
                    width=terminal_width,
                ),
                self._panel(
                    _syntax(data["instructions"], "markdown", width=panel_width),
                    title="[muted]instructions[/]",
                    border_style="dim",
                    padding=(0, 1),
                    width=terminal_width,
                ),
                self._panel(
                    _syntax(data["input_text"], "markdown", width=panel_width),
                    title="[muted]input[/]",
                    border_style="dim",
                    padding=(0, 1),
                    width=terminal_width,
                ),
            ),
            title=_role_panel_title("agent", subtitle="input"),
            border_style=_BRAND_STYLE,
            padding=(0, 0),
            width=terminal_width,
        )

    def _redraw_all(self) -> None:
        if not self._history:
            return
        self._console.clear()
        for kind, data in self._history:
            for item in self._iter_history_block(kind, data):
                if item == "":
                    self._console.print()
                else:
                    self._console.print(item)

    def _request_redraw(self) -> None:
        if not self._history or not self._is_interactive_tty():
            return
        if self._transient_ui:
            self._pending_redraw = True
            return
        with self._display_lock:
            self._redraw_all()
            if self._at_prompt:
                self._print_header("user")

    def _sync_redraw(self) -> None:
        if not self._pending_redraw or not self._history or not self._is_interactive_tty():
            self._pending_redraw = False
            return
        with self._display_lock:
            self._redraw_all()
        self._pending_redraw = False

    def _record(self, kind: str, **data: Any) -> None:
        self._history.append((kind, data))
        if self._is_interactive_tty():
            with self._display_lock:
                self._redraw_all()
            return
        for item in self._iter_history_block(kind, data):
            if item == "":
                self._console.print()
            else:
                self._console.print(item)

    def _kv_table(
        self,
        rows: list[tuple[str, str]],
        *,
        box=ROUNDED,
        fit: bool = False,
        nested: bool = False,
    ) -> Table:
        """Key/value table sized to the terminal or parent panel."""
        table_kwargs: dict = {
            "show_header": False,
            "box": box,
            "border_style": "dim",
            "padding": (0, 1),
        }
        if not fit:
            table_kwargs["width"] = (
                self._panel_text_width() if nested else self._fresh_terminal_width()
            )
            table_kwargs["expand"] = True
        table = Table(**table_kwargs)
        table.add_column(style="muted", no_wrap=True)
        table.add_column(overflow="fold")
        for key, value in rows:
            table.add_row(key, value)
        return table

    @property
    def session_model(self) -> str:
        return self._session_model or get_settings().default_model

    def set_session_model(self, model: str) -> None:
        """Update the model shown on the startup banner for this session."""
        self._session_model = model
        for index, (kind, data) in enumerate(self._history):
            if kind == "startup":
                self._history[index] = ("startup", {"model": model})
                break
        self._sync_redraw()

    @property
    def session_reasoning(self) -> str | None:
        return self._session_reasoning

    def set_session_reasoning(self, effort: str | None) -> None:
        """Update the default reasoning effort used for slash-command completion."""
        self._session_reasoning = effort

    def print_startup(self) -> None:
        self._record("startup", model=self.session_model)

    def clear_session(self) -> None:
        """Clear the on-screen transcript and show the startup banner again."""
        self._history.clear()
        self._pending_redraw = False
        self._stop_requested = False
        if self._is_interactive_tty():
            with self._display_lock:
                self._console.clear()
        self._record("startup", model=self.session_model)

    def print_task_complete(self) -> None:
        if not self.verbose:
            return
        self._record("status", markup="[success]✓[/] [success]Task complete[/]")

    def print_failed(self, message: str | None = None) -> None:
        if not self.verbose:
            if message:
                self.print_agent(message)
            return
        self._record("status", markup="[error]✗[/] [error]Failed[/]")
        if message:
            self._err.print(
                self._panel(message, border_style="red", width=self._fresh_terminal_width())
            )

    def print_stopped(self, message: str) -> None:
        if not self.verbose:
            if message:
                self.print_agent(message)
            return
        self._record("status", markup="[warning]⚠[/] [warning]Stopped[/]")
        self._err.print(
            self._panel(message, border_style="yellow", width=self._fresh_terminal_width())
        )

    def print_notice(self, message: str) -> None:
        self._err.print(f"[warning]{message}[/]")

    def newline(self) -> None:
        self._console.print()

    def _print_header(self, role: str, *, subtitle: str | None = None) -> None:
        self._console.print()
        self._console.print(self._header_title(role, subtitle=subtitle))

    def prompt_user(self) -> str:
        """Read user input; Enter sends; multiline paste shows a collapsed label."""
        self._sync_redraw()
        self._at_prompt = True
        try:
            self._print_header("user")
            paste_state = _PasteState()
            result = _prompt_user_line(
                paste_state,
                workspace=self._workspace or resolve_workspace(),
                show_commands=self.show_commands,
                auto_model_switch=self.auto_model_switch,
                default_reasoning_effort=self._session_reasoning,
            )
            if result is None:
                raise click.Abort()
            text = paste_state.resolve(result).strip()
            workspace = self._workspace or resolve_workspace()
            text = expand_file_mentions(text, workspace)
            self._record(
                "user_message",
                text=text,
                paste_label=paste_state.label,
            )
            return text
        finally:
            self._at_prompt = False
            self._sync_redraw()

    def print_user(self, text: str) -> None:
        self._record("user_message", text=text.rstrip() or "")

    def print_agent(
        self,
        text: str,
        *,
        subtitle: str | None = None,
    ) -> None:
        self._record("agent_message", text=text, subtitle=subtitle)

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

        self._record(
            "llm_request",
            turn=turn,
            model=model,
            reasoning_effort=reasoning_effort,
            instructions=instructions,
            input_text=input_text,
        )

    def print_llm_response(self, *, turn: int, raw: str) -> None:
        """Print raw model response without truncation (debug -o)."""
        if not self.debug_output:
            return
        self._record("llm_response", turn=turn, raw=raw)

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
            if self._show_thoughts and step.thought:
                self._record("thought", text=step.thought)
            if step.message and step.action in (
                AgentAction.NEED_USER_INPUT,
                AgentAction.TASK_COMPLETE,
                AgentAction.FAILED,
            ):
                self.print_agent(step.message)
            if step.action == AgentAction.SWITCH_MODEL and step.model:
                self.print_notice(
                    f"Model set to {step.model} for the next turn."
                )
            elif step.action in (
                AgentAction.SWITCH_TOOLS,
                AgentAction.SWITCH_API,
            ) and step.tools:
                self.print_notice(
                    f"Hosted tools enabled: {', '.join(step.tools)}."
                )
            elif self._show_tool_commands and step.action == AgentAction.RUN_SHELL and step.command:
                self.print_agent(step.command, subtitle="shell")
            elif self._show_tool_commands and step.action == AgentAction.RUN_PYTHON and step.code:
                preview = step.code.strip()
                if "\n" in preview:
                    preview = preview.splitlines()[0] + " ..."
                self.print_agent(preview, subtitle="python")
            elif self.show_model and (model or reasoning):
                routing_rows: list[tuple[str, str]] = []
                if model:
                    routing_rows.append(("model", model))
                if reasoning:
                    routing_rows.append(("reasoning", reasoning))
                self._record("routing", rows=routing_rows)
            return

        show_routing = (
            (self.show_model and model)
            or next_model
            or next_reasoning
            or hosted_tools
        )
        if show_routing:
            routing_rows = []
            if self.show_model and model:
                routing_rows.append(("model", model))
            if self.show_model and reasoning:
                routing_rows.append(("reasoning", reasoning))
            if hosted_tools:
                routing_rows.append(("hosted tools", ", ".join(hosted_tools)))
            if next_model:
                routing_rows.append(("next model", next_model))
            if next_reasoning:
                routing_rows.append(("next reasoning", next_reasoning))
            self._record("routing", rows=routing_rows)

        if self._show_thoughts and step.thought:
            self._record("thought", text=step.thought)

        if self._show_tool_commands and step.action == AgentAction.RUN_SHELL and step.command:
            self.print_agent(step.command, subtitle="shell")
        elif self._show_tool_commands and step.action == AgentAction.RUN_PYTHON and step.code:
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
    def _show_thoughts(self) -> bool:
        return self.show_thoughts or self.verbose

    @property
    def _show_tool_commands(self) -> bool:
        return self.show_commands or self.debug

    @property
    def _show_tool_output(self) -> bool:
        return self.debug

    def print_tool_result(self, result: ToolResult) -> None:
        """Brief tool output summary after execution (-d)."""
        if not self._show_tool_output:
            return
        self._record(
            "tool_result",
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
        )

    @contextmanager
    def thinking(self, label: str = "is thinking") -> Iterator[None]:
        """Show a spinner while the agent waits on the LLM."""
        if not sys.stdout.isatty():
            self._console.print(f"[agent]◆ GoodBoy[/] [muted]{label}…[/]")
            yield
            return

        self._transient_ui = True
        try:
            with self._escape_stop_listener():
                with self._console.status(
                    f"[agent]🐶 GoodBoy[/] [muted]{label}…[/]",
                    spinner="dots",
                ):
                    yield
        finally:
            self._transient_ui = False
            self._sync_redraw()


@contextmanager
def tool_activity(ui: ConversationUI, subtitle: str) -> Iterator[None]:
    """Spinner while a shell or Python tool runs (TTY only)."""
    if not sys.stderr.isatty():
        yield
        return
    with ui.thinking(label=f"is running {subtitle}"):
        yield
