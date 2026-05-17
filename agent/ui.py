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
from dataclasses import dataclass
from typing import Any, Iterator, Literal

import click
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document, PasteMode
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition, has_focus, in_paste_mode
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import (
    ConditionalContainer,
    Dimension,
    Float,
    FloatContainer,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
from prompt_toolkit.shortcuts import PromptSession
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
    is_repl_slash_command,
    search_slash_commands,
    slash_command_display_meta,
)
from agent.workspace import resolve_workspace
from settings import get_settings
from agent.types import AgentAction, AgentStep, ToolResult
from llm import TokenUsage

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

_USER_INPUT_PLACEHOLDER = "Ask anything"
_USER_INPUT_FOOTER = "@ files, / commands"
_USER_INPUT_MENU_RESERVE = 8
_USER_INPUT_MAX_LINES = 16

_MAX_ACTIVITY_DIFF_LINES = 40

_resize_poller_installed = False


@dataclass
class ThinkingUpdater:
    """Update the GoodBoy loading line while the model streams a step."""

    _label: str
    _console: Console
    _status: Any = None
    _non_tty_printed: bool = False

    def update(self, label: str) -> None:
        cleaned = label.strip().rstrip(".…")
        if not cleaned or cleaned == self._label:
            return
        self._label = cleaned
        if self._status is not None:
            self._status.update(self._render())
        elif not self._non_tty_printed:
            self._non_tty_printed = True
            self._console.print(self._render())

    def _render(self) -> str:
        return f"[agent]🐶 GoodBoy[/] [muted]{self._label}…[/]"


def progress_label_for_step(step: AgentStep) -> str:
    """Spinner label while a harness tool runs; prefers model ``status``."""
    if step.status:
        return step.status.strip().rstrip(".…")
    return activity_label(step, phase="progress")


def activity_label(step: AgentStep, *, phase: Literal["progress", "done"]) -> str:
    """Human-facing status text for harness tool progress or completion."""
    name = Path(step.path).name if step.path else None

    if step.action == AgentAction.RUN_SHELL:
        return (
            "running shell command"
            if phase == "progress"
            else "ran shell command"
        )
    if step.action == AgentAction.RUN_PYTHON:
        return "running Python" if phase == "progress" else "ran Python"
    if step.action == AgentAction.READ_FILE:
        if name:
            return f"reading {name}" if phase == "progress" else f"read {name}"
        return "reading file" if phase == "progress" else "read file"
    if step.action in (AgentAction.STR_REPLACE, AgentAction.APPLY_PATCH):
        if name:
            return f"writing {name}" if phase == "progress" else f"wrote {name}"
        return "writing file" if phase == "progress" else "wrote file"

    action = step.action.value.replace("_", " ")
    return action if phase == "progress" else f"finished {action}"


def _truncate_diff_lines(diff: str, *, max_lines: int = _MAX_ACTIVITY_DIFF_LINES) -> str:
    lines = diff.splitlines()
    if len(lines) <= max_lines:
        return diff
    kept = lines[:max_lines]
    omitted = len(lines) - max_lines
    return "\n".join(kept) + f"\n... [{omitted} more lines]"


def _extract_unified_diff(stdout: str) -> str | None:
    """Pull a unified diff from tool stdout (e.g. after str_replace)."""
    marker = stdout.find("\n--- ")
    if marker >= 0:
        return stdout[marker + 1 :]
    if stdout.startswith("--- "):
        return stdout
    return None


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
    extra_lines = line_count - 1
    return f"[Pasted text #{paste_id} · +{extra_lines} lines]"


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
        stream_output: bool = False,
        session_model: str | None = None,
        default_reasoning_effort: str | None = None,
    ) -> None:
        self._workspace = workspace.resolve()
        self._show_commands = show_commands
        self._auto_model_switch = auto_model_switch
        self._stream_output = stream_output
        self._session_model = session_model
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
                        stream_output=self._stream_output,
                        session_model=self._session_model,
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


def _accept_active_completion(buffer: Buffer) -> bool:
    """Apply the highlighted completion and close the menu."""
    state = buffer.complete_state
    if state is None or not state.completions:
        return False
    completion = state.current_completion or state.completions[0]
    buffer.apply_completion(completion)
    return True


_KITTY_CSI_U_ENTER_RE = re.compile(r"^\x1b\[13(?:;(\d+))?(?::\d+)?u$")
_KITTY_CSI_U_D_RE = re.compile(r"^\x1b\[100(?:;(\d+))?(?::\d+)?u$")
_XTERM_MODIFYOTHERKEYS_RE = re.compile(r"^\x1b\[27;(\d+);(\d+)~$")
_ENHANCED_KEYBOARD_SEQUENCES_REGISTERED = False


def _is_shift_enter_data(data: str) -> bool:
    """True for terminal encodings of Shift+Enter (not plain Enter)."""
    # Line-feed, or VS Code sendSequence ESC+LF workaround.
    if data in ("\n", "\x1b\n"):
        return True
    # xterm modifyOtherKeys: CSI 27 ; 2 ; 13 ~
    if data.startswith("\x1b[27;2;"):
        return True
    # Kitty keyboard protocol: CSI 13 ; modifiers u
    match = _KITTY_CSI_U_ENTER_RE.match(data)
    if match is not None:
        modifier = match.group(1)
        if modifier is None:
            return False
        # Encoded value is 1 + bitmask; shift is bit 0.
        return (int(modifier) - 1) & 1 != 0
    return False


def _is_shift_enter(event: KeyPressEvent) -> bool:
    """True when Enter was pressed with Shift (terminal-specific encoding)."""
    return _is_shift_enter_data(event.data)


def _register_enhanced_keyboard_sequences() -> None:
    """Teach prompt_toolkit Kitty CSI u Enter sequences (not in defaults)."""
    global _ENHANCED_KEYBOARD_SEQUENCES_REGISTERED
    if _ENHANCED_KEYBOARD_SEQUENCES_REGISTERED:
        return
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.keys import Keys

    for sequence in (
        "\x1b[13u",
        "\x1b[13;2u",
        "\x1b[13;2:1u",
        "\x1b[13;2:3u",
        "\x1b[100;9u",
        "\x1b[27;9;100~",
    ):
        key = Keys.ControlM if sequence.startswith("\x1b[13") else Keys.ControlD
        ANSI_SEQUENCES.setdefault(sequence, key)
    _ENHANCED_KEYBOARD_SEQUENCES_REGISTERED = True


@contextmanager
def _enhanced_keyboard_reporting() -> Iterator[None]:
    """Ask the terminal to report Shift+Enter distinctly from Enter."""
    if not sys.stdout.isatty():
        yield
        return
    _register_enhanced_keyboard_sequences()
    try:
        # Kitty keyboard protocol (flag 1 = disambiguate legacy keys).
        sys.stdout.write("\x1b[>1u")
        # xterm modifyOtherKeys mode 2.
        sys.stdout.write("\x1b[>4;2m")
        sys.stdout.flush()
        yield
    finally:
        try:
            sys.stdout.write("\x1b[<1u\x1b[>4;0m")
            sys.stdout.flush()
        except OSError:
            pass


def _insert_input_newline(buffer: Buffer) -> None:
    buffer.newline(copy_margin=not in_paste_mode())


def _clear_user_input(buffer: Buffer) -> None:
    buffer.reset(Document())


@Condition
def _input_has_text() -> bool:
    return bool(get_app().current_buffer.text)


def _is_cmd_d_data(data: str) -> bool:
    """True for Cmd+D / Ctrl+D encodings (clear input, not EOF)."""
    if data == "\x04":
        return True
    match = _KITTY_CSI_U_D_RE.match(data)
    if match is not None:
        modifier = match.group(1)
        if modifier is None:
            return False
        # Kitty: encoded value is 1 + bitmask; super (Cmd) is bit 3.
        return (int(modifier) - 1) & 8 != 0
    match = _XTERM_MODIFYOTHERKEYS_RE.match(data)
    if match is not None and int(match.group(2)) == 100:
        modifier = int(match.group(1))
        return (modifier - 1) & 8 != 0
    return False


def _prompt_clear_input_key_bindings() -> KeyBindings:
    """Cmd+D / Ctrl+D clears the prompt when it has text."""
    kb = KeyBindings()

    @kb.add("c-d", filter=_input_has_text, eager=True)
    def _clear_on_cmd_d(event: KeyPressEvent) -> None:
        if not _is_cmd_d_data(event.data):
            return
        _clear_user_input(event.current_buffer)

    return kb


def _prompt_enter_key_bindings() -> KeyBindings:
    """Enter submits; Shift+Enter inserts a newline; Enter accepts completions."""
    kb = KeyBindings()

    @kb.add("enter", eager=True)
    def _handle_enter(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        if _is_shift_enter(event):
            _insert_input_newline(buffer)
            return
        if buffer.complete_state is not None and buffer.complete_state.completions:
            _accept_active_completion(buffer)
            return
        buffer.validate_and_handle()

    return kb


def _prompt_newline_key_bindings() -> KeyBindings:
    """Shift+Enter (often sent as c-j / line-feed) inserts a newline."""
    kb = KeyBindings()

    @kb.add("c-j", eager=True)
    def _insert_newline(event: KeyPressEvent) -> None:
        _insert_input_newline(event.current_buffer)

    return kb


def _input_window_line_count(buffer: Buffer) -> int:
    return max(1, min(buffer.document.line_count, _USER_INPUT_MAX_LINES))


def _input_window_height(buffer: Buffer) -> Dimension:
    return Dimension.exact(_input_window_line_count(buffer))


_PROMPT_TOOLKIT_CONFIGURED = False


def _configure_prompt_toolkit() -> None:
    """Avoid CPR on terminals that mishandle cursor position reports."""
    global _PROMPT_TOOLKIT_CONFIGURED
    if _PROMPT_TOOLKIT_CONFIGURED:
        return
    _PROMPT_TOOLKIT_CONFIGURED = True
    if os.environ.get("PROMPT_TOOLKIT_NO_CPR") == "1":
        return
    term = os.environ.get("TERM", "").lower()
    if term in ("dumb", "unknown"):
        os.environ["PROMPT_TOOLKIT_NO_CPR"] = "1"
        return
    # Cursor and VS Code integrated terminals often leak CPR as visible input (e.g. "28;1R").
    if os.environ.get("TERM_PROGRAM", "").lower() == "vscode":
        os.environ["PROMPT_TOOLKIT_NO_CPR"] = "1"


def _terminal_columns() -> int:
    try:
        if sys.stdout.isatty():
            return max(os.get_terminal_size(sys.stdout.fileno()).columns, 1)
    except OSError:
        pass
    return 80


def _input_horizontal_rule() -> str:
    return "─" * _terminal_columns()


def _user_input_placeholder() -> AnyFormattedText:
    return [("class:placeholder", _USER_INPUT_PLACEHOLDER)]


def _input_border_fragments() -> AnyFormattedText:
    return [("class:input-border", _input_horizontal_rule())]


def _input_footer_fragments() -> AnyFormattedText:
    return [("class:input-footer", _USER_INPUT_FOOTER)]


def _run_framed_user_prompt(
    buffer: Buffer,
    *,
    style: Style,
) -> str | None:
    """Prompt with top/bottom rules and footer in the layout (works without CPR)."""
    show_placeholder = Condition(lambda: buffer.text == "")

    input_control = BufferControl(
        buffer=buffer,
        input_processors=[
            ConditionalProcessor(
                AfterInput(_user_input_placeholder),
                show_placeholder,
            ),
        ],
    )
    input_window = Window(
        input_control,
        height=lambda: _input_window_height(buffer),
        wrap_lines=False,
    )

    def _accept(buff: Buffer) -> bool:
        get_app().exit(result=buff.text)
        return True

    buffer.accept_handler = _accept

    def _frame_row(text_fn: Any) -> Window:
        return Window(
            FormattedTextControl(text_fn),
            height=Dimension.exact(1),
            dont_extend_height=True,
        )

    top_window = _frame_row(_input_border_fragments)
    bottom_window = _frame_row(_input_border_fragments)
    footer_window = _frame_row(_input_footer_fragments)

    completions_open = Condition(lambda: buffer.complete_state is not None)
    frame_footer_visible = Condition(lambda: buffer.complete_state is None)

    # Reserve space below the input so the completion menu opens downward
    # (prompt_toolkit flips upward when there is more room above the cursor).
    menu_spacer = ConditionalContainer(
        Window(height=Dimension(min=_USER_INPUT_MENU_RESERVE)),
        filter=completions_open,
    )
    bottom_row = ConditionalContainer(bottom_window, filter=frame_footer_visible)
    footer_row = ConditionalContainer(footer_window, filter=frame_footer_visible)

    input_frame = HSplit(
        [top_window, input_window, menu_spacer, bottom_row, footer_row]
    )

    root_container = FloatContainer(
        input_frame,
        floats=[
            Float(
                xcursor=True,
                ycursor=True,
                transparent=True,
                content=CompletionsMenu(
                    max_height=16,
                    scroll_offset=1,
                    extra_filter=has_focus(input_control),
                ),
            ),
        ],
    )

    app = Application(
        layout=Layout(root_container, input_window),
        key_bindings=merge_key_bindings(
            [
                _prompt_newline_key_bindings(),
                load_key_bindings(),
                _prompt_enter_key_bindings(),
                _prompt_clear_input_key_bindings(),
            ]
        ),
        style=style,
        full_screen=False,
        erase_when_done=False,
    )
    try:
        with _enhanced_keyboard_reporting():
            return app.run()
    except (KeyboardInterrupt, EOFError):
        return None


def _prompt_user_line(
    paste_state: _PasteState,
    *,
    workspace: Path | None = None,
    show_commands: bool = False,
    auto_model_switch: bool = False,
    stream_output: bool = False,
    session_model: str | None = None,
    default_reasoning_effort: str | None = None,
) -> str | None:
    """Prompt: Enter sends; Shift+Enter (c-j) adds a line; multiline paste collapses."""
    _configure_prompt_toolkit()
    root = (workspace or resolve_workspace()).resolve()
    style = merge_styles(
        [
            DEFAULT_STYLE,
            Style.from_dict(
                {
                    "": "#ffffff",
                    "placeholder": "dim",
                    "input-border": "dim",
                    "input-footer": "dim",
                    "mention.choice": "ansibrightblue",
                    "completion-menu": "bg:#1e1e1e",
                    "completion-menu.completion": "",
                    "completion-menu.completion.current": "bg:ansiblue",
                }
            ),
        ]
    )
    completer = _UserInputCompleter(
        root,
        show_commands=show_commands,
        auto_model_switch=auto_model_switch,
        stream_output=stream_output,
        session_model=session_model,
        default_reasoning_effort=default_reasoning_effort,
    )

    if not sys.stdout.isatty():
        session = PromptSession(
            style=style,
            multiline=False,
            placeholder=_user_input_placeholder(),
            key_bindings=merge_key_bindings(
                [
                    _prompt_newline_key_bindings(),
                    _prompt_enter_key_bindings(),
                    _prompt_clear_input_key_bindings(),
                ]
            ),
            completer=completer,
            complete_while_typing=True,
        )
        _attach_paste_handler(session.default_buffer, paste_state)
        try:
            return session.prompt()
        except (KeyboardInterrupt, EOFError):
            return None

    buffer = Buffer(
        completer=completer,
        complete_while_typing=True,
        multiline=False,
    )
    _attach_paste_handler(buffer, paste_state)
    return _run_framed_user_prompt(buffer, style=style)


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


_ESC_CONTINUATION_TIMEOUT = 0.05
_ESC_CSI_TERMINATORS = frozenset(range(0x40, 0x7F))


def _drain_escape_sequence(fd: int, stop_event: threading.Event) -> bool:
    """Drain bytes that follow an ESC byte to detect multi-byte key sequences.

    Returns True when the ESC was part of a sequence (CSI ``ESC [``, SS3
    ``ESC O``, Option/Meta ``ESC <char>``, or any other follow-up that arrives
    within ``_ESC_CONTINUATION_TIMEOUT``). Returns False when no continuation
    arrives — i.e. the user pressed a bare Escape.
    """
    readable, _, _ = select.select([fd], [], [], _ESC_CONTINUATION_TIMEOUT)
    if not readable:
        return False
    try:
        nxt = os.read(fd, 1)
    except OSError:
        return False
    if not nxt:
        return False
    if nxt in (b"[", b"O"):
        # CSI / SS3 introducer — drain until a final byte (0x40..0x7E).
        while not stop_event.is_set():
            r2, _, _ = select.select([fd], [], [], _ESC_CONTINUATION_TIMEOUT)
            if not r2:
                return True
            try:
                b = os.read(fd, 1)
            except OSError:
                return True
            if not b:
                return True
            if b[0] in _ESC_CSI_TERMINATORS:
                return True
    return True


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
        stream_output: bool = False,
        show_activity: bool = True,
        console: Console | None = None,
        workspace: Path | str | None = None,
        model: str | None = None,
    ) -> None:
        self.show_thoughts = show_thoughts
        self.verbose = verbose
        self.show_activity = show_activity
        self.show_model = show_model
        self.show_commands = show_commands
        self.auto_model_switch = auto_model_switch
        self.debug = debug
        self.debug_input = debug_input
        self.debug_output = debug_output
        self.stream_output = stream_output
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
                    if ch != b"\x1b":
                        continue
                    # An ESC byte may start a CSI/SS3 sequence (arrow keys, F-keys,
                    # Option-shortcuts, bracketed paste, mouse reports). Only treat
                    # a *bare* Escape as a stop request — drain any continuation
                    # bytes that arrive within a short window and ignore them.
                    if _drain_escape_sequence(fd, stop_event):
                        continue
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
            yield self._panel(
                Text.from_markup(self._startup_markup(data)),
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
        if kind == "activity":
            style = "error" if data.get("failed") else "muted"
            yield f"[{style}]◦ {data['text']}[/]"
            return
        if kind == "file_diff":
            yield ""
            panel_width = self._panel_text_width()
            path = data.get("path", "file")
            yield self._panel(
                _syntax(data["diff"], "diff", width=panel_width),
                title=_role_panel_title("agent", subtitle=f"diff · {path}"),
                border_style="dim",
                padding=(0, 1),
            )
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

    def _startup_data(self) -> dict[str, Any]:
        return {
            "model": self.session_model,
            "auto_model_switch": self.auto_model_switch,
            "reasoning_effort": self._session_reasoning,
        }

    def _startup_markup(self, data: dict[str, Any] | None = None) -> str:
        payload = data or self._startup_data()
        return format_startup(
            model=payload.get("model", self.session_model),
            auto_model_switch=bool(payload.get("auto_model_switch")),
            reasoning_effort=payload.get("reasoning_effort"),
        )

    def refresh_startup_banner(self) -> None:
        """Update the startup panel after session routing toggles."""
        data = self._startup_data()
        for index, (kind, _item) in enumerate(self._history):
            if kind == "startup":
                self._history[index] = ("startup", data)
                break
        else:
            return
        self._sync_redraw()

    def set_session_model(self, model: str) -> None:
        """Update the model shown on the startup banner for this session."""
        self._session_model = model
        self.refresh_startup_banner()

    @property
    def session_reasoning(self) -> str | None:
        return self._session_reasoning

    def set_session_reasoning(self, effort: str | None) -> None:
        """Update the default reasoning effort used for slash-command completion."""
        self._session_reasoning = effort
        self.refresh_startup_banner()

    def print_startup(self) -> None:
        self._record("startup", **self._startup_data())

    def clear_session(self) -> None:
        """Clear the on-screen transcript and show the startup banner again."""
        self._history.clear()
        self._pending_redraw = False
        self._stop_requested = False
        if self._is_interactive_tty():
            with self._display_lock:
                self._console.clear()
        self._record("startup", **self._startup_data())

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

    def print_session_log_path(self, path: Path | str) -> None:
        """Show where this session is being logged (debug mode only)."""
        if not self.debug:
            return
        self._err.print(f"[muted]Session log: {path}[/]")

    def print_turn_usage(
        self,
        *,
        turn: int,
        model: str,
        usage: TokenUsage,
    ) -> None:
        """Brief token usage line after each LLM call when verbose or show_model."""
        if not (self.verbose or self.show_model):
            return
        self._err.print(
            f"[muted]turn {turn} · {model} · "
            f"in {usage.input_tokens:,} · out {usage.output_tokens:,} · "
            f"total {usage.total_tokens:,}[/]"
        )

    def newline(self) -> None:
        self._console.print()

    def prompt_user(self) -> str:
        """Read user input; Enter sends; Shift+Enter adds a line; paste collapses."""
        self._sync_redraw()
        # Drop any stop flag left over from a prior step's escape listener so
        # it can't abort the next task before it runs.
        self._stop_requested = False
        self._at_prompt = True
        try:
            paste_state = _PasteState()
            result = _prompt_user_line(
                paste_state,
                workspace=self._workspace or resolve_workspace(),
                show_commands=self.show_commands,
                auto_model_switch=self.auto_model_switch,
                stream_output=self.stream_output,
                session_model=self.session_model,
                default_reasoning_effort=self._session_reasoning,
            )
            if result is None:
                raise click.Abort()
            text = paste_state.resolve(result).strip()
            workspace = self._workspace or resolve_workspace()
            text = expand_file_mentions(text, workspace)
            if not is_repl_slash_command(text):
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

    def print_llm_response(
        self, *, turn: int, raw: str, streamed: bool = False
    ) -> None:
        """Print raw model response without truncation (debug -o)."""
        if not self.debug_output or streamed:
            return
        self._record("llm_response", turn=turn, raw=raw)

    def begin_model_stream(self, *, turn: int) -> None:
        """Start streaming model output to the console (-s)."""
        if not self.stream_output:
            return
        self._err.print(f"[dim]Model output (turn {turn}):[/]")

    def write_model_stream_delta(self, text: str) -> None:
        """Emit one chunk of streamed model text to stderr."""
        if not self.stream_output or not text:
            return
        self._err.file.write(text)
        self._err.file.flush()

    def end_model_stream(self) -> None:
        """Finish a streamed model response with a newline."""
        if not self.stream_output:
            return
        self._err.file.write("\n")
        self._err.file.flush()

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
                self.print_agent(step.code.strip(), subtitle="python")
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
            self.print_agent(step.code.strip(), subtitle="python")
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

    def _diff_for_edit_step(self, step: AgentStep, result: ToolResult) -> str | None:
        if step.patch and step.patch.strip():
            return step.patch
        extracted = _extract_unified_diff(result.stdout or "")
        if extracted and extracted.strip():
            return extracted
        return None

    def print_file_diff(self, path: str, diff: str) -> None:
        """Show a truncated unified diff for a file edit (default mode)."""
        if not diff.strip():
            return
        display_path = Path(path).name if path else "file"
        self._record(
            "file_diff",
            path=display_path,
            diff=_truncate_diff_lines(diff),
        )

    def print_harness_activity(self, step: AgentStep, result: ToolResult) -> None:
        """Record a one-line summary after a harness tool finishes (default mode)."""
        if not self.show_activity or self.verbose:
            return

        failed = result.timed_out or result.exit_code not in (0, None)
        if (
            not failed
            and step.action in (AgentAction.RUN_SHELL, AgentAction.RUN_PYTHON)
        ):
            return

        label = activity_label(step, phase="done")
        if failed:
            err_hint = (result.stderr or "failed").strip().splitlines()[0]
            if len(err_hint) > 80:
                err_hint = err_hint[:77] + "..."
            text = f"{label} — {err_hint}"
        else:
            text = label

        self._record("activity", text=text, failed=failed)

        if (
            not self.debug
            and not failed
            and step.action in (AgentAction.STR_REPLACE, AgentAction.APPLY_PATCH)
        ):
            diff = self._diff_for_edit_step(step, result)
            if diff:
                self.print_file_diff(step.path or "file", diff)

    @contextmanager
    def thinking(self, label: str = "working on your task") -> Iterator[ThinkingUpdater]:
        """Show a spinner while the agent waits on the LLM.

        Yields a :class:`ThinkingUpdater` so callers can refresh the label when
        ``status`` arrives from a streamed model response.
        """
        updater = ThinkingUpdater(
            _label=label.strip().rstrip(".…") or "working on your task",
            _console=self._console,
        )
        if not sys.stdout.isatty():
            self._console.print(
                f"[agent]◆ GoodBoy[/] [muted]{updater._label}…[/]"
            )
            yield updater
            return

        self._transient_ui = True
        try:
            with self._escape_stop_listener():
                with self._console.status(
                    updater._render(),
                    spinner="dots",
                ) as status:
                    updater._status = status
                    yield updater
        finally:
            self._transient_ui = False
            self._sync_redraw()


@contextmanager
def tool_activity(ui: ConversationUI, label: str) -> Iterator[None]:
    """Spinner while a harness tool runs (interactive TTY only).

    ``label`` is human-facing text from :func:`activity_label` (e.g.
    ``reading foo.py``).
    """
    if not ui._is_interactive_tty():
        yield
        return
    with ui.thinking(label=label):
        yield
