"""Slash commands available in the interactive harness prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SLASH_COMMAND_LINE = re.compile(r"^/([^\s]*)$")


@dataclass(frozen=True)
class ReplCommand:
    name: str
    description: str
    aliases: tuple[str, ...] = ()


REPL_COMMANDS: tuple[ReplCommand, ...] = (
    ReplCommand(
        "help",
        "List slash commands and current session settings",
        ("?", "h"),
    ),
    ReplCommand(
        "clear",
        "Clear conversation history",
        ("new", "reset"),
    ),
    ReplCommand(
        "retry",
        "Re-run the previous task",
        ("again",),
    ),
    ReplCommand(
        "model",
        "Choose the LLM for this session",
    ),
    ReplCommand(
        "reasoning",
        "Choose default reasoning effort for reasoning models",
        ("reason",),
    ),
    ReplCommand(
        "plan",
        "Cycle plan mode (auto/off/always)",
        ("planmode",),
    ),
    ReplCommand(
        "commands",
        "Toggle showing shell/Python commands (no output)",
        ("cmds", "show-commands"),
    ),
    ReplCommand(
        "autoswitch",
        "Toggle automatic model switching",
        ("auto",),
    ),
    ReplCommand(
        "stream",
        "Toggle streaming model output as it is generated",
        ("streaming",),
    ),
    ReplCommand(
        "exit",
        "Exit GoodBoy",
        ("quit", "q"),
    ),
)

MODEL_COMMAND_NAMES = frozenset(
    name
    for command in REPL_COMMANDS
    if command.name == "model"
    for name in (command.name, *command.aliases)
)
REASONING_COMMAND_NAMES = frozenset(
    name
    for command in REPL_COMMANDS
    if command.name == "reasoning"
    for name in (command.name, *command.aliases)
)
PLAN_COMMAND_NAMES = frozenset(
    name
    for command in REPL_COMMANDS
    if command.name == "plan"
    for name in (command.name, *command.aliases)
)

HELP_COMMAND_NAMES = frozenset(
    name
    for command in REPL_COMMANDS
    if command.name == "help"
    for name in (command.name, *command.aliases)
)
RETRY_COMMAND_NAMES = frozenset(
    name
    for command in REPL_COMMANDS
    if command.name == "retry"
    for name in (command.name, *command.aliases)
)
CLEAR_COMMAND_NAMES = frozenset(
    name
    for command in REPL_COMMANDS
    if command.name == "clear"
    for name in (command.name, *command.aliases)
)
EXIT_COMMAND_NAMES = frozenset(
    name
    for command in REPL_COMMANDS
    if command.name == "exit"
    for name in (command.name, *command.aliases)
)
COMMANDS_COMMAND_NAMES = frozenset(
    name
    for command in REPL_COMMANDS
    if command.name == "commands"
    for name in (command.name, *command.aliases)
)
AUTOSWITCH_COMMAND_NAMES = frozenset(
    name
    for command in REPL_COMMANDS
    if command.name == "autoswitch"
    for name in (command.name, *command.aliases)
)
STREAM_COMMAND_NAMES = frozenset(
    name
    for command in REPL_COMMANDS
    if command.name == "stream"
    for name in (command.name, *command.aliases)
)

_TOGGLE_COMMAND_STATE: dict[str, str] = {
    "commands": "show_commands",
    "autoswitch": "auto_model_switch",
    "stream": "stream_output",
}

_PLAN_MODES: tuple[str, ...] = ("auto", "off", "always")

_ALL_COMMAND_NAMES: frozenset[str] = frozenset(
    name
    for command in REPL_COMMANDS
    for name in (command.name, *command.aliases)
)


def is_repl_slash_command(text: str) -> bool:
    """Return True when ``text`` is a recognized ``/command`` invocation."""
    normalized = text.strip().lower()
    if not normalized.startswith("/"):
        return False
    return normalized[1:] in _ALL_COMMAND_NAMES


def next_plan_mode(current: str | None) -> str:
    """Return the next plan mode in the user-facing cycle."""
    normalized = (current or "auto").strip().lower()
    try:
        index = _PLAN_MODES.index(normalized)
    except ValueError:
        index = 0
    return _PLAN_MODES[(index + 1) % len(_PLAN_MODES)]


def slash_command_display_meta(
    command: ReplCommand,
    *,
    show_commands: bool = False,
    auto_model_switch: bool = False,
    stream_output: bool = False,
    session_model: str | None = None,
    default_reasoning_effort: str | None = None,
    plan_mode: str = "auto",
) -> str:
    """Completion description; toggle commands include current on/off (default off)."""
    if command.name == "reasoning":
        current = default_reasoning_effort or "not set"
        if auto_model_switch:
            return f"{command.description} (current: {current})"
        return (
            f"{command.description} (session: {current}; agent cannot change per turn)"
        )
    if command.name == "model":
        current = session_model or "not set"
        if auto_model_switch:
            return f"{command.description} (current: {current})"
        return (
            f"{command.description} (session: {current}; agent cannot change per turn)"
        )
    if command.name == "plan":
        current = (plan_mode or "auto").strip().lower() or "auto"
        return f"{command.description} (current: {current})"
    state_key = _TOGGLE_COMMAND_STATE.get(command.name)
    if state_key is None:
        return command.description
    toggle_state = {
        "show_commands": show_commands,
        "auto_model_switch": auto_model_switch,
        "stream_output": stream_output,
    }
    enabled = toggle_state[state_key]
    state = "on" if enabled else "off"
    return f"{command.description} ({state})"


def active_slash_command_query(text_before_cursor: str) -> tuple[str, int] | None:
    """Return (query after /, start_position) when the cursor is in a slash command."""
    line_start = text_before_cursor.rfind("\n") + 1
    line_before = text_before_cursor[line_start:]
    match = _SLASH_COMMAND_LINE.match(line_before)
    if match is None:
        return None
    query = match.group(1)
    return query, -len(query)


def format_help_text(
    *,
    show_commands: bool = False,
    auto_model_switch: bool = False,
    stream_output: bool = False,
    session_model: str | None = None,
    default_reasoning_effort: str | None = None,
    plan_mode: str = "auto",
) -> str:
    """Human-readable slash-command reference for ``/help``."""
    lines = ["Slash commands:"]
    for command in REPL_COMMANDS:
        names = "/".join((command.name, *command.aliases))
        meta = slash_command_display_meta(
            command,
            show_commands=show_commands,
            auto_model_switch=auto_model_switch,
            stream_output=stream_output,
            session_model=session_model,
            default_reasoning_effort=default_reasoning_effort,
            plan_mode=plan_mode,
        )
        lines.append(f"  /{names} — {meta}")
    lines.append("")
    lines.append(f"Model: {session_model or 'not set'}")
    lines.append(f"Reasoning: {default_reasoning_effort or 'not set'}")
    lines.append(f"Plan mode: {plan_mode}")
    return "\n".join(lines)


def search_slash_commands(query: str) -> list[ReplCommand]:
    """Return slash commands matching query (prefix on name or alias)."""
    query_fold = query.casefold()
    if not query_fold:
        return list(REPL_COMMANDS)

    scored: list[tuple[tuple[int, str], ReplCommand]] = []

    for command in REPL_COMMANDS:
        rank = _command_rank(command, query_fold)
        if rank is None:
            continue
        scored.append((rank, command))

    scored.sort(key=lambda item: item[0])
    return [command for _, command in scored]


def _command_rank(command: ReplCommand, query_fold: str) -> tuple[int, str] | None:
    if not query_fold:
        return (0, command.name)
    names = (command.name, *command.aliases)
    best: tuple[int, str] | None = None
    for name in names:
        name_fold = name.casefold()
        if name_fold == query_fold:
            tier = 0
        elif name_fold.startswith(query_fold):
            tier = 1
        elif query_fold in name_fold:
            tier = 2
        else:
            continue
        candidate = (tier, name)
        if best is None or candidate < best:
            best = candidate
    return best
