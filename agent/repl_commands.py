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
        "clear",
        "Clear conversation history",
        ("new", "reset"),
    ),
    ReplCommand(
        "model",
        "Choose the LLM for this session",
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

_TOGGLE_COMMAND_STATE: dict[str, str] = {
    "commands": "show_commands",
    "autoswitch": "auto_model_switch",
}


def slash_command_display_meta(
    command: ReplCommand,
    *,
    show_commands: bool = False,
    auto_model_switch: bool = False,
) -> str:
    """Completion description; toggle commands include current on/off (default off)."""
    state_key = _TOGGLE_COMMAND_STATE.get(command.name)
    if state_key is None:
        return command.description
    enabled = show_commands if state_key == "show_commands" else auto_model_switch
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
