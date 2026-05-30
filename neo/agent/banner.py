"""Startup banner: ASCII art, version, tool description, and matrix rain intro."""

from __future__ import annotations

import os
import random
import re
import time
from importlib.metadata import PackageNotFoundError, version

from rich.console import Console, RenderableType
from rich.live import Live
from rich.text import Text

from settings import PACKAGE_DIR

_BRAND_STYLE = "rgb(0,180,50)"

# Half-width katakana, digits, and ascii punctuation — same set the web banner
# uses for its canvas rain so the terminal intro reads as the same effect.
MATRIX_GLYPHS = (
    "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
    "0123456789<>/\\|=+*-_:;."
)

MATRIX_ASCII = r"""
███╗   ██╗ ███████╗  ██████╗ 
████╗  ██║ ██╔════╝ ██╔═══██╗
██╔██╗ ██║ █████╗   ██║   ██║
██║╚██╗██║ ██╔══╝   ██║   ██║
██║ ╚████║ ███████╗ ╚██████╔╝
╚═╝  ╚═══╝ ╚══════╝  ╚═════╝ 
"""


def _matrix_art() -> str:
    return "\n".join(
        f"[{_BRAND_STYLE}]{line}[/]" for line in MATRIX_ASCII.rstrip().splitlines()
    )


TIPS = [
    "Tip: Press Ctrl+C to interrupt safely",
    "Tip: Use `/help` to list all slash commands",
    "Tip: Use `/model` to switch LLM providers",
    "Tip: Use `/reasoning` to set reasoning effort",
    "Tip: Use `/plan` to toggle plan mode (auto/off/always)",
    "Tip: Use `/clear` to reset conversation history",
    "Tip: Use `/retry` to re-run the last task",
]


def get_random_tip() -> str:
    return random.choice(TIPS)


def get_version() -> str:
    try:
        return version("neo")
    except PackageNotFoundError:
        text = (PACKAGE_DIR / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        return match.group(1) if match else "unknown"


def format_startup(
    *,
    model: str,
    reasoning_effort: str | None = None,
) -> str:
    ver = get_version()
    reasoning_line = ""
    if reasoning_effort:
        reasoning_line = (
            f"\n[dim]Reasoning[/] [{_BRAND_STYLE}]{reasoning_effort}[/]"
        )
    return (
        f"{_matrix_art()}\n"
        f"[bold {_BRAND_STYLE}]Neo[/] [dim]v{ver}[/]\n"
        f"[dim]Model[/] [{_BRAND_STYLE}]{model}[/]{reasoning_line}\n"
        f"{get_random_tip()}"
    )


def matrix_intro_enabled() -> bool:
    """True when the dropping-character intro should play.

    Skipped when ``NEO_NO_ANIMATION=1`` or ``NO_COLOR`` is set, and on dumb
    terminals.
    """
    if os.environ.get("NEO_NO_ANIMATION") == "1":
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    term = os.environ.get("TERM", "").lower()
    if term in ("dumb", "unknown"):
        return False
    return True


# Trail lengths follow the web rain: bright head, then 2 saturated cells, then
# a fading tail. Values are 1-indexed cells *behind* the leading glyph.
_TRAIL_HEAD = "bold rgb(220,255,225)"
_TRAIL_BRIGHT = _BRAND_STYLE
_TRAIL_MID = "rgb(0,160,40)"
_TRAIL_DIM = "rgb(0,90,20)"
_TRAIL_LENGTH = 7


def _render_rain_frame(
    *,
    drops: list[int],
    height: int,
    width: int,
) -> Text:
    """Build one frame of falling green characters sized to ``width × height``."""
    choice = random.choice
    glyphs = MATRIX_GLYPHS
    text = Text(no_wrap=True, overflow="crop")
    for row in range(height):
        for col in range(width):
            head = drops[col]
            # ``cell`` is how far above the head this row sits. 0 = head row,
            # positive = part of the trail rising up the column. Negative means
            # the head has not reached this row yet (blank cell).
            cell = head - row
            if cell < 0:
                text.append(" ")
                continue
            if cell == 0:
                text.append(choice(glyphs), style=_TRAIL_HEAD)
            elif cell <= 2:
                text.append(choice(glyphs), style=_TRAIL_BRIGHT)
            elif cell <= _TRAIL_LENGTH - 2:
                text.append(choice(glyphs), style=_TRAIL_MID)
            elif cell <= _TRAIL_LENGTH:
                text.append(choice(glyphs), style=_TRAIL_DIM)
            else:
                text.append(" ")
        if row < height - 1:
            text.append("\n")
    return text


def play_matrix_intro(
    console: Console,
    *,
    height: int,
    width: int,
    duration: float = 1.0,
    frame_rate: int = 24,
) -> None:
    """Animate dropping green katakana characters in a fixed ``height × width`` region.

    The animation is rendered through :class:`rich.live.Live` in transient
    mode, so when it finishes the area is cleared and the caller can render
    the final banner panel in its place. No-op on non-TTY consoles or when
    ``matrix_intro_enabled()`` is False.
    """
    if not console.is_terminal or not matrix_intro_enabled():
        return
    if height <= 0 or width <= 0:
        return

    # Stagger initial drops above the visible area so the columns fall in
    # from the top rather than appearing in a single horizontal band.
    drops = [random.randint(-height * 2, -1) for _ in range(width)]

    interval = 1.0 / max(frame_rate, 1)
    total_frames = max(1, int(duration * frame_rate))

    initial = _render_rain_frame(drops=drops, height=height, width=width)
    try:
        with Live(
            initial,
            console=console,
            refresh_per_second=frame_rate,
            transient=True,
        ) as live:
            for _ in range(total_frames):
                time.sleep(interval)
                for col in range(width):
                    drops[col] += 1
                    if drops[col] > height + 4 and random.random() > 0.7:
                        drops[col] = -random.randint(0, height)
                live.update(
                    _render_rain_frame(drops=drops, height=height, width=width)
                )
    except (KeyboardInterrupt, OSError):
        # Leave the live region as-is; the caller will redraw the static
        # banner over whatever the terminal currently shows.
        return


def measure_renderable_height(
    console: Console,
    renderable: RenderableType,
    *,
    width: int | None = None,
) -> int:
    """Number of terminal rows ``renderable`` would occupy on ``console``."""
    options = console.options
    if width is not None:
        options = options.update(width=width)
    return len(console.render_lines(renderable, options))


def render_rain_snapshot(
    *,
    height: int,
    width: int,
    rng: random.Random | None = None,
) -> Text:
    """A still frame from an ongoing matrix rain (drops at random positions).

    Each column is given a head row in the range ``[0, height + trail)`` so
    some columns are still empty while others have already finished falling.
    The result is sized exactly to ``height × width`` cells so it can sit in
    a fixed-width panel column without wrapping.
    """
    if height <= 0 or width <= 0:
        return Text("")
    pick = (rng or random).choice
    rand = (rng or random).random
    randint = (rng or random).randint

    # Pre-pick head rows for each column; some columns get a "no drop" sentinel
    # to keep the snapshot from looking like a solid wall of glyphs.
    heads: list[int | None] = []
    for _ in range(width):
        if rand() < 0.85:
            heads.append(randint(0, height + _TRAIL_LENGTH))
        else:
            heads.append(None)

    text = Text(no_wrap=True, overflow="crop")
    for row in range(height):
        for col in range(width):
            head = heads[col]
            if head is None:
                text.append(" ")
                continue
            cell = head - row
            if cell < 0:
                text.append(" ")
            elif cell == 0:
                text.append(pick(MATRIX_GLYPHS), style=_TRAIL_HEAD)
            elif cell <= 2:
                text.append(pick(MATRIX_GLYPHS), style=_TRAIL_BRIGHT)
            elif cell <= _TRAIL_LENGTH - 2:
                text.append(pick(MATRIX_GLYPHS), style=_TRAIL_MID)
            elif cell <= _TRAIL_LENGTH:
                text.append(pick(MATRIX_GLYPHS), style=_TRAIL_DIM)
            else:
                text.append(" ")
        if row < height - 1:
            text.append("\n")
    return text
