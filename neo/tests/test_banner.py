"""Tests for startup banner."""

import io
import re

from rich.console import Console

from agent.banner import (
    MATRIX_GLYPHS,
    format_startup,
    get_version,
    get_random_tip,
    matrix_intro_enabled,
    play_matrix_intro,
    _render_rain_frame,
)
from settings import PACKAGE_DIR


def test_get_version_matches_pyproject():
    text = (PACKAGE_DIR / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    expected = match.group(1) if match else "unknown"
    assert get_version() == expected


def test_format_startup_includes_version_model_and_tip():
    text = format_startup(model="gpt-5.4-nano")
    assert "Neo" in text
    assert get_version() in text
    assert "gpt-5.4-nano" in text
    assert "Model" in text
    assert "Routing" not in text
    assert "session" not in text.lower()
    # The banner now shows a random tip from TIPS instead of a static DESCRIPTION
    from agent.banner import TIPS
    assert any(tip in text for tip in TIPS)


def test_format_startup_renders_reasoning_effort_when_provided():
    text = format_startup(model="gpt-5.4-nano", reasoning_effort="low")
    assert "Routing" not in text
    # The random tip may contain slash commands like /model or /reasoning,
    # so we only assert that "low" is rendered as the reasoning effort line.
    assert "low" in text


def test_matrix_glyph_set_matches_web_rain():
    # The web rain uses half-width katakana plus digits/punctuation. Keep the
    # terminal intro in sync so both surfaces read as the same effect.
    for char in "ｱｲｳｴｵﾝ":
        assert char in MATRIX_GLYPHS
    for char in "0123456789":
        assert char in MATRIX_GLYPHS


def test_matrix_intro_disabled_when_no_animation_env_set(monkeypatch):
    monkeypatch.setenv("NEO_NO_ANIMATION", "1")
    assert matrix_intro_enabled() is False


def test_matrix_intro_disabled_when_no_color_env_set(monkeypatch):
    monkeypatch.delenv("NEO_NO_ANIMATION", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert matrix_intro_enabled() is False


def test_matrix_intro_disabled_for_dumb_terminals(monkeypatch):
    monkeypatch.delenv("NEO_NO_ANIMATION", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert matrix_intro_enabled() is False


def test_matrix_intro_enabled_by_default(monkeypatch):
    monkeypatch.delenv("NEO_NO_ANIMATION", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert matrix_intro_enabled() is True


def test_render_rain_frame_emits_matrix_glyphs():
    frame = _render_rain_frame(drops=[3, 3, 3, 3], height=4, width=4)
    rendered = frame.plain
    assert len(rendered.splitlines()) == 4
    # At least one rendered cell should be a known matrix glyph for a fully
    # active column at the head row.
    assert any(ch in MATRIX_GLYPHS for ch in rendered)


def test_play_matrix_intro_is_noop_on_non_tty():
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=40)
    play_matrix_intro(console, height=4, width=20, duration=0.1, frame_rate=10)
    assert buffer.getvalue() == ""


def test_play_matrix_intro_respects_no_animation_env(monkeypatch):
    monkeypatch.setenv("NEO_NO_ANIMATION", "1")
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=40)
    play_matrix_intro(console, height=4, width=20, duration=0.1, frame_rate=10)
    assert buffer.getvalue() == ""
