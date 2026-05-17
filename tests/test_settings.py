"""Tests for .env-backed settings."""

from settings import (
    DEFAULT_CONTEXT_RECENT_FULL_TURNS,
    GOODBOY_AUTO_MODEL_SWITCH_VAR,
    GOODBOY_CONTEXT_RECENT_FULL_TURNS_VAR,
    GOODBOY_REASONING_EFFORT_VAR,
    GOODBOY_SHOW_COMMANDS_VAR,
    Settings,
    get_settings,
)


def test_settings_show_commands_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{GOODBOY_SHOW_COMMANDS_VAR}=true\n", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    get_settings.cache_clear()
    assert Settings.from_env().show_commands is True
    get_settings.cache_clear()


def test_settings_reasoning_effort_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{GOODBOY_REASONING_EFFORT_VAR}=medium\n", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    get_settings.cache_clear()
    assert Settings.from_env().default_reasoning_effort == "medium"
    get_settings.cache_clear()


def test_settings_auto_model_switch_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{GOODBOY_AUTO_MODEL_SWITCH_VAR}=true\n", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    get_settings.cache_clear()
    assert Settings.from_env().auto_model_switch is True
    get_settings.cache_clear()


def test_settings_recent_full_turns_default(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    monkeypatch.delenv(GOODBOY_CONTEXT_RECENT_FULL_TURNS_VAR, raising=False)
    get_settings.cache_clear()
    assert (
        Settings.from_env().context_recent_full_turns
        == DEFAULT_CONTEXT_RECENT_FULL_TURNS
    )
    get_settings.cache_clear()


def test_settings_recent_full_turns_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"{GOODBOY_CONTEXT_RECENT_FULL_TURNS_VAR}=5\n", encoding="utf-8"
    )
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    get_settings.cache_clear()
    assert Settings.from_env().context_recent_full_turns == 5
    get_settings.cache_clear()


def test_settings_recent_full_turns_floor_at_one(monkeypatch, tmp_path):
    """Window must always include at least the current turn."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"{GOODBOY_CONTEXT_RECENT_FULL_TURNS_VAR}=0\n", encoding="utf-8"
    )
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    get_settings.cache_clear()
    assert Settings.from_env().context_recent_full_turns == 1
    get_settings.cache_clear()
