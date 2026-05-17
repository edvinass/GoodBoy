"""Tests for .env-backed settings."""

from settings import (
    GOODBOY_AUTO_MODEL_SWITCH_VAR,
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
