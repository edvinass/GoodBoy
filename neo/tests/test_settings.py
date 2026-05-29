"""Tests for .env-backed settings."""

from settings import (
    DEFAULT_CONTEXT_RECENT_FULL_TURNS,
    DEFAULT_LOCAL_RECENT_FULL_TURNS,
    NEO_CONTEXT_RECENT_FULL_TURNS_VAR,
    NEO_LOCAL_RECENT_FULL_TURNS_VAR,
    NEO_REASONING_EFFORT_VAR,
    NEO_RESPONSE_CHAIN_VAR,
    NEO_SHOW_COMMANDS_VAR,
    Settings,
    get_settings,
)


def test_settings_response_chain_default_enabled(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    monkeypatch.delenv(NEO_RESPONSE_CHAIN_VAR, raising=False)
    get_settings.cache_clear()
    assert Settings.from_env().response_chain_enabled is True
    get_settings.cache_clear()


def test_settings_show_commands_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{NEO_SHOW_COMMANDS_VAR}=true\n", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    get_settings.cache_clear()
    assert Settings.from_env().show_commands is True
    get_settings.cache_clear()


def test_settings_reasoning_effort_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{NEO_REASONING_EFFORT_VAR}=medium\n", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    get_settings.cache_clear()
    assert Settings.from_env().default_reasoning_effort == "medium"
    get_settings.cache_clear()


def test_settings_recent_full_turns_default(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    monkeypatch.delenv(NEO_CONTEXT_RECENT_FULL_TURNS_VAR, raising=False)
    get_settings.cache_clear()
    assert (
        Settings.from_env().context_recent_full_turns
        == DEFAULT_CONTEXT_RECENT_FULL_TURNS
    )
    get_settings.cache_clear()


def test_settings_recent_full_turns_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"{NEO_CONTEXT_RECENT_FULL_TURNS_VAR}=5\n", encoding="utf-8"
    )
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    get_settings.cache_clear()
    assert Settings.from_env().context_recent_full_turns == 5
    get_settings.cache_clear()


def test_settings_recent_full_turns_floor_at_one(monkeypatch, tmp_path):
    """Window must always include at least the current turn."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"{NEO_CONTEXT_RECENT_FULL_TURNS_VAR}=0\n", encoding="utf-8"
    )
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    get_settings.cache_clear()
    assert Settings.from_env().context_recent_full_turns == 1
    get_settings.cache_clear()


def test_settings_local_recent_full_turns_default(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    monkeypatch.delenv(NEO_LOCAL_RECENT_FULL_TURNS_VAR, raising=False)
    get_settings.cache_clear()
    assert (
        Settings.from_env().local_recent_full_turns
        == DEFAULT_LOCAL_RECENT_FULL_TURNS
    )
    get_settings.cache_clear()


def test_settings_local_recent_full_turns_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"{NEO_LOCAL_RECENT_FULL_TURNS_VAR}=6\n", encoding="utf-8"
    )
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    get_settings.cache_clear()
    assert Settings.from_env().local_recent_full_turns == 6
    get_settings.cache_clear()


def test_settings_local_recent_full_turns_floor_at_one(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"{NEO_LOCAL_RECENT_FULL_TURNS_VAR}=0\n", encoding="utf-8"
    )
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    get_settings.cache_clear()
    assert Settings.from_env().local_recent_full_turns == 1
    get_settings.cache_clear()


def test_settings_local_default_smaller_than_cloud_default():
    """Local window must be tighter than the cloud window by design."""
    assert DEFAULT_LOCAL_RECENT_FULL_TURNS < DEFAULT_CONTEXT_RECENT_FULL_TURNS
