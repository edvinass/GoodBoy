"""Isolate tests from the developer's repo ``.env`` and shell exports."""

from __future__ import annotations

import os

import pytest

from settings import ENV_FILE, get_settings, load_env

# Keys that ``Settings.from_env`` / ``load_env`` may read from the environment.
_ISOLATED_ENV_PREFIXES = ("GOODBOY_", "OPENAI_", "DEEPSEEK_")
_ISOLATED_ENV_KEYS = frozenset(
    {
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
    }
)


def _clear_isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(_ISOLATED_ENV_PREFIXES) or key in _ISOLATED_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _isolated_settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Use an empty ``.env`` and drop GoodBoy-related env vars before each test."""
    empty_env = tmp_path / ".env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", empty_env)
    _clear_isolated_env(monkeypatch)
    # Lets model-list validation run without a real API key.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    load_env()
    yield
    get_settings.cache_clear()
