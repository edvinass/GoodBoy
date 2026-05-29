"""Tests for TLS / proxy configuration of the OpenAI HTTP client."""

from unittest.mock import MagicMock, patch

import pytest

from llm import _resolve_ssl_verify, clear_openai_client_cache, get_client
from settings import get_settings


def test_resolve_ssl_verify_uses_ca_bundle(tmp_path, monkeypatch):
    ca = tmp_path / "corp-ca.pem"
    ca.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    monkeypatch.setenv("NEO_SSL_CA_BUNDLE", str(ca))
    get_settings.cache_clear()
    clear_openai_client_cache()

    assert _resolve_ssl_verify() == str(ca.resolve())


def test_resolve_ssl_verify_disabled(monkeypatch):
    monkeypatch.setenv("NEO_SSL_VERIFY", "false")
    get_settings.cache_clear()
    clear_openai_client_cache()

    assert _resolve_ssl_verify() is False


def test_resolve_ssl_verify_missing_bundle_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("NEO_SSL_CA_BUNDLE", str(tmp_path / "missing.pem"))
    get_settings.cache_clear()
    clear_openai_client_cache()

    with pytest.raises(Exception) as exc_info:
        _resolve_ssl_verify()
    assert "not found" in str(exc_info.value).lower()


def test_get_client_passes_verify_to_httpx(tmp_path, monkeypatch):
    ca = tmp_path / "corp-ca.pem"
    ca.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    monkeypatch.setenv("NEO_SSL_CA_BUNDLE", str(ca))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    clear_openai_client_cache()

    with patch("llm.httpx.Client") as mock_client_cls, patch("llm.OpenAI") as mock_openai:
        get_client()
        mock_client_cls.assert_called_once_with(verify=str(ca.resolve()))
        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["http_client"] is mock_client_cls.return_value
