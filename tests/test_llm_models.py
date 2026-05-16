"""Tests for curated model list (no API extras in allowlist)."""

from unittest.mock import MagicMock, patch

from llm import MODEL_CHOICES, get_curated_models


def test_get_curated_models_excludes_api_extras():
    mock_client = MagicMock()
    mock_client.models.list.return_value = [
        MagicMock(id="gpt-4o-mini"),
        MagicMock(id="gpt-3.5-turbo"),
        MagicMock(id="gpt-4.1-2025-04-14"),
        MagicMock(id="gpt-5.5-2026-04-23"),
    ]

    with patch("llm.get_client", return_value=mock_client):
        models = get_curated_models()

    assert models == ["gpt-4o-mini"]
    assert "gpt-3.5-turbo" not in models
    assert "gpt-4.1-2025-04-14" not in models
    assert all(m in MODEL_CHOICES for m in models)


def test_format_models_section_only_curated_entries():
    from agent.models import format_models_section

    text = format_models_section(["gpt-4o-mini", "gpt-3.5-turbo", "gpt-5.5"])
    assert "gpt-4o-mini" in text
    assert "gpt-5.5" in text
    assert "gpt-3.5-turbo" not in text
    assert "No curated guidance" not in text
    assert "Other allowed IDs" not in text
