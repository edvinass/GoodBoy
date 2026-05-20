"""Tests for DeepSeek provider integration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.deepseek_llm import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL_IDS,
    DEEPSEEK_MODELS,
    clear_deepseek_client_cache,
    complete_deepseek,
    complete_structured_deepseek,
    deepseek_model_label,
    get_deepseek_client,
    get_deepseek_model_spec,
    is_deepseek_model,
)
from agent.loop import AgentLoop
from agent.models import MODEL_CATALOG, get_model_spec, resolve_reasoning_effort
from llm import get_selectable_models, is_deepseek_model as llm_is_deepseek
from settings import DEEPSEEK_API_KEY_VAR, Settings, get_settings


def _stub_settings(monkeypatch, tmp_path, *, deepseek_key: str | None = "ds-test"):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    if deepseek_key is None:
        monkeypatch.delenv(DEEPSEEK_API_KEY_VAR, raising=False)
    else:
        monkeypatch.setenv(DEEPSEEK_API_KEY_VAR, deepseek_key)
    get_settings.cache_clear()
    clear_deepseek_client_cache()


def test_is_deepseek_model_matches_prefix():
    assert is_deepseek_model("deepseek-v4-pro")
    assert is_deepseek_model("deepseek-chat")
    assert llm_is_deepseek("deepseek-v4-flash")
    assert not is_deepseek_model("gpt-5.4-mini")
    assert not is_deepseek_model("local:qwen2.5-coder-7b-q4")
    assert not is_deepseek_model(None)


def test_deepseek_catalog_present_in_model_catalog():
    for model_id in DEEPSEEK_MODEL_IDS:
        spec = get_model_spec(model_id)
        assert spec is not None, f"Missing catalog entry for {model_id}"
        assert MODEL_CATALOG[model_id].id == model_id


def test_deepseek_model_label_falls_back_to_id():
    assert deepseek_model_label("deepseek-v4-pro").startswith("DeepSeek V4 Pro")
    assert deepseek_model_label("unknown-model") == "unknown-model"


def test_deepseek_settings_reads_env(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path, deepseek_key="sk-deepseek-xyz")
    cfg = Settings.from_env()
    assert cfg.deepseek_api_key == "sk-deepseek-xyz"
    assert cfg.openai_api_key is None
    get_settings.cache_clear()


def test_deepseek_settings_strips_whitespace(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path, deepseek_key="   ")
    cfg = Settings.from_env()
    assert cfg.deepseek_api_key is None
    get_settings.cache_clear()


def test_get_deepseek_client_requires_key(monkeypatch, tmp_path):
    import click

    _stub_settings(monkeypatch, tmp_path, deepseek_key=None)
    get_settings.cache_clear()
    with pytest.raises(click.ClickException):
        get_deepseek_client()
    get_settings.cache_clear()


def test_get_deepseek_client_uses_correct_base_url(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path)

    captured: dict[str, object] = {}

    def fake_openai(*, api_key, base_url, http_client):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        captured["http_client"] = http_client
        return MagicMock()

    with patch("agent.deepseek_llm.OpenAI", fake_openai):
        client = get_deepseek_client()
    assert captured["api_key"] == "ds-test"
    assert captured["base_url"] == DEEPSEEK_BASE_URL
    assert client is not None
    clear_deepseek_client_cache()
    get_settings.cache_clear()


def _fake_chat_response(text: str, *, prompt: int = 7, completion: int = 11):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def test_complete_deepseek_passes_messages_and_returns_text(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    fake_create = MagicMock(return_value=_fake_chat_response("hello"))
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    def make_client(*_args, **_kwargs):
        return fake_client

    with patch("agent.deepseek_llm._deepseek_client", make_client):
        text = complete_deepseek(
            "explain",
            model="deepseek-v4-pro",
            instructions="You are helpful",
            reasoning_effort="medium",
        )
    assert text == "hello"
    fake_create.assert_called_once()
    call_kwargs = fake_create.call_args.kwargs
    captured.update(call_kwargs)
    assert captured["model"] == "deepseek-v4-pro"
    assert captured["messages"][0] == {"role": "system", "content": "You are helpful"}
    assert captured["messages"][1] == {"role": "user", "content": "explain"}
    assert captured.get("reasoning_effort") == "medium"
    extra = captured.get("extra_body") or {}
    assert extra.get("thinking", {}).get("type") == "enabled"
    clear_deepseek_client_cache()
    get_settings.cache_clear()


def test_complete_structured_deepseek_uses_json_object_format(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path)
    payload = {"action": "task_complete", "message": "done", "status": "Done"}
    fake_create = MagicMock(return_value=_fake_chat_response(json.dumps(payload)))
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    with patch("agent.deepseek_llm._deepseek_client", lambda *_a, **_k: fake_client):
        text, response_id, usage = complete_structured_deepseek(
            model="deepseek-v4-flash",
            input="Do the thing",
            instructions="Follow rules.",
            json_schema={"type": "object", "properties": {"action": {"type": "string"}}},
        )
    assert json.loads(text) == payload
    assert response_id is None
    assert usage is not None and usage.input_tokens == 7

    call_kwargs = fake_create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-v4-flash"
    assert call_kwargs["response_format"] == {"type": "json_object"}
    system_msg = call_kwargs["messages"][0]
    assert "JSON Schema" in system_msg["content"]
    clear_deepseek_client_cache()
    get_settings.cache_clear()


def test_reasoning_effort_resolution_for_deepseek():
    # Reasoning models map medium/high through; xhigh collapses to high.
    assert resolve_reasoning_effort("deepseek-v4-pro", "medium") == "medium"
    assert resolve_reasoning_effort("deepseek-v4-pro", "high") == "high"
    # Non-reasoning legacy alias drops effort silently.
    assert resolve_reasoning_effort("deepseek-chat", "medium") is None


def test_selectable_models_include_deepseek_when_key_present(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "agent.local_llm.list_installed_models", lambda: []
    )
    models = get_selectable_models()
    for ds in DEEPSEEK_MODEL_IDS:
        assert ds in models
    clear_deepseek_client_cache()
    get_settings.cache_clear()


def test_selectable_models_exclude_deepseek_when_no_key(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path, deepseek_key=None)
    monkeypatch.setattr("agent.local_llm.list_installed_models", lambda: [])
    monkeypatch.setattr("llm.get_curated_models", lambda *_a, **_k: ["gpt-5.4-nano"])
    models = get_selectable_models(api_key="sk-openai")
    for ds in DEEPSEEK_MODEL_IDS:
        assert ds not in models
    clear_deepseek_client_cache()
    get_settings.cache_clear()


def test_loop_disables_response_chain_for_deepseek(tmp_path, monkeypatch):
    _stub_settings(monkeypatch, tmp_path)

    def fake_llm(**_kwargs):
        return (
            '{"action":"task_complete","message":"done","status":"Done"}',
            None,
            None,
        )

    loop = AgentLoop(
        workspace=tmp_path,
        model="deepseek-v4-pro",
        allowed_models=["deepseek-v4-pro"],
        llm_call=fake_llm,
    )
    assert loop._chain_enabled is False
    clear_deepseek_client_cache()
    get_settings.cache_clear()


def test_loop_does_not_send_hosted_tools_for_deepseek(tmp_path, monkeypatch):
    _stub_settings(monkeypatch, tmp_path)
    captured: list[dict] = []

    def tracking_llm(**kwargs):
        captured.append(dict(kwargs))
        return (
            '{"action":"task_complete","message":"ok","status":"Done"}',
            None,
            None,
        )

    loop = AgentLoop(
        workspace=tmp_path,
        model="deepseek-v4-pro",
        allowed_models=["deepseek-v4-pro"],
        llm_call=tracking_llm,
    )
    loop.run("ping")
    assert captured
    call = captured[0]
    assert "tools" not in call
    assert "previous_response_id" not in call
    clear_deepseek_client_cache()
    get_settings.cache_clear()


def test_deepseek_models_listed_in_catalog_label_format():
    for spec in DEEPSEEK_MODELS:
        assert spec.id.startswith("deepseek-")
        assert spec.label
