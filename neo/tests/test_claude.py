"""Tests for Anthropic Claude provider integration."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.claude_llm import (
    CLAUDE_BASE_URL,
    CLAUDE_MODEL_IDS,
    CLAUDE_MODELS,
    claude_model_label,
    clear_claude_client_cache,
    complete_claude,
    complete_structured_claude,
    get_claude_client,
    get_claude_model_spec,
    is_claude_model,
)
from agent.loop import AgentLoop
from agent.models import MODEL_CATALOG, get_model_spec, resolve_reasoning_effort
from llm import get_selectable_models, is_claude_model as llm_is_claude
from settings import ANTHROPIC_API_KEY_VAR, Settings, get_settings


def _stub_settings(monkeypatch, tmp_path, *, anthropic_key: str | None = "ant-test"):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("settings.ENV_FILE", env_file)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    if anthropic_key is None:
        monkeypatch.delenv(ANTHROPIC_API_KEY_VAR, raising=False)
    else:
        monkeypatch.setenv(ANTHROPIC_API_KEY_VAR, anthropic_key)
    get_settings.cache_clear()
    clear_claude_client_cache()


def test_is_claude_model_matches_prefix():
    assert is_claude_model("claude-opus-4-8")
    assert is_claude_model("claude-sonnet-4-6")
    assert llm_is_claude("claude-haiku-4-5")
    assert not is_claude_model("gpt-5.4-mini")
    assert not is_claude_model("deepseek-v4-pro")
    assert not is_claude_model("local:qwen2.5-coder-7b-q4")
    assert not is_claude_model(None)


def test_claude_catalog_present_in_model_catalog():
    for model_id in CLAUDE_MODEL_IDS:
        spec = get_model_spec(model_id)
        assert spec is not None, f"Missing catalog entry for {model_id}"
        assert MODEL_CATALOG[model_id].id == model_id


def test_claude_model_label_falls_back_to_id():
    assert claude_model_label("claude-opus-4-8").startswith("Claude Opus 4.8")
    assert claude_model_label("unknown-model") == "unknown-model"


def test_claude_settings_reads_env(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path, anthropic_key="sk-ant-xyz")
    cfg = Settings.from_env()
    assert cfg.anthropic_api_key == "sk-ant-xyz"
    assert cfg.openai_api_key is None
    get_settings.cache_clear()


def test_claude_settings_strips_whitespace(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path, anthropic_key="   ")
    cfg = Settings.from_env()
    assert cfg.anthropic_api_key is None
    get_settings.cache_clear()


def test_get_claude_client_requires_key(monkeypatch, tmp_path):
    import click

    _stub_settings(monkeypatch, tmp_path, anthropic_key=None)
    get_settings.cache_clear()
    with pytest.raises(click.ClickException):
        get_claude_client()
    get_settings.cache_clear()


def test_get_claude_client_uses_correct_base_url(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path)

    captured: dict[str, object] = {}

    def fake_anthropic(*, api_key, base_url, http_client):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        captured["http_client"] = http_client
        return MagicMock()

    with patch("anthropic.Anthropic", fake_anthropic):
        client = get_claude_client()
    assert captured["api_key"] == "ant-test"
    assert captured["base_url"] == CLAUDE_BASE_URL
    assert client is not None
    clear_claude_client_cache()
    get_settings.cache_clear()


def _fake_message_response(text: str, *, input_tokens: int = 9, output_tokens: int = 12):
    block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn")


def _fake_message_response_with_thinking(text: str):
    thinking_block = SimpleNamespace(type="thinking", thinking="internal scratchpad")
    text_block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=5, output_tokens=7)
    return SimpleNamespace(content=[thinking_block, text_block], usage=usage)


def test_complete_claude_passes_system_and_returns_text(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path)
    fake_create = MagicMock(return_value=_fake_message_response("hello world"))
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=fake_create)
    )

    def make_client(*_args, **_kwargs):
        return fake_client

    with patch("agent.claude_llm._claude_client", make_client):
        text = complete_claude(
            "explain entropy",
            model="claude-sonnet-4-6",
            instructions="You are helpful",
            reasoning_effort="medium",
        )
    assert text == "hello world"
    fake_create.assert_called_once()
    call_kwargs = fake_create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["system"] == "You are helpful"
    assert call_kwargs["messages"][0] == {"role": "user", "content": "explain entropy"}
    assert call_kwargs["max_tokens"] > 0
    # Reasoning effort should be forwarded as adaptive thinking + output_config.effort.
    assert call_kwargs.get("thinking") == {"type": "adaptive"}
    assert call_kwargs.get("output_config") == {"effort": "medium"}
    clear_claude_client_cache()
    get_settings.cache_clear()


def test_complete_claude_skips_thinking_blocks(monkeypatch, tmp_path):
    """Multi-block responses must drop scratchpad and return only the visible text."""
    _stub_settings(monkeypatch, tmp_path)
    fake_create = MagicMock(
        return_value=_fake_message_response_with_thinking("user-facing answer")
    )
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    with patch("agent.claude_llm._claude_client", lambda *_a, **_k: fake_client):
        text = complete_claude(
            "explain",
            model="claude-opus-4-8",
            instructions="sys",
            reasoning_effort="high",
        )
    assert text == "user-facing answer"
    clear_claude_client_cache()
    get_settings.cache_clear()


def test_complete_claude_skips_reasoning_for_non_reasoning_model(
    monkeypatch, tmp_path
):
    """Haiku 4.5 has no extended thinking — request must omit thinking kwargs."""
    _stub_settings(monkeypatch, tmp_path)
    fake_create = MagicMock(return_value=_fake_message_response("ok"))
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    with patch("agent.claude_llm._claude_client", lambda *_a, **_k: fake_client):
        complete_claude(
            "hi",
            model="claude-haiku-4-5",
            instructions="be brief",
            reasoning_effort="high",
        )
    call_kwargs = fake_create.call_args.kwargs
    assert "thinking" not in call_kwargs
    assert "output_config" not in call_kwargs
    clear_claude_client_cache()
    get_settings.cache_clear()


def test_complete_structured_claude_uses_output_config_format(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path)
    payload = {"action": "task_complete", "message": "done", "status": "Done"}
    fake_create = MagicMock(return_value=_fake_message_response(json.dumps(payload)))
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))

    with patch("agent.claude_llm._claude_client", lambda *_a, **_k: fake_client):
        text, response_id, usage = complete_structured_claude(
            model="claude-sonnet-4-6",
            input="Do the thing",
            instructions="Follow rules.",
            json_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "message": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
            reasoning_effort="medium",
        )
    assert json.loads(text) == payload
    assert response_id is None
    assert usage is not None and usage.input_tokens == 9

    call_kwargs = fake_create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["system"] == "Follow rules."
    output_config = call_kwargs["output_config"]
    assert output_config["effort"] == "medium"
    fmt = output_config["format"]
    assert fmt["type"] == "json_schema"
    schema = fmt["schema"]
    # Object schemas must be tightened for Anthropic's GA structured outputs.
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"action", "message", "status"}
    assert call_kwargs.get("thinking") == {"type": "adaptive"}
    clear_claude_client_cache()
    get_settings.cache_clear()


@contextmanager
def _fake_stream(chunks: list[str], final_response):
    class _StreamCtx:
        text_stream = iter(chunks)

        def get_final_message(self):
            return final_response

        def close(self):
            pass

    yield _StreamCtx()


def test_complete_structured_claude_streams_text(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path)
    payload = {"action": "task_complete", "message": "done", "status": "Done"}
    chunks = ['{"action":"task_complete",', '"message":"done",', '"status":"Done"}']
    final = _fake_message_response(json.dumps(payload), input_tokens=4, output_tokens=8)

    def fake_stream(**_kwargs):
        return _fake_stream(chunks, final)

    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=MagicMock(), stream=fake_stream)
    )
    seen: list[str] = []

    with patch("agent.claude_llm._claude_client", lambda *_a, **_k: fake_client):
        text, response_id, usage = complete_structured_claude(
            model="claude-haiku-4-5",
            input="Stream please",
            instructions="System",
            json_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "message": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
            stream=True,
            on_text_delta=seen.append,
        )
    assert json.loads(text) == payload
    assert response_id is None
    assert seen == chunks
    assert usage is not None and usage.output_tokens == 8
    clear_claude_client_cache()
    get_settings.cache_clear()


def test_reasoning_effort_resolution_for_claude():
    assert resolve_reasoning_effort("claude-opus-4-8", "medium") == "medium"
    assert resolve_reasoning_effort("claude-opus-4-8", "high") == "high"
    assert resolve_reasoning_effort("claude-opus-4-8", "xhigh") == "xhigh"
    # Haiku 4.5 is catalogued as non-reasoning, so any effort collapses to None.
    assert resolve_reasoning_effort("claude-haiku-4-5", "high") is None


def test_selectable_models_include_claude_when_key_present(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path)
    monkeypatch.setattr("agent.local_llm.list_installed_models", lambda: [])
    monkeypatch.setattr("llm.get_curated_models", lambda *_a, **_k: ["gpt-5.4-nano"])
    models = get_selectable_models(api_key="sk-openai")
    for cm in CLAUDE_MODEL_IDS:
        assert cm in models
    clear_claude_client_cache()
    get_settings.cache_clear()


def test_selectable_models_exclude_claude_when_no_key(monkeypatch, tmp_path):
    _stub_settings(monkeypatch, tmp_path, anthropic_key=None)
    monkeypatch.setattr("agent.local_llm.list_installed_models", lambda: [])
    monkeypatch.setattr("llm.get_curated_models", lambda *_a, **_k: ["gpt-5.4-nano"])
    models = get_selectable_models(api_key="sk-openai")
    for cm in CLAUDE_MODEL_IDS:
        assert cm not in models
    clear_claude_client_cache()
    get_settings.cache_clear()


def test_loop_disables_response_chain_for_claude(tmp_path, monkeypatch):
    _stub_settings(monkeypatch, tmp_path)

    def fake_llm(**_kwargs):
        return (
            '{"action":"task_complete","message":"done","status":"Done"}',
            None,
            None,
        )

    loop = AgentLoop(
        workspace=tmp_path,
        model="claude-opus-4-8",
        allowed_models=["claude-opus-4-8"],
        llm_call=fake_llm,
    )
    assert loop._chain_enabled is False
    clear_claude_client_cache()
    get_settings.cache_clear()


def test_loop_does_not_send_hosted_tools_for_claude(tmp_path, monkeypatch):
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
        model="claude-sonnet-4-6",
        allowed_models=["claude-sonnet-4-6"],
        llm_call=tracking_llm,
    )
    loop.run("ping")
    assert captured
    call = captured[0]
    assert "tools" not in call
    assert "previous_response_id" not in call
    clear_claude_client_cache()
    get_settings.cache_clear()


def test_claude_models_listed_in_catalog_label_format():
    for spec in CLAUDE_MODELS:
        assert spec.id.startswith("claude-")
        assert spec.label
