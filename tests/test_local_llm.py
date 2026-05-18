"""Tests for local GGUF model helpers."""

import sys
from unittest.mock import MagicMock, patch

from agent.local_llm import (
    complete_structured_local,
    download_model,
    get_catalog_spec,
    has_installed_local_model,
    is_local_model,
    list_installed_models,
    resolve_local_model,
)
from llm import complete_structured_with_id, get_selectable_models, is_local_model as llm_is_local


def test_is_local_model_prefix():
    assert is_local_model("local:qwen2.5-coder-7b-q4")
    assert not is_local_model("gpt-5.4-nano")
    assert llm_is_local("local:qwen2.5-coder-7b-q4")


def test_list_installed_models_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODBOY_MODELS_DIR", str(tmp_path))
    from settings import get_settings

    get_settings.cache_clear()
    assert list_installed_models() == []


def test_list_installed_models_dropped_gguf(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODBOY_MODELS_DIR", str(tmp_path))
    from settings import get_settings

    get_settings.cache_clear()
    (tmp_path / "my-custom-model.gguf").write_bytes(b"gguf")
    ids = list_installed_models()
    assert ids == ["local:file:my-custom-model.gguf"]


def test_resolve_dropped_gguf(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODBOY_MODELS_DIR", str(tmp_path))
    from settings import get_settings

    get_settings.cache_clear()
    path = tmp_path / "custom.Q4_K_M.gguf"
    path.write_bytes(b"x" * 1000)
    ref = resolve_local_model("local:file:custom.Q4_K_M.gguf")
    assert ref is not None
    assert ref.path == path
    assert "custom.Q4_K_M.gguf" in ref.label


def test_catalog_takes_priority_over_duplicate_filename(tmp_path, monkeypatch):
    spec = get_catalog_spec("local:qwen2.5-coder-7b-q4")
    assert spec is not None
    monkeypatch.setenv("GOODBOY_MODELS_DIR", str(tmp_path))
    from settings import get_settings

    get_settings.cache_clear()
    (tmp_path / spec.filename).write_bytes(b"gguf")
    ids = list_installed_models()
    assert ids == ["local:qwen2.5-coder-7b-q4"]
    assert "local:file:" not in ids[0]


def test_list_installed_models_when_file_present(tmp_path, monkeypatch):
    spec = get_catalog_spec("local:qwen2.5-coder-7b-q4")
    assert spec is not None
    monkeypatch.setenv("GOODBOY_MODELS_DIR", str(tmp_path))
    from settings import get_settings

    get_settings.cache_clear()
    (tmp_path / spec.filename).write_bytes(b"gguf")
    assert "local:qwen2.5-coder-7b-q4" in list_installed_models()
    assert has_installed_local_model("local:qwen2.5-coder-7b-q4")


def test_download_model_skips_when_present(tmp_path, monkeypatch):
    spec = get_catalog_spec("local:qwen2.5-coder-7b-q4")
    assert spec is not None
    monkeypatch.setenv("GOODBOY_MODELS_DIR", str(tmp_path))
    from settings import get_settings

    get_settings.cache_clear()
    dest = tmp_path / spec.filename
    dest.write_bytes(b"existing")

    fake_hub = MagicMock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    with patch("agent.local_llm.require_local_deps"):
        path = download_model(spec.id)
    fake_hub.hf_hub_download.assert_not_called()
    assert path == dest


def test_download_model_calls_http_download(tmp_path, monkeypatch):
    spec = get_catalog_spec("local:qwen2.5-coder-7b-q4")
    assert spec is not None
    monkeypatch.setenv("GOODBOY_MODELS_DIR", str(tmp_path))
    from settings import get_settings

    get_settings.cache_clear()

    dest = tmp_path / spec.filename

    def _fake_download(spec_arg, dest_arg, **kwargs):
        assert spec_arg.id == spec.id
        dest_arg.write_bytes(b"gguf")

    with patch("agent.local_llm.require_local_deps"), patch(
        "agent.local_llm._download_gguf_file", side_effect=_fake_download
    ) as mock_dl:
        path = download_model(spec.id)
    mock_dl.assert_called_once()
    assert path == dest
    assert dest.is_file()


def test_download_reports_progress(tmp_path, monkeypatch):
    spec = get_catalog_spec("local:qwen2.5-coder-7b-q4")
    assert spec is not None
    monkeypatch.setenv("GOODBOY_MODELS_DIR", str(tmp_path))
    from settings import get_settings

    get_settings.cache_clear()
    dest = tmp_path / spec.filename
    seen: list[tuple[int, int | None]] = []

    def _fake_download(_spec, dest_arg, *, on_progress=None, **kwargs):
        if on_progress:
            on_progress(50, 100)
            on_progress(100, 100)
        dest_arg.write_bytes(b"gguf")

    with patch("agent.local_llm.require_local_deps"), patch(
        "agent.local_llm._download_gguf_file", side_effect=_fake_download
    ):
        download_model(
            spec.id,
            on_progress=lambda cur, tot: seen.append((cur, tot)),
        )

    assert seen == [(50, 100), (100, 100)]


def test_complete_structured_local_invokes_runner():
    mock_runner = MagicMock()
    mock_runner.complete_chat.return_value = (
        '{"action":"task_complete","message":"ok"}',
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )
    with patch("agent.local_llm.get_runner", return_value=mock_runner), patch(
        "agent.local_llm.require_local_deps"
    ), patch("agent.local_llm._try_json_grammar", return_value=None):
        text, rid, usage = complete_structured_local(
            model="local:qwen2.5-coder-7b-q4",
            input="task",
            instructions="sys",
            json_schema={"type": "object"},
        )
    assert "task_complete" in text
    assert rid is None
    assert usage is not None
    assert usage.total_tokens == 3


def test_complete_structured_with_id_dispatches_local():
    with patch(
        "agent.local_llm.complete_structured_local",
        return_value=('{"action":"task_complete"}', None, None),
    ) as mock_local:
        text, rid, usage = complete_structured_with_id(
            input="task",
            instructions="sys",
            json_schema={"type": "object"},
            model="local:qwen2.5-coder-7b-q4",
        )
    mock_local.assert_called_once()
    assert rid is None


def test_get_selectable_models_merges_local_and_cloud(tmp_path, monkeypatch):
    spec = get_catalog_spec("local:qwen2.5-coder-7b-q4")
    assert spec is not None
    monkeypatch.setenv("GOODBOY_MODELS_DIR", str(tmp_path))
    (tmp_path / spec.filename).write_bytes(b"gguf")
    from settings import get_settings

    get_settings.cache_clear()

    with patch("llm.get_curated_models", return_value=["gpt-5.4-nano"]):
        models = get_selectable_models(api_key="sk-test")
    assert "local:qwen2.5-coder-7b-q4" in models
    assert "gpt-5.4-nano" in models


def test_is_configured_with_local_only(tmp_path, monkeypatch):
    spec = get_catalog_spec("local:qwen2.5-coder-7b-q4")
    assert spec is not None
    monkeypatch.setenv("GOODBOY_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_MODEL", spec.id)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from settings import get_settings, is_configured

    get_settings.cache_clear()
    (tmp_path / spec.filename).write_bytes(b"gguf")
    assert is_configured()
