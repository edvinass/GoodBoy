"""In-process local GGUF inference via llama-cpp-python (no external daemon)."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from settings import get_settings

LOCAL_MODEL_PREFIX = "local:"

ProgressCallback = Callable[[int, int | None], None]
StreamTextCallback = Callable[[str], None]


@dataclass(frozen=True)
class LocalModelSpec:
    """Catalog entry for a downloadable GGUF model."""

    id: str
    label: str
    repo_id: str
    filename: str
    n_ctx: int = 8192
    n_gpu_layers: int = -1

    @property
    def short_id(self) -> str:
        if self.id.startswith(LOCAL_MODEL_PREFIX):
            return self.id[len(LOCAL_MODEL_PREFIX) :]
        return self.id


LOCAL_MODEL_CATALOG: dict[str, LocalModelSpec] = {
    "qwen2.5-coder-7b-q4": LocalModelSpec(
        id="local:qwen2.5-coder-7b-q4",
        label="Qwen2.5 Coder 7B (Q4, ~4GB) — coding agent",
        repo_id="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        n_ctx=8192,
        n_gpu_layers=-1,
    ),
}

LOCAL_MODEL_LABELS: dict[str, str] = {
    spec.id: spec.label for spec in LOCAL_MODEL_CATALOG.values()
}

_DEFAULT_CATALOG_ID = "qwen2.5-coder-7b-q4"

_runner_lock = threading.Lock()
_runner: LocalModelRunner | None = None


def is_local_model(model_id: str | None) -> bool:
    return bool(model_id and model_id.startswith(LOCAL_MODEL_PREFIX))


def get_catalog_spec(model_id: str) -> LocalModelSpec | None:
    if not is_local_model(model_id):
        return None
    short = model_id[len(LOCAL_MODEL_PREFIX) :]
    return LOCAL_MODEL_CATALOG.get(short)


def default_catalog_spec() -> LocalModelSpec:
    return LOCAL_MODEL_CATALOG[_DEFAULT_CATALOG_ID]


def models_dir() -> Path:
    cfg = get_settings()
    base = cfg.models_dir
    base.mkdir(parents=True, exist_ok=True)
    return base


def model_gguf_path(spec: LocalModelSpec) -> Path:
    return models_dir() / spec.filename


def resolve_model_path(model_id: str) -> Path | None:
    spec = get_catalog_spec(model_id)
    if spec is None:
        return None
    path = model_gguf_path(spec)
    return path if path.is_file() else None


def list_installed_models() -> list[str]:
    return [
        spec.id
        for spec in LOCAL_MODEL_CATALOG.values()
        if model_gguf_path(spec).is_file()
    ]


def has_installed_local_model(model_id: str | None) -> bool:
    if not is_local_model(model_id):
        return False
    return resolve_model_path(model_id) is not None


def require_local_deps() -> None:
    try:
        import llama_cpp  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise click.ClickException(
            "Local models require optional dependencies. Install with:\n"
            '  pip install -e ".[local]"'
        ) from exc


def download_model(
    model_id: str,
    *,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download a catalog model to the GoodBoy models directory."""
    require_local_deps()
    from huggingface_hub import hf_hub_download

    spec = get_catalog_spec(model_id)
    if spec is None:
        raise click.ClickException(f"Unknown local model: {model_id}")

    dest = model_gguf_path(spec)
    if dest.is_file():
        return dest

    def _progress(current: int, total: int | None) -> None:
        if on_progress is not None:
            on_progress(current, total)

    click.echo(f"Downloading {spec.label}…")
    click.echo(f"  Repository: {spec.repo_id}")
    click.echo(f"  File: {spec.filename}")

    downloaded = hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.filename,
        local_dir=str(models_dir()),
        local_dir_use_symlinks=False,
    )
    path = Path(downloaded)
    if path.resolve() != dest.resolve() and path.is_file():
        dest.write_bytes(path.read_bytes())
    if not dest.is_file():
        raise click.ClickException(f"Download failed: expected file at {dest}")
    return dest


class LocalModelRunner:
    """Lazy-loaded llama-cpp model instance (one active model at a time)."""

    def __init__(self) -> None:
        self._llama: Any = None
        self._loaded_id: str | None = None
        self._lock = threading.Lock()

    def unload(self) -> None:
        with self._lock:
            self._llama = None
            self._loaded_id = None

    def _ensure_loaded(self, model_id: str) -> Any:
        require_local_deps()
        from llama_cpp import Llama

        path = resolve_model_path(model_id)
        if path is None:
            raise click.ClickException(
                f"Local model weights not found for {model_id}. "
                "Run: goodboy setup"
            )
        spec = get_catalog_spec(model_id)
        assert spec is not None

        with self._lock:
            if self._llama is not None and self._loaded_id == model_id:
                return self._llama
            if self._llama is not None:
                self._llama = None
                self._loaded_id = None

            click.echo(
                click.style(f"Loading local model {spec.label}…", fg="yellow"),
                err=True,
            )
            self._llama = Llama(
                model_path=str(path),
                n_ctx=spec.n_ctx,
                n_gpu_layers=spec.n_gpu_layers,
                verbose=False,
            )
            self._loaded_id = model_id
            return self._llama

    def complete_chat(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        stream: bool = False,
        on_text_delta: StreamTextCallback | None = None,
        grammar: Any | None = None,
    ) -> tuple[str, Any]:
        llama = self._ensure_loaded(model_id)
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if grammar is not None:
            kwargs["grammar"] = grammar

        if stream and on_text_delta is not None:
            chunks: list[str] = []
            stream_out = llama.create_chat_completion(stream=True, **kwargs)
            for part in stream_out:
                delta = (
                    part.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content")
                )
                if delta:
                    chunks.append(delta)
                    on_text_delta(delta)
            text = "".join(chunks)
            return text, None

        response = llama.create_chat_completion(stream=False, **kwargs)
        text = response["choices"][0]["message"]["content"]
        usage = response.get("usage")
        return text or "", usage


def get_runner() -> LocalModelRunner:
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = LocalModelRunner()
        return _runner


def clear_runner_cache() -> None:
    global _runner
    with _runner_lock:
        if _runner is not None:
            _runner.unload()
            _runner = None


def _build_messages(instructions: str, input_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": input_text},
    ]


def _append_schema(instructions: str, json_schema: dict[str, Any]) -> str:
    schema_text = json.dumps(json_schema, indent=2)
    return f"{instructions}\n\n## Schema (JSON Schema)\n{schema_text}\n"


def _try_json_grammar(json_schema: dict[str, Any]) -> Any | None:
    try:
        from llama_cpp import LlamaGrammar

        return LlamaGrammar.from_json_schema(json.dumps(json_schema))
    except Exception:
        return None


def _usage_from_llama(usage: Any) -> Any:
    from llm import TokenUsage

    if usage is None:
        return None
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or 0)
    else:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or 0)
    if not (prompt or completion or total):
        return None
    if total == 0 and (prompt or completion):
        total = prompt + completion
    return TokenUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
    )


def complete_structured_local(
    *,
    model: str,
    input: str,
    instructions: str,
    json_schema: dict[str, Any],
    stream: bool = False,
    on_text_delta: StreamTextCallback | None = None,
) -> tuple[str, None, Any]:
    """Run structured agent completion on a local GGUF model."""
    require_local_deps()
    full_instructions = _append_schema(instructions, json_schema)
    messages = _build_messages(full_instructions, input)
    grammar = _try_json_grammar(json_schema)
    runner = get_runner()

    use_stream = stream and on_text_delta is not None
    text, raw_usage = runner.complete_chat(
        model_id=model,
        messages=messages,
        stream=use_stream,
        on_text_delta=on_text_delta,
        grammar=grammar,
    )
    return text, None, _usage_from_llama(raw_usage)
