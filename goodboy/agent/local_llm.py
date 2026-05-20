"""In-process local GGUF inference via llama-cpp-python (no external daemon)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from agent.tools import _smart_truncate
from settings import get_settings

_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_CONTEXT_SAFETY_TOKENS = 256
_LOCAL_DEFAULT_MAX_TOKENS = 2048

LOCAL_MODEL_PREFIX = "local:"
LOCAL_FILE_PREFIX = "local:file:"

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


@dataclass(frozen=True)
class LocalModelRef:
    """Resolved local model (catalog entry or dropped GGUF file)."""

    id: str
    label: str
    path: Path
    n_ctx: int = 8192
    n_gpu_layers: int = -1


# Ordered small → large for setup menus (bartowski GGUF repos, Q4_K_M quants).
_LOCAL_MODEL_SPECS: tuple[LocalModelSpec, ...] = (
    LocalModelSpec(
        id="local:llama-3.2-3b-q4",
        label="Llama 3.2 3B Instruct (Q4, ~2GB) — fast, light RAM",
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        n_ctx=8192,
    ),
    LocalModelSpec(
        id="local:phi-3.5-mini-q4",
        label="Phi-3.5 Mini Instruct (Q4, ~2.4GB) — compact general",
        repo_id="bartowski/Phi-3.5-mini-instruct-GGUF",
        filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        n_ctx=8192,
    ),
    LocalModelSpec(
        id="local:qwen2.5-coder-7b-q4",
        label="Qwen2.5 Coder 7B (Q4, ~4.7GB) — coding agent (default)",
        repo_id="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        n_ctx=8192,
    ),
    LocalModelSpec(
        id="local:qwen2.5-7b-q4",
        label="Qwen2.5 7B Instruct (Q4, ~4.7GB) — strong general",
        repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        n_ctx=8192,
    ),
    LocalModelSpec(
        id="local:mistral-7b-q4",
        label="Mistral 7B Instruct v0.3 (Q4, ~4.4GB) — general chat",
        repo_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        filename="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        n_ctx=8192,
    ),
    LocalModelSpec(
        id="local:granite-3.1-8b-q4",
        label="Granite 3.1 8B Instruct (Q4, ~4.9GB) — enterprise general",
        repo_id="bartowski/granite-3.1-8b-instruct-GGUF",
        filename="granite-3.1-8b-instruct-Q4_K_M.gguf",
        n_ctx=8192,
    ),
    LocalModelSpec(
        id="local:gemma-2-9b-q4",
        label="Gemma 2 9B IT (Q4, ~5.8GB) — capable general",
        repo_id="bartowski/gemma-2-9b-it-GGUF",
        filename="gemma-2-9b-it-Q4_K_M.gguf",
        n_ctx=8192,
    ),
    LocalModelSpec(
        id="local:qwen2.5-coder-14b-q4",
        label="Qwen2.5 Coder 14B (Q4, ~9GB) — best coding, 16GB+ RAM",
        repo_id="bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
        filename="Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        n_ctx=8192,
    ),
    LocalModelSpec(
        id="local:deepseek-coder-v2-lite-q4",
        label="DeepSeek Coder V2 Lite (Q4, ~10GB) — MoE coding, 16GB+ RAM",
        repo_id="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        filename="DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        n_ctx=8192,
    ),
)

LOCAL_MODEL_CATALOG: dict[str, LocalModelSpec] = {
    spec.short_id: spec for spec in _LOCAL_MODEL_SPECS
}

LOCAL_MODEL_LABELS: dict[str, str] = {
    spec.id: spec.label for spec in _LOCAL_MODEL_SPECS
}

_DEFAULT_CATALOG_ID = "qwen2.5-coder-7b-q4"


def iter_catalog() -> tuple[LocalModelSpec, ...]:
    """Catalog entries in setup-menu order (smallest download first)."""
    return _LOCAL_MODEL_SPECS

_runner_lock = threading.Lock()
_runner: LocalModelRunner | None = None

# One-shot throttle for the "context full" warning. The agent loop calls
# `reset_context_full_notice()` at the start of every task so the user sees a
# single actionable notice per task instead of one per LLM call.
_truncation_state: dict[str, int] = {"events": 0, "warned_at": 0}


def reset_context_full_notice() -> None:
    """Re-arm the per-task context-full warning."""
    _truncation_state["events"] = 0
    _truncation_state["warned_at"] = 0


def context_full_events() -> int:
    """How many LLM calls in the current task had to trim prompt content."""
    return _truncation_state["events"]


def _emit_context_full_notice() -> None:
    """Track a truncation event; warn once per task with actionable advice."""
    _truncation_state["events"] += 1
    if _truncation_state["warned_at"]:
        return
    _truncation_state["warned_at"] = _truncation_state["events"]
    click.echo(
        click.style(
            "Local model context is tight — older tool output is being "
            "truncated. For better results: /clear to drop conversation, "
            "narrow the task, or /model a larger-context model.",
            fg="yellow",
        ),
        err=True,
    )


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


def _scan_gguf_files() -> dict[str, Path]:
    """Map filename → path for every ``.gguf`` in the models directory (top level)."""
    base = models_dir()
    return {
        path.name: path
        for path in sorted(base.glob("*.gguf"))
        if path.is_file()
    }


def _label_for_dropped_file(filename: str) -> str:
    size_gb = ""
    try:
        path = models_dir() / filename
        gib = path.stat().st_size / (1024**3)
        size_gb = f", ~{gib:.1f}GB"
    except OSError:
        pass
    return f"{filename}{size_gb} — dropped GGUF"


def _model_id_for_file(filename: str) -> str:
    return f"{LOCAL_FILE_PREFIX}{filename}"


def is_dropped_file_model(model_id: str) -> bool:
    return model_id.startswith(LOCAL_FILE_PREFIX)


def resolve_local_model(model_id: str) -> LocalModelRef | None:
    """Resolve a local model id to weights path and runtime settings."""
    if not is_local_model(model_id):
        return None

    files = _scan_gguf_files()
    spec = get_catalog_spec(model_id)
    if spec is not None and spec.filename in files:
        return LocalModelRef(
            id=spec.id,
            label=spec.label,
            path=files[spec.filename],
            n_ctx=spec.n_ctx,
            n_gpu_layers=spec.n_gpu_layers,
        )

    if is_dropped_file_model(model_id):
        filename = model_id[len(LOCAL_FILE_PREFIX) :]
        if filename in files:
            return LocalModelRef(
                id=model_id,
                label=_label_for_dropped_file(filename),
                path=files[filename],
            )

    return None


def resolve_model_path(model_id: str) -> Path | None:
    ref = resolve_local_model(model_id)
    return ref.path if ref is not None else None


def local_model_label(model_id: str) -> str:
    ref = resolve_local_model(model_id)
    if ref is not None:
        return ref.label
    return LOCAL_MODEL_LABELS.get(model_id, model_id)


def list_installed_models() -> list[str]:
    """Catalog matches first, then any other ``.gguf`` files in the models directory."""
    files = _scan_gguf_files()
    ids: list[str] = []
    claimed: set[Path] = set()

    for spec in _LOCAL_MODEL_SPECS:
        path = files.get(spec.filename)
        if path is None:
            continue
        ids.append(spec.id)
        claimed.add(path.resolve())

    for filename in sorted(files):
        path = files[filename]
        if path.resolve() in claimed:
            continue
        ids.append(_model_id_for_file(filename))

    return ids


def has_installed_local_model(model_id: str | None) -> bool:
    if not is_local_model(model_id):
        return False
    return resolve_local_model(model_id) is not None


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _local_deps_available() -> bool:
    try:
        import llama_cpp  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False
    return True


def ensure_local_deps(*, install: bool = True) -> None:
    """Import local-model dependencies, optionally pip-installing [local] extras."""
    if _local_deps_available():
        return
    if not install:
        raise click.ClickException(_local_deps_install_hint())

    pkg = _package_root()
    pyproject = pkg / "pyproject.toml"
    if not pyproject.is_file():
        raise click.ClickException(_local_deps_install_hint())

    click.echo(
        "Installing local model dependencies (llama-cpp-python, huggingface-hub)…"
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-e", f"{pkg}[local]"],
    )
    if not _local_deps_available():
        raise click.ClickException(_local_deps_install_hint())


def _local_deps_install_hint() -> str:
    pkg = _package_root()
    return (
        "Local models require optional dependencies. Install with:\n"
        f'  {sys.executable} -m pip install -e "{pkg}[local]"'
    )


def require_local_deps() -> None:
    ensure_local_deps(install=True)


def _resolve_hf_token() -> str | bool | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None


def _prefer_http_download_env() -> None:
    """Xet transfers often hang at 0% on some networks; HTTP CDN is more reliable."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _download_gguf_file(
    spec: LocalModelSpec,
    dest: Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Stream a GGUF from the Hub CDN with a live progress bar."""
    import httpx
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers, hf_raise_for_status

    _prefer_http_download_env()
    token = _resolve_hf_token()

    hub_url = hf_hub_url(repo_id=spec.repo_id, filename=spec.filename)
    metadata = get_hf_file_metadata(hub_url, token=token)
    download_url = metadata.location
    total_size = metadata.size

    partial = dest.with_suffix(dest.suffix + ".part")
    resume_from = partial.stat().st_size if partial.is_file() else 0
    if resume_from > 0 and total_size is not None and resume_from >= total_size:
        partial.replace(dest)
        return

    headers = build_hf_headers(token=token)
    headers["Accept-Encoding"] = "identity"
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    dest.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task_id = progress.add_task(
            spec.filename,
            total=total_size,
            completed=resume_from,
        )

        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=None)) as client:
            with client.stream("GET", download_url, headers=headers) as response:
                if resume_from > 0 and response.status_code == 416:
                    partial.unlink(missing_ok=True)
                    return _download_gguf_file(spec, dest, on_progress=on_progress)
                if resume_from > 0 and response.status_code == 200:
                    partial.unlink(missing_ok=True)
                    return _download_gguf_file(spec, dest, on_progress=on_progress)
                hf_raise_for_status(response)
                if total_size is None:
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        total_size = int(content_length) + resume_from
                        progress.update(task_id, total=total_size)

                mode = "ab" if resume_from > 0 else "wb"
                with partial.open(mode) as out:
                    for chunk in response.iter_bytes(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                        if not chunk:
                            continue
                        out.write(chunk)
                        progress.update(task_id, advance=len(chunk))
                        if on_progress is not None:
                            on_progress(partial.stat().st_size, total_size)

    partial.replace(dest)


def download_model(
    model_id: str,
    *,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download a catalog model to the GoodBoy models directory."""
    require_local_deps()

    spec = get_catalog_spec(model_id)
    if spec is None:
        raise click.ClickException(f"Unknown local model: {model_id}")

    dest = model_gguf_path(spec)
    if dest.is_file():
        return dest

    click.echo(f"Downloading {spec.label}…")
    click.echo(f"  Repository: {spec.repo_id}")
    click.echo(f"  File: {spec.filename}")
    if _resolve_hf_token() is None:
        click.echo(
            click.style(
                "  Tip: set HF_TOKEN for faster Hub downloads "
                "(https://huggingface.co/settings/tokens)",
                fg="yellow",
            )
        )

    _download_gguf_file(spec, dest, on_progress=on_progress)

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

        ref = resolve_local_model(model_id)
        if ref is None:
            raise click.ClickException(
                f"Local model weights not found for {model_id}. "
                f"Place a .gguf in {models_dir()} or run: goodboy setup"
            )

        with self._lock:
            if self._llama is not None and self._loaded_id == model_id:
                return self._llama
            if self._llama is not None:
                self._llama = None
                self._loaded_id = None

            click.echo(
                click.style(f"Loading local model {ref.label}…", fg="yellow"),
                err=True,
            )
            self._llama = Llama(
                model_path=str(ref.path),
                n_ctx=ref.n_ctx,
                n_gpu_layers=ref.n_gpu_layers,
                verbose=False,
            )
            self._loaded_id = model_id
            return self._llama

    def complete_chat(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = _LOCAL_DEFAULT_MAX_TOKENS,
        stream: bool = False,
        on_text_delta: StreamTextCallback | None = None,
        grammar: Any | None = None,
        abort_check: Any | None = None,
    ) -> tuple[str, Any]:
        from llm import UserAbort

        llama = self._ensure_loaded(model_id)
        n_ctx = int(llama.n_ctx())
        max_tokens = _effective_max_tokens(n_ctx, max_tokens)
        messages = _fit_messages_to_context(
            llama, messages, max_tokens=max_tokens, n_ctx=n_ctx
        )
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if grammar is not None:
            kwargs["grammar"] = grammar

        try:
            if abort_check is not None and abort_check():
                raise UserAbort()
            if stream and on_text_delta is not None:
                chunks: list[str] = []
                stream_out = llama.create_chat_completion(stream=True, **kwargs)
                for part in stream_out:
                    if abort_check is not None and abort_check():
                        try:
                            stream_out.close()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        raise UserAbort()
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
            if abort_check is not None and abort_check():
                raise UserAbort()
            text = response["choices"][0]["message"]["content"]
            usage = response.get("usage")
            return text or "", usage
        except ValueError as exc:
            if "exceed context window" in str(exc).lower():
                raise click.ClickException(
                    "Local model context window exceeded. Try /clear, a smaller "
                    "task, or switch to a cloud model with /model."
                ) from exc
            raise


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


def _resolve_chat_formatter(llama: Any) -> Callable[..., Any]:
    """Return the chat template formatter backing ``create_chat_completion``."""
    import llama_cpp.llama_chat_format as lcf

    handler = (
        llama.chat_handler
        or getattr(llama, "_chat_handlers", {}).get(llama.chat_format)
        or lcf.get_chat_completion_handler(llama.chat_format)
    )
    closure = getattr(handler, "__closure__", None)
    if closure:
        formatter = closure[0].cell_contents
        if callable(formatter):
            return formatter
    raise RuntimeError("Could not resolve chat formatter for local model")


def _count_chat_tokens(llama: Any, messages: list[dict[str, str]]) -> int:
    formatter = _resolve_chat_formatter(llama)
    result = formatter(messages=messages)
    tokens = llama.tokenize(
        result.prompt.encode("utf-8"),
        add_bos=not result.added_special,
        special=True,
    )
    return len(tokens)


def _effective_max_tokens(n_ctx: int, requested: int) -> int:
    """Cap completion tokens so a reasonable prompt budget remains."""
    cap = max(512, (n_ctx * 2) // 5)
    return min(requested, cap)


def _fit_messages_to_context(
    llama: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    n_ctx: int,
) -> list[dict[str, str]]:
    """Truncate chat content so the rendered prompt fits in ``n_ctx``."""
    limit = max(512, n_ctx - max_tokens - _CONTEXT_SAFETY_TOKENS)
    fitted = [dict(m) for m in messages]
    if _count_chat_tokens(llama, fitted) <= limit:
        return fitted

    original_user = next(
        (m.get("content") or "") for m in fitted if m.get("role") == "user"
    )
    user_msg = next(m for m in fitted if m.get("role") == "user")
    encoded = original_user.encode("utf-8", errors="replace")
    budget = len(encoded)
    while budget > 2048 and _count_chat_tokens(llama, fitted) > limit:
        budget = int(budget * 0.85)
        head = budget // 2
        tail = budget - head
        user_msg["content"] = _smart_truncate(
            original_user, head_bytes=head, tail_bytes=tail
        )

    if _count_chat_tokens(llama, fitted) > limit:
        system_msg = next((m for m in fitted if m.get("role") == "system"), None)
        if system_msg is not None:
            original_system = system_msg.get("content") or ""
            sys_encoded = original_system.encode("utf-8", errors="replace")
            sys_budget = len(sys_encoded)
            while sys_budget > 2048 and _count_chat_tokens(llama, fitted) > limit:
                sys_budget = int(sys_budget * 0.85)
                head = sys_budget // 2
                tail = sys_budget - head
                system_msg["content"] = _smart_truncate(
                    original_system, head_bytes=head, tail_bytes=tail
                )

    if _count_chat_tokens(llama, fitted) > limit:
        raise click.ClickException(
            "Local model context window is too small for this prompt even after "
            "truncation. Try /clear, shorten the task, or switch to a cloud model "
            "with /model."
        )

    if user_msg.get("content") != original_user:
        _emit_context_full_notice()
    return fitted


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
    abort_check: Any | None = None,
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
        abort_check=abort_check,
    )
    return text, None, _usage_from_llama(raw_usage)
