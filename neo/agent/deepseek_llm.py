"""DeepSeek API integration (OpenAI-compatible Chat Completions).

DeepSeek exposes an OpenAI-compatible REST endpoint at ``https://api.deepseek.com``
with ``chat/completions`` semantics. We use the official ``openai`` SDK with a
custom ``base_url`` so we can reuse the same HTTP/SSL plumbing as the OpenAI
provider without depending on any DeepSeek-specific package.

See: https://api-docs.deepseek.com/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import click
import httpx
from openai import APIConnectionError, OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_PREFIX = "deepseek-"


@dataclass(frozen=True)
class DeepSeekModelSpec:
    """Static metadata for a selectable DeepSeek model."""

    id: str
    label: str
    reasoning: bool
    price_input_per_1m: float
    price_cached_input_per_1m: float
    price_output_per_1m: float
    enable_thinking: bool = False


# https://api-docs.deepseek.com/quick_start/pricing (cache-miss prices, no
# promotional discount). ``deepseek-chat`` / ``deepseek-reasoner`` remain
# selectable for compatibility — they map to non-thinking / thinking modes of
# ``deepseek-v4-flash`` server-side.
DEEPSEEK_MODELS: tuple[DeepSeekModelSpec, ...] = (
    DeepSeekModelSpec(
        id="deepseek-v4-flash",
        label="DeepSeek V4 Flash — cost-efficient, 1M ctx",
        reasoning=True,
        price_input_per_1m=0.14,
        price_cached_input_per_1m=0.0028,
        price_output_per_1m=0.28,
    ),
    DeepSeekModelSpec(
        id="deepseek-v4-pro",
        label="DeepSeek V4 Pro — frontier reasoning, 1M ctx",
        reasoning=True,
        price_input_per_1m=1.74,
        price_cached_input_per_1m=0.0145,
        price_output_per_1m=3.48,
        enable_thinking=True,
    ),
    DeepSeekModelSpec(
        id="deepseek-chat",
        label="DeepSeek Chat (legacy alias for V4 Flash non-thinking)",
        reasoning=False,
        price_input_per_1m=0.14,
        price_cached_input_per_1m=0.0028,
        price_output_per_1m=0.28,
    ),
    DeepSeekModelSpec(
        id="deepseek-reasoner",
        label="DeepSeek Reasoner (legacy alias for V4 Flash thinking)",
        reasoning=True,
        price_input_per_1m=0.14,
        price_cached_input_per_1m=0.0028,
        price_output_per_1m=0.28,
        enable_thinking=True,
    ),
)

DEEPSEEK_MODEL_IDS: tuple[str, ...] = tuple(spec.id for spec in DEEPSEEK_MODELS)
DEEPSEEK_MODEL_LABELS: dict[str, str] = {
    spec.id: spec.label for spec in DEEPSEEK_MODELS
}
_DEEPSEEK_MODEL_INDEX: dict[str, DeepSeekModelSpec] = {
    spec.id: spec for spec in DEEPSEEK_MODELS
}


def is_deepseek_model(model_id: str | None) -> bool:
    """True when ``model_id`` should be routed to the DeepSeek API."""
    if not model_id:
        return False
    return model_id.startswith(DEEPSEEK_MODEL_PREFIX)


def get_deepseek_model_spec(model_id: str) -> DeepSeekModelSpec | None:
    return _DEEPSEEK_MODEL_INDEX.get(model_id)


def deepseek_model_label(model_id: str) -> str:
    return DEEPSEEK_MODEL_LABELS.get(model_id, model_id)


def _resolve_ssl_verify() -> bool | str:
    """Mirror llm._resolve_ssl_verify so DeepSeek inherits proxy/CA config."""
    from settings import get_settings

    cfg = get_settings()
    if not cfg.ssl_verify:
        return False
    if cfg.ssl_ca_bundle:
        path = Path(cfg.ssl_ca_bundle).expanduser().resolve()
        if not path.is_file():
            raise click.ClickException(
                f"SSL CA bundle not found: {path}\n"
                "Set GOODBOY_SSL_CA_BUNDLE (or SSL_CERT_FILE) to your proxy/root CA .pem file."
            )
        return str(path)
    return True


def _build_http_client() -> httpx.Client:
    return httpx.Client(verify=_resolve_ssl_verify())


@lru_cache(maxsize=8)
def _deepseek_client(
    resolved_key: str, ssl_verify: bool, ssl_ca_bundle: str | None
) -> OpenAI:
    return OpenAI(
        api_key=resolved_key,
        base_url=DEEPSEEK_BASE_URL,
        http_client=_build_http_client(),
    )


def clear_deepseek_client_cache() -> None:
    """Drop cached DeepSeek clients (e.g. after key/SSL changes)."""
    _deepseek_client.cache_clear()


def get_deepseek_client(*, api_key: str | None = None) -> OpenAI:
    from settings import get_settings

    cfg = get_settings()
    resolved_key = api_key or cfg.deepseek_api_key
    if not resolved_key:
        raise click.ClickException(
            "DEEPSEEK_API_KEY is not set. Run: goodboy setup"
        )
    return _deepseek_client(resolved_key, cfg.ssl_verify, cfg.ssl_ca_bundle)


def _extract_token_usage(response: Any) -> Any:
    from llm import TokenUsage

    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0)
    if total == 0 and (prompt or completion):
        total = prompt + completion
    if not (prompt or completion or total):
        return None
    return TokenUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
    )


def _extract_chat_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise click.ClickException("DeepSeek returned no choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise click.ClickException("DeepSeek returned no message.")
    content = getattr(message, "content", None) or ""
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = getattr(part, "text", None) or (
                part.get("text") if isinstance(part, dict) else None
            )
            if text:
                parts.append(text)
        content = "".join(parts)
    if not content:
        raise click.ClickException("DeepSeek returned empty content.")
    return content


def _build_chat_messages(
    instructions: str | None, input_text: str
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": input_text})
    return messages


def _reasoning_kwargs(
    spec: DeepSeekModelSpec | None, reasoning_effort: str | None
) -> dict[str, Any]:
    """Translate harness reasoning effort to DeepSeek chat parameters."""
    kwargs: dict[str, Any] = {}
    if spec is None or not spec.reasoning:
        return kwargs

    effort = reasoning_effort
    if effort is None or effort == "" or effort == "none":
        if spec.enable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        return kwargs

    if effort in ("minimal", "low"):
        return kwargs

    deepseek_effort = "high" if effort == "xhigh" else effort
    if deepseek_effort in ("medium", "high"):
        kwargs["reasoning_effort"] = deepseek_effort
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    return kwargs


def complete_deepseek(
    prompt: str,
    *,
    model: str,
    instructions: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Non-streaming text completion via the DeepSeek Chat Completions API."""
    from llm import format_api_connection_error

    client = get_deepseek_client()
    spec = get_deepseek_model_spec(model)
    messages = _build_chat_messages(instructions, prompt)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    kwargs.update(_reasoning_kwargs(spec, reasoning_effort))

    try:
        response = client.chat.completions.create(**kwargs)
    except APIConnectionError as exc:
        raise click.ClickException(
            format_api_connection_error(exc, provider="DeepSeek")
        ) from exc
    return _extract_chat_text(response)


def _append_schema_to_instructions(
    instructions: str, json_schema: dict[str, Any]
) -> str:
    """Inline the JSON schema so DeepSeek can honour ``json_object`` mode."""
    schema_text = json.dumps(json_schema, indent=2)
    return (
        f"{instructions}\n\n"
        "## Output format\n"
        "Respond with a single JSON object that conforms to this JSON Schema. "
        "Do not include any text outside the JSON object.\n\n"
        f"```json\n{schema_text}\n```\n"
    )


def complete_structured_deepseek(
    *,
    model: str,
    input: str,
    instructions: str,
    json_schema: dict[str, Any],
    reasoning_effort: str | None = None,
    stream: bool = False,
    on_text_delta: Any | None = None,
    abort_check: Any | None = None,
) -> tuple[str, None, Any]:
    """Structured JSON completion via DeepSeek (returns ``(text, None, usage)``).

    DeepSeek only supports legacy ``response_format={"type": "json_object"}``
    JSON mode (not the OpenAI Responses API JSON-Schema format), so the harness
    schema is appended to the system prompt and the model is asked to honour
    it. The chained-response id is always ``None`` because DeepSeek does not
    expose a ``previous_response_id`` mechanism.
    """
    from llm import format_api_connection_error

    client = get_deepseek_client()
    spec = get_deepseek_model_spec(model)
    full_instructions = _append_schema_to_instructions(instructions, json_schema)
    messages = _build_chat_messages(full_instructions, input)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    kwargs.update(_reasoning_kwargs(spec, reasoning_effort))

    from llm import UserAbort

    use_stream = bool(stream and on_text_delta is not None)
    try:
        if use_stream:
            stream_kwargs = dict(kwargs)
            stream_kwargs["stream"] = True
            stream_kwargs["stream_options"] = {"include_usage": True}
            chunks: list[str] = []
            final_response: Any = None
            response = client.chat.completions.create(**stream_kwargs)
            for event in response:
                if abort_check is not None and abort_check():
                    try:
                        response.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    raise UserAbort()
                final_response = event
                choices = getattr(event, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if content:
                    chunks.append(content)
                    on_text_delta(content)
            text = "".join(chunks)
            usage = _extract_token_usage(final_response) if final_response is not None else None
            return text, None, usage

        if abort_check is not None and abort_check():
            raise UserAbort()
        kwargs["stream"] = False
        response = client.chat.completions.create(**kwargs)
    except UserAbort:
        raise
    except APIConnectionError as exc:
        raise click.ClickException(
            format_api_connection_error(exc, provider="DeepSeek")
        ) from exc

    return _extract_chat_text(response), None, _extract_token_usage(response)
