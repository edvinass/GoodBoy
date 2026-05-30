"""Anthropic Claude API integration (Messages API).

Uses the official ``anthropic`` SDK against ``https://api.anthropic.com``. The
Messages API has a different shape from OpenAI's Responses / Chat Completions:

* ``client.messages.create(model=..., max_tokens=..., messages=[...], system=...)``
* Multi-block ``content`` response (``content[0].text``).
* Structured JSON via ``output_config={"format": {"type": "json_schema", ...}}``
  (GA since Jan 2026).
* Extended reasoning via ``thinking={"type": "adaptive"}`` + an effort hint on
  ``output_config``. Opus 4.7+ only support adaptive thinking; older models
  also accept manual ``budget_tokens`` but that is deprecated.
* Streaming via ``client.messages.stream(...)`` with ``text_stream``.

See:
* https://platform.claude.com/docs/en/api/sdks/python
* https://platform.claude.com/docs/en/build-with-claude/structured-outputs
* https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
* https://platform.claude.com/docs/en/build-with-claude/streaming
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import click
import httpx

CLAUDE_BASE_URL = "https://api.anthropic.com"
CLAUDE_MODEL_PREFIX = "claude-"

# Generous output cap that fits comfortably inside every current Claude model's
# per-request output token limit. Callers can override with ``max_output_tokens``.
DEFAULT_CLAUDE_MAX_OUTPUT_TOKENS = 8192


@dataclass(frozen=True)
class ClaudeModelSpec:
    """Static metadata for a selectable Anthropic Claude model."""

    id: str
    label: str
    reasoning: bool
    price_input_per_1m: float
    price_cached_input_per_1m: float
    price_output_per_1m: float
    # Adaptive thinking is the only mode supported on Opus 4.7+; older models
    # still accept the legacy ``{"type": "enabled", "budget_tokens": N}`` form
    # but adaptive is the recommended path and what we use everywhere.
    supports_adaptive_thinking: bool = True
    # Max output tokens per single request (Anthropic enforces a per-model cap).
    max_output_tokens: int = 8192


# Pricing and capability matrix as of May 2026. Sources:
#   https://www.anthropic.com/pricing
#   https://dev.to/hiyoyok/claude-api-cheatsheet-2026-models-pricing-limits-in-one-place-4jf2
#   https://platform.claude.com/docs/en/about-claude/models
CLAUDE_MODELS: tuple[ClaudeModelSpec, ...] = (
    ClaudeModelSpec(
        id="claude-opus-4-8",
        label="Claude Opus 4.8 — flagship reasoning, 1M ctx",
        reasoning=True,
        price_input_per_1m=5.00,
        price_cached_input_per_1m=0.50,
        price_output_per_1m=25.00,
        supports_adaptive_thinking=True,
        max_output_tokens=32000,
    ),
    ClaudeModelSpec(
        id="claude-opus-4-7",
        label="Claude Opus 4.7 — previous flagship, 1M ctx",
        reasoning=True,
        price_input_per_1m=5.00,
        price_cached_input_per_1m=0.50,
        price_output_per_1m=25.00,
        supports_adaptive_thinking=True,
        max_output_tokens=32000,
    ),
    ClaudeModelSpec(
        id="claude-sonnet-4-6",
        label="Claude Sonnet 4.6 — balanced default, 1M ctx",
        reasoning=True,
        price_input_per_1m=3.00,
        price_cached_input_per_1m=0.30,
        price_output_per_1m=15.00,
        supports_adaptive_thinking=True,
        max_output_tokens=64000,
    ),
    ClaudeModelSpec(
        id="claude-haiku-4-5",
        label="Claude Haiku 4.5 — fastest/cheapest, 200K ctx",
        reasoning=False,
        price_input_per_1m=1.00,
        price_cached_input_per_1m=0.10,
        price_output_per_1m=5.00,
        supports_adaptive_thinking=False,
        max_output_tokens=8192,
    ),
)

CLAUDE_MODEL_IDS: tuple[str, ...] = tuple(spec.id for spec in CLAUDE_MODELS)
CLAUDE_MODEL_LABELS: dict[str, str] = {
    spec.id: spec.label for spec in CLAUDE_MODELS
}
_CLAUDE_MODEL_INDEX: dict[str, ClaudeModelSpec] = {
    spec.id: spec for spec in CLAUDE_MODELS
}


def is_claude_model(model_id: str | None) -> bool:
    """True when ``model_id`` should be routed to the Anthropic API."""
    if not model_id:
        return False
    return model_id.startswith(CLAUDE_MODEL_PREFIX)


def get_claude_model_spec(model_id: str) -> ClaudeModelSpec | None:
    return _CLAUDE_MODEL_INDEX.get(model_id)


def claude_model_label(model_id: str) -> str:
    return CLAUDE_MODEL_LABELS.get(model_id, model_id)


def _resolve_ssl_verify() -> bool | str:
    """Mirror llm._resolve_ssl_verify so Claude inherits proxy/CA config."""
    from settings import get_settings

    cfg = get_settings()
    if not cfg.ssl_verify:
        return False
    if cfg.ssl_ca_bundle:
        path = Path(cfg.ssl_ca_bundle).expanduser().resolve()
        if not path.is_file():
            raise click.ClickException(
                f"SSL CA bundle not found: {path}\n"
                "Set NEO_SSL_CA_BUNDLE (or SSL_CERT_FILE) to your proxy/root CA .pem file."
            )
        return str(path)
    return True


def _build_http_client() -> httpx.Client:
    return httpx.Client(verify=_resolve_ssl_verify())


@lru_cache(maxsize=8)
def _claude_client(
    resolved_key: str, ssl_verify: bool, ssl_ca_bundle: str | None
) -> Any:
    from anthropic import Anthropic

    return Anthropic(
        api_key=resolved_key,
        base_url=CLAUDE_BASE_URL,
        http_client=_build_http_client(),
    )


def clear_claude_client_cache() -> None:
    """Drop cached Claude clients (e.g. after key/SSL changes)."""
    _claude_client.cache_clear()


def get_claude_client(*, api_key: str | None = None) -> Any:
    from settings import get_settings

    cfg = get_settings()
    resolved_key = api_key or cfg.anthropic_api_key
    if not resolved_key:
        raise click.ClickException(
            "ANTHROPIC_API_KEY is not set. Run: neo setup"
        )
    return _claude_client(resolved_key, cfg.ssl_verify, cfg.ssl_ca_bundle)


def _extract_token_usage(response: Any) -> Any:
    from llm import TokenUsage

    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    # Anthropic returns input_tokens / output_tokens (no total_tokens field).
    # Cache read/write tokens are reported separately and are not added back
    # into the input total, so we surface the raw input count as-is.
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    if not (input_tokens or output_tokens):
        return None
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _extract_message_text(response: Any) -> str:
    """Concatenate all ``text`` blocks from a Claude Messages response.

    With extended thinking enabled, the content list may also contain
    ``thinking`` blocks. We skip those — callers want the user-visible answer,
    not the model's internal scratchpad. Structured-output responses still
    arrive in a single ``text`` block whose body is valid JSON.
    """
    content = getattr(response, "content", None) or []
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type != "text":
            continue
        text = getattr(block, "text", None) or (
            block.get("text") if isinstance(block, dict) else None
        )
        if text:
            parts.append(text)
    if not parts:
        raise click.ClickException("Claude returned no text content.")
    return "".join(parts)


def _thinking_kwargs(
    spec: ClaudeModelSpec | None, reasoning_effort: str | None
) -> dict[str, Any]:
    """Map harness reasoning effort to Claude adaptive-thinking + output_config.

    Returns kwargs that can be merged into the request body. We use adaptive
    thinking everywhere (the only supported mode on Opus 4.7+), and pass the
    effort through ``output_config.effort`` so Claude picks a budget.
    """
    kwargs: dict[str, Any] = {}
    if spec is None or not spec.reasoning:
        return kwargs

    effort = reasoning_effort
    if effort is None or effort == "" or effort == "none":
        return kwargs

    # Map our harness effort levels to Claude's adaptive effort scale. Claude
    # accepts: low / medium / high / xhigh / max.
    effort_map = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
    }
    claude_effort = effort_map.get(effort)
    if claude_effort is None:
        return kwargs

    if spec.supports_adaptive_thinking:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": claude_effort}
    return kwargs


def _resolve_max_tokens(
    spec: ClaudeModelSpec | None, requested: int | None
) -> int:
    """Pick max_tokens, clamped to the model's per-request cap."""
    cap = spec.max_output_tokens if spec is not None else DEFAULT_CLAUDE_MAX_OUTPUT_TOKENS
    if requested is None:
        return cap
    return max(1, min(requested, cap))


def complete_claude(
    prompt: str,
    *,
    model: str,
    instructions: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Non-streaming text completion via the Claude Messages API."""
    from anthropic import APIConnectionError

    from llm import format_api_connection_error

    client = get_claude_client()
    spec = get_claude_model_spec(model)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": _resolve_max_tokens(spec, max_output_tokens),
        "messages": [{"role": "user", "content": prompt}],
    }
    if instructions:
        kwargs["system"] = instructions
    # Opus 4.7+ rejects non-default temperature/top_p. We only forward them
    # when the caller asked for a specific value, and let Anthropic surface
    # a 400 if the model rejects it (rare in practice for our harness).
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p

    thinking_kwargs = _thinking_kwargs(spec, reasoning_effort)
    # Merge instead of overwrite — _thinking_kwargs may set output_config too.
    for key, value in thinking_kwargs.items():
        kwargs[key] = value

    try:
        response = client.messages.create(**kwargs)
    except APIConnectionError as exc:
        raise click.ClickException(
            format_api_connection_error(exc, provider="Anthropic")
        ) from exc
    return _extract_message_text(response)


def _normalize_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Tighten a JSON schema so Anthropic's structured-output validator accepts it.

    Anthropic's GA structured outputs require ``additionalProperties: false`` on
    every object and ``required`` to enumerate every declared property.
    OpenAI's harness schema usually already meets this in strict mode, so this
    is a defensive deep-copy that fills in the gaps.
    """
    if not isinstance(schema, dict):
        return schema

    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            cleaned[key] = _normalize_json_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _normalize_json_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value

    if cleaned.get("type") == "object" and isinstance(cleaned.get("properties"), dict):
        cleaned.setdefault("additionalProperties", False)
        cleaned.setdefault("required", list(cleaned["properties"].keys()))
    return cleaned


def _build_structured_output_config(
    spec: ClaudeModelSpec | None,
    json_schema: dict[str, Any],
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Combine effort hint (from thinking) with json_schema format."""
    base = _thinking_kwargs(spec, reasoning_effort)
    output_config: dict[str, Any] = dict(base.get("output_config", {}))
    output_config["format"] = {
        "type": "json_schema",
        "schema": _normalize_json_schema(json_schema),
    }
    return output_config


def complete_structured_claude(
    *,
    model: str,
    input: str,
    instructions: str,
    json_schema: dict[str, Any],
    reasoning_effort: str | None = None,
    stream: bool = False,
    on_text_delta: Any | None = None,
    abort_check: Any | None = None,
    max_output_tokens: int | None = None,
) -> tuple[str, None, Any]:
    """Structured JSON completion via Claude (returns ``(text, None, usage)``).

    Anthropic does not expose a ``previous_response_id`` mechanism comparable to
    OpenAI's Responses API, so the chained-response id is always ``None`` and
    the caller (the agent loop) replays the full transcript each turn.
    """
    from anthropic import APIConnectionError

    from llm import UserAbort, format_api_connection_error

    client = get_claude_client()
    spec = get_claude_model_spec(model)

    output_config = _build_structured_output_config(
        spec, json_schema, reasoning_effort
    )
    thinking_kwargs = _thinking_kwargs(spec, reasoning_effort)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": _resolve_max_tokens(spec, max_output_tokens),
        "system": instructions,
        "messages": [{"role": "user", "content": input}],
        "output_config": output_config,
    }
    if "thinking" in thinking_kwargs:
        kwargs["thinking"] = thinking_kwargs["thinking"]

    use_stream = bool(stream and on_text_delta is not None)
    try:
        if use_stream:
            chunks: list[str] = []
            final_message: Any = None
            with client.messages.stream(**kwargs) as message_stream:
                for text_chunk in message_stream.text_stream:
                    if abort_check is not None and abort_check():
                        try:
                            message_stream.close()
                        except Exception:
                            pass
                        raise UserAbort()
                    if text_chunk:
                        chunks.append(text_chunk)
                        on_text_delta(text_chunk)
                final_message = message_stream.get_final_message()
            text = "".join(chunks)
            if not text and final_message is not None:
                # Fallback: streaming yielded no deltas (rare), but the
                # accumulated message still has the JSON body.
                text = _extract_message_text(final_message)
            usage = (
                _extract_token_usage(final_message)
                if final_message is not None
                else None
            )
            return text, None, usage

        if abort_check is not None and abort_check():
            raise UserAbort()
        response = client.messages.create(**kwargs)
    except UserAbort:
        raise
    except APIConnectionError as exc:
        raise click.ClickException(
            format_api_connection_error(exc, provider="Anthropic")
        ) from exc

    return _extract_message_text(response), None, _extract_token_usage(response)
