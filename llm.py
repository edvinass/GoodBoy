"""OpenAI Responses API client with selectable models and parameters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable
from typing import Any

StreamTextCallback = Callable[[str], None]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

import click
import httpx
import questionary
from openai import APIConnectionError, BadRequestError, OpenAI

from settings import get_settings

# https://platform.openai.com/docs/models
class OpenAIModel(str, Enum):
    GPT_5_5 = "gpt-5.5"
    GPT_5_4 = "gpt-5.4"
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_4_NANO = "gpt-5.4-nano"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5 = "gpt-5"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"


# https://platform.openai.com/docs/guides/reasoning
REASONING_EFFORT = ("none", "minimal", "low", "medium", "high", "xhigh")

MODEL_CHOICES = [m.value for m in OpenAIModel]

MODEL_LABELS: dict[str, str] = {
    OpenAIModel.GPT_5_5.value: "GPT-5.5 — advanced coding and professional work",
    OpenAIModel.GPT_5_4.value: "GPT-5.4 — affordable coding and professional work",
    OpenAIModel.GPT_5_4_MINI.value: "GPT-5.4 mini — coding, computer use, subagents",
    OpenAIModel.GPT_5_4_NANO.value: "GPT-5.4 nano — cheapest GPT-5.4-class",
    OpenAIModel.GPT_5_MINI.value: "GPT-5 mini — near-frontier, cost-sensitive volume",
    OpenAIModel.GPT_5_NANO.value: "GPT-5 nano — fastest, most cost-efficient GPT-5",
    OpenAIModel.GPT_5.value: "GPT-5 — previous reasoning model (configurable effort)",
    OpenAIModel.GPT_4_1.value: "GPT-4.1 — smartest non-reasoning model",
    OpenAIModel.GPT_4_1_MINI.value: "GPT-4.1 mini",
    OpenAIModel.GPT_4_1_NANO.value: "GPT-4.1 nano",
}

_CHAT_MODEL_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")
_CHAT_MODEL_EXCLUDE = (
    "embed",
    "embedding",
    "audio",
    "realtime",
    "tts",
    "transcribe",
    "search",
    "image",
    "instruct",
    "dall-e",
    "whisper",
    "moderation",
)


def _is_selectable_chat_model(model_id: str) -> bool:
    lid = model_id.lower()
    if any(token in lid for token in _CHAT_MODEL_EXCLUDE):
        return False
    return any(lid.startswith(prefix) for prefix in _CHAT_MODEL_PREFIXES)


def get_curated_models(*, api_key: str | None = None) -> list[str]:
    """Return main catalog models available on the account (no dated snapshots or extras)."""
    try:
        client = get_client(api_key=api_key)
        api_ids = {
            m.id for m in client.models.list() if _is_selectable_chat_model(m.id)
        }
        curated = [m.value for m in OpenAIModel if m.value in api_ids]
        if curated:
            return curated
    except click.ClickException:
        return MODEL_CHOICES.copy()
    except Exception as exc:
        click.echo(
            click.style(
                f"Could not fetch models from API ({exc}); using built-in list.",
                fg="yellow",
            ),
            err=True,
        )
    return MODEL_CHOICES.copy()


def get_available_models(*, api_key: str | None = None) -> list[str]:
    """Alias for get_selectable_models (setup menu and agent allowlist)."""
    return get_selectable_models(api_key=api_key)


def is_local_model(model_id: str | None) -> bool:
    from agent.local_llm import is_local_model as _is_local

    return _is_local(model_id)


def get_selectable_models(*, api_key: str | None = None) -> list[str]:
    """Return OpenAI cloud models (when configured) plus installed local models."""
    from agent.local_llm import LOCAL_MODEL_LABELS, list_installed_models

    local = list_installed_models()
    cloud: list[str] = []
    if api_key:
        try:
            cloud = get_curated_models(api_key=api_key)
        except click.ClickException:
            cloud = MODEL_CHOICES.copy()
    elif get_settings().openai_api_key:
        cloud = get_curated_models()
    merged: list[str] = []
    seen: set[str] = set()
    for model_id in local + cloud:
        if model_id not in seen:
            seen.add(model_id)
            merged.append(model_id)
    if merged:
        return merged
    if api_key or get_settings().openai_api_key:
        return get_curated_models(api_key=api_key)
    return MODEL_CHOICES.copy()


def local_model_label(model_id: str) -> str:
    from agent.local_llm import LOCAL_MODEL_LABELS

    return LOCAL_MODEL_LABELS.get(model_id, model_id)


def _model_choice(model_id: str) -> questionary.Choice:
    from agent.models import format_model_select_label

    if is_local_model(model_id):
        description = local_model_label(model_id)
        label = f"[Local] {description}"
    else:
        description = MODEL_LABELS.get(model_id, model_id)
        label = format_model_select_label(model_id, description)
    return questionary.Choice(title=label, value=model_id)


def select_model_interactive(
    models: list[str] | None = None,
    *,
    default: str | None = None,
    api_key: str | None = None,
) -> str:
    """Show an arrow-key menu and return the chosen model ID."""
    options = models if models is not None else get_selectable_models(api_key=api_key)
    if not options:
        raise click.ClickException("No models available to select.")

    resolved_default = default if default is not None else get_settings().default_model
    default_value = (
        resolved_default if resolved_default in options else options[0]
    )

    choice = questionary.select(
        "Select a model",
        choices=[_model_choice(m) for m in options],
        default=default_value,
        use_indicator=True,
        use_arrow_keys=True,
        instruction="(↑↓ to move, Enter to confirm)",
    ).ask()

    if choice is None:
        raise click.ClickException("Model selection cancelled.")
    return choice


_REASONING_LABELS: dict[str, str] = {
    "none": "None — no reasoning tokens",
    "minimal": "Minimal — very light planning",
    "low": "Low — routine coding and tool use",
    "medium": "Medium — non-trivial agent steps",
    "high": "High — complex debugging",
    "xhigh": "Extra high — hardest problems",
}


def _reasoning_choice(level: str) -> questionary.Choice:
    label = _REASONING_LABELS.get(level, level)
    return questionary.Choice(title=label, value=level)


def select_reasoning_interactive(*, default: str | None = None) -> str | None:
    """Show an arrow-key menu and return the chosen reasoning effort, or None to clear."""
    resolved_default = (
        default if default is not None else get_settings().default_reasoning_effort
    )
    choices: list[questionary.Choice] = [
        questionary.Choice(title="Not set (model API default)", value=""),
    ]
    choices.extend(_reasoning_choice(level) for level in REASONING_EFFORT)
    default_value = resolved_default if resolved_default in REASONING_EFFORT else ""

    choice = questionary.select(
        "Select default reasoning effort",
        choices=choices,
        default=default_value,
        use_indicator=True,
        use_arrow_keys=True,
        instruction="(↑↓ to move, Enter to confirm)",
    ).ask()

    if choice is None:
        raise click.ClickException("Reasoning effort selection cancelled.")
    return choice or None


def _resolve_ssl_verify() -> bool | str:
    """Return httpx verify: True (default CAs), path to CA bundle, or False."""
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


def clear_openai_client_cache() -> None:
    """Drop cached OpenAI clients (e.g. after .env SSL settings change)."""
    _openai_client.cache_clear()


@lru_cache(maxsize=8)
def _openai_client(
    resolved_key: str, ssl_verify: bool, ssl_ca_bundle: str | None
) -> OpenAI:
    return OpenAI(api_key=resolved_key, http_client=_build_http_client())


def format_api_connection_error(exc: BaseException) -> str:
    """User-facing hint when TLS/proxy blocks the OpenAI API."""
    parts: list[str] = []
    cause: BaseException | None = exc
    while cause is not None:
        parts.append(str(cause))
        cause = cause.__cause__  # type: ignore[assignment]
    text = " ".join(parts).lower()
    if "certificate_verify_failed" in text or "self-signed certificate" in text:
        return (
            "Could not reach OpenAI: TLS certificate verification failed "
            "(common behind corporate HTTPS proxies).\n\n"
            "Fix: save your organization's CA certificate to a .pem file, then add to .env:\n"
            "  GOODBOY_SSL_CA_BUNDLE=/path/to/corporate-ca.pem\n\n"
            "Or export before running goodboy:\n"
            "  export SSL_CERT_FILE=/path/to/corporate-ca.pem\n\n"
            "Last resort only (insecure): GOODBOY_SSL_VERIFY=false"
        )
    return f"Could not reach OpenAI: {exc}"


def get_client(*, api_key: str | None = None) -> OpenAI:
    cfg = get_settings()
    resolved_key = api_key or cfg.openai_api_key
    if not resolved_key:
        raise click.ClickException(
            "OPENAI_API_KEY is not set. Run: goodboy setup"
        )
    return _openai_client(resolved_key, cfg.ssl_verify, cfg.ssl_ca_bundle)


def _extract_token_usage(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0)
    if total == 0 and (input_tokens or output_tokens):
        total = input_tokens + output_tokens
    if not (input_tokens or output_tokens or total):
        return None
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
    )


def _extract_response_text(response: Any) -> str:
    if getattr(response, "output_text", None):
        return response.output_text

    output = getattr(response, "output", None) or []
    for item in output:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                return text

    raise click.ClickException("Model returned no text output.")


def complete(
    prompt: str,
    *,
    model: str | None = None,
    instructions: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Call OpenAI Responses API and return assistant text."""
    client = get_client()
    resolved_model = model or get_settings().default_model

    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "input": prompt,
    }
    if instructions:
        kwargs["instructions"] = instructions
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if reasoning_effort is not None and reasoning_effort != "none":
        kwargs["reasoning"] = {"effort": reasoning_effort}

    try:
        response = client.responses.create(**kwargs)
    except APIConnectionError as exc:
        raise click.ClickException(format_api_connection_error(exc)) from exc
    return _extract_response_text(response)


def _structured_request_kwargs(
    *,
    input: str,
    instructions: str,
    json_schema: dict[str, Any],
    model: str,
    reasoning_effort: str | None,
    tools: list[str] | None,
    strict_json: bool = True,
) -> dict[str, Any]:
    text_config: dict[str, Any] = {
        "format": {
            "type": "json_schema",
            "name": "agent_step",
            "schema": json_schema,
            "strict": strict_json,
        }
    }
    kwargs: dict[str, Any] = {
        "model": model,
        "input": input,
        "instructions": instructions,
        "text": text_config,
    }
    if reasoning_effort is not None and reasoning_effort != "none":
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if tools:
        kwargs["tools"] = [{"type": tool} for tool in tools]
    return kwargs


def _complete_structured_stream(
    client: OpenAI,
    kwargs: dict[str, Any],
    *,
    on_text_delta: StreamTextCallback,
) -> tuple[str, str | None, TokenUsage | None]:
    with client.responses.stream(**kwargs) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                on_text_delta(event.delta)
        final = stream.get_final_response()
        return (
            _extract_response_text(final),
            getattr(final, "id", None),
            _extract_token_usage(final),
        )


def complete_structured(
    *,
    input: str,
    instructions: str,
    json_schema: dict[str, Any],
    model: str | None = None,
    reasoning_effort: str | None = None,
    tools: list[str] | None = None,
    stream: bool = False,
    on_text_delta: StreamTextCallback | None = None,
    previous_response_id: str | None = None,
) -> str:
    """Call Responses API with JSON schema output; return assistant text only."""
    text, _, _ = complete_structured_with_id(
        input=input,
        instructions=instructions,
        json_schema=json_schema,
        model=model,
        reasoning_effort=reasoning_effort,
        tools=tools,
        stream=stream,
        on_text_delta=on_text_delta,
        previous_response_id=previous_response_id,
    )
    return text


def complete_structured_with_id(
    *,
    input: str,
    instructions: str,
    json_schema: dict[str, Any],
    model: str | None = None,
    reasoning_effort: str | None = None,
    tools: list[str] | None = None,
    stream: bool = False,
    on_text_delta: StreamTextCallback | None = None,
    previous_response_id: str | None = None,
) -> tuple[str, str | None, TokenUsage | None]:
    """Like complete_structured but also returns the OpenAI response id and usage.

    The id can be passed to a subsequent call as ``previous_response_id`` to
    chain a stateful conversation server-side, allowing the caller to send only
    the new turn's data instead of the full transcript each time.

    Local models (``local:`` prefix) run in-process via llama-cpp-python and
    always return ``(text, None, usage)``.
    """
    from agent.prompt import system_prompt_with_schema

    resolved_model = model or get_settings().default_model
    if is_local_model(resolved_model):
        from agent.local_llm import complete_structured_local

        return complete_structured_local(
            model=resolved_model,
            input=input,
            instructions=instructions,
            json_schema=json_schema,
            stream=stream,
            on_text_delta=on_text_delta,
        )

    client = get_client()
    cfg = get_settings()
    kwargs = _structured_request_kwargs(
        input=input,
        instructions=instructions,
        json_schema=json_schema,
        model=resolved_model,
        reasoning_effort=reasoning_effort,
        tools=tools,
        strict_json=cfg.strict_json_schema,
    )
    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id
    use_stream = stream and on_text_delta is not None

    def _call(strict: bool) -> tuple[str, str | None, TokenUsage | None]:
        call_kwargs = dict(kwargs)
        call_kwargs["text"] = dict(kwargs["text"])
        call_kwargs["text"]["format"] = dict(kwargs["text"]["format"])
        call_kwargs["text"]["format"]["strict"] = strict
        if use_stream:
            return _complete_structured_stream(
                client, call_kwargs, on_text_delta=on_text_delta
            )
        response = client.responses.create(**call_kwargs)
        return (
            _extract_response_text(response),
            getattr(response, "id", None),
            _extract_token_usage(response),
        )

    try:
        try:
            return _call(cfg.strict_json_schema)
        except Exception:
            if not cfg.strict_json_schema:
                raise
            return _call(False)
    except APIConnectionError as exc:
        raise click.ClickException(format_api_connection_error(exc)) from exc
    except BadRequestError as exc:
        if previous_response_id is not None and _looks_like_chain_broken(exc):
            raise ResponseChainBroken(str(exc)) from exc
        raise
    except Exception:
        fallback_instructions = system_prompt_with_schema()
        return (
            complete(
                input,
                model=resolved_model,
                instructions=fallback_instructions,
                reasoning_effort=reasoning_effort,
            ),
            None,
            None,
        )


class ResponseChainBroken(Exception):
    """Raised when the OpenAI server rejected previous_response_id (expired/missing)."""


_CHAIN_BROKEN_MARKERS = (
    "previous_response_id",
    "previous response",
    "response not found",
    "no such response",
)


def _looks_like_chain_broken(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CHAIN_BROKEN_MARKERS)


@click.command()
@click.argument("prompt")
@click.option(
    "--model",
    "-m",
    type=click.Choice(MODEL_CHOICES, case_sensitive=False),
    default=None,
    show_default="from .env (OPENAI_MODEL)",
    help="OpenAI model ID; defaults to saved model from goodboy setup.",
)
@click.option(
    "--instructions",
    "-i",
    default=None,
    help="System-level instructions for the model.",
)
@click.option(
    "--temperature",
    "-t",
    type=click.FloatRange(0, 2),
    default=None,
    help="Sampling temperature (0–2).",
)
@click.option(
    "--top-p",
    type=click.FloatRange(0, 1),
    default=None,
    help="Nucleus sampling; alternative to temperature.",
)
@click.option(
    "--max-output-tokens",
    type=click.IntRange(1),
    default=None,
    help="Upper bound on generated tokens (output + reasoning).",
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(REASONING_EFFORT, case_sensitive=False),
    default=None,
    help="Reasoning effort for reasoning models.",
)
def main(
    prompt: str,
    model: str,
    instructions: str | None,
    temperature: float | None,
    top_p: float | None,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
) -> None:
    """Send PROMPT to OpenAI and print the model response."""
    text = complete(
        prompt,
        model=model,
        instructions=instructions,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    click.echo(text)


if __name__ == "__main__":
    main()
