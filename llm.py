"""OpenAI Responses API client with selectable models and parameters."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import click
import httpx
import questionary
from openai import APIConnectionError, OpenAI

from settings import get_settings

# https://platform.openai.com/docs/models
class OpenAIModel(str, Enum):
    GPT_5_5 = "gpt-5.5"
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_4_NANO = "gpt-5.4-nano"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O3 = "o3"
    O4_MINI = "o4-mini"


# https://platform.openai.com/docs/guides/reasoning
REASONING_EFFORT = ("none", "minimal", "low", "medium", "high", "xhigh")

MODEL_CHOICES = [m.value for m in OpenAIModel]

MODEL_LABELS: dict[str, str] = {
    OpenAIModel.GPT_5_5.value: "GPT-5.5",
    OpenAIModel.GPT_5_4_MINI.value: "GPT-5.4 mini",
    OpenAIModel.GPT_5_4_NANO.value: "GPT-5.4 nano",
    OpenAIModel.GPT_4_1.value: "GPT-4.1",
    OpenAIModel.GPT_4_1_MINI.value: "GPT-4.1 mini",
    OpenAIModel.GPT_4_1_NANO.value: "GPT-4.1 nano",
    OpenAIModel.GPT_4O.value: "GPT-4o",
    OpenAIModel.GPT_4O_MINI.value: "GPT-4o mini (default)",
    OpenAIModel.O3.value: "o3 (reasoning)",
    OpenAIModel.O4_MINI.value: "o4-mini (reasoning)",
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
    """Alias for get_curated_models (setup menu and agent allowlist)."""
    return get_curated_models(api_key=api_key)


def _model_choice(model_id: str) -> questionary.Choice:
    label = MODEL_LABELS.get(model_id, model_id)
    return questionary.Choice(title=label, value=model_id)


def select_model_interactive(
    models: list[str] | None = None,
    *,
    default: str | None = None,
    api_key: str | None = None,
) -> str:
    """Show an arrow-key menu and return the chosen model ID."""
    options = models if models is not None else get_available_models(api_key=api_key)
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
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    try:
        response = client.responses.create(**kwargs)
    except APIConnectionError as exc:
        raise click.ClickException(format_api_connection_error(exc)) from exc
    return _extract_response_text(response)


def complete_structured(
    *,
    input: str,
    instructions: str,
    json_schema: dict[str, Any],
    model: str | None = None,
    reasoning_effort: str | None = None,
    tools: list[str] | None = None,
) -> str:
    """Call Responses API with JSON schema output; fall back to plain completion."""
    from agent.prompt import system_prompt_with_schema

    client = get_client()
    resolved_model = model or get_settings().default_model

    text_config: dict[str, Any] = {
        "format": {
            "type": "json_schema",
            "name": "agent_step",
            "schema": json_schema,
            "strict": False,
        }
    }
    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "input": input,
        "instructions": instructions,
        "text": text_config,
    }
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if tools:
        kwargs["tools"] = [{"type": tool} for tool in tools]

    try:
        response = client.responses.create(**kwargs)
        return _extract_response_text(response)
    except APIConnectionError as exc:
        raise click.ClickException(format_api_connection_error(exc)) from exc
    except Exception:
        fallback_instructions = system_prompt_with_schema()
        return complete(
            input,
            model=resolved_model,
            instructions=fallback_instructions,
            reasoning_effort=reasoning_effort,
        )


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
