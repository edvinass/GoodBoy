"""OpenAI Responses API client with selectable models and parameters."""

from __future__ import annotations

from enum import Enum
from typing import Any

import click
import questionary
from openai import OpenAI

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


def get_available_models() -> list[str]:
    """Return chat models from the API, curated list first; fallback if unreachable."""
    try:
        client = get_client()
        api_ids = {
            m.id for m in client.models.list() if _is_selectable_chat_model(m.id)
        }
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

    curated = [m.value for m in OpenAIModel if m.value in api_ids]
    extras = sorted(api_ids - set(curated))
    if curated or extras:
        return curated + extras
    return MODEL_CHOICES.copy()


def _model_choice(model_id: str) -> questionary.Choice:
    label = MODEL_LABELS.get(model_id, model_id)
    return questionary.Choice(title=label, value=model_id)


def select_model_interactive(
    models: list[str] | None = None,
    *,
    default: str = OpenAIModel.GPT_4O_MINI.value,
) -> str:
    """Show an arrow-key menu and return the chosen model ID."""
    options = models if models is not None else get_available_models()
    if not options:
        raise click.ClickException("No models available to select.")

    default_value = default if default in options else options[0]

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


def get_client() -> OpenAI:
    api_key = get_settings().openai_api_key
    if not api_key:
        raise click.ClickException(
            "OPENAI_API_KEY is not set. Add it to .env in the project root."
        )
    return OpenAI(api_key=api_key)


def complete(
    prompt: str,
    *,
    model: str = OpenAIModel.GPT_4O_MINI.value,
    instructions: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Call OpenAI Responses API and return assistant text."""
    client = get_client()

    kwargs: dict[str, Any] = {
        "model": model,
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

    response = client.responses.create(**kwargs)

    if getattr(response, "output_text", None):
        return response.output_text

    for item in response.output:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                return text

    raise click.ClickException("Model returned no text output.")


@click.command()
@click.argument("prompt")
@click.option(
    "--model",
    "-m",
    type=click.Choice(MODEL_CHOICES, case_sensitive=False),
    default=OpenAIModel.GPT_4O_MINI.value,
    show_default=True,
    help="OpenAI model ID.",
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
