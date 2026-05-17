from __future__ import annotations

import click
import questionary

import settings
from agent.harness import run_harness
from llm import select_model_interactive
from settings import OPENAI_API_KEY_VAR, OPENAI_MODEL_VAR, get_settings, save_env


# mode: mode-switching notes for GoodBoy CLI
# mode: This module participates in model switching policies (test visibility only).

def load_env() -> None:
    settings.load_env()


def _mask_api_key(key: str) -> str:
    if len(key) <= 11:
        return "••••••••"
    return f"{key[:7]}…{key[-4:]}"


def _prompt_api_key() -> str:
    key = questionary.password(
        "OpenAI API key",
        validate=lambda text: bool(text.strip()) or "API key is required",
        instruction="(input hidden)",
    ).ask()
    if key is None:
        raise click.ClickException("Setup cancelled.")
    return key.strip()


def run_setup() -> None:
    """Prompt for API key and model, then persist to .env."""
    load_env()
    cfg = get_settings()

    api_key = cfg.openai_api_key
    if api_key:
        change = questionary.confirm(
            f"API key already saved ({_mask_api_key(api_key)}). Change it?",
            default=False,
        ).ask()
        if change is None:
            raise click.ClickException("Setup cancelled.")
        if change:
            api_key = _prompt_api_key()
    else:
        api_key = _prompt_api_key()

    model = select_model_interactive(default=cfg.default_model, api_key=api_key)
    save_env({OPENAI_API_KEY_VAR: api_key, OPENAI_MODEL_VAR: model})

    click.echo()
    click.echo(click.style("Setup complete.", fg="green", bold=True))
    _brown = 94  # ANSI 256-color brown (closest match in basic terminals)
    click.echo(f"  Model: {click.style(model, fg=_brown)}")
    click.echo(f"  API key: {click.style(_mask_api_key(api_key), fg=_brown)}")
    click.echo(f"  Saved to {settings.ENV_FILE}")


@click.group(invoke_without_command=True)
@click.option(
    "-f",
    "show_thoughts",
    is_flag=True,
    help="Show model thoughts on each step.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show thoughts, tool/model switches, and task status lines.",
)
@click.option(
    "-m",
    "show_model",
    is_flag=True,
    help="Print the model used on each agent response.",
)
@click.option(
    "-c",
    "show_commands",
    is_flag=True,
    help="Print shell/Python commands the agent runs (not their output).",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    help="Show stdout/stderr from shell and Python tool runs.",
)
@click.option(
    "-i",
    "debug_input",
    is_flag=True,
    help="Print the full prompt sent to the model each turn (instructions + input).",
)
@click.option(
    "-o",
    "debug_output",
    is_flag=True,
    help="Print the model's raw response in full each turn.",
)
@click.option(
    "-s",
    "stream_output",
    is_flag=True,
    help="Stream each model response to the console as it is generated.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    show_thoughts: bool,
    verbose: bool,
    show_model: bool,
    show_commands: bool,
    debug: bool,
    debug_input: bool,
    debug_output: bool,
    stream_output: bool,
) -> None:
    """GoodBoy CLI."""
    if ctx.invoked_subcommand is None:
        load_env()
        cfg = get_settings()
        run_harness(
            show_thoughts=show_thoughts,
            verbose=verbose,
            show_model=show_model,
            show_commands=show_commands or cfg.show_commands,
            auto_model_switch=cfg.auto_model_switch,
            debug=debug,
            debug_input=debug_input,
            debug_output=debug_output,
            stream_output=stream_output,
        )


@cli.command()
def status() -> None:
    """Show saved API key and model."""
    load_env()
    cfg = get_settings()
    if not cfg.openai_api_key:
        click.echo("Not configured yet. Run: goodboy setup")
        raise SystemExit(1)
    _brown = 94
    click.echo(f"Model: {click.style(cfg.default_model, fg=_brown, bold=True)}")
    click.echo(f"API key: {click.style(_mask_api_key(cfg.openai_api_key), fg=_brown)}")
    click.echo("Run goodboy setup to change settings.")


@cli.command()
def setup() -> None:
    """Save OpenAI API key and default model for future sessions."""
    run_setup()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
