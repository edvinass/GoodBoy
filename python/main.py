from __future__ import annotations

import click
import questionary

import settings
from agent.harness import run_harness
from agent.local_llm import (
    default_catalog_spec,
    download_model,
    is_local_model,
    iter_catalog,
    list_installed_models,
    resolve_model_path,
)
from llm import get_selectable_models, select_model_interactive
from settings import OPENAI_API_KEY_VAR, OPENAI_MODEL_VAR, get_settings, is_configured, save_env


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


def _prompt_setup_mode() -> str:
    choice = questionary.select(
        "How do you want to use GoodBoy?",
        choices=[
            questionary.Choice("Cloud (OpenAI API)", value="cloud"),
            questionary.Choice("Local (on this machine, no API key)", value="local"),
            questionary.Choice("Both (OpenAI + local models)", value="both"),
        ],
        use_indicator=True,
        use_arrow_keys=True,
    ).ask()
    if choice is None:
        raise click.ClickException("Setup cancelled.")
    return choice


def _select_local_catalog_model() -> str:
    default_spec = default_catalog_spec()
    choices = [
        questionary.Choice(title=spec.label, value=spec.id)
        for spec in iter_catalog()
    ]
    choice = questionary.select(
        "Select a local model to download",
        choices=choices,
        default=default_spec.id,
        use_indicator=True,
        use_arrow_keys=True,
    ).ask()
    if choice is None:
        raise click.ClickException("Setup cancelled.")
    return choice


def _download_local_model(model_id: str) -> None:
    existing = resolve_model_path(model_id)
    if existing is not None:
        click.echo(f"Already installed: {existing}")
        return
    download_model(model_id)


def run_setup() -> None:
    """Prompt for cloud/local configuration, download local weights, pick default model."""
    load_env()
    cfg = get_settings()

    mode = _prompt_setup_mode()
    api_key = cfg.openai_api_key

    if mode in ("cloud", "both"):
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
    else:
        if api_key:
            keep = questionary.confirm(
                f"Keep saved API key ({_mask_api_key(api_key)}) for optional cloud use?",
                default=True,
            ).ask()
            if keep is None:
                raise click.ClickException("Setup cancelled.")
            if not keep:
                api_key = None
        else:
            api_key = None

    if mode in ("local", "both"):
        local_id = _select_local_catalog_model()
        _download_local_model(local_id)

    selectable = get_selectable_models(api_key=api_key)
    if not selectable:
        raise click.ClickException(
            "No models available. Configure OpenAI or install a local model."
        )

    default_for_picker = cfg.default_model
    if mode == "local" and list_installed_models():
        default_for_picker = list_installed_models()[0]
    elif mode == "both" and list_installed_models() and not api_key:
        default_for_picker = list_installed_models()[0]

    model = select_model_interactive(
        models=selectable,
        default=default_for_picker,
        api_key=api_key,
    )

    updates: dict[str, str] = {OPENAI_MODEL_VAR: model}
    if api_key:
        updates[OPENAI_API_KEY_VAR] = api_key
    elif mode == "local":
        updates[OPENAI_API_KEY_VAR] = ""

    save_env(updates)

    click.echo()
    click.echo(click.style("Setup complete.", fg="green", bold=True))
    _brown = 94
    click.echo(f"  Model: {click.style(model, fg=_brown)}")
    if api_key:
        click.echo(f"  API key: {click.style(_mask_api_key(api_key), fg=_brown)}")
    elif is_local_model(model):
        click.echo("  API key: (not set — using local model)")
    if list_installed_models():
        click.echo(
            f"  Local models: {', '.join(list_installed_models())}"
        )
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
    """Show saved API key, model, and local model status."""
    load_env()
    cfg = get_settings()
    if not is_configured(cfg):
        click.echo("Not configured yet. Run: goodboy setup")
        raise SystemExit(1)
    _brown = 94
    model = cfg.default_model
    click.echo(f"Model: {click.style(model, fg=_brown, bold=True)}")
    if is_local_model(model):
        path = resolve_model_path(model)
        click.echo(f"Provider: {click.style('local (in-process)', fg=_brown)}")
        if path:
            click.echo(f"Weights: {path}")
    else:
        click.echo(f"Provider: {click.style('OpenAI', fg=_brown)}")
    installed = list_installed_models()
    if installed:
        click.echo(f"Installed local: {', '.join(installed)}")
    if cfg.openai_api_key:
        click.echo(f"API key: {click.style(_mask_api_key(cfg.openai_api_key), fg=_brown)}")
    else:
        click.echo("API key: (not set)")
    click.echo(f"Models dir: {cfg.models_dir}")
    click.echo("Run goodboy setup to change settings.")


@cli.command()
def setup() -> None:
    """Save API key, download local models, and choose default model."""
    run_setup()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
