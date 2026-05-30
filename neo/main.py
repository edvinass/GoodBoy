from __future__ import annotations

import click
import questionary

import settings
from agent.claude_llm import (
    CLAUDE_MODEL_IDS,
    is_claude_model,
)
from agent.deepseek_llm import (
    DEEPSEEK_MODEL_IDS,
    is_deepseek_model,
)
from agent.harness import run_harness
from agent.local_llm import (
    default_catalog_spec,
    download_model,
    ensure_local_deps,
    is_local_model,
    iter_catalog,
    list_installed_models,
    resolve_model_path,
)
from llm import get_selectable_models, select_model_interactive
from settings import (
    ANTHROPIC_API_KEY_VAR,
    DEEPSEEK_API_KEY_VAR,
    OPENAI_API_KEY_VAR,
    OPENAI_MODEL_VAR,
    get_settings,
    is_configured,
    save_env,
)


def load_env() -> None:
    settings.load_env()


def _mask_api_key(key: str) -> str:
    if len(key) <= 11:
        return "••••••••"
    return f"{key[:7]}…{key[-4:]}"


def _prompt_api_key(label: str = "OpenAI API key") -> str:
    key = questionary.password(
        label,
        validate=lambda text: bool(text.strip()) or "API key is required",
        instruction="(input hidden)",
    ).ask()
    if key is None:
        raise click.ClickException("Setup cancelled.")
    return key.strip()


def _prompt_setup_mode() -> str:
    choice = questionary.select(
        "How do you want to use Neo?",
        choices=[
            questionary.Choice("Cloud (OpenAI API)", value="cloud"),
            questionary.Choice("DeepSeek (api.deepseek.com)", value="deepseek"),
            questionary.Choice("Anthropic (api.anthropic.com)", value="anthropic"),
            questionary.Choice("Local (on this machine, no API key)", value="local"),
            questionary.Choice(
                "Multiple providers (OpenAI / DeepSeek / Anthropic / local)",
                value="multi",
            ),
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


def _prompt_provider_key(
    label: str, existing: str | None, *, required: bool
) -> str | None:
    """Reuse or replace a provider key; return None when not used."""
    if existing:
        change = questionary.confirm(
            f"{label} already saved ({_mask_api_key(existing)}). Change it?",
            default=False,
        ).ask()
        if change is None:
            raise click.ClickException("Setup cancelled.")
        if change:
            return _prompt_api_key(label)
        return existing
    if required:
        return _prompt_api_key(label)
    add = questionary.confirm(f"Configure {label} now?", default=False).ask()
    if add is None:
        raise click.ClickException("Setup cancelled.")
    return _prompt_api_key(label) if add else None


def run_setup() -> None:
    """Prompt for cloud/local configuration, download local weights, pick default model."""
    load_env()
    cfg = get_settings()

    mode = _prompt_setup_mode()
    openai_key = cfg.openai_api_key
    deepseek_key = cfg.deepseek_api_key
    anthropic_key = cfg.anthropic_api_key

    if mode == "cloud":
        openai_key = _prompt_provider_key(
            "OpenAI API key", openai_key, required=True
        )
    elif mode == "deepseek":
        deepseek_key = _prompt_provider_key(
            "DeepSeek API key", deepseek_key, required=True
        )
    elif mode == "anthropic":
        anthropic_key = _prompt_provider_key(
            "Anthropic API key", anthropic_key, required=True
        )
    elif mode == "multi":
        openai_key = _prompt_provider_key(
            "OpenAI API key", openai_key, required=False
        )
        deepseek_key = _prompt_provider_key(
            "DeepSeek API key", deepseek_key, required=False
        )
        anthropic_key = _prompt_provider_key(
            "Anthropic API key", anthropic_key, required=False
        )
    else:  # local-only
        if openai_key:
            keep = questionary.confirm(
                f"Keep saved OpenAI key ({_mask_api_key(openai_key)}) for optional cloud use?",
                default=True,
            ).ask()
            if keep is None:
                raise click.ClickException("Setup cancelled.")
            if not keep:
                openai_key = None
        if deepseek_key:
            keep = questionary.confirm(
                f"Keep saved DeepSeek key ({_mask_api_key(deepseek_key)}) for optional cloud use?",
                default=True,
            ).ask()
            if keep is None:
                raise click.ClickException("Setup cancelled.")
            if not keep:
                deepseek_key = None
        if anthropic_key:
            keep = questionary.confirm(
                f"Keep saved Anthropic key ({_mask_api_key(anthropic_key)}) for optional cloud use?",
                default=True,
            ).ask()
            if keep is None:
                raise click.ClickException("Setup cancelled.")
            if not keep:
                anthropic_key = None

    if mode in ("local", "multi"):
        offer_local = (
            mode == "local"
            or questionary.confirm(
                "Download a local GGUF model now?",
                default=False,
            ).ask()
        )
        if offer_local is None:
            raise click.ClickException("Setup cancelled.")
        if offer_local:
            ensure_local_deps()
            local_id = _select_local_catalog_model()
            _download_local_model(local_id)

    # Persist credentials before listing selectable models so the OpenAI,
    # DeepSeek, and Anthropic clients can fetch their catalogs with the new
    # keys.
    pending_updates: dict[str, str] = {}
    if openai_key:
        pending_updates[OPENAI_API_KEY_VAR] = openai_key
    elif mode in ("local", "deepseek", "anthropic") and cfg.openai_api_key:
        pending_updates[OPENAI_API_KEY_VAR] = ""
    if deepseek_key:
        pending_updates[DEEPSEEK_API_KEY_VAR] = deepseek_key
    elif mode in ("local", "cloud", "anthropic") and cfg.deepseek_api_key:
        pending_updates[DEEPSEEK_API_KEY_VAR] = ""
    if anthropic_key:
        pending_updates[ANTHROPIC_API_KEY_VAR] = anthropic_key
    elif mode in ("local", "cloud", "deepseek") and cfg.anthropic_api_key:
        pending_updates[ANTHROPIC_API_KEY_VAR] = ""
    if pending_updates:
        save_env(pending_updates)

    selectable = get_selectable_models(api_key=openai_key)
    if not selectable:
        raise click.ClickException(
            "No models available. Configure OpenAI, DeepSeek, Anthropic, or install a local model."
        )

    default_for_picker = cfg.default_model
    if mode == "local" and list_installed_models():
        default_for_picker = list_installed_models()[0]
    elif mode == "deepseek":
        default_for_picker = next(iter(DEEPSEEK_MODEL_IDS))
    elif mode == "anthropic":
        default_for_picker = next(iter(CLAUDE_MODEL_IDS))
    elif (
        mode == "multi"
        and list_installed_models()
        and not openai_key
        and not deepseek_key
        and not anthropic_key
    ):
        default_for_picker = list_installed_models()[0]

    model = select_model_interactive(
        models=selectable,
        default=default_for_picker,
        api_key=openai_key,
    )

    save_env({OPENAI_MODEL_VAR: model})

    click.echo()
    click.echo(click.style("Setup complete.", fg="green", bold=True))
    _brand = 46
    click.echo(f"  Model: {click.style(model, fg=_brand)}")
    if openai_key:
        click.echo(
            f"  OpenAI key: {click.style(_mask_api_key(openai_key), fg=_brand)}"
        )
    if deepseek_key:
        click.echo(
            f"  DeepSeek key: {click.style(_mask_api_key(deepseek_key), fg=_brand)}"
        )
    if anthropic_key:
        click.echo(
            f"  Anthropic key: {click.style(_mask_api_key(anthropic_key), fg=_brand)}"
        )
    if (
        not openai_key
        and not deepseek_key
        and not anthropic_key
        and is_local_model(model)
    ):
        click.echo("  API key: (not set — using local model)")
    if list_installed_models():
        click.echo(
            f"  Local models: {', '.join(list_installed_models())}"
        )
    click.echo(f"  Saved to {settings.ENV_FILE}")


def ensure_configured() -> None:
    """Run interactive setup when API key and default model are not configured."""
    load_env()
    if is_configured(get_settings()):
        return
    run_setup()
    if not is_configured(get_settings()):
        raise SystemExit(1)


@click.group(invoke_without_command=True)
@click.option(
    "-f",
    "--show-thoughts",
    "show_thoughts",
    flag_value=True,
    default=True,
    help="Show model thoughts on each step (enabled by default).",
)
@click.option(
    "--hide-thoughts",
    "show_thoughts",
    flag_value=False,
    help="Hide model thoughts on each step.",
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
    help="Print the tool commands the agent runs (not their output).",
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
    """Neo CLI."""
    if ctx.invoked_subcommand is None:
        ensure_configured()
        cfg = get_settings()
        run_harness(
            show_thoughts=show_thoughts,
            verbose=verbose,
            show_model=show_model,
            show_commands=show_commands or cfg.show_commands,
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
        click.echo("Not configured yet. Run: neo")
        raise SystemExit(1)
    _brand = 46
    model = cfg.default_model
    click.echo(f"Model: {click.style(model, fg=_brand, bold=True)}")
    if is_local_model(model):
        path = resolve_model_path(model)
        click.echo(f"Provider: {click.style('local (in-process)', fg=_brand)}")
        if path:
            click.echo(f"Weights: {path}")
    elif is_deepseek_model(model):
        click.echo(f"Provider: {click.style('DeepSeek', fg=_brand)}")
    elif is_claude_model(model):
        click.echo(f"Provider: {click.style('Anthropic', fg=_brand)}")
    else:
        click.echo(f"Provider: {click.style('OpenAI', fg=_brand)}")
    installed = list_installed_models()
    if installed:
        click.echo(f"Installed local: {', '.join(installed)}")
    if cfg.openai_api_key:
        click.echo(
            f"OpenAI key: {click.style(_mask_api_key(cfg.openai_api_key), fg=_brand)}"
        )
    else:
        click.echo("OpenAI key: (not set)")
    if cfg.deepseek_api_key:
        click.echo(
            f"DeepSeek key: {click.style(_mask_api_key(cfg.deepseek_api_key), fg=_brand)}"
        )
    else:
        click.echo("DeepSeek key: (not set)")
    if cfg.anthropic_api_key:
        click.echo(
            f"Anthropic key: {click.style(_mask_api_key(cfg.anthropic_api_key), fg=_brand)}"
        )
    else:
        click.echo("Anthropic key: (not set)")
    click.echo(f"Models dir: {cfg.models_dir}")
    click.echo("Run neo setup to change settings.")


@cli.command()
def setup() -> None:
    """Save API key, download local models, and choose default model."""
    run_setup()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
