import click
import settings
from llm import select_model_interactive


def load_env() -> None:
    settings.load_env()


@click.command()
def main() -> None:
    load_env()
    model = select_model_interactive()
    click.echo(f"Selected model: {click.style(model, fg='green', bold=True)}")


if __name__ == "__main__":
    main()
