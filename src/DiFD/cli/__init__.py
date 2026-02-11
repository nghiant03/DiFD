"""DiFD CLI - Centralized command-line interface using Typer."""

import typer

from DiFD.cli.evaluate import app as evaluate_app
from DiFD.cli.inject import app as inject_app
from DiFD.cli.optimize import app as optimize_app
from DiFD.cli.train import app as train_app
from DiFD.logging import configure_logging

app = typer.Typer(
    name="difd",
    help="DiFD - Deep Learning Fault Diagnosis CLI",
    no_args_is_help=True,
)


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Configure global options."""
    level = "DEBUG" if debug else "INFO"
    configure_logging(level=level, verbose=verbose)


app.add_typer(inject_app, name="inject", help="Inject faults into sensor datasets")
app.add_typer(train_app, name="train", help="Train deep learning models")
app.add_typer(evaluate_app, name="evaluate", help="Evaluate trained models")
app.add_typer(optimize_app, name="optimize", help="Hyperparameter optimization with Optuna")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
