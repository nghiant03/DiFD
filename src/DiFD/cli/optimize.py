"""CLI subcommand for hyperparameter optimization."""

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command("run")
def optimize_run(
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to injected dataset (.npz)"),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model architecture to optimize"),
    ] = "lstm",
    n_trials: Annotated[
        int,
        typer.Option("--n-trials", "-n", help="Number of Optuna trials"),
    ] = 100,
    timeout: Annotated[
        Optional[int],
        typer.Option("--timeout", "-t", help="Optimization timeout in seconds"),
    ] = None,
    study_name: Annotated[
        Optional[str],
        typer.Option("--study-name", help="Optuna study name"),
    ] = None,
    storage: Annotated[
        Optional[str],
        typer.Option("--storage", help="Optuna storage URL (e.g., sqlite:///optuna.db)"),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", "-s", help="Random seed"),
    ] = 42,
) -> None:
    """Run hyperparameter optimization with Optuna."""
    typer.echo(f"Loading data from: {data}")
    typer.echo(f"Optimizing {model} model")
    typer.echo(f"Running {n_trials} trials")
    if timeout:
        typer.echo(f"Timeout: {timeout}s")
    typer.echo("Optimization not yet implemented")


@app.command("show")
def optimize_show(
    study_name: Annotated[
        str,
        typer.Argument(help="Name of the study to show"),
    ],
    storage: Annotated[
        str,
        typer.Option("--storage", help="Optuna storage URL"),
    ] = "sqlite:///optuna.db",
) -> None:
    """Show results from an optimization study."""
    typer.echo(f"Study: {study_name}")
    typer.echo(f"Storage: {storage}")
    typer.echo("Study visualization not yet implemented")
