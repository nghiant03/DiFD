"""CLI subcommand for model training."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from DiFD.logging import logger
from DiFD.schema import TrainConfig

app = typer.Typer(no_args_is_help=True)

# Pre-instantiate defaults for help text
_defaults = TrainConfig()


@app.command("run")
def train_run(
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to injected dataset (.npz)"),
    ],
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help=f"Model architecture to use (default: {_defaults.model})"),
    ] = None,
    epochs: Annotated[
        Optional[int],
        typer.Option("--epochs", "-e", help=f"Number of training epochs (default: {_defaults.epochs})"),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", "-b", help=f"Training batch size (default: {_defaults.batch_size})"),
    ] = None,
    learning_rate: Annotated[
        Optional[float],
        typer.Option("--lr", help=f"Learning rate (default: {_defaults.learning_rate})"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path for trained model"),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", "-s", help=f"Random seed (default: {_defaults.seed})"),
    ] = None,
) -> None:
    """Train a fault diagnosis model."""
    # Build config: CLI args override schema defaults
    config = TrainConfig(
        model=model if model is not None else _defaults.model,
        epochs=epochs if epochs is not None else _defaults.epochs,
        batch_size=batch_size if batch_size is not None else _defaults.batch_size,
        learning_rate=learning_rate if learning_rate is not None else _defaults.learning_rate,
        seed=seed if seed is not None else _defaults.seed,
    )

    logger.info("Loading data from: {}", data)
    logger.info("Training {} model for {} epochs", config.model, config.epochs)
    logger.info("Batch size: {}, Learning rate: {}", config.batch_size, config.learning_rate)
    logger.warning("Training not yet implemented")


@app.command("list-models")
def train_list_models() -> None:
    """List available model architectures."""
    from rich.console import Console
    from rich.table import Table

    models = ["lstm", "gru", "transformer", "cnn", "tcn"]
    console = Console()
    table = Table(title="Available Models", show_header=True)
    table.add_column("Name", style="cyan")
    for m in models:
        table.add_row(m)
    console.print(table)
