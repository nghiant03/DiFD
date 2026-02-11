"""CLI subcommand for model training."""

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command("run")
def train_run(
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to injected dataset (.npz)"),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model architecture to use"),
    ] = "lstm",
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of training epochs"),
    ] = 100,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Training batch size"),
    ] = 32,
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="Learning rate"),
    ] = 0.001,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path for trained model"),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", "-s", help="Random seed"),
    ] = 42,
) -> None:
    """Train a fault diagnosis model."""
    typer.echo(f"Loading data from: {data}")
    typer.echo(f"Training {model} model for {epochs} epochs")
    typer.echo(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    typer.echo("Training not yet implemented")


@app.command("list-models")
def train_list_models() -> None:
    """List available model architectures."""
    models = ["lstm", "gru", "transformer", "cnn", "tcn"]
    typer.echo("Available models:")
    for m in models:
        typer.echo(f"  - {m}")
