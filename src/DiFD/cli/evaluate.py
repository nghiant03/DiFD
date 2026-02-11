"""CLI subcommand for model evaluation."""

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command("run")
def evaluate_run(
    model: Annotated[
        Path,
        typer.Option("--model", "-m", help="Path to trained model"),
    ],
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to test dataset (.npz)"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path for evaluation results"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Evaluation batch size"),
    ] = 64,
) -> None:
    """Evaluate a trained model on test data."""
    typer.echo(f"Loading model from: {model}")
    typer.echo(f"Evaluating on: {data}")
    typer.echo("Evaluation not yet implemented")


@app.command("metrics")
def evaluate_metrics() -> None:
    """List available evaluation metrics."""
    metrics = ["accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"]
    typer.echo("Available metrics:")
    for m in metrics:
        typer.echo(f"  - {m}")
