"""CLI subcommand for model evaluation."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from DiFD.logging import logger
from DiFD.schema import EvaluateConfig

app = typer.Typer(no_args_is_help=True)

# Pre-instantiate defaults for help text
_defaults = EvaluateConfig()


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
        Optional[int],
        typer.Option("--batch-size", "-b", help=f"Evaluation batch size (default: {_defaults.batch_size})"),
    ] = None,
) -> None:
    """Evaluate a trained model on test data."""
    # Build config: CLI args override schema defaults
    config = EvaluateConfig(
        batch_size=batch_size if batch_size is not None else _defaults.batch_size,
    )

    logger.info("Loading model from: {}", model)
    logger.info("Evaluating on: {}", data)
    logger.info("Batch size: {}", config.batch_size)
    logger.warning("Evaluation not yet implemented")


@app.command("metrics")
def evaluate_metrics() -> None:
    """List available evaluation metrics."""
    from rich.console import Console
    from rich.table import Table

    metrics = ["accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"]
    console = Console()
    table = Table(title="Available Metrics", show_header=True)
    table.add_column("Name", style="cyan")
    for m in metrics:
        table.add_row(m)
    console.print(table)
