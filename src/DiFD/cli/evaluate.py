"""CLI subcommand for model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from DiFD.datasets import InjectedDataset
from DiFD.evaluation import Evaluator
from DiFD.logging import logger
from DiFD.models import create_model
from DiFD.schema import EvaluateConfig
from DiFD.schema.types import FaultType

app = typer.Typer(no_args_is_help=True)

_defaults = EvaluateConfig()


@app.command("run")
def evaluate_run(
    model: Annotated[
        Path,
        typer.Option("--model", "-m", help="Path to trained model (.pt)"),
    ],
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to injected dataset (.npz)"),
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
    import json

    import torch

    from DiFD.models.base import BaseModel

    config = EvaluateConfig(
        batch_size=batch_size if batch_size is not None else _defaults.batch_size,
    )

    logger.info("Loading data from: {}", data)
    dataset = InjectedDataset.load(data)
    dataset.print_summary()

    if dataset.X_test is None or len(dataset.X_test) == 0:
        logger.error("No test data available in dataset")
        raise typer.Exit(code=1)

    logger.info("Loading model from: {}", model)
    checkpoint = torch.load(model, weights_only=True)
    model_name = checkpoint.get("model_name", "lstm")

    input_size = dataset.X_test.shape[-1]
    num_classes = FaultType.count()
    net = create_model(model_name, input_size=input_size, num_classes=num_classes)
    assert isinstance(net, BaseModel)
    net.load_state_dict(checkpoint["state_dict"])
    logger.info("Model: {} ({:,} parameters)", net.name, net.count_parameters())

    evaluator = Evaluator(config=config)
    logger.info("Evaluating with batch_size={}", config.batch_size)
    result = evaluator.evaluate(net, dataset.X_test, dataset.y_test)
    evaluator.log_results(result)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        results_dict = {
            "loss": result.loss,
            "accuracy": result.accuracy,
            "macro_f1": result.macro_f1,
            "per_class": {
                name: {
                    "precision": result.class_metrics.precision[i],
                    "recall": result.class_metrics.recall[i],
                    "f1": result.class_metrics.f1[i],
                    "support": result.class_metrics.support[i],
                }
                for i, name in enumerate(FaultType.names())
                if i < len(result.class_metrics.precision)
            },
        }
        output.write_text(json.dumps(results_dict, indent=2))
        logger.info("Results saved to: {}", output)


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
