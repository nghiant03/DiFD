"""CLI subcommand for model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from DiFD.datasets import InjectedDataset
from DiFD.evaluation import Evaluator
from DiFD.logging import logger
from DiFD.schema import EvaluateConfig
from DiFD.schema.types import FaultType
from DiFD.training.windowing import prepare_data

app = typer.Typer(no_args_is_help=True)

_defaults = EvaluateConfig()


@app.command("run")
def evaluate_run(
    model: Annotated[
        Path,
        typer.Option("--model", "-m", help="Path to trained model directory"),
    ],
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to injected dataset directory"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory for evaluation results"),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", "-b", help=f"Evaluation batch size (default: {_defaults.batch_size})"),
    ] = None,
) -> None:
    """Evaluate a trained model on test data."""
    import torch

    from DiFD.models.base import BaseModel

    config = EvaluateConfig(
        batch_size=batch_size if batch_size is not None else _defaults.batch_size,
    )

    logger.info("Loading data from: {}", data)
    dataset = InjectedDataset.load(data)
    dataset.print_summary()

    logger.info("Loading model from: {}", model)
    meta = BaseModel.load_metadata(model)
    model_name = str(meta.get("model_name", "lstm"))
    model_config = meta.get("model_config", {})
    assert isinstance(model_config, dict)

    train_cfg = meta.get("train_config")
    saved_features: list[str] | None = None
    if isinstance(train_cfg, dict):
        saved_features = train_cfg.get("features")

    _, _, _, _, X_test, y_test = prepare_data(dataset, features=saved_features)

    if len(X_test) == 0:
        logger.error("No test data available in dataset")
        raise typer.Exit(code=1)

    from DiFD.models import create_model

    input_size = X_test.shape[-1]
    num_classes = FaultType.count()
    net = create_model(model_name, input_size=input_size, num_classes=num_classes)
    assert isinstance(net, BaseModel)
    net.load_state_dict(
        torch.load(model / "weight.pt", weights_only=True)
    )
    logger.info("Model: {} ({:,} parameters)", net.name, net.count_parameters())

    evaluator = Evaluator(config=config)
    logger.info("Evaluating with batch_size={}", config.batch_size)
    result = evaluator.evaluate(net, X_test, y_test)
    evaluator.log_results(result)

    save_dir = output if output is not None else model
    result.save(
        save_dir,
        train_config=train_cfg,  # type: ignore[arg-type]
        injection_config=dataset.config.to_dict(),
    )
    logger.info("Results saved to: {}", save_dir)


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
