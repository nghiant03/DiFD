"""CLI subcommand for model training."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from DiFD.datasets import InjectedDataset
from DiFD.evaluation import Evaluator
from DiFD.logging import logger
from DiFD.models import create_model
from DiFD.schema import EvaluateConfig, TrainConfig
from DiFD.schema.types import FaultType
from DiFD.training import CheckpointCallback, EarlyStoppingCallback, LoggingCallback, Trainer
from DiFD.training.trainer import _build_loss

app = typer.Typer(no_args_is_help=True)

_defaults = TrainConfig()


@app.command("run")
def train_run(
    data: Annotated[
        Path,
        typer.Argument(help="Path to injected dataset (.npz)"),
    ],
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help=f"Model architecture (default: {_defaults.model})"),
    ] = None,
    epochs: Annotated[
        Optional[int],
        typer.Option("--epochs", "-e", help=f"Training epochs (default: {_defaults.epochs})"),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", "-b", help=f"Batch size (default: {_defaults.batch_size})"),
    ] = None,
    learning_rate: Annotated[
        Optional[float],
        typer.Option("--lr", help=f"Learning rate (default: {_defaults.learning_rate})"),
    ] = None,
    use_focal_loss: Annotated[
        Optional[bool],
        typer.Option("--focal-loss/--no-focal-loss", help="Use focal loss instead of cross-entropy"),
    ] = None,
    focal_gamma: Annotated[
        Optional[float],
        typer.Option("--focal-gamma", help=f"Focal loss gamma (default: {_defaults.focal_gamma})"),
    ] = None,
    oversample: Annotated[
        Optional[bool],
        typer.Option("--oversample/--no-oversample", help="Oversample minority classes"),
    ] = None,
    oversample_ratio: Annotated[
        Optional[float],
        typer.Option(
            "--oversample-ratio",
            help=f"Target minority/majority ratio (default: {_defaults.oversample_ratio})",
        ),
    ] = None,
    val_ratio: Annotated[
        Optional[float],
        typer.Option(
            "--val-ratio",
            help=f"Fraction of training data for validation (default: {_defaults.val_ratio})",
        ),
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
    config = TrainConfig(
        model=model if model is not None else _defaults.model,
        epochs=epochs if epochs is not None else _defaults.epochs,
        batch_size=batch_size if batch_size is not None else _defaults.batch_size,
        learning_rate=learning_rate if learning_rate is not None else _defaults.learning_rate,
        use_focal_loss=use_focal_loss if use_focal_loss is not None else _defaults.use_focal_loss,
        focal_gamma=focal_gamma if focal_gamma is not None else _defaults.focal_gamma,
        oversample=oversample if oversample is not None else _defaults.oversample,
        oversample_ratio=oversample_ratio if oversample_ratio is not None else _defaults.oversample_ratio,
        val_ratio=val_ratio if val_ratio is not None else _defaults.val_ratio,
        seed=seed if seed is not None else _defaults.seed,
    )
    logger.debug("TrainConfig: {}", config.to_dict())

    logger.info("Loading data from: {}", data)
    dataset = InjectedDataset.load(data)
    dataset.print_summary()
    logger.debug(
        "Dataset shapes: X_train={}, y_train={}, X_test={}, y_test={}",
        dataset.X_train.shape,
        dataset.y_train.shape,
        dataset.X_test.shape,
        dataset.y_test.shape,
    )

    input_size = dataset.X_train.shape[-1]
    num_classes = FaultType.count()
    logger.debug(
        "Creating model: arch={}, input_size={}, num_classes={}",
        config.model,
        input_size,
        num_classes,
    )
    net = create_model(
        config.model,
        input_size=input_size,
        num_classes=num_classes,
    )
    logger.info(
        "Model: {} ({:,} parameters)", net.name, net.count_parameters()
    )

    output_path = output if output is not None else Path(f"models/{config.model}.pt")
    logger.debug("Output path: {}", output_path)

    callbacks = [
        LoggingCallback(),
        EarlyStoppingCallback(patience=10),
        CheckpointCallback(save_path=output_path, config_dict=config.to_dict()),
    ]

    trainer = Trainer(config=config, callbacks=callbacks)

    logger.info(
        "Training for {} epochs | batch_size={} | lr={} | focal_loss={} | oversample={}",
        config.epochs,
        config.batch_size,
        config.learning_rate,
        config.use_focal_loss,
        config.oversample,
    )

    result = trainer.fit(
        model=net,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
    )

    logger.info(
        "Training complete at epoch {} | best_val_loss={:.4f}",
        result.stopped_epoch,
        result.best_val_loss if result.best_val_loss is not None else float("nan"),
    )
    logger.info("Model saved to: {}", output_path)

    if dataset.X_test is not None and len(dataset.X_test) > 0:
        logger.info("--- Final Test Evaluation ---")
        evaluator = Evaluator(
            config=EvaluateConfig(batch_size=config.batch_size),
            device=str(trainer.device),
        )
        criterion = _build_loss(config, trainer.device)
        result = evaluator.evaluate(net, dataset.X_test, dataset.y_test, criterion=criterion)
        evaluator.log_results(result)


@app.command("list-models")
def train_list_models() -> None:
    """List available model architectures."""
    from rich.console import Console
    from rich.table import Table

    from DiFD.models import list_models

    console = Console()
    table = Table(title="Available Models", show_header=True)
    table.add_column("Name", style="cyan")
    for m in list_models():
        table.add_row(m)
    console.print(table)
