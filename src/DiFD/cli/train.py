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
from DiFD.training.windowing import prepare_data

app = typer.Typer(no_args_is_help=True)

_FIELD_DEFAULTS = TrainConfig.model_fields


def _field_default(name: str) -> object:
    """Get the default value for a TrainConfig field."""
    return _FIELD_DEFAULTS[name].default


@app.command("run")
def train_run(
    model: Annotated[
        str,
        typer.Argument(help="Model architecture"),
    ],
    data: Annotated[
        Path,
        typer.Argument(help="Path to injected dataset directory"),
    ],
    epochs: Annotated[
        Optional[int],
        typer.Option("--epochs", "-e", help=f"Training epochs (default: {_field_default('epochs')})"),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", "-b", help=f"Batch size (default: {_field_default('batch_size')})"),
    ] = None,
    learning_rate: Annotated[
        Optional[float],
        typer.Option("--lr", help=f"Learning rate (default: {_field_default('learning_rate')})"),
    ] = None,
    use_focal_loss: Annotated[
        Optional[bool],
        typer.Option("--focal-loss/--no-focal-loss", help="Use focal loss instead of cross-entropy"),
    ] = None,
    focal_gamma: Annotated[
        Optional[float],
        typer.Option("--focal-gamma", help=f"Focal loss gamma (default: {_field_default('focal_gamma')})"),
    ] = None,
    oversample: Annotated[
        Optional[bool],
        typer.Option("--oversample/--no-oversample", help="Oversample minority classes"),
    ] = None,
    oversample_ratio: Annotated[
        Optional[float],
        typer.Option(
            "--oversample-ratio",
            help=f"Target minority/majority ratio (default: {_field_default('oversample_ratio')})",
        ),
    ] = None,
    val_ratio: Annotated[
        Optional[float],
        typer.Option(
            "--val-ratio",
            help=f"Fraction of training data for validation (default: {_field_default('val_ratio')})",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory for trained model"),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", "-s", help=f"Random seed (default: {_field_default('seed')})"),
    ] = None,
    features: Annotated[
        Optional[list[str]],
        typer.Option("--features", "-f", help="Feature(s) to train on (default: all)"),
    ] = None,
) -> None:
    """Train a fault diagnosis model."""
    defaults = TrainConfig(model=model)
    config = TrainConfig(
        model=model,
        epochs=epochs if epochs is not None else defaults.epochs,
        batch_size=batch_size if batch_size is not None else defaults.batch_size,
        learning_rate=learning_rate if learning_rate is not None else defaults.learning_rate,
        use_focal_loss=use_focal_loss if use_focal_loss is not None else defaults.use_focal_loss,
        focal_gamma=focal_gamma if focal_gamma is not None else defaults.focal_gamma,
        oversample=oversample if oversample is not None else defaults.oversample,
        oversample_ratio=oversample_ratio if oversample_ratio is not None else defaults.oversample_ratio,
        val_ratio=val_ratio if val_ratio is not None else defaults.val_ratio,
        seed=seed if seed is not None else defaults.seed,
        features=features if features is not None else defaults.features,
    )
    logger.debug("TrainConfig: {}", config.to_dict())

    logger.info("Loading data from: {}", data)
    dataset = InjectedDataset.load(data)
    dataset.print_summary()

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        dataset, features=config.features
    )
    logger.debug(
        "Windowed shapes: X_train={}, y_train={}, X_val={}, y_val={}, X_test={}, y_test={}",
        X_train.shape,
        y_train.shape,
        X_val.shape,
        y_val.shape,
        X_test.shape,
        y_test.shape,
    )

    input_size = X_train.shape[-1]
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

    output_path = output if output is not None else Path(f"models/{config.model}")
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
        X_train=X_train,
        y_train=y_train,
        X_val=X_val if len(X_val) > 0 else None,
        y_val=y_val if len(y_val) > 0 else None,
    )

    logger.info(
        "Training complete at epoch {} | best_val_loss={:.4f}",
        result.stopped_epoch,
        result.best_val_loss if result.best_val_loss is not None else float("nan"),
    )
    logger.info("Model saved to: {}", output_path)

    if len(X_test) > 0:
        logger.info("--- Final Test Evaluation ---")
        evaluator = Evaluator(
            config=EvaluateConfig(batch_size=config.batch_size),
            device=str(trainer.device),
        )
        criterion = _build_loss(config, trainer.device)
        eval_result = evaluator.evaluate(net, X_test, y_test, criterion=criterion)
        evaluator.log_results(eval_result)

        eval_result.save(
            output_path,
            train_config=config.to_dict(),
            injection_config=dataset.config.to_dict(),
        )
        logger.info("Results saved to: {}", output_path)


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
