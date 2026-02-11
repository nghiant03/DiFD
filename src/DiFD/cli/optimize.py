"""CLI subcommand for hyperparameter optimization."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from DiFD.logging import logger
from DiFD.schema import OptimizeConfig

app = typer.Typer(no_args_is_help=True)

# Pre-instantiate defaults for help text
_defaults = OptimizeConfig()


@app.command("run")
def optimize_run(
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to injected dataset (.npz)"),
    ],
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help=f"Model architecture to optimize (default: {_defaults.model})"),
    ] = None,
    n_trials: Annotated[
        Optional[int],
        typer.Option("--n-trials", "-n", help=f"Number of Optuna trials (default: {_defaults.n_trials})"),
    ] = None,
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
        typer.Option("--storage", help=f"Optuna storage URL (default: {_defaults.storage})"),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", "-s", help=f"Random seed (default: {_defaults.seed})"),
    ] = None,
) -> None:
    """Run hyperparameter optimization with Optuna."""
    # Build config: CLI args override schema defaults
    config = OptimizeConfig(
        model=model if model is not None else _defaults.model,
        n_trials=n_trials if n_trials is not None else _defaults.n_trials,
        seed=seed if seed is not None else _defaults.seed,
        storage=storage if storage is not None else _defaults.storage,
    )

    logger.info("Loading data from: {}", data)
    logger.info("Optimizing {} model", config.model)
    logger.info("Running {} trials", config.n_trials)
    if timeout:
        logger.info("Timeout: {}s", timeout)
    logger.warning("Optimization not yet implemented")


@app.command("show")
def optimize_show(
    study_name: Annotated[
        str,
        typer.Argument(help="Name of the study to show"),
    ],
    storage: Annotated[
        Optional[str],
        typer.Option("--storage", help=f"Optuna storage URL (default: {_defaults.storage})"),
    ] = None,
) -> None:
    """Show results from an optimization study."""
    # Use schema default if not provided
    resolved_storage = storage if storage is not None else _defaults.storage

    logger.info("Study: {}", study_name)
    logger.info("Storage: {}", resolved_storage)
    logger.warning("Study visualization not yet implemented")
