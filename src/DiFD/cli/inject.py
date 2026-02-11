"""CLI subcommand for fault injection."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from DiFD.schema import FaultConfig, FaultType, InjectionConfig, MarkovConfig, WindowConfig
from DiFD.datasets import get_dataset, list_datasets

app = typer.Typer(no_args_is_help=True)


@app.command("run")
def inject_run(
    dataset: Annotated[
        str,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = "intel_lab",
    data_path: Annotated[
        Optional[Path],
        typer.Option("--data-path", help="Path to raw data file"),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for injected dataset"),
    ] = Path("data/injected/dataset.npz"),
    seed: Annotated[
        int,
        typer.Option("--seed", "-s", help="Random seed for reproducibility"),
    ] = 42,
    resample_freq: Annotated[
        str,
        typer.Option("--resample-freq", help="Resampling frequency (e.g., '30s', '1min')"),
    ] = "30s",
    interpolation: Annotated[
        str,
        typer.Option(
            "--interpolation",
            help="Interpolation method for missing values",
        ),
    ] = "linear",
    target_features: Annotated[
        Optional[list[str]],
        typer.Option("--target-features", "-t", help="Features to inject faults into"),
    ] = None,
    all_features: Annotated[
        Optional[list[str]],
        typer.Option("--all-features", "-a", help="All features to include in output"),
    ] = None,
    window_size: Annotated[
        int,
        typer.Option("--window-size", "-w", help="Window size in timesteps"),
    ] = 60,
    train_stride: Annotated[
        int,
        typer.Option("--train-stride", help="Stride for training windows"),
    ] = 10,
    test_stride: Annotated[
        int,
        typer.Option("--test-stride", help="Stride for test windows"),
    ] = 60,
    train_ratio: Annotated[
        float,
        typer.Option("--train-ratio", help="Fraction of data for training"),
    ] = 0.8,
    spike_prob: Annotated[
        float,
        typer.Option("--spike-prob", help="Transition probability to spike fault"),
    ] = 0.02,
    spike_duration: Annotated[
        int,
        typer.Option("--spike-duration", help="Average duration of spike faults"),
    ] = 2,
    spike_magnitude_min: Annotated[
        float,
        typer.Option("--spike-magnitude-min", help="Minimum spike magnitude"),
    ] = -20.0,
    spike_magnitude_max: Annotated[
        float,
        typer.Option("--spike-magnitude-max", help="Maximum spike magnitude"),
    ] = 20.0,
    drift_prob: Annotated[
        float,
        typer.Option("--drift-prob", help="Transition probability to drift fault"),
    ] = 0.01,
    drift_duration: Annotated[
        int,
        typer.Option("--drift-duration", help="Average duration of drift faults"),
    ] = 60,
    drift_rate: Annotated[
        float,
        typer.Option("--drift-rate", help="Drift rate per timestep"),
    ] = 0.1,
    stuck_prob: Annotated[
        float,
        typer.Option("--stuck-prob", help="Transition probability to stuck fault"),
    ] = 0.015,
    stuck_duration: Annotated[
        int,
        typer.Option("--stuck-duration", help="Average duration of stuck faults"),
    ] = 20,
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to JSON config file (CLI args override)"),
    ] = None,
) -> None:
    """Run fault injection on a dataset."""
    from DiFD.injection import FaultInjector

    if config and config.exists():
        with open(config) as f:
            config_dict = json.load(f)
        base_config = InjectionConfig.from_dict(config_dict)
    else:
        base_config = None

    target_features_list = target_features or (
        base_config.target_features if base_config else ["temp"]
    )
    all_features_list = all_features or (
        base_config.all_features if base_config else ["temp", "humid", "light", "volt"]
    )

    fault_configs = [
        FaultConfig(
            fault_type=FaultType.SPIKE,
            transition_prob=spike_prob,
            average_duration=spike_duration,
            params={"magnitude_range": (spike_magnitude_min, spike_magnitude_max)},
        ),
        FaultConfig(
            fault_type=FaultType.DRIFT,
            transition_prob=drift_prob,
            average_duration=drift_duration,
            params={"drift_rate": drift_rate},
        ),
        FaultConfig(
            fault_type=FaultType.STUCK,
            transition_prob=stuck_prob,
            average_duration=stuck_duration,
            params={},
        ),
    ]

    injection_config = InjectionConfig(
        markov=MarkovConfig(fault_configs=fault_configs, seed=seed),
        window=WindowConfig(
            window_size=window_size,
            train_stride=train_stride,
            test_stride=test_stride,
            train_ratio=train_ratio,
        ),
        resample_freq=resample_freq,
        target_features=target_features_list,
        all_features=all_features_list,
        interpolation_method=interpolation,
        seed=seed,
    )

    typer.echo(f"Loading dataset: {dataset}")
    ds = get_dataset(dataset, str(data_path) if data_path else None)

    typer.echo(f"Running fault injection with seed={injection_config.seed}")
    injector = FaultInjector(injection_config)
    result = injector.run(ds)

    output.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Saving to: {output}")
    result.save(output)
    result.print_summary()

    config_path = output.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(injection_config.to_dict(), f, indent=2)
    typer.echo(f"Config saved to: {config_path}")


@app.command("list-datasets")
def inject_list_datasets() -> None:
    """List available datasets."""
    datasets = list_datasets()
    typer.echo("Available datasets:")
    for ds in datasets:
        typer.echo(f"  - {ds}")
