"""CLI subcommand for fault injection."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from DiFD.logging import logger
from DiFD.schema import FaultConfig, FaultType, InjectionConfig, MarkovConfig, WindowConfig

app = typer.Typer(no_args_is_help=True)

# Pre-instantiate defaults for help text
_injection_defaults = InjectionConfig()
_window_defaults = WindowConfig()
_markov_defaults = MarkovConfig()
_spike_defaults = _markov_defaults.get_config(FaultType.SPIKE)
_drift_defaults = _markov_defaults.get_config(FaultType.DRIFT)
_stuck_defaults = _markov_defaults.get_config(FaultType.STUCK)
assert _spike_defaults and _drift_defaults and _stuck_defaults


@app.command("run")
def inject_run(
    dataset: Annotated[
        str,
        typer.Argument(help="Dataset to use"),
    ],
    data_path: Annotated[
        Path,
        typer.Argument(help="Path to raw data file"),
    ],
    output: Annotated[
        Path,
        typer.Argument(help="Output path for injected dataset"),
    ],
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", "-s", help="Random seed for reproducibility"),
    ] = None,
    resample_freq: Annotated[
        Optional[str],
        typer.Option(
            "--resample-freq",
            help=f"Resampling frequency (default: {_injection_defaults.resample_freq})",
        ),
    ] = None,
    interpolation: Annotated[
        Optional[str],
        typer.Option(
            "--interpolation",
            help=f"Interpolation method for missing values (default: {_injection_defaults.interpolation_method})",
        ),
    ] = None,
    target_features: Annotated[
        Optional[list[str]],
        typer.Option(
            "--target-features", "-t",
            help=f"Features to inject faults into (default: {_injection_defaults.target_features})",
        ),
    ] = None,
    all_features: Annotated[
        Optional[list[str]],
        typer.Option(
            "--all-features", "-a",
            help=f"All features to include in output (default: {_injection_defaults.all_features})",
        ),
    ] = None,
    window_size: Annotated[
        Optional[int],
        typer.Option(
            "--window-size", "-w",
            help=f"Window size in timesteps (default: {_window_defaults.window_size})",
        ),
    ] = None,
    train_stride: Annotated[
        Optional[int],
        typer.Option(
            "--train-stride",
            help=f"Stride for training windows (default: {_window_defaults.train_stride})",
        ),
    ] = None,
    test_stride: Annotated[
        Optional[int],
        typer.Option(
            "--test-stride",
            help=f"Stride for test windows (default: {_window_defaults.test_stride})",
        ),
    ] = None,
    train_ratio: Annotated[
        Optional[float],
        typer.Option(
            "--train-ratio",
            help=f"Fraction of data for training (default: {_window_defaults.train_ratio})",
        ),
    ] = None,
    spike_prob: Annotated[
        Optional[float],
        typer.Option(
            "--spike-prob",
            help=f"Transition probability to spike fault (default: {_spike_defaults.transition_prob})",
        ),
    ] = None,
    spike_duration: Annotated[
        Optional[int],
        typer.Option(
            "--spike-duration",
            help=f"Average duration of spike faults (default: {_spike_defaults.average_duration})",
        ),
    ] = None,
    spike_magnitude_min: Annotated[
        Optional[float],
        typer.Option(
            "--spike-magnitude-min",
            help=f"Minimum spike magnitude (default: {_spike_defaults.params.get('magnitude_range', (-20.0, 20.0))[0]})",
        ),
    ] = None,
    spike_magnitude_max: Annotated[
        Optional[float],
        typer.Option(
            "--spike-magnitude-max",
            help=f"Maximum spike magnitude (default: {_spike_defaults.params.get('magnitude_range', (-20.0, 20.0))[1]})",
        ),
    ] = None,
    drift_prob: Annotated[
        Optional[float],
        typer.Option(
            "--drift-prob",
            help=f"Transition probability to drift fault (default: {_drift_defaults.transition_prob})",
        ),
    ] = None,
    drift_duration: Annotated[
        Optional[int],
        typer.Option(
            "--drift-duration",
            help=f"Average duration of drift faults (default: {_drift_defaults.average_duration})",
        ),
    ] = None,
    drift_rate: Annotated[
        Optional[float],
        typer.Option(
            "--drift-rate",
            help=f"Drift rate per timestep (default: {_drift_defaults.params.get('drift_rate', 0.1)})",
        ),
    ] = None,
    stuck_prob: Annotated[
        Optional[float],
        typer.Option(
            "--stuck-prob",
            help=f"Transition probability to stuck fault (default: {_stuck_defaults.transition_prob})",
        ),
    ] = None,
    stuck_duration: Annotated[
        Optional[int],
        typer.Option(
            "--stuck-duration",
            help=f"Average duration of stuck faults (default: {_stuck_defaults.average_duration})",
        ),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to JSON config file (CLI args override)"),
    ] = None,
) -> None:
    """Run fault injection on a dataset."""
    from DiFD.datasets import get_dataset
    from DiFD.injection import FaultInjector

    # Load base config from file if provided
    if config and config.exists():
        with open(config) as f:
            config_dict = json.load(f)
        base_config = InjectionConfig.from_dict(config_dict)
    else:
        base_config = InjectionConfig()

    defaults = WindowConfig()
    window_config = WindowConfig(
        window_size=window_size if window_size is not None else base_config.window.window_size if base_config.window.window_size != defaults.window_size else defaults.window_size,
        train_stride=train_stride if train_stride is not None else base_config.window.train_stride if base_config.window.train_stride != defaults.train_stride else defaults.train_stride,
        test_stride=test_stride if test_stride is not None else base_config.window.test_stride if base_config.window.test_stride != defaults.test_stride else defaults.test_stride,
        train_ratio=train_ratio if train_ratio is not None else base_config.window.train_ratio if base_config.window.train_ratio != defaults.train_ratio else defaults.train_ratio,
    )

    default_markov = MarkovConfig()
    default_spike = default_markov.get_config(FaultType.SPIKE)
    default_drift = default_markov.get_config(FaultType.DRIFT)
    default_stuck = default_markov.get_config(FaultType.STUCK)
    assert default_spike is not None and default_drift is not None and default_stuck is not None

    base_spike = base_config.markov.get_config(FaultType.SPIKE) or default_spike
    base_drift = base_config.markov.get_config(FaultType.DRIFT) or default_drift
    base_stuck = base_config.markov.get_config(FaultType.STUCK) or default_stuck

    spike_trans = spike_prob if spike_prob is not None else base_spike.transition_prob
    spike_dur = spike_duration if spike_duration is not None else base_spike.average_duration
    spike_mag_min = spike_magnitude_min if spike_magnitude_min is not None else base_spike.params.get("magnitude_range", (-20.0, 20.0))[0]
    spike_mag_max = spike_magnitude_max if spike_magnitude_max is not None else base_spike.params.get("magnitude_range", (-20.0, 20.0))[1]

    drift_trans = drift_prob if drift_prob is not None else base_drift.transition_prob
    drift_dur = drift_duration if drift_duration is not None else base_drift.average_duration
    drift_r = drift_rate if drift_rate is not None else base_drift.params.get("drift_rate", 0.1)

    stuck_trans = stuck_prob if stuck_prob is not None else base_stuck.transition_prob
    stuck_dur = stuck_duration if stuck_duration is not None else base_stuck.average_duration

    fault_configs = [
        FaultConfig(
            fault_type=FaultType.SPIKE,
            transition_prob=spike_trans,
            average_duration=spike_dur,
            params={"magnitude_range": (spike_mag_min, spike_mag_max)},
        ),
        FaultConfig(
            fault_type=FaultType.DRIFT,
            transition_prob=drift_trans,
            average_duration=drift_dur,
            params={"drift_rate": drift_r},
        ),
        FaultConfig(
            fault_type=FaultType.STUCK,
            transition_prob=stuck_trans,
            average_duration=stuck_dur,
            params={},
        ),
    ]

    resolved_seed = seed if seed is not None else base_config.seed

    markov_config = MarkovConfig(fault_configs=fault_configs, seed=resolved_seed)

    resolved_resample = resample_freq if resample_freq is not None else base_config.resample_freq
    resolved_interpolation = interpolation if interpolation is not None else base_config.interpolation_method
    resolved_target = target_features if target_features is not None else base_config.target_features
    resolved_all = all_features if all_features is not None else base_config.all_features

    injection_config = InjectionConfig(
        markov=markov_config,
        window=window_config,
        resample_freq=resolved_resample,
        target_features=resolved_target,
        all_features=resolved_all,
        interpolation_method=resolved_interpolation,
        seed=resolved_seed,
    )

    logger.info("Loading dataset: {}", dataset)
    ds = get_dataset(dataset, data_path)

    logger.info("Running fault injection with seed={}", injection_config.seed)
    injector = FaultInjector(injection_config)
    result = injector.run(ds)

    output.mkdir(parents=True, exist_ok=True)
    logger.info("Saving to: {}", output)
    result.save(output)
    result.print_summary()

    config_path = (output / "injected_config").with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(injection_config.to_dict(), f, indent=2)
    logger.info("Config saved to: {}", config_path)


@app.command("list-datasets")
def inject_list_datasets() -> None:
    """List available datasets."""
    from rich.console import Console
    from rich.table import Table

    from DiFD.datasets import list_datasets

    console = Console()
    datasets = list_datasets()
    table = Table(title="Available Datasets", show_header=True)
    table.add_column("Name", style="cyan")
    for ds in datasets:
        table.add_row(ds)
    console.print(table)
