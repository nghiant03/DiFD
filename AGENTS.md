# AGENTS.md

AGENTS MUST KEEP THIS FILE UP TO DATE AFTER A CODE CHANGE.
This repository is a research project for fault diagnosis analysis.

## Environment Management

- Use **uv** for Python environment management.

## Code Quality

- Use **ruff** for linting and formatting `.py` source files.
- Use **pyright** for type checking `.py` source files.
- Run tests with `uv run pytest`.

## Notebook Conventions

- In every Jupyter notebook, the **import block must be in the top cell**.

## Project Structure

```
src/DiFD/
├── schema/            # Pydantic config models: FaultType, FaultConfig, MarkovConfig, WindowConfig, InjectionConfig
├── cli/               # Typer CLI with subcommands (inject, train, evaluate, optimize)
├── injection/         # Fault injection: Markov generator, fault injectors, registry
├── datasets/          # Dataset loaders (Intel Lab + extensible registry) + InjectedDataset container
├── models/            # Deep learning model definitions
├── training/          # Trainer, focal loss, oversampling, and callbacks
├── evaluation/        # Metrics and evaluator
├── optimization/      # Optuna hyperparameter sweep

data/                  # Raw datasets and injected outputs
tests/                 # Unit tests per module
notebooks/             # Jupyter notebooks for analysis
```

## Schema Module (`schema/`)

The `schema/` module contains Pydantic configuration models used by injection, training, and evaluation:

- `FaultType` - Enum: NORMAL=0, SPIKE=1, DRIFT=2, STUCK=3
- `FaultConfig` - Configuration for a single fault type (transition prob, duration, params)
- `MarkovConfig` - Markov chain configuration (list of fault configs, seed)
- `WindowConfig` - Sliding window parameters (size, strides, train ratio)
- `InjectionConfig` - Complete injection pipeline config (serializable as metadata)
- `TrainConfig` - Training configuration (model, epochs, batch_size, learning_rate, use_focal_loss, focal_gamma, focal_alpha, oversample, oversample_ratio, seed)
- `EvaluateConfig` - Evaluation configuration (batch_size)
- `OptimizeConfig` - Optimization configuration (model, n_trials, seed, storage)

## Configuration Design Pattern

**Single Source of Truth (SSOT)**: All default values live exclusively in Pydantic schema classes (`schema/config.py`, `schema/types.py`). CLI modules use `None` as default and fall back to schema defaults.

**Pattern**:
```python
# CLI: Use None defaults
@app.command()
def run(
    window_size: Optional[int] = None,  # NOT = 60
    ...
):
    defaults = WindowConfig()
    config = WindowConfig(
        window_size=window_size if window_size is not None else defaults.window_size,
    )
```

**Rationale**:
- Prevents value drift between CLI and schema
- Single place to update defaults
- Schema classes document the canonical defaults
- CLI `--help` can reference schema or show "default: from config"

## Datasets Module (`datasets/`)

- `InjectedDataset` - Container with X/y train/test arrays + config + save/load

## Training Module (`training/`)

- `FocalLoss` (`loss.py`) - Focal loss for imbalanced multi-class classification. gamma=0 recovers CE.
- `oversample_minority` (`oversampling.py`) - Window-level oversampling: duplicates windows containing any non-NORMAL label until minority count reaches `ratio * majority_count`.
- `Trainer` (`trainer.py`) - Full training loop with Adam optimizer, optional focal loss, optional oversampling, and callback hooks. Returns `TrainResult` with per-epoch history.
- `TrainingCallback` (`callbacks.py`) - Abstract base; implementations: `LoggingCallback`, `EarlyStoppingCallback`, `CheckpointCallback`.

## Workflow

1. **Fault Injection**: `uv run difd inject run intel_lab data/raw/Intel/data.txt data/injected/intel_lab.npz`
2. **Training**: `uv run difd train run --data data/injected/intel_lab.npz --model lstm`
3. **Evaluation**: `uv run difd evaluate run --model models/lstm.pt --data data/injected/intel_lab.npz`
4. **Optimization**: `uv run difd optimize run --data data/injected/intel_lab.npz --n-trials 100`

## CLI Structure

The CLI uses **Typer** with a centralized command namespace:

```
difd                    # Main entry point
├── inject              # Fault injection subcommands
│   ├── run             # Run fault injection
│   └── list-datasets   # List available datasets
├── train               # Training subcommands
│   ├── run             # Train a model
│   └── list-models     # List available models
├── evaluate            # Evaluation subcommands
│   ├── run             # Evaluate a model
│   └── metrics         # List available metrics
└── optimize            # Hyperparameter optimization
    ├── run             # Run Optuna optimization
    └── show            # Show study results
```

Run `difd --help` or `difd <subcommand> --help` for detailed options.

## Adding New Fault Types

1. Add new value to `FaultType` enum in `schema/types.py`.
2. Create injector class in `injection/faults.py` subclassing `BaseFaultInjector`.
3. Register in `injection/registry.py` with `register_fault()`.
4. Add default config in `MarkovConfig._default_fault_configs()`.

## Adding New Datasets

1. Implement a new dataset class in `src/DiFD/datasets/` subclassing `BaseDataset`.
2. Implement: `name`, `feature_columns`, `group_column`, `timestamp_column`, `load()`, `preprocess()`.
3. Register in `datasets/registry.py` with `register_dataset()`.

## CLI Options (inject run)

CLI options use `None` defaults; actual defaults come from schema classes.

```
DATASET                Dataset name (required positional argument)
DATA_PATH              Path to raw data file (required positional argument)
OUTPUT                 Output path for .npz file (required positional argument)
-s, --seed             Random seed for reproducibility
--resample-freq        Resampling frequency (default from InjectionConfig: 30s)
-t, --target-features  Features to inject faults into
-a, --all-features     All features to include in output
-w, --window-size      Window size in timesteps (default from WindowConfig: 60)
--train-stride         Stride for training windows (default from WindowConfig: 10)
--test-stride          Stride for test windows (default from WindowConfig: 60)
--spike-prob           Transition probability to spike (default from MarkovConfig: 0.02)
--drift-prob           Transition probability to drift (default from MarkovConfig: 0.01)
--stuck-prob           Transition probability to stuck (default from MarkovConfig: 0.015)
-c, --config           Path to JSON config file (CLI args override)
```
