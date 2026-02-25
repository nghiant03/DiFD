"""Pipeline configuration classes.

This module defines configuration classes for all pipeline phases:
injection, training, evaluation, and optimization.

Each config class owns its default values (Single Source of Truth).
CLI modules should use None defaults and fall back to these schema defaults.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from DiFD.schema.types import FaultType, MarkovConfig, WindowConfig


class InjectionConfig(BaseModel):
    """Complete configuration for fault injection pipeline.

    This is the main config object that gets serialized as metadata.

    Attributes:
        markov: Markov chain configuration.
        window: Windowing configuration.
        resample_freq: Resampling frequency string (e.g., "30s").
        target_features: Features to inject faults into.
        all_features: All features to include in the output.
        interpolation_method: Method for interpolating missing values.
        group_column: Column to group by (e.g., "moteid").
        seed: Global random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    markov: MarkovConfig = Field(default_factory=MarkovConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    resample_freq: str = "5min"
    target_features: list[str] = Field(default_factory=lambda: ["temp"])
    all_features: list[str] = Field(
        default_factory=lambda: ["temp", "humid", "light", "volt"]
    )
    interpolation_method: str = "linear"
    group_column: str = "moteid"
    seed: int | None = None

    @model_validator(mode="after")
    def _propagate_seed(self) -> "InjectionConfig":
        """Propagate seed to markov config if not set."""
        if self.seed is not None and self.markov.seed is None:
            object.__setattr__(
                self,
                "markov",
                self.markov.model_copy(update={"seed": self.seed}),
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "markov": self.markov.to_dict(),
            "window": self.window.to_dict(),
            "resample_freq": self.resample_freq,
            "target_features": self.target_features,
            "all_features": self.all_features,
            "interpolation_method": self.interpolation_method,
            "group_column": self.group_column,
            "seed": self.seed,
            "fault_type_mapping": {ft.name: ft.value for ft in FaultType},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InjectionConfig":
        """Reconstruct from dictionary."""
        return cls(
            markov=MarkovConfig.from_dict(data.get("markov", {})),
            window=WindowConfig.from_dict(data.get("window", {})),
            resample_freq=data.get("resample_freq", "30s"),
            target_features=data.get("target_features", ["temp"]),
            all_features=data.get("all_features", ["temp", "humid", "light", "volt"]),
            interpolation_method=data.get("interpolation_method", "linear"),
            group_column=data.get("group_column", "moteid"),
            seed=data.get("seed"),
        )


class TrainConfig(BaseModel):
    """Configuration for model training.

    Attributes:
        model: Model architecture name.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        use_focal_loss: Whether to use focal loss instead of cross-entropy.
        focal_gamma: Focusing parameter for focal loss (higher = more focus on hard examples).
        focal_alpha: Per-class balancing weights for focal loss. None means uniform.
        oversample: Whether to oversample minority (non-NORMAL) classes.
        oversample_ratio: Target ratio of minority to majority samples (1.0 = balanced).
        val_ratio: Fraction of training data to use for validation (0.0 = no split).
        seed: Random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    model: str = "lstm"
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    use_focal_loss: bool = False
    focal_gamma: float = Field(default=2.0, ge=0.0)
    focal_alpha: list[float] | None = None
    oversample: bool = False
    oversample_ratio: float = Field(default=1.0, gt=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "use_focal_loss": self.use_focal_loss,
            "focal_gamma": self.focal_gamma,
            "focal_alpha": self.focal_alpha,
            "oversample": self.oversample,
            "oversample_ratio": self.oversample_ratio,
            "val_ratio": self.val_ratio,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        """Reconstruct from dictionary."""
        defaults = cls()
        return cls(
            model=data.get("model", defaults.model),
            epochs=data.get("epochs", defaults.epochs),
            batch_size=data.get("batch_size", defaults.batch_size),
            learning_rate=data.get("learning_rate", defaults.learning_rate),
            use_focal_loss=data.get("use_focal_loss", defaults.use_focal_loss),
            focal_gamma=data.get("focal_gamma", defaults.focal_gamma),
            focal_alpha=data.get("focal_alpha", defaults.focal_alpha),
            oversample=data.get("oversample", defaults.oversample),
            oversample_ratio=data.get("oversample_ratio", defaults.oversample_ratio),
            val_ratio=data.get("val_ratio", defaults.val_ratio),
            seed=data.get("seed", defaults.seed),
        )


class EvaluateConfig(BaseModel):
    """Configuration for model evaluation.

    Attributes:
        batch_size: Evaluation batch size.
    """

    model_config = ConfigDict(frozen=True)

    batch_size: int = Field(default=64, ge=1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluateConfig":
        """Reconstruct from dictionary."""
        defaults = cls()
        return cls(
            batch_size=data.get("batch_size", defaults.batch_size),
        )


class OptimizeConfig(BaseModel):
    """Configuration for hyperparameter optimization.

    Attributes:
        model: Model architecture to optimize.
        n_trials: Number of Optuna trials.
        seed: Random seed for reproducibility.
        storage: Optuna storage URL.
    """

    model_config = ConfigDict(frozen=True)

    model: str = "lstm"
    n_trials: int = Field(default=100, ge=1)
    seed: int = 42
    storage: str = "sqlite:///optuna.db"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "n_trials": self.n_trials,
            "seed": self.seed,
            "storage": self.storage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizeConfig":
        """Reconstruct from dictionary."""
        defaults = cls()
        return cls(
            model=data.get("model", defaults.model),
            n_trials=data.get("n_trials", defaults.n_trials),
            seed=data.get("seed", defaults.seed),
            storage=data.get("storage", defaults.storage),
        )
