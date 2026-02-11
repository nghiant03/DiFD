"""Injection pipeline configuration.

This module defines the complete configuration for the fault injection
pipeline, which gets serialized as metadata with the output dataset.
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
    resample_freq: str = "30s"
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
