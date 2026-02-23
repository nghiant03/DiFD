"""Core type definitions for fault diagnosis.

This module defines the fundamental types shared across injection,
training, and evaluation phases.
"""

from enum import IntEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FaultType(IntEnum):
    """Fault type enumeration.

    Integer values are used directly as labels in the dataset.
    New fault types should be added with sequential integer values.
    """

    NORMAL = 0
    SPIKE = 1
    DRIFT = 2
    STUCK = 3

    @classmethod
    def from_string(cls, name: str) -> "FaultType":
        """Convert string name to FaultType."""
        return cls[name.upper()]

    @classmethod
    def names(cls) -> list[str]:
        """Return list of all fault type names."""
        return [ft.name for ft in cls]

    @classmethod
    def fault_names(cls) -> list[str]:
        """Return list of fault type names excluding NORMAL."""
        return [ft.name for ft in cls if ft != cls.NORMAL]

    @classmethod
    def count(cls) -> int:
        """Return total number of fault types including NORMAL."""
        return len(cls)


class FaultConfig(BaseModel):
    """Configuration for a specific fault type.

    Attributes:
        fault_type: The type of fault.
        transition_prob: Probability of transitioning from NORMAL to this fault.
        average_duration: Expected duration in timesteps before returning to NORMAL.
        params: Fault-specific parameters (e.g., magnitude for spike).
    """

    model_config = ConfigDict(frozen=True, use_enum_values=False)

    fault_type: FaultType
    transition_prob: float = Field(default=0.02, ge=0.0, le=1.0)
    average_duration: int = Field(default=10, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)

    def return_prob(self) -> float:
        """Probability of returning to NORMAL at each timestep."""
        return 1.0 / self.average_duration

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fault_type": self.fault_type.name,
            "transition_prob": self.transition_prob,
            "average_duration": self.average_duration,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FaultConfig":
        """Reconstruct from dictionary."""
        return cls(
            fault_type=FaultType.from_string(data["fault_type"]),
            transition_prob=data["transition_prob"],
            average_duration=data["average_duration"],
            params=data.get("params", {}),
        )


class MarkovConfig(BaseModel):
    """Configuration for the Markov chain state generator.

    Attributes:
        fault_configs: List of fault configurations (excluding NORMAL).
        seed: Random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    fault_configs: list[FaultConfig] = Field(default_factory=list)
    seed: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _set_default_configs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("fault_configs"):
                data = dict(data)
                data["fault_configs"] = cls._default_fault_configs()
        return data

    @staticmethod
    def _default_fault_configs() -> list["FaultConfig"]:
        """Return default fault configurations."""
        return [
            FaultConfig(
                fault_type=FaultType.SPIKE,
                transition_prob=0.02,
                average_duration=2,
                params={"magnitude_range": (5.0, 10.0)},
            ),
            FaultConfig(
                fault_type=FaultType.DRIFT,
                transition_prob=0.001,
                average_duration=40,
                params={"drift_rate": 0.1},
            ),
            FaultConfig(
                fault_type=FaultType.STUCK,
                transition_prob=0.0025,
                average_duration=15,
                params={},
            ),
        ]

    def get_config(self, fault_type: FaultType) -> FaultConfig | None:
        """Get configuration for a specific fault type."""
        for cfg in self.fault_configs:
            if cfg.fault_type == fault_type:
                return cfg
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "seed": self.seed,
            "fault_configs": [cfg.to_dict() for cfg in self.fault_configs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarkovConfig":
        """Reconstruct from dictionary."""
        fault_configs = [
            FaultConfig.from_dict(fc) for fc in data.get("fault_configs", [])
        ]
        return cls(
            fault_configs=fault_configs if fault_configs else [],
            seed=data.get("seed"),
        )


class WindowConfig(BaseModel):
    """Configuration for sliding window dataset creation.

    Attributes:
        window_size: Number of timesteps per window.
        train_stride: Stride for training windows (allows overlap).
        test_stride: Stride for testing windows (typically no overlap).
        train_ratio: Fraction of data for training (chronological split).
    """

    model_config = ConfigDict(frozen=True)

    window_size: int = Field(default=60, ge=1)
    train_stride: int = Field(default=10, ge=1)
    test_stride: int = Field(default=60, ge=1)
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "window_size": self.window_size,
            "train_stride": self.train_stride,
            "test_stride": self.test_stride,
            "train_ratio": self.train_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowConfig":
        """Reconstruct from dictionary."""
        return cls(
            window_size=data.get("window_size", 60),
            train_stride=data.get("train_stride", 10),
            test_stride=data.get("test_stride", 60),
            train_ratio=data.get("train_ratio", 0.8),
        )
