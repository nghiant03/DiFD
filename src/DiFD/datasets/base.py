"""Base class for dataset loaders."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseDataset(ABC):
    """Abstract base class for raw dataset loaders.

    Each dataset should implement this interface to provide
    standardized access to sensor data for fault injection.
    """

    def __init__(self, data_path: str | Path | None = None) -> None:
        """Initialize the dataset loader.

        Args:
            data_path: Path to the raw data file or directory.
        """
        if data_path is not None:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name."""
        ...

    @property
    @abstractmethod
    def feature_columns(self) -> list[str]:
        """Return list of available feature column names."""
        ...

    @property
    @abstractmethod
    def group_column(self) -> str:
        """Return the column name used for grouping (e.g., sensor ID)."""
        ...

    @property
    @abstractmethod
    def timestamp_column(self) -> str:
        """Return the column name for timestamps."""
        ...

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and return the raw dataset.

        Returns:
            DataFrame with at minimum: timestamp, group, and feature columns.
        """
        ...

    @abstractmethod
    def preprocess(
        self,
        df: pd.DataFrame,
        resample_freq: str = "30s",
        interpolation_method: str = "linear",
    ) -> pd.DataFrame:
        """Preprocess the dataset: resample and interpolate.

        Args:
            df: Raw dataframe from load().
            resample_freq: Frequency string for resampling (e.g., "30s").
            interpolation_method: Method for interpolation (e.g., "linear").

        Returns:
            Preprocessed DataFrame with regular time intervals.
        """
        ...
