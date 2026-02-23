"""Injected dataset container.

This module defines the InjectedDataset class which is the output of
the injection pipeline and input to the training pipeline.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

from DiFD.schema import FaultType, InjectionConfig


@dataclass
class InjectedDataset:
    """Container for the final injected dataset.

    This is the output of the injection pipeline and input to training.

    Attributes:
        X_train: Training features (num_samples, window_size, num_features).
        y_train: Training labels (num_samples, window_size).
        X_test: Test features (num_samples, window_size, num_features).
        y_test: Test labels (num_samples, window_size).
        config: The configuration used to generate this dataset.
        feature_names: Names of features in order.
    """

    X_train: NDArray[np.float32]
    y_train: NDArray[np.int32]
    X_test: NDArray[np.float32]
    y_test: NDArray[np.int32]
    config: InjectionConfig
    feature_names: list[str]

    def save(self, path: str | Path) -> None:
        """Save dataset to .npz file with metadata."""
        path = Path(path) / "injected_dataset.npz"
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            config=json.dumps(self.config.to_dict()),
            feature_names=json.dumps(self.feature_names),
        )

    @classmethod
    def load(cls, path: str | Path) -> "InjectedDataset":
        """Load dataset from .npz file."""
        data = np.load(path, allow_pickle=False)
        config = InjectionConfig.from_dict(json.loads(str(data["config"])))
        feature_names = json.loads(str(data["feature_names"]))

        return cls(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            config=config,
            feature_names=feature_names,
        )

    def print_summary(self) -> None:
        """Print dataset summary statistics using rich formatting."""
        console = Console()

        # Shapes table
        shapes_table = Table(title="Injected Dataset Summary", show_header=True)
        shapes_table.add_column("Array", style="cyan")
        shapes_table.add_column("Shape", style="green")
        shapes_table.add_row("X_train", str(self.X_train.shape))
        shapes_table.add_row("y_train", str(self.y_train.shape))
        shapes_table.add_row("X_test", str(self.X_test.shape))
        shapes_table.add_row("y_test", str(self.y_test.shape))
        console.print(shapes_table)

        console.print(f"\n[bold]Features:[/bold] {self.feature_names}")

        # Class distribution tables
        console.print("\n[bold]Class Distribution (Train):[/bold]")
        console.print(self._build_class_dist_table(self.y_train))
        console.print("\n[bold]Class Distribution (Test):[/bold]")
        console.print(self._build_class_dist_table(self.y_test))

    def _build_class_dist_table(self, y: NDArray[np.int32]) -> Table:
        """Build a rich Table for class distribution."""
        table = Table(show_header=True)
        table.add_column("Fault Type", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")

        flat = y.flatten()
        total = len(flat)
        for ft in FaultType:
            count = int(np.sum(flat == ft.value))
            pct = 100.0 * count / total if total > 0 else 0.0
            table.add_row(ft.name, f"{count:,}", f"{pct:.2f}%")
        return table

    def get_class_weights(self, split: str = "train") -> dict[int, float]:
        """Compute inverse frequency class weights for imbalanced learning.

        Args:
            split: Which split to compute weights for ("train" or "test").

        Returns:
            Dictionary mapping class index to weight.
        """
        y = self.y_train if split == "train" else self.y_test
        flat = y.flatten()
        total = len(flat)
        weights = {}
        for ft in FaultType:
            count = int(np.sum(flat == ft.value))
            if count > 0:
                weights[ft.value] = total / (FaultType.count() * count)
            else:
                weights[ft.value] = 1.0
        return weights
