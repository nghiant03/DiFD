"""Training callbacks for monitoring and controlling training.

Callbacks are invoked by the Trainer at specific points during the
training loop to provide extensible hooks for logging, early stopping,
checkpointing, and other side effects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from DiFD.logging import logger
from DiFD.models.base import BaseModel


@dataclass
class ClassMetrics:
    """Per-class precision, recall, and F1 score.

    Attributes:
        precision: Per-class precision array.
        recall: Per-class recall array.
        f1: Per-class F1 score array.
        support: Per-class sample count array.
    """

    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]


@dataclass
class TrainMetrics:
    """Metrics collected during a single epoch.

    Attributes:
        epoch: Current epoch number (1-indexed).
        train_loss: Average training loss for the epoch.
        val_loss: Average validation loss for the epoch (None if no val set).
        train_acc: Training accuracy for the epoch.
        val_acc: Validation accuracy for the epoch (None if no val set).
        train_macro_f1: Macro-averaged F1 on training set.
        val_macro_f1: Macro-averaged F1 on validation set (None if no val set).
        train_class_metrics: Per-class metrics on training set (None if not computed).
        val_class_metrics: Per-class metrics on validation set (None if not computed).
    """

    epoch: int
    train_loss: float
    val_loss: float | None = None
    train_acc: float | None = None
    val_acc: float | None = None
    train_macro_f1: float | None = None
    val_macro_f1: float | None = None
    train_class_metrics: ClassMetrics | None = None
    val_class_metrics: ClassMetrics | None = None


class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""

    @abstractmethod
    def on_epoch_end(self, metrics: TrainMetrics, model: BaseModel) -> bool:
        """Called at the end of each epoch.

        Args:
            metrics: Collected metrics for the epoch.
            model: The model being trained.

        Returns:
            True to continue training, False to stop early.
        """
        ...


class LoggingCallback(TrainingCallback):
    """Logs training metrics at each epoch."""

    def on_epoch_end(self, metrics: TrainMetrics, model: BaseModel) -> bool:
        parts = [
            f"Epoch {metrics.epoch}",
            f"train_loss={metrics.train_loss:.4f}",
        ]
        if metrics.train_acc is not None:
            parts.append(f"train_acc={metrics.train_acc:.4f}")
        if metrics.train_macro_f1 is not None:
            parts.append(f"train_f1={metrics.train_macro_f1:.4f}")
        if metrics.val_loss is not None:
            parts.append(f"val_loss={metrics.val_loss:.4f}")
        if metrics.val_acc is not None:
            parts.append(f"val_acc={metrics.val_acc:.4f}")
        if metrics.val_macro_f1 is not None:
            parts.append(f"val_f1={metrics.val_macro_f1:.4f}")

        logger.info(" | ".join(parts))
        return True


@dataclass
class EarlyStoppingCallback(TrainingCallback):
    """Stops training when validation loss stops improving.

    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
    """

    patience: int = 10
    min_delta: float = 1e-4
    _best_loss: float = field(default=float("inf"), init=False, repr=False)
    _counter: int = field(default=0, init=False, repr=False)

    def on_epoch_end(self, metrics: TrainMetrics, model: BaseModel) -> bool:
        val_loss = metrics.val_loss
        if val_loss is None:
            return True

        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                logger.info(
                    "Early stopping triggered after {} epochs without improvement",
                    self.patience,
                )
                return False
        return True


@dataclass
class CheckpointCallback(TrainingCallback):
    """Saves model checkpoint when validation loss improves.

    Saves to a directory with weight.pt and config.json.

    Attributes:
        save_path: Directory path to save the best model checkpoint.
        config_dict: Optional config dictionary to include in checkpoint.
    """

    save_path: str | Path = "best_model"
    config_dict: dict[str, object] | None = None
    _best_loss: float = field(default=float("inf"), init=False, repr=False)

    def on_epoch_end(self, metrics: TrainMetrics, model: BaseModel) -> bool:
        val_loss = metrics.val_loss if metrics.val_loss is not None else metrics.train_loss

        if val_loss < self._best_loss:
            self._best_loss = val_loss
            model.save(self.save_path, config_dict=self.config_dict)
            logger.info("Saved checkpoint to {} (val_loss={:.4f})", self.save_path, val_loss)
        return True
