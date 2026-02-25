"""Trainer for fault diagnosis models.

Handles the full training loop including:
- Optional oversampling of minority classes
- Configurable loss function (cross-entropy or focal loss)
- Callback-driven logging, early stopping, and checkpointing
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from DiFD.logging import logger
from DiFD.models.base import BaseModel
from DiFD.schema import TrainConfig
from DiFD.training.callbacks import (
    LoggingCallback,
    TrainMetrics,
    TrainingCallback,
)
from DiFD.training.loss import FocalLoss
from DiFD.training.oversampling import oversample_minority


def _build_loss(
    config: TrainConfig,
    device: torch.device,
) -> nn.Module:
    """Build the loss function from config.

    Args:
        config: Training configuration.
        device: Target device for tensors.

    Returns:
        Loss module ready for ``(logits, targets)`` inputs.
    """
    if config.use_focal_loss:
        alpha = (
            torch.tensor(config.focal_alpha, dtype=torch.float32).to(device)
            if config.focal_alpha is not None
            else None
        )
        return FocalLoss(gamma=config.focal_gamma, alpha=alpha)
    return nn.CrossEntropyLoss()


def _prepare_data(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    config: TrainConfig,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Apply oversampling if enabled.

    Args:
        X: Feature array ``(N, seq_len, features)``.
        y: Label array ``(N, seq_len)``.
        config: Training configuration.

    Returns:
        Possibly oversampled ``(X, y)`` tuple.
    """
    if config.oversample:
        return oversample_minority(
            X, y, ratio=config.oversample_ratio, seed=config.seed
        )
    return X, y


@dataclass
class TrainResult:
    """Result container returned after training completes.

    Attributes:
        history: Per-epoch metrics collected during training.
        best_val_loss: Lowest validation loss seen (``None`` if no val data).
        stopped_epoch: Epoch at which training stopped (may be < total if early stopped).
    """

    history: list[TrainMetrics] = field(default_factory=list)
    best_val_loss: float | None = None
    stopped_epoch: int = 0


class Trainer:
    """Trains a fault-diagnosis model.

    Args:
        config: Training configuration (loss, oversampling, hyperparams).
        callbacks: Optional sequence of callbacks. If ``None``, a
            :class:`LoggingCallback` is used by default.
        device: PyTorch device string. ``None`` auto-selects CUDA if available.
    """

    def __init__(
        self,
        config: TrainConfig,
        callbacks: Sequence[TrainingCallback] | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.callbacks: list[TrainingCallback] = (
            list(callbacks) if callbacks is not None else [LoggingCallback()]
        )
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def fit(
        self,
        model: BaseModel,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
    ) -> TrainResult:
        """Train the model.

        Args:
            model: Model instance to train (modified in-place).
            X_train: Training features ``(N, seq_len, features)``.
            y_train: Training labels ``(N, seq_len)``.
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            :class:`TrainResult` with full training history.
        """
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        X_train, y_train = _prepare_data(X_train, y_train, self.config)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = (
            self._make_loader(X_val, y_val, shuffle=False)
            if X_val is not None and y_val is not None
            else None
        )

        model = model.to(self.device)
        criterion = _build_loss(self.config, self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        result = TrainResult()

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = (
                self._eval_epoch(model, val_loader, criterion)
                if val_loader is not None
                else (None, None)
            )

            metrics = TrainMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
            )
            result.history.append(metrics)

            if val_loss is not None and (
                result.best_val_loss is None or val_loss < result.best_val_loss
            ):
                result.best_val_loss = val_loss

            should_continue = all(cb.on_epoch_end(metrics, model) for cb in self.callbacks)
            if not should_continue:
                result.stopped_epoch = epoch
                logger.info("Training stopped at epoch {}", epoch)
                break
        else:
            result.stopped_epoch = self.config.epochs

        return result

    def _make_loader(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        shuffle: bool,
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        """Create a DataLoader from numpy arrays."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def _train_epoch(
        self,
        model: BaseModel,
        loader: DataLoader[tuple[torch.Tensor, ...]],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        """Run one training epoch.

        Returns:
            ``(avg_loss, accuracy)`` over the epoch.
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            logits = model(X_batch)

            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()

        avg_loss = total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: BaseModel,
        loader: DataLoader[tuple[torch.Tensor, ...]],
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Run one evaluation epoch.

        Returns:
            ``(avg_loss, accuracy)`` over the dataset.
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = model(X_batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()

        avg_loss = total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy
