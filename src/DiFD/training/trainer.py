"""Trainer for fault diagnosis models.

Handles the full training loop including:
- Optional oversampling of minority classes
- Configurable loss function (cross-entropy or focal loss)
- Callback-driven logging, early stopping, and checkpointing
- Automatic train/val split when no validation data is provided
- Per-class precision, recall, F1 metrics
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
    ClassMetrics,
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
        logger.debug(
            "Using FocalLoss with gamma={}, alpha={}",
            config.focal_gamma,
            config.focal_alpha,
        )
        return FocalLoss(gamma=config.focal_gamma, alpha=alpha)
    logger.debug("Using CrossEntropyLoss")
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
        logger.debug(
            "Oversampling minority classes with ratio={}, seed={}",
            config.oversample_ratio,
            config.seed,
        )
        X_out, y_out = oversample_minority(
            X, y, ratio=config.oversample_ratio, seed=config.seed
        )
        logger.info(
            "Oversampled: {} -> {} windows",
            len(X),
            len(X_out),
        )
        return X_out, y_out
    return X, y


def _split_train_val(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    val_ratio: float,
    seed: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
]:
    """Split data into train and validation sets.

    Args:
        X: Feature array ``(N, seq_len, features)``.
        y: Label array ``(N, seq_len)``.
        val_ratio: Fraction of data to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        ``(X_train, y_train, X_val, y_val)`` tuple.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    n_val = max(1, int(n * val_ratio))
    indices = rng.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _compute_class_metrics(
    all_preds: list[torch.Tensor],
    all_targets: list[torch.Tensor],
    num_classes: int,
) -> ClassMetrics:
    """Compute per-class precision, recall, and F1 from collected predictions.

    Args:
        all_preds: List of prediction tensors.
        all_targets: List of target tensors.
        num_classes: Number of classes.

    Returns:
        :class:`ClassMetrics` with per-class metrics.
    """
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    precision = []
    recall = []
    f1 = []
    support = []

    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision.append(p)
        recall.append(r)
        f1.append(f)
        support.append(int(((targets == c).sum().item())))

    return ClassMetrics(precision=precision, recall=recall, f1=f1, support=support)


def _macro_f1(class_metrics: ClassMetrics) -> float:
    """Compute macro-averaged F1 from per-class metrics."""
    if not class_metrics.f1:
        return 0.0
    return sum(class_metrics.f1) / len(class_metrics.f1)


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

        If no validation data is provided and ``config.val_ratio > 0``,
        the training data is automatically split.

        Args:
            model: Model instance to train (modified in-place).
            X_train: Training features ``(N, seq_len, features)``.
            y_train: Training labels ``(N, seq_len)``.
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            :class:`TrainResult` with full training history.
        """
        logger.info("Setting random seed to {}", self.config.seed)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        logger.debug(
            "Training data shape: X={}, y={}",
            X_train.shape,
            y_train.shape,
        )
        X_train, y_train = _prepare_data(X_train, y_train, self.config)

        if X_val is None and y_val is None and self.config.val_ratio > 0:
            X_train, y_train, X_val, y_val = _split_train_val(
                X_train, y_train, self.config.val_ratio, self.config.seed
            )
            logger.info(
                "Split training data: train={}, val={} (val_ratio={})",
                len(X_train),
                len(X_val),
                self.config.val_ratio,
            )

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = (
            self._make_loader(X_val, y_val, shuffle=False)
            if X_val is not None and y_val is not None
            else None
        )
        logger.debug(
            "Train batches: {}, Val batches: {}",
            len(train_loader),
            len(val_loader) if val_loader is not None else 0,
        )

        model = model.to(self.device)

        num_classes = model(
            torch.zeros(1, X_train.shape[1], X_train.shape[2], device=self.device)
        ).size(-1)
        logger.info("Using device: {}", self.device)
        criterion = _build_loss(self.config, self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        logger.debug("Optimizer: Adam(lr={})", self.config.learning_rate)

        result = TrainResult()

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc, train_cm = self._train_epoch(
                model, train_loader, criterion, optimizer
            )
            train_class_metrics = _compute_class_metrics(
                train_cm[0], train_cm[1], num_classes
            )
            train_macro_f1 = _macro_f1(train_class_metrics)

            val_loss: float | None = None
            val_acc: float | None = None
            val_macro_f1: float | None = None
            val_class_metrics: ClassMetrics | None = None

            if val_loader is not None:
                val_loss, val_acc, val_cm = self._eval_epoch(
                    model, val_loader, criterion
                )
                val_class_metrics = _compute_class_metrics(
                    val_cm[0], val_cm[1], num_classes
                )
                val_macro_f1 = _macro_f1(val_class_metrics)

            metrics = TrainMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                train_macro_f1=train_macro_f1,
                val_macro_f1=val_macro_f1,
                train_class_metrics=train_class_metrics,
                val_class_metrics=val_class_metrics,
            )
            result.history.append(metrics)

            if val_loss is not None and (
                result.best_val_loss is None or val_loss < result.best_val_loss
            ):
                result.best_val_loss = val_loss

            should_continue = all(cb.on_epoch_end(metrics, model) for cb in self.callbacks)
            if not should_continue:
                result.stopped_epoch = epoch
                logger.info("Training stopped early at epoch {}", epoch)
                break
        else:
            result.stopped_epoch = self.config.epochs
            logger.info("Training completed all {} epochs", self.config.epochs)

        self._log_final_metrics(result)
        return result

    def _log_final_metrics(self, result: TrainResult) -> None:
        """Log a summary of final training metrics."""
        if not result.history:
            return

        last = result.history[-1]
        logger.info("--- Training Summary ---")
        logger.info(
            "Stopped at epoch {} | train_loss={:.4f} | train_acc={:.4f} | train_f1={:.4f}",
            result.stopped_epoch,
            last.train_loss,
            last.train_acc if last.train_acc is not None else 0.0,
            last.train_macro_f1 if last.train_macro_f1 is not None else 0.0,
        )
        if last.val_loss is not None:
            logger.info(
                "val_loss={:.4f} | val_acc={:.4f} | val_f1={:.4f} | best_val_loss={:.4f}",
                last.val_loss,
                last.val_acc if last.val_acc is not None else 0.0,
                last.val_macro_f1 if last.val_macro_f1 is not None else 0.0,
                result.best_val_loss if result.best_val_loss is not None else float("nan"),
            )
        if last.val_class_metrics is not None:
            self._log_class_metrics("Validation", last.val_class_metrics)
        elif last.train_class_metrics is not None:
            self._log_class_metrics("Training", last.train_class_metrics)

    @staticmethod
    def _log_class_metrics(split_name: str, cm: ClassMetrics) -> None:
        """Log per-class metrics table."""
        from DiFD.schema.types import FaultType

        names = FaultType.names()
        logger.info("--- {} Per-Class Metrics ---", split_name)
        logger.info("{:<10s}  {:>9s}  {:>9s}  {:>9s}  {:>9s}", "Class", "Precision", "Recall", "F1", "Support")
        for i, name in enumerate(names):
            if i < len(cm.precision):
                logger.info(
                    "{:<10s}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9d}",
                    name,
                    cm.precision[i],
                    cm.recall[i],
                    cm.f1[i],
                    cm.support[i],
                )

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
    ) -> tuple[float, float, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Run one training epoch.

        Returns:
            ``(avg_loss, accuracy, (all_preds, all_targets))`` over the epoch.
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

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

            all_preds.append(preds.detach().cpu().reshape(-1))
            all_targets.append(y_batch.detach().cpu().reshape(-1))

        avg_loss = total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy, (all_preds, all_targets)

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: BaseModel,
        loader: DataLoader[tuple[torch.Tensor, ...]],
        criterion: nn.Module,
    ) -> tuple[float, float, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Run one evaluation epoch.

        Returns:
            ``(avg_loss, accuracy, (all_preds, all_targets))`` over the dataset.
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = model(X_batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()

            all_preds.append(preds.detach().cpu().reshape(-1))
            all_targets.append(y_batch.detach().cpu().reshape(-1))

        avg_loss = total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy, (all_preds, all_targets)
