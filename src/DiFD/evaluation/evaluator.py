"""Evaluator for fault diagnosis models.

Runs inference on a dataset and computes classification metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from DiFD.logging import logger
from DiFD.models.base import BaseModel
from DiFD.schema import EvaluateConfig
from DiFD.schema.types import FaultType
from DiFD.training.callbacks import ClassMetrics

from .metrics import compute_class_metrics, macro_f1


@dataclass
class EvalResult:
    """Result container returned after evaluation.

    Attributes:
        loss: Average loss over the evaluation set.
        accuracy: Overall accuracy.
        macro_f1: Macro-averaged F1.
        class_metrics: Per-class precision, recall, F1, and support.
    """

    loss: float
    accuracy: float
    macro_f1: float
    class_metrics: ClassMetrics


class Evaluator:
    """Evaluates a fault-diagnosis model on a dataset.

    Args:
        config: Evaluation configuration.
        device: PyTorch device string. ``None`` auto-selects CUDA if available.
    """

    def __init__(
        self,
        config: EvaluateConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config if config is not None else EvaluateConfig()
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: BaseModel,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        criterion: nn.Module | None = None,
    ) -> EvalResult:
        """Evaluate the model on the given data.

        Args:
            model: Trained model to evaluate.
            X: Feature array ``(N, seq_len, features)``.
            y: Label array ``(N, seq_len)``.
            criterion: Loss function. Defaults to ``CrossEntropyLoss``.

        Returns:
            :class:`EvalResult` with loss, accuracy, and per-class metrics.
        """
        model = model.to(self.device)
        model.eval()

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        loader = self._make_loader(X, y)

        num_classes = model(
            torch.zeros(1, X.shape[1], X.shape[2], device=self.device)
        ).size(-1)

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
        class_metrics = compute_class_metrics(all_preds, all_targets, num_classes)
        f1 = macro_f1(class_metrics)

        return EvalResult(
            loss=avg_loss,
            accuracy=accuracy,
            macro_f1=f1,
            class_metrics=class_metrics,
        )

    def log_results(self, result: EvalResult, split_name: str = "Test") -> None:
        """Log evaluation results.

        Args:
            result: Evaluation result to log.
            split_name: Name of the data split (e.g. "Test", "Validation").
        """
        logger.info(
            "{}: loss={:.4f} | acc={:.4f} | f1={:.4f}",
            split_name,
            result.loss,
            result.accuracy,
            result.macro_f1,
        )
        names = FaultType.names()
        cm = result.class_metrics
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
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        """Create a DataLoader from numpy arrays."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
