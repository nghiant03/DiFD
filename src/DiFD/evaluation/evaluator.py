"""Evaluator for fault diagnosis models.

Runs inference on a dataset and computes classification metrics.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        y_true: Ground truth labels ``(total_timesteps,)``.
        y_pred: Predicted labels ``(total_timesteps,)``.
        y_prob: Predicted class probabilities ``(total_timesteps, num_classes)``.
    """

    loss: float
    accuracy: float
    macro_f1: float
    class_metrics: ClassMetrics
    y_true: NDArray[np.int32] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    y_pred: NDArray[np.int32] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    y_prob: NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))

    def save(
        self,
        path: str | Path,
        train_config: dict[str, Any] | None = None,
        injection_config: dict[str, Any] | None = None,
    ) -> None:
        """Save evaluation results, predictions, and configs to a directory.

        Writes:
            - ``eval_metrics.json``: aggregate and per-class metrics + configs
            - ``predictions.npz``: y_true, y_pred, y_prob arrays

        Args:
            path: Directory to save into (created if needed).
            train_config: Training config dict to embed.
            injection_config: Injection config dict to embed.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        names = FaultType.names()
        per_class = {}
        for i, name in enumerate(names):
            if i < len(self.class_metrics.precision):
                per_class[name] = {
                    "precision": self.class_metrics.precision[i],
                    "recall": self.class_metrics.recall[i],
                    "f1": self.class_metrics.f1[i],
                    "support": self.class_metrics.support[i],
                }

        metrics_dict: dict[str, Any] = {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": per_class,
        }
        if train_config is not None:
            metrics_dict["train_config"] = train_config
        if injection_config is not None:
            metrics_dict["injection_config"] = injection_config

        (directory / "eval_metrics.json").write_text(json.dumps(metrics_dict, indent=2))

        prob_columns = [f"prob_{n.lower()}" for n in names]
        with (directory / "predictions.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["y_true", "y_pred"] + prob_columns)
            for idx in range(len(self.y_true)):
                true_name = names[self.y_true[idx]] if self.y_true[idx] < len(names) else str(self.y_true[idx])
                pred_name = names[self.y_pred[idx]] if self.y_pred[idx] < len(names) else str(self.y_pred[idx])
                probs = [f"{p:.6f}" for p in self.y_prob[idx]] if idx < len(self.y_prob) else []
                writer.writerow([true_name, pred_name] + probs)

    @classmethod
    def load(cls, path: str | Path) -> EvalResult:
        """Load evaluation results from a directory.

        Args:
            path: Directory containing ``eval_metrics.json`` and ``predictions.npz``.

        Returns:
            Reconstructed EvalResult.
        """
        directory = Path(path)
        meta = json.loads((directory / "eval_metrics.json").read_text())

        per_class = meta.get("per_class", {})
        names = FaultType.names()
        precision = [per_class.get(n, {}).get("precision", 0.0) for n in names]
        recall = [per_class.get(n, {}).get("recall", 0.0) for n in names]
        f1_scores = [per_class.get(n, {}).get("f1", 0.0) for n in names]
        support = [per_class.get(n, {}).get("support", 0) for n in names]

        csv_path = directory / "predictions.csv"
        npz_path = directory / "predictions.npz"
        if csv_path.exists():
            name_to_idx = {ft.name: ft.value for ft in FaultType}
            with csv_path.open(newline="") as f:
                reader = csv.reader(f)
                next(reader)
                rows = list(reader)
            if rows:
                y_true = np.array(
                    [name_to_idx[r[0]] if r[0] in name_to_idx else int(r[0]) for r in rows], dtype=np.int32
                )
                y_pred = np.array(
                    [name_to_idx[r[1]] if r[1] in name_to_idx else int(r[1]) for r in rows], dtype=np.int32
                )
                y_prob = np.array(
                    [[float(v) for v in r[2:]] for r in rows], dtype=np.float32
                )
            else:
                y_true = np.empty(0, dtype=np.int32)
                y_pred = np.empty(0, dtype=np.int32)
                y_prob = np.empty((0, 0), dtype=np.float32)
        elif npz_path.exists():
            preds = np.load(npz_path)
            y_true = preds["y_true"]
            y_pred = preds["y_pred"]
            y_prob = preds["y_prob"]
        else:
            y_true = np.empty(0, dtype=np.int32)
            y_pred = np.empty(0, dtype=np.int32)
            y_prob = np.empty((0, 0), dtype=np.float32)

        return cls(
            loss=meta["loss"],
            accuracy=meta["accuracy"],
            macro_f1=meta["macro_f1"],
            class_metrics=ClassMetrics(
                precision=precision,
                recall=recall,
                f1=f1_scores,
                support=support,
            ),
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )


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
            :class:`EvalResult` with loss, accuracy, per-class metrics, and predictions.
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
        all_probs: list[torch.Tensor] = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = model(X_batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=-1)
            probs = torch.softmax(logits, dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()

            all_preds.append(preds.detach().cpu().reshape(-1))
            all_targets.append(y_batch.detach().cpu().reshape(-1))
            all_probs.append(probs.detach().cpu().reshape(-1, num_classes))

        avg_loss = total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
        accuracy = correct / max(total, 1)
        class_metrics = compute_class_metrics(all_preds, all_targets, num_classes)
        f1 = macro_f1(class_metrics)

        y_true = torch.cat(all_targets).numpy().astype(np.int32)
        y_pred = torch.cat(all_preds).numpy().astype(np.int32)
        y_prob = torch.cat(all_probs).numpy().astype(np.float32)

        return EvalResult(
            loss=avg_loss,
            accuracy=accuracy,
            macro_f1=f1,
            class_metrics=class_metrics,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
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
