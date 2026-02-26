"""Evaluation metrics for fault diagnosis models.

Provides functions for computing per-class and aggregate metrics
from model predictions.
"""

from __future__ import annotations

import torch

from DiFD.training.callbacks import ClassMetrics


def compute_class_metrics(
    all_preds: list[torch.Tensor],
    all_targets: list[torch.Tensor],
    num_classes: int,
) -> ClassMetrics:
    """Compute per-class precision, recall, and F1 from collected predictions.

    Args:
        all_preds: List of prediction tensors (flattened per batch).
        all_targets: List of target tensors (flattened per batch).
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


def macro_f1(class_metrics: ClassMetrics) -> float:
    """Compute macro-averaged F1 from per-class metrics."""
    if not class_metrics.f1:
        return 0.0
    return sum(class_metrics.f1) / len(class_metrics.f1)
