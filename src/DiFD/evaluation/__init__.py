"""Evaluation module for fault diagnosis models.

Provides the Evaluator class and metric computation utilities.
"""

from DiFD.evaluation.evaluator import EvalResult, Evaluator
from DiFD.evaluation.metrics import compute_class_metrics, macro_f1

__all__ = [
    "EvalResult",
    "Evaluator",
    "compute_class_metrics",
    "macro_f1",
]
