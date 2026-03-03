"""Training module for fault diagnosis models.

Provides the Trainer class, loss functions, oversampling utilities,
and training callbacks.
"""

from DiFD.training.callbacks import (
    CheckpointCallback,
    ClassMetrics,
    EarlyStoppingCallback,
    LoggingCallback,
    TrainMetrics,
    TrainingCallback,
)
from DiFD.training.loss import FocalLoss
from DiFD.training.oversampling import oversample_minority
from DiFD.training.trainer import TrainResult, Trainer
from DiFD.training.windowing import prepare_data

__all__ = [
    "CheckpointCallback",
    "ClassMetrics",
    "EarlyStoppingCallback",
    "FocalLoss",
    "LoggingCallback",
    "TrainMetrics",
    "TrainResult",
    "Trainer",
    "TrainingCallback",
    "oversample_minority",
    "prepare_data",
]
