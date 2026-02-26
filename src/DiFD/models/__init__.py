"""Deep learning models for fault diagnosis.

This module provides model architectures and a registry system for
managing different model implementations.
"""

from DiFD.models.base import BaseModel
from DiFD.models.gru import GRUClassifier
from DiFD.models.lstm import LSTMClassifier
from DiFD.models.registry import (
    create_model,
    get_model_class,
    is_registered,
    list_models,
    register_model,
)

__all__ = [
    "BaseModel",
    "GRUClassifier",
    "LSTMClassifier",
    "create_model",
    "get_model_class",
    "is_registered",
    "list_models",
    "register_model",
]
