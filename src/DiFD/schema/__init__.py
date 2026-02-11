"""Schema module for fault diagnosis configuration.

This module exports the fundamental types and configuration classes
shared across all phases: injection, training, and evaluation.
"""

from DiFD.schema.config import InjectionConfig
from DiFD.schema.types import FaultConfig, FaultType, MarkovConfig, WindowConfig

__all__ = [
    "FaultType",
    "FaultConfig",
    "MarkovConfig",
    "WindowConfig",
    "InjectionConfig",
]
