"""Dataset loaders for fault diagnosis.

Provides standardized access to sensor datasets for fault injection.
"""

from DiFD.datasets.base import BaseDataset
from DiFD.datasets.injected import InjectedDataset
from DiFD.datasets.intel_lab import IntelLabDataset
from DiFD.datasets.registry import get_dataset, list_datasets, register_dataset

__all__ = [
    "BaseDataset",
    "InjectedDataset",
    "IntelLabDataset",
    "get_dataset",
    "list_datasets",
    "register_dataset",
]
