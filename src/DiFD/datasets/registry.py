"""Dataset registry.

Provides a central registry for dataset loaders,
allowing new datasets to be added dynamically.
"""

from pathlib import Path

from DiFD.datasets.base import BaseDataset
from DiFD.datasets.intel_lab import IntelLabDataset

_REGISTRY: dict[str, type[BaseDataset]] = {}


def register_dataset(name: str, dataset_cls: type[BaseDataset]) -> None:
    """Register a dataset loader class.

    Args:
        name: Unique name for the dataset.
        dataset_cls: The dataset class to register.
    """
    _REGISTRY[name.lower()] = dataset_cls


def get_dataset(name: str, data_path: str | Path) -> BaseDataset:
    """Get an instance of a registered dataset loader.

    Args:
        name: Name of the dataset to load.
        data_path: Path to the dataset file.

    Returns:
        Instance of the registered dataset loader.

    Raises:
        KeyError: If no dataset is registered with the given name.
    """
    name_lower = name.lower()
    if name_lower not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(f"Unknown dataset: {name}. Available: {available}")

    return _REGISTRY[name_lower](data_path)


def list_datasets() -> list[str]:
    """Return list of registered dataset names."""
    return list(_REGISTRY.keys())


register_dataset("intel_lab", IntelLabDataset)
