"""Model registry for dynamic model lookup.

Provides a central registry for model classes, allowing new architectures
to be added dynamically without modifying core training code.
"""

from typing import Callable

from DiFD.models.autoformer import AutoformerClassifier
from DiFD.models.base import BaseModel
from DiFD.models.gru import GRUClassifier
from DiFD.models.informer import InformerClassifier
from DiFD.models.lstm import LSTMClassifier
from DiFD.models.transformer import TransformerClassifier

ModelFactory = Callable[..., BaseModel]

_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str, model_cls: type[BaseModel]) -> None:
    """Register a model class with a name.

    Args:
        name: Unique name for the model (used in CLI and configs).
        model_cls: The model class to register.

    Raises:
        ValueError: If name is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"Model '{name}' is already registered")
    _REGISTRY[name] = model_cls


def get_model_class(name: str) -> type[BaseModel]:
    """Get a registered model class by name.

    Args:
        name: The registered model name.

    Returns:
        The model class.

    Raises:
        KeyError: If no model is registered with the given name.
    """
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return _REGISTRY[name]


def create_model(name: str, **kwargs: object) -> BaseModel:
    """Create a model instance by name.

    Args:
        name: The registered model name.
        **kwargs: Arguments passed to the model constructor.

    Returns:
        Instantiated model.
    """
    model_cls = get_model_class(name)
    return model_cls(**kwargs)


def list_models() -> list[str]:
    """List all registered model names.

    Returns:
        List of registered model names.
    """
    return list(_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if a model name is registered.

    Args:
        name: The model name to check.

    Returns:
        True if registered, False otherwise.
    """
    return name in _REGISTRY


register_model("lstm", LSTMClassifier)
register_model("gru", GRUClassifier)
register_model("autoformer", AutoformerClassifier)
register_model("transformer", TransformerClassifier)
register_model("informer", InformerClassifier)
