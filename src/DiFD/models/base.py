"""Base class for deep learning models.

All model implementations should inherit from BaseModel and implement
the required abstract methods.
"""

import json
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for fault diagnosis models.

    All models must implement:
        - name: Property returning the model's registered name
        - forward: Standard PyTorch forward pass

    The base class provides common utilities for model management.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the registered name of this model."""
        ...

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, seq_len, features).

        Returns:
            Output tensor of shape (batch, seq_len, num_classes) for
            many-to-many classification.
        """
        ...

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str, config_dict: dict[str, object] | None = None) -> None:
        """Save model state dict and optional config to file.

        Args:
            path: Path to save the model checkpoint.
            config_dict: Optional training config to embed in checkpoint.
        """
        checkpoint: dict[str, object] = {
            "model_name": self.name,
            "state_dict": self.state_dict(),
        }
        if config_dict is not None:
            checkpoint["config"] = json.dumps(config_dict)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, **kwargs: object) -> "BaseModel":
        """Load model from checkpoint.

        Args:
            path: Path to the model checkpoint.
            **kwargs: Additional arguments passed to model constructor.

        Returns:
            Loaded model instance.
        """
        checkpoint = torch.load(path, weights_only=True)
        model = cls(**kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @staticmethod
    def load_config(path: str) -> dict[str, object] | None:
        """Load training config from a checkpoint file.

        Args:
            path: Path to the model checkpoint.

        Returns:
            Config dictionary if present, None otherwise.
        """
        checkpoint = torch.load(path, weights_only=True)
        config_json = checkpoint.get("config")
        if config_json is not None:
            return json.loads(config_json)  # type: ignore[no-any-return]
        return None
