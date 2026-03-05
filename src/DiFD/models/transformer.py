"""Vanilla Transformer model for many-to-many fault classification.

This module implements a standard Transformer encoder architecture with
positional encoding for per-timestep fault diagnosis.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from DiFD.models.base import BaseModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe: torch.Tensor = self.pe  # type: ignore[assignment]
        x = x + pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(BaseModel):
    """Vanilla Transformer encoder for many-to-many sequence classification.

    Architecture:
        Input -> Linear(input_size, d_model) -> PositionalEncoding
        -> N x TransformerEncoderLayer -> LayerNorm -> Dropout
        -> Linear(d_model, num_classes) -> Output

    Args:
        input_size: Number of input features per timestep.
        d_model: Dimension of the transformer hidden states.
        num_layers: Number of encoder layers.
        num_classes: Number of output classes (fault types).
        n_heads: Number of attention heads.
        d_ff: Dimension of the feed-forward layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        n_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_prob = dropout

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    @property
    def name(self) -> str:
        return "transformer"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for many-to-many classification.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Logits tensor of shape (batch, seq_len, num_classes).
        """
        hidden = self.input_proj(x)
        hidden = self.pos_encoding(hidden)
        hidden = self.encoder(hidden)
        hidden = self.dropout_layer(hidden)
        logits = self.fc(hidden)
        return logits

    def get_config(self) -> dict[str, object]:
        """Return model configuration for serialization."""
        return {
            "input_size": self.input_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout_prob,
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> TransformerClassifier:
        """Load model from a saved directory.

        Args:
            path: Path to the model directory.

        Returns:
            Loaded TransformerClassifier instance.
        """
        directory = Path(path)
        meta = BaseModel.load_metadata(directory)
        config = meta["model_config"]
        assert isinstance(config, dict)
        model = cls(
            input_size=int(config["input_size"]),
            d_model=int(config["d_model"]),
            num_layers=int(config["num_layers"]),
            num_classes=int(config["num_classes"]),
            n_heads=int(config["n_heads"]),
            d_ff=int(config["d_ff"]),
            dropout=float(config["dropout"]),
        )
        model.load_state_dict(torch.load(directory / "weight.pt", weights_only=True))
        return model
