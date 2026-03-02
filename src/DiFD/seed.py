"""Reproducibility utilities for seeding all random number generators."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from DiFD.logging import logger


def seed_everything(seed: int) -> None:
    """Seed all RNGs for reproducibility.

    Seeds Python's :mod:`random`, NumPy's legacy global RNG,
    PyTorch CPU and CUDA generators, and sets environment variables
    for deterministic CuDNN behaviour.

    Args:
        seed: Non-negative integer seed value.
    """
    logger.info("Seeding everything with seed={}", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
