"""Oversampling utilities for imbalanced datasets.

Provides window-level oversampling of minority (non-NORMAL) classes
by duplicating windows that contain at least one non-normal label.
"""

import numpy as np
from numpy.typing import NDArray

from DiFD.logging import logger
from DiFD.schema.types import FaultType


def oversample_minority(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    ratio: float = 1.0,
    seed: int | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Oversample windows containing non-NORMAL labels.

    Each window is classified as "minority" if any timestep in that
    window has a non-NORMAL label.  Minority windows are duplicated
    until the number of minority windows reaches ``ratio * majority_count``.

    Args:
        X: Feature array of shape ``(N, seq_len, features)``.
        y: Label array of shape ``(N, seq_len)``.
        ratio: Target ratio of minority to majority windows.
            1.0 means balanced; 0.5 means half as many minority as majority.
        seed: Random seed for reproducible shuffling of duplicated samples.

    Returns:
        Tuple of ``(X_oversampled, y_oversampled)`` with duplicated windows appended.
    """
    normal_val = FaultType.NORMAL.value

    is_minority = np.any(y != normal_val, axis=1)
    minority_idx = np.where(is_minority)[0]
    majority_idx = np.where(~is_minority)[0]

    n_minority = len(minority_idx)
    n_majority = len(majority_idx)

    if n_minority == 0:
        logger.warning("No minority samples found, skipping oversampling")
        return X, y

    target_minority = int(n_majority * ratio)
    n_to_add = max(0, target_minority - n_minority)

    if n_to_add == 0:
        logger.info("Minority already meets target ratio, no oversampling needed")
        return X, y

    rng = np.random.default_rng(seed)
    extra_idx = rng.choice(minority_idx, size=n_to_add, replace=True)

    X_out = np.concatenate([X, X[extra_idx]], axis=0)
    y_out = np.concatenate([y, y[extra_idx]], axis=0)

    shuffle = rng.permutation(len(X_out))
    X_out = X_out[shuffle]
    y_out = y_out[shuffle]

    logger.info(
        "Oversampled: {} -> {} windows (added {} minority copies)",
        len(X),
        len(X_out),
        n_to_add,
    )

    return X_out, y_out
