"""Windowing and train/test splitting utilities.

Converts an InjectedDataset (DataFrame) into windowed numpy arrays
suitable for model training and evaluation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from DiFD.datasets.injected import InjectedDataset
from DiFD.schema.types import WindowConfig


def _create_windows(
    data: NDArray[np.float32],
    labels: NDArray[np.int32],
    window_size: int,
    stride: int,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Create sliding windows from contiguous data.

    Args:
        data: Feature array of shape ``(timesteps, features)``.
        labels: Label array of shape ``(timesteps,)``.
        window_size: Number of timesteps per window.
        stride: Step size between consecutive windows.

    Returns:
        Tuple of ``(X, y)`` where X has shape ``(num_windows, window_size, features)``
        and y has shape ``(num_windows, window_size)``.
    """
    if len(data) < window_size:
        return (
            np.empty((0, window_size, data.shape[1]), dtype=np.float32),
            np.empty((0, window_size), dtype=np.int32),
        )

    starts = list(range(0, len(data) - window_size + 1, stride))
    X = np.stack([data[i : i + window_size] for i in starts])
    y = np.stack([labels[i : i + window_size] for i in starts])
    return X.astype(np.float32), y.astype(np.int32)


def prepare_data(
    dataset: InjectedDataset,
    window_config: WindowConfig | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:
    """Convert an InjectedDataset into windowed train/test arrays.

    For each group in the DataFrame:
        1. Chronologically split into train/test by ``train_ratio``
        2. Extract sliding windows with appropriate stride

    Args:
        dataset: InjectedDataset containing the injected DataFrame.
        window_config: Windowing configuration. Falls back to ``dataset.config.window``.

    Returns:
        Tuple of ``(X_train, y_train, X_test, y_test)`` numpy arrays.
    """
    wc = window_config if window_config is not None else dataset.config.window

    df = dataset.df
    features = dataset.feature_names
    group_col = dataset.group_column

    train_X_parts: list[NDArray[np.float32]] = []
    train_y_parts: list[NDArray[np.int32]] = []
    test_X_parts: list[NDArray[np.float32]] = []
    test_y_parts: list[NDArray[np.int32]] = []

    groups = df.groupby(group_col) if group_col in df.columns else [(None, df)]

    for _, group_df in groups:
        group_features = group_df[features].to_numpy(dtype=np.float32)
        group_labels = group_df["fault_state"].to_numpy(dtype=np.int32)

        split_idx = int(len(group_features) * wc.train_ratio)

        train_data = group_features[:split_idx]
        train_labels = group_labels[:split_idx]
        test_data = group_features[split_idx:]
        test_labels = group_labels[split_idx:]

        X_tr, y_tr = _create_windows(train_data, train_labels, wc.window_size, wc.train_stride)
        X_te, y_te = _create_windows(test_data, test_labels, wc.window_size, wc.test_stride)

        if len(X_tr) > 0:
            train_X_parts.append(X_tr)
            train_y_parts.append(y_tr)
        if len(X_te) > 0:
            test_X_parts.append(X_te)
            test_y_parts.append(y_te)

    X_train = np.concatenate(train_X_parts) if train_X_parts else np.empty((0, wc.window_size, len(features)), dtype=np.float32)
    y_train = np.concatenate(train_y_parts) if train_y_parts else np.empty((0, wc.window_size), dtype=np.int32)
    X_test = np.concatenate(test_X_parts) if test_X_parts else np.empty((0, wc.window_size, len(features)), dtype=np.float32)
    y_test = np.concatenate(test_y_parts) if test_y_parts else np.empty((0, wc.window_size), dtype=np.int32)

    return X_train, y_train, X_test, y_test
