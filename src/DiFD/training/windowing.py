"""Windowing and train/val/test splitting utilities.

Converts an InjectedDataset (DataFrame) into windowed numpy arrays
suitable for model training and evaluation.

The split is **chronological per group** to prevent information leakage:
  1. First ``train_ratio`` fraction → train
  2. Last ``val_ratio`` fraction of the train portion → val
  3. Remainder after ``train_ratio`` → test

Windows are created *after* the split so overlapping strides cannot
leak across partitions.
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
    features: list[str] | None = None,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
]:
    """Convert an InjectedDataset into windowed train/val/test arrays.

    For each group in the DataFrame:
        1. Chronologically split into train/val/test
        2. Extract sliding windows with appropriate stride

    The validation set is carved from the *end* of the training portion
    so that no overlapping windows can leak between splits.

    Args:
        dataset: InjectedDataset containing the injected DataFrame.
        window_config: Windowing configuration. Falls back to ``dataset.config.window``.
        features: Subset of feature names to use. When ``None`` (default),
            all features from ``dataset.feature_names`` are used.

    Returns:
        Tuple of ``(X_train, y_train, X_val, y_val, X_test, y_test)`` numpy arrays.

    Raises:
        ValueError: If any name in *features* is not in the dataset.
    """
    wc = window_config if window_config is not None else dataset.config.window

    df = dataset.df
    if features is not None:
        unknown = set(features) - set(dataset.feature_names)
        if unknown:
            msg = f"Unknown features: {sorted(unknown)}. Available: {dataset.feature_names}"
            raise ValueError(msg)
        selected_features = list(features)
    else:
        selected_features = dataset.feature_names
    group_col = dataset.group_column

    train_X_parts: list[NDArray[np.float32]] = []
    train_y_parts: list[NDArray[np.int32]] = []
    val_X_parts: list[NDArray[np.float32]] = []
    val_y_parts: list[NDArray[np.int32]] = []
    test_X_parts: list[NDArray[np.float32]] = []
    test_y_parts: list[NDArray[np.int32]] = []

    groups = df.groupby(group_col) if group_col in df.columns else [(None, df)]

    for _, group_df in groups:
        group_features = group_df[selected_features].to_numpy(dtype=np.float32)
        group_labels = group_df["fault_state"].to_numpy(dtype=np.int32)

        n = len(group_features)
        train_end = int(n * wc.train_ratio)

        if wc.val_ratio > 0:
            val_len = int(train_end * wc.val_ratio)
            val_start = train_end - val_len
        else:
            val_start = train_end

        train_data = group_features[:val_start]
        train_labels = group_labels[:val_start]

        val_data = group_features[val_start:train_end]
        val_labels = group_labels[val_start:train_end]

        test_data = group_features[train_end:]
        test_labels = group_labels[train_end:]

        X_tr, y_tr = _create_windows(train_data, train_labels, wc.window_size, wc.train_stride)
        X_va, y_va = _create_windows(val_data, val_labels, wc.window_size, wc.test_stride)
        X_te, y_te = _create_windows(test_data, test_labels, wc.window_size, wc.test_stride)

        if len(X_tr) > 0:
            train_X_parts.append(X_tr)
            train_y_parts.append(y_tr)
        if len(X_va) > 0:
            val_X_parts.append(X_va)
            val_y_parts.append(y_va)
        if len(X_te) > 0:
            test_X_parts.append(X_te)
            test_y_parts.append(y_te)

    n_feat = len(selected_features)
    X_train = np.concatenate(train_X_parts) if train_X_parts else np.empty((0, wc.window_size, n_feat), dtype=np.float32)
    y_train = np.concatenate(train_y_parts) if train_y_parts else np.empty((0, wc.window_size), dtype=np.int32)
    X_val = np.concatenate(val_X_parts) if val_X_parts else np.empty((0, wc.window_size, n_feat), dtype=np.float32)
    y_val = np.concatenate(val_y_parts) if val_y_parts else np.empty((0, wc.window_size), dtype=np.int32)
    X_test = np.concatenate(test_X_parts) if test_X_parts else np.empty((0, wc.window_size, n_feat), dtype=np.float32)
    y_test = np.concatenate(test_y_parts) if test_y_parts else np.empty((0, wc.window_size), dtype=np.int32)

    return X_train, y_train, X_val, y_val, X_test, y_test
