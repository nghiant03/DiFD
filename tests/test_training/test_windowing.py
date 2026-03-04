"""Tests for windowing and 3-way chronological split."""

import numpy as np
import pandas as pd
import pytest

from DiFD.datasets import InjectedDataset
from DiFD.schema import InjectionConfig
from DiFD.schema.types import WindowConfig
from DiFD.training.windowing import prepare_data


def _make_injected_dataset(
    n_timesteps: int = 500,
    n_groups: int = 2,
    n_features: int = 2,
    seed: int = 0,
) -> InjectedDataset:
    rng = np.random.default_rng(seed)
    rows = []
    feature_names = [f"f{i}" for i in range(n_features)]
    for g in range(1, n_groups + 1):
        for t in range(n_timesteps):
            row = {"moteid": g}
            for f in feature_names:
                row[f] = float(rng.standard_normal())
            row["fault_state"] = int(rng.integers(0, 4))
            rows.append(row)

    df = pd.DataFrame(rows)
    config = InjectionConfig(
        target_features=feature_names,
        all_features=feature_names,
        group_column="moteid",
    )
    return InjectedDataset(df=df, config=config, feature_names=feature_names)


class TestThreeWaySplit:
    def test_returns_six_arrays(self) -> None:
        ds = _make_injected_dataset()
        result = prepare_data(ds)
        assert len(result) == 6

    def test_no_overlap_between_splits(self) -> None:
        ds = _make_injected_dataset(n_timesteps=500, n_groups=1, seed=42)
        wc = WindowConfig(
            window_size=10,
            train_stride=5,
            test_stride=10,
            train_ratio=0.8,
            val_ratio=0.2,
        )
        X_train, _, X_val, _, X_test, _ = prepare_data(ds, window_config=wc)

        train_set = {tuple(w.flatten()) for w in X_train}
        val_set = {tuple(w.flatten()) for w in X_val}
        test_set = {tuple(w.flatten()) for w in X_test}

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_val_ratio_zero_gives_empty_val(self) -> None:
        ds = _make_injected_dataset(n_timesteps=200, n_groups=1)
        wc = WindowConfig(
            window_size=10,
            train_stride=5,
            test_stride=10,
            train_ratio=0.8,
            val_ratio=0.0,
        )
        _, _, X_val, y_val, _, _ = prepare_data(ds, window_config=wc)
        assert len(X_val) == 0
        assert len(y_val) == 0

    def test_chronological_ordering(self) -> None:
        ds = _make_injected_dataset(n_timesteps=300, n_groups=1, n_features=1, seed=7)
        wc = WindowConfig(
            window_size=10,
            train_stride=10,
            test_stride=10,
            train_ratio=0.8,
            val_ratio=0.1,
        )
        X_train, _, X_val, _, X_test, _ = prepare_data(ds, window_config=wc)

        if len(X_train) > 0 and len(X_val) > 0:
            last_train_val = X_train[-1, -1, 0]
            first_val_val = X_val[0, 0, 0]
            df = ds.df[ds.feature_names].to_numpy(dtype=np.float32).flatten()
            train_last_idx = np.where(df == last_train_val)[0]
            val_first_idx = np.where(df == first_val_val)[0]
            assert train_last_idx.max() < val_first_idx.min()

        if len(X_val) > 0 and len(X_test) > 0:
            last_val_val = X_val[-1, -1, 0]
            first_test_val = X_test[0, 0, 0]
            df = ds.df[ds.feature_names].to_numpy(dtype=np.float32).flatten()
            val_last_idx = np.where(df == last_val_val)[0]
            test_first_idx = np.where(df == first_test_val)[0]
            assert val_last_idx.max() < test_first_idx.min()

    def test_all_splits_nonempty_with_enough_data(self) -> None:
        ds = _make_injected_dataset(n_timesteps=500, n_groups=1)
        wc = WindowConfig(
            window_size=10,
            train_stride=5,
            test_stride=10,
            train_ratio=0.8,
            val_ratio=0.1,
        )
        X_train, _, X_val, _, X_test, _ = prepare_data(ds, window_config=wc)
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0

    def test_shapes_consistent(self) -> None:
        ds = _make_injected_dataset(n_timesteps=300, n_groups=2, n_features=3)
        wc = WindowConfig(
            window_size=15,
            train_stride=5,
            test_stride=15,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(ds, window_config=wc)

        for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
            if len(X) > 0:
                assert X.shape[1] == 15
                assert X.shape[2] == 3
                assert y.shape[1] == 15
                assert X.shape[0] == y.shape[0]
