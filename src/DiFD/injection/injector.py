"""Main fault injection orchestrator.

Coordinates the full injection pipeline: load data, generate states,
apply faults, create windowed dataset.
"""

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from DiFD.datasets import InjectedDataset
from DiFD.datasets.base import BaseDataset
from DiFD.injection.markov import MarkovStateGenerator
from DiFD.injection.registry import get_injector
from DiFD.schema import InjectionConfig


class FaultInjector:
    """Orchestrates the fault injection pipeline.

    Workflow:
        1. Load and preprocess raw dataset
        2. Generate Markov state sequences per group
        3. Apply fault injectors based on states
        4. Create sliding windows with train/test split
        5. Return InjectedDataset with metadata
    """

    def __init__(self, config: InjectionConfig) -> None:
        """Initialize the fault injector.

        Args:
            config: Complete injection configuration.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.markov_gen = MarkovStateGenerator(config.markov)

    def run(self, dataset: BaseDataset) -> InjectedDataset:
        """Execute the full injection pipeline.

        Args:
            dataset: A dataset loader instance.

        Returns:
            InjectedDataset with train/test splits and metadata.
        """
        df = dataset.load()
        df = dataset.preprocess(
            df,
            resample_freq=self.config.resample_freq,
            interpolation_method=self.config.interpolation_method,
        )

        df, states = self._inject_faults(df, dataset.group_column)

        features = [f for f in self.config.all_features if f in df.columns]

        X_train, y_train, X_test, y_test = self._create_windows(
            df, states, features, dataset.group_column
        )

        return InjectedDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=self.config,
            feature_names=features,
        )

    def _inject_faults(
        self, df: pd.DataFrame, group_column: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate fault states and apply fault injectors.

        Args:
            df: Preprocessed DataFrame.
            group_column: Column name for grouping.

        Returns:
            Tuple of (modified DataFrame, fault state Series).
        """
        df = df.copy()
        all_states = np.zeros(len(df), dtype=np.int32)

        for group_id, group_df in df.groupby(group_column):
            indices = group_df.index.to_numpy()
            length = len(indices)

            states = self.markov_gen.generate(length)
            all_states[indices] = states

            for target_feature in self.config.target_features:
                if target_feature not in df.columns:
                    continue

                data = df.loc[indices, target_feature].to_numpy(dtype=np.float64).copy()

                for fault_config in self.config.markov.fault_configs:
                    logger.debug(f"Injecting {fault_config.fault_type} into group {group_id}, feature {target_feature}")
                    fault_type = fault_config.fault_type
                    mask = states == fault_type.value

                    if not np.any(mask):
                        continue

                    injector = get_injector(fault_type)
                    data = injector.apply(data, mask, fault_config.params, self.rng)

                df.loc[indices, target_feature] = data

        df["fault_state"] = all_states
        return df, pd.Series(all_states, index=df.index)

    def _create_windows(
        self,
        df: pd.DataFrame,
        states: pd.Series,
        features: list[str],
        group_column: str,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int32],
        NDArray[np.float32],
        NDArray[np.int32],
    ]:
        """Create sliding windows with train/test split.

        Processes each group separately to avoid mixing data.
        Split is chronological within each group.

        Args:
            df: DataFrame with injected faults.
            states: Fault state series.
            features: Feature columns to include.
            group_column: Column name for grouping.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test).
        """
        window_size = self.config.window.window_size
        train_stride = self.config.window.train_stride
        test_stride = self.config.window.test_stride
        train_ratio = self.config.window.train_ratio

        X_train_list: list[NDArray[np.float32]] = []
        y_train_list: list[NDArray[np.int32]] = []
        X_test_list: list[NDArray[np.float32]] = []
        y_test_list: list[NDArray[np.int32]] = []

        for _, group_df in df.groupby(group_column):
            group_indices = group_df.index.to_numpy()
            group_features = group_df[features].to_numpy(dtype=np.float32)
            group_states = states.loc[group_indices].to_numpy(dtype=np.int32)

            n_samples = len(group_indices)
            if n_samples < window_size:
                continue

            split_idx = int(n_samples * train_ratio)

            train_features = group_features[:split_idx]
            train_states = group_states[:split_idx]
            test_features = group_features[split_idx:]
            test_states = group_states[split_idx:]

            train_X, train_y = self._extract_windows(
                train_features, train_states, window_size, train_stride
            )
            if train_X is not None and train_y is not None:
                X_train_list.append(train_X)
                y_train_list.append(train_y)

            test_X, test_y = self._extract_windows(
                test_features, test_states, window_size, test_stride
            )
            if test_X is not None and test_y is not None:
                X_test_list.append(test_X)
                y_test_list.append(test_y)

        X_train = (
            np.concatenate(X_train_list, axis=0)
            if X_train_list
            else np.empty((0, window_size, len(features)), dtype=np.float32)
        )
        y_train = (
            np.concatenate(y_train_list, axis=0)
            if y_train_list
            else np.empty((0, window_size), dtype=np.int32)
        )
        X_test = (
            np.concatenate(X_test_list, axis=0)
            if X_test_list
            else np.empty((0, window_size, len(features)), dtype=np.float32)
        )
        y_test = (
            np.concatenate(y_test_list, axis=0)
            if y_test_list
            else np.empty((0, window_size), dtype=np.int32)
        )

        return X_train, y_train, X_test, y_test

    @staticmethod
    def _extract_windows(
        features: NDArray[np.float32],
        states: NDArray[np.int32],
        window_size: int,
        stride: int,
    ) -> tuple[NDArray[np.float32] | None, NDArray[np.int32] | None]:
        """Extract sliding windows from a single group's data.

        Args:
            features: Feature array of shape (n_samples, n_features).
            states: State array of shape (n_samples,).
            window_size: Number of timesteps per window.
            stride: Step size between windows.

        Returns:
            Tuple of (X, y) arrays or (None, None) if no windows fit.
        """
        n_samples = len(features)
        if n_samples < window_size:
            return None, None

        n_windows = (n_samples - window_size) // stride + 1
        if n_windows <= 0:
            return None, None

        n_features = features.shape[1]
        X = np.zeros((n_windows, window_size, n_features), dtype=np.float32)
        y = np.zeros((n_windows, window_size), dtype=np.int32)

        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            X[i] = features[start:end]
            y[i] = states[start:end]

        return X, y
