"""Main fault injection orchestrator.

Coordinates the full injection pipeline: load data, generate states,
apply faults, return injected DataFrame.
"""

import numpy as np
import pandas as pd
from loguru import logger

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
        4. Return InjectedDataset wrapping the injected DataFrame
    """

    def __init__(self, config: InjectionConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.markov_gen = MarkovStateGenerator(config.markov, self.rng)

    def run(self, dataset: BaseDataset) -> InjectedDataset:
        """Execute the full injection pipeline.

        Args:
            dataset: A dataset loader instance.

        Returns:
            InjectedDataset wrapping the injected DataFrame with fault_state column.
        """
        df = dataset.load()
        df = dataset.preprocess(
            df,
            resample_freq=self.config.resample_freq,
            interpolation_method=self.config.interpolation_method,
        )

        df, _ = self._inject_faults(df, dataset.group_column)

        features = [f for f in self.config.all_features if f in df.columns]

        return InjectedDataset(
            df=df,
            config=self.config,
            feature_names=features,
        )

    def _inject_faults(
        self, df: pd.DataFrame, group_column: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate fault states and apply fault injectors."""
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
