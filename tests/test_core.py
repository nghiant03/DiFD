"""Tests for schema types, configuration, and InjectedDataset."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from DiFD.datasets import InjectedDataset
from DiFD.schema import (
    FaultConfig,
    FaultType,
    InjectionConfig,
    MarkovConfig,
    WindowConfig,
)


class TestFaultType:
    """Tests for FaultType enum."""

    def test_values(self) -> None:
        assert FaultType.NORMAL.value == 0
        assert FaultType.SPIKE.value == 1
        assert FaultType.DRIFT.value == 2
        assert FaultType.STUCK.value == 3

    def test_from_string(self) -> None:
        assert FaultType.from_string("normal") == FaultType.NORMAL
        assert FaultType.from_string("SPIKE") == FaultType.SPIKE

    def test_names(self) -> None:
        names = FaultType.names()
        assert "NORMAL" in names
        assert "SPIKE" in names
        assert len(names) == 4

    def test_count(self) -> None:
        assert FaultType.count() == 4


class TestFaultConfig:
    """Tests for FaultConfig."""

    def test_return_prob(self) -> None:
        cfg = FaultConfig(fault_type=FaultType.SPIKE, average_duration=10)
        assert cfg.return_prob() == pytest.approx(0.1)

    def test_return_prob_zero_duration_raises(self) -> None:
        with pytest.raises(Exception):  # Pydantic ValidationError
            FaultConfig(fault_type=FaultType.SPIKE, average_duration=0)

    def test_to_from_dict(self) -> None:
        cfg = FaultConfig(
            fault_type=FaultType.DRIFT,
            transition_prob=0.05,
            average_duration=30,
            params={"drift_rate": 0.2},
        )
        d = cfg.to_dict()
        restored = FaultConfig.from_dict(d)
        assert restored.fault_type == cfg.fault_type
        assert restored.transition_prob == cfg.transition_prob
        assert restored.params == cfg.params


class TestMarkovConfig:
    """Tests for MarkovConfig."""

    def test_default_fault_configs(self) -> None:
        cfg = MarkovConfig()
        assert len(cfg.fault_configs) == 3
        assert cfg.get_config(FaultType.SPIKE) is not None

    def test_custom_configs(self) -> None:
        custom = [FaultConfig(fault_type=FaultType.SPIKE)]
        cfg = MarkovConfig(fault_configs=custom)
        assert len(cfg.fault_configs) == 1


class TestWindowConfig:
    """Tests for WindowConfig."""

    def test_defaults(self) -> None:
        cfg = WindowConfig()
        assert cfg.window_size == 60
        assert cfg.train_stride == 10
        assert cfg.test_stride == 60
        assert cfg.train_ratio == 0.8
        assert cfg.val_ratio == 0.1


class TestInjectionConfig:
    """Tests for InjectionConfig."""

    def test_seed_propagation(self) -> None:
        cfg = InjectionConfig(seed=123)
        assert cfg.markov.seed == 123

    def test_to_from_dict(self) -> None:
        cfg = InjectionConfig(
            seed=42,
            target_features=["temp", "humid"],
            resample_freq="1min",
        )
        d = cfg.to_dict()
        restored = InjectionConfig.from_dict(d)
        assert restored.seed == cfg.seed
        assert restored.target_features == cfg.target_features
        assert restored.resample_freq == cfg.resample_freq

    def test_dict_contains_fault_mapping(self) -> None:
        cfg = InjectionConfig()
        d = cfg.to_dict()
        assert "fault_type_mapping" in d
        assert d["fault_type_mapping"]["NORMAL"] == 0


def _make_injected_df(n_rows: int = 600, n_groups: int = 3) -> pd.DataFrame:
    """Create a synthetic injected DataFrame for testing."""
    rng = np.random.default_rng(42)
    rows_per_group = n_rows // n_groups
    dfs = []
    for g in range(1, n_groups + 1):
        df = pd.DataFrame({
            "moteid": g,
            "temp": rng.standard_normal(rows_per_group).astype(np.float32),
            "humid": rng.standard_normal(rows_per_group).astype(np.float32),
            "light": rng.standard_normal(rows_per_group).astype(np.float32),
            "volt": rng.standard_normal(rows_per_group).astype(np.float32),
            "fault_state": rng.integers(0, 4, rows_per_group, dtype=np.int32),
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class TestInjectedDataset:
    """Tests for InjectedDataset."""

    def test_save_load(self) -> None:
        config = InjectionConfig(seed=42)
        df = _make_injected_df()
        dataset = InjectedDataset(
            df=df,
            config=config,
            feature_names=["temp", "humid", "light", "volt"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dataset"
            dataset.save(path)

            loaded = InjectedDataset.load(path)

            pd.testing.assert_frame_equal(loaded.df, dataset.df)
            assert loaded.feature_names == dataset.feature_names
            assert loaded.config.seed == config.seed

    def test_properties(self) -> None:
        config = InjectionConfig(seed=42)
        df = _make_injected_df(n_rows=600, n_groups=3)
        dataset = InjectedDataset(
            df=df,
            config=config,
            feature_names=["temp", "humid", "light", "volt"],
        )

        assert dataset.num_groups == 3
        assert dataset.total_timesteps == 600
        assert dataset.num_features == 4

    def test_get_class_weights(self) -> None:
        config = InjectionConfig()
        df = _make_injected_df()
        dataset = InjectedDataset(
            df=df,
            config=config,
            feature_names=["temp", "humid", "light", "volt"],
        )

        weights = dataset.get_class_weights()
        assert FaultType.NORMAL.value in weights
        assert all(w > 0 for w in weights.values())
