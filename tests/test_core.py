"""Tests for schema types, configuration, and InjectedDataset."""

import tempfile
from pathlib import Path

import numpy as np
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


class TestInjectedDataset:
    """Tests for InjectedDataset."""

    def test_save_load(self) -> None:
        config = InjectionConfig(seed=42)
        dataset = InjectedDataset(
            X_train=np.random.randn(100, 60, 4).astype(np.float32),
            y_train=np.random.randint(0, 4, (100, 60)).astype(np.int32),
            X_test=np.random.randn(20, 60, 4).astype(np.float32),
            y_test=np.random.randint(0, 4, (20, 60)).astype(np.int32),
            config=config,
            feature_names=["temp", "humid", "light", "volt"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dataset.npz"
            dataset.save(path)

            loaded = InjectedDataset.load(path)

            assert np.allclose(loaded.X_train, dataset.X_train)
            assert np.allclose(loaded.y_train, dataset.y_train)
            assert loaded.feature_names == dataset.feature_names
            assert loaded.config.seed == config.seed

    def test_get_class_weights(self) -> None:
        y_train = np.zeros((10, 60), dtype=np.int32)
        y_train[0, :30] = 1

        dataset = InjectedDataset(
            X_train=np.zeros((10, 60, 4), dtype=np.float32),
            y_train=y_train,
            X_test=np.zeros((2, 60, 4), dtype=np.float32),
            y_test=np.zeros((2, 60), dtype=np.int32),
            config=InjectionConfig(),
            feature_names=["temp", "humid", "light", "volt"],
        )

        weights = dataset.get_class_weights("train")
        assert FaultType.NORMAL.value in weights
        assert weights[FaultType.SPIKE.value] > weights[FaultType.NORMAL.value]
