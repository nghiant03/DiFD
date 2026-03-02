"""Tests for fault injection module."""

import numpy as np
import pytest

from DiFD.schema import FaultType, MarkovConfig
from DiFD.injection import (
    DriftFaultInjector,
    MarkovStateGenerator,
    SpikeFaultInjector,
    StuckFaultInjector,
    get_injector,
)


class TestMarkovStateGenerator:
    """Tests for MarkovStateGenerator."""

    def test_generate_length(self) -> None:
        gen = MarkovStateGenerator(MarkovConfig(seed=42), np.random.default_rng(42))
        states = gen.generate(1000)
        assert len(states) == 1000
        assert states.dtype == np.int32

    def test_valid_states(self) -> None:
        gen = MarkovStateGenerator(MarkovConfig(seed=42), np.random.default_rng(42))
        states = gen.generate(1000)
        unique_states = set(states)
        valid_states = {ft.value for ft in FaultType}
        assert unique_states.issubset(valid_states)

    def test_no_direct_fault_transitions(self) -> None:
        """Verify no direct transitions between fault states."""
        gen = MarkovStateGenerator(MarkovConfig(seed=42), np.random.default_rng(42))
        states = gen.generate(10000)
        for i in range(1, len(states)):
            prev, curr = states[i - 1], states[i]
            if prev != FaultType.NORMAL.value and curr != FaultType.NORMAL.value:
                assert prev == curr, "Direct transition between different faults"

    def test_reproducible_with_seed(self) -> None:
        gen1 = MarkovStateGenerator(MarkovConfig(seed=123), np.random.default_rng(123))
        gen2 = MarkovStateGenerator(MarkovConfig(seed=123), np.random.default_rng(123))
        states1 = gen1.generate(100)
        states2 = gen2.generate(100)
        assert np.array_equal(states1, states2)


class TestSpikeFaultInjector:
    """Tests for SpikeFaultInjector."""

    def test_apply_modifies_masked_values(self) -> None:
        injector = SpikeFaultInjector()
        data = np.zeros(10, dtype=np.float64)
        mask = np.array([False, False, True, True, False] * 2)
        params = {"magnitude_range": (5.0, 10.0)}
        rng = np.random.default_rng(42)

        result = injector.apply(data, mask, params, rng)

        assert np.all(result[~mask] == 0.0)
        assert np.all(np.abs(result[mask]) >= 5.0)


class TestDriftFaultInjector:
    """Tests for DriftFaultInjector."""

    def test_apply_creates_trend(self) -> None:
        injector = DriftFaultInjector()
        data = np.zeros(10, dtype=np.float64)
        mask = np.array([False, False, True, True, True, True, False, False, False, False])
        params = {"drift_rate": 1.0}
        rng = np.random.default_rng(42)

        result = injector.apply(data, mask, params, rng)

        drifted = result[mask]
        assert len(set(np.abs(drifted))) > 1


class TestStuckFaultInjector:
    """Tests for StuckFaultInjector."""

    def test_apply_freezes_value(self) -> None:
        injector = StuckFaultInjector()
        data = np.arange(10, dtype=np.float64)
        mask = np.array([False, False, True, True, True, False, False, False, False, False])
        rng = np.random.default_rng(42)

        result = injector.apply(data, mask, {}, rng)

        stuck_values = result[mask]
        assert np.all(stuck_values == stuck_values[0])
        assert stuck_values[0] == 2.0


class TestRegistry:
    """Tests for fault registry."""

    def test_get_injector(self) -> None:
        for fault_type in [FaultType.SPIKE, FaultType.DRIFT, FaultType.STUCK]:
            injector = get_injector(fault_type)
            assert injector is not None

    def test_get_unknown_injector(self) -> None:
        with pytest.raises(KeyError):
            get_injector(FaultType.NORMAL)
