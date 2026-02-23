"""Concrete fault injector implementations.

Each class implements a specific fault injection strategy.
New fault types can be added by subclassing BaseFaultInjector
and registering with the fault registry.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from DiFD.injection.base import BaseFaultInjector


class SpikeFaultInjector(BaseFaultInjector):
    """Injects spike faults: sudden large deviations from normal values.

    Parameters:
        magnitude_range: Tuple of (min, max) for random offset.
    """

    @property
    def fault_name(self) -> str:
        return "SPIKE"

    def apply(
        self,
        data: NDArray[np.float64],
        mask: NDArray[np.bool_],
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        magnitude_range = params.get("magnitude_range", (-5.0, 5.0))
        min_mag, max_mag = magnitude_range

        spike_count = int(np.sum(mask))
        if spike_count > 0:
            offsets = rng.uniform(min_mag, max_mag, size=spike_count)
            signs = rng.choice([-1, 1], size=spike_count)
            data[mask] += np.abs(offsets) * signs

        return data


class DriftFaultInjector(BaseFaultInjector):
    """Injects drift faults: gradual linear trend over fault duration.

    Parameters:
        drift_rate: Amount added per timestep within the fault.
    """

    @property
    def fault_name(self) -> str:
        return "DRIFT"

    def apply(
        self,
        data: NDArray[np.float64],
        mask: NDArray[np.bool_],
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        drift_rate = params.get("drift_rate", 0.1)

        indices = np.where(mask)[0]
        if len(indices) == 0:
            return data

        segments = self._find_contiguous_segments(indices)

        for segment in segments:
            direction = rng.choice([-1, 1])
            for i, idx in enumerate(segment):
                data[idx] += direction * drift_rate * (i + 1)

        return data

    @staticmethod
    def _find_contiguous_segments(indices: NDArray[np.intp]) -> list[list[int]]:
        """Split indices into contiguous segments."""
        if len(indices) == 0:
            return []

        segments: list[list[int]] = []
        current_segment: list[int] = [int(indices[0])]

        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                current_segment.append(int(indices[i]))
            else:
                segments.append(current_segment)
                current_segment = [int(indices[i])]

        segments.append(current_segment)
        return segments


class StuckFaultInjector(BaseFaultInjector):
    """Injects stuck faults: value freezes at the start of the fault.

    No additional parameters required.
    """

    @property
    def fault_name(self) -> str:
        return "STUCK"

    def apply(
        self,
        data: NDArray[np.float64],
        mask: NDArray[np.bool_],
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return data

        segments = DriftFaultInjector._find_contiguous_segments(indices)

        for segment in segments:
            stuck_value = data[segment[0]]
            for idx in segment:
                data[idx] = stuck_value

        return data
