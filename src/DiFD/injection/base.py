"""Base class for fault injectors."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseFaultInjector(ABC):
    """Abstract base class for fault injection strategies.

    Each fault type should implement this interface to define how
    faults are applied to sensor data.
    """

    @property
    @abstractmethod
    def fault_name(self) -> str:
        """Return the name of this fault type."""
        ...

    @abstractmethod
    def apply(
        self,
        data: NDArray[np.float64],
        mask: NDArray[np.bool_],
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Apply fault to data where mask is True.

        Args:
            data: Original sensor data array (will be modified in place).
            mask: Boolean mask indicating where this fault is active.
            params: Fault-specific parameters from FaultConfig.
            rng: Random number generator for reproducibility.

        Returns:
            Modified data array.
        """
        ...

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
