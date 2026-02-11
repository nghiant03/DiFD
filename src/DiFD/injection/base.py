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
