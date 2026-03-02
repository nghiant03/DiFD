"""Markov chain state generator for fault injection.

Implements a coupled Markov model where faults cannot overlap:
transitions from one fault type to another must go through NORMAL first.
"""

import numpy as np
from numpy.typing import NDArray

from DiFD.schema import FaultType, MarkovConfig


class MarkovStateGenerator:
    """Generates fault state sequences using a Markov chain.

    The model ensures:
    - From NORMAL: can transition to any fault type with configured probability
    - From any fault: can only transition back to NORMAL (based on average duration)
    - No direct transitions between different fault types (no overlap)
    """

    def __init__(self, config: MarkovConfig, rng: np.random.Generator) -> None:
        """Initialize the generator.

        Args:
            config: Markov chain configuration with fault configs.
            rng: Random number generator for reproducibility.
        """
        self.config = config
        self.rng = rng

        self._fault_types = [cfg.fault_type for cfg in config.fault_configs]
        self._transition_probs = np.array(
            [cfg.transition_prob for cfg in config.fault_configs]
        )
        self._return_probs = np.array(
            [cfg.return_prob() for cfg in config.fault_configs]
        )

    def generate(self, length: int) -> NDArray[np.int32]:
        """Generate a sequence of fault states.

        Args:
            length: Number of timesteps to generate.

        Returns:
            Integer array of shape (length,) with values from FaultType enum.
        """
        states = np.zeros(length, dtype=np.int32)
        current_state = FaultType.NORMAL

        for i in range(length):
            states[i] = current_state

            if current_state == FaultType.NORMAL:
                current_state = self._transition_from_normal()
            else:
                current_state = self._transition_from_fault(current_state)

        return states

    def _transition_from_normal(self) -> FaultType:
        """Determine next state when currently in NORMAL."""
        rand = self.rng.random()
        cumulative = 0.0

        for fault_type, prob in zip(
            self._fault_types, self._transition_probs, strict=True
        ):
            cumulative += prob
            if rand < cumulative:
                return fault_type

        return FaultType.NORMAL

    def _transition_from_fault(self, current: FaultType) -> FaultType:
        """Determine next state when currently in a fault state."""
        fault_idx = self._fault_types.index(current)
        return_prob = self._return_probs[fault_idx]

        if self.rng.random() < return_prob:
            return FaultType.NORMAL
        return current

    def generate_for_groups(
        self, group_lengths: list[int]
    ) -> dict[int, NDArray[np.int32]]:
        """Generate independent state sequences for multiple groups.

        Args:
            group_lengths: List of sequence lengths for each group.

        Returns:
            Dictionary mapping group index to state array.
        """
        return {i: self.generate(length) for i, length in enumerate(group_lengths)}
