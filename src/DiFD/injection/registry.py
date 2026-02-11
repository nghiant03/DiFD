"""Fault injector registry.

Provides a central registry for fault injection strategies,
allowing new fault types to be added dynamically.
"""

from DiFD.schema import FaultType
from DiFD.injection.base import BaseFaultInjector
from DiFD.injection.faults import (
    DriftFaultInjector,
    SpikeFaultInjector,
    StuckFaultInjector,
)

_REGISTRY: dict[FaultType, type[BaseFaultInjector]] = {}


def register_fault(fault_type: FaultType, injector_cls: type[BaseFaultInjector]) -> None:
    """Register a fault injector class for a fault type.

    Args:
        fault_type: The fault type enum value.
        injector_cls: The injector class to register.
    """
    _REGISTRY[fault_type] = injector_cls


def get_injector(fault_type: FaultType) -> BaseFaultInjector:
    """Get an instance of the injector for a fault type.

    Args:
        fault_type: The fault type to get injector for.

    Returns:
        Instance of the registered injector.

    Raises:
        KeyError: If no injector is registered for the fault type.
    """
    if fault_type not in _REGISTRY:
        raise KeyError(f"No injector registered for fault type: {fault_type.name}")
    return _REGISTRY[fault_type]()


def get_all_injectors() -> dict[FaultType, BaseFaultInjector]:
    """Get instances of all registered injectors.

    Returns:
        Dictionary mapping fault type to injector instance.
    """
    return {ft: cls() for ft, cls in _REGISTRY.items()}


register_fault(FaultType.SPIKE, SpikeFaultInjector)
register_fault(FaultType.DRIFT, DriftFaultInjector)
register_fault(FaultType.STUCK, StuckFaultInjector)
