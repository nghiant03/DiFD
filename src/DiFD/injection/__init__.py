"""Fault injection module for generating synthetic fault datasets.

This module provides tools for injecting faults into sensor data
using a Markov chain model to generate realistic fault sequences.
"""

from DiFD.injection.base import BaseFaultInjector
from DiFD.injection.faults import (
    DriftFaultInjector,
    SpikeFaultInjector,
    StuckFaultInjector,
)
from DiFD.injection.injector import FaultInjector
from DiFD.injection.markov import MarkovStateGenerator
from DiFD.injection.registry import get_injector, register_fault

__all__ = [
    "BaseFaultInjector",
    "SpikeFaultInjector",
    "DriftFaultInjector",
    "StuckFaultInjector",
    "FaultInjector",
    "MarkovStateGenerator",
    "register_fault",
    "get_injector",
]
