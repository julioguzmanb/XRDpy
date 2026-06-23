"""GUI-facing adapters for simulation backend metadata."""
from __future__ import annotations
from .simulation_service import GeometryInfo, MotorInfo, SimulationService
from .path_service import SimulationPathService

__all__ = [
    "GeometryInfo",
    "MotorInfo",
    "SimulationService",
    "SimulationPathService",
]
