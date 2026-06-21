"""
SPring-8 / SACLA-specific analysis tools for trxrdpy.

This subpackage contains beamline-specific code for:
- data reduction
- azimuthal-integration helpers
- PBS / parallel job submission helpers

Notes
-----
`datared` is not imported eagerly because SACLA reduction may require:
- an older Python environment
- SACLA-only dependencies
- VPN / HPC access

Import submodules explicitly when needed, e.g.
    from trxrdpy.analysis.Spring8_SACLA import datared
    from trxrdpy.analysis.Spring8_SACLA import azimint
"""
from __future__ import annotations

from importlib import import_module

__all__ = [
    "datared",
    "azimint",
    "pbs",
]


def __getattr__(name: str):
    """Lazily import and return a SACLA data-reduction attribute."""
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return names exposed by the lazy SACLA module wrapper."""
    return sorted(list(globals().keys()) + __all__)
