"""
SPring-8 / SACLA-specific analysis tools for XRDpy.

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
    from XRDpy.analysis.Spring8_SACLA import datared
    from XRDpy.analysis.Spring8_SACLA import azimint
"""

from importlib import import_module

__all__ = [
    "datared",
    "azimint",
    "pbs",
]


def __getattr__(name: str):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)