"""
MAX IV / FemtoMAX-specific analysis tools for trxrdpy.

This subpackage contains beamline-specific logic for:
- data reduction from FemtoMAX raw data
- facility-facing azimuthal-integration wrappers for the shared 2D workflow

Public modules
--------------
- datared : user-facing data-reduction API
- azimint : user-facing azimuthal-integration helpers / wrappers

Internal modules
----------------
- datared_utils : implementation details used by datared
"""

from . import datared
from . import azimint

__all__ = [
    "datared",
    "azimint",
]