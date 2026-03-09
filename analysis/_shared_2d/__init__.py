"""
Shared analysis helpers for beamlines using the common 2D-image workflow.

This subpackage contains reusable user-facing and helper modules for facilities
whose reduced data are organized around homogenized 2D images and shared XY-file
generation.

Current scope
-------------
- azimint : shared azimuthal-integration API for 2D-image-based beamlines

Intended users
--------------
This subpackage is meant to be reused by beamline-specific analysis packages such as:
- XRDpy.analysis.MaxIV_FemtoMAX
- XRDpy.analysis.Spring8_SACLA

Notes
-----
Beamlines with a different reduced-data model may need their own implementation
outside this subpackage.
"""

from . import azimint

__all__ = [
    "azimint",
]