"""
Analysis subpackage for trxrdpy.

Structure
---------
- common                : facility-independent analysis logic
- fitting               : user-facing peak-fitting API
- differential_analysis : user-facing differential-analysis API
- MaxIV_FemtoMAX        : MAX IV / FemtoMAX-specific data access and reduction
- Spring8_SACLA         : SPring-8 / SACLA-specific data access and reduction
- ESRF_ID09             : ESRF ID09-specific data access and reduction
"""

from . import common
from . import fitting
from . import differential_analysis
from . import MaxIV_FemtoMAX
from . import Spring8_SACLA
from . import ESRF_ID09

__all__ = [
    "common",
    "fitting",
    "differential_analysis",
    "MaxIV_FemtoMAX",
    "Spring8_SACLA",
    "ESRF_ID09",
]