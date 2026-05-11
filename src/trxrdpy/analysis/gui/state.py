"""
Shared state for the analysis GUI.

This module contains lightweight state containers only.
It should not perform file I/O, data processing, plotting, or Qt widget construction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AnalysisGuiState:
    """
    Mutable state shared across the analysis GUI.

    The goal is to keep GUI widgets thin and avoid storing workflow state
    directly inside large QWidget classes.
    """

    # Facility
    facility: Optional[str] = None

    # Legacy Session tab paths
    path_root: Optional[Path] = None
    analysis_subdir: str = "analysis"
    raw_subdir: str = ""

    # Calibration and shared geometry
    poni_path: Optional[Path] = None
    mask_edf_path: Optional[Path] = None
    azim_offset_deg: float = -90.0

    # Compatibility aliases for newer internal naming.
    # These can be removed later if we decide on only one naming convention.
    root_path: Optional[Path] = None
    data_path: Optional[Path] = None
    output_path: Optional[Path] = None
    calibration_path: Optional[Path] = None

    # Selected experiment/run metadata
    selected_run: Optional[str] = None
    selected_scan: Optional[str] = None

    # Generic storage for later tabs
    metadata: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)