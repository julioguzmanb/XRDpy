"""
Shared state for the analysis GUI.

This module contains lightweight state containers only.
It should not perform file I/O, data processing, plotting, or Qt widget construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AnalysisGuiState:
    """Mutable state shared across the analysis GUI.

    The goal is to keep GUI widgets thin and avoid storing workflow state
    directly inside large QWidget classes.

    Attributes
    ----------
    facility : str or None
        Stable facility key selected for the active session.
    path_root : pathlib.Path or None
        Experiment root used to construct raw and analysis paths.
    analysis_subdir, raw_subdir : str
        Processed and raw directory names below ``path_root``.
    femtomax_ping_reference_path : pathlib.Path or None
        Optional custom FemtoMAX scan-to-ping reference table.
    poni_path, mask_edf_path : pathlib.Path or None
        Shared pyFAI geometry and detector-mask files.
    azim_offset_deg : float
        Package-to-pyFAI azimuthal coordinate offset in degrees.
    polarization_enabled : bool
        Whether integration workflows apply polarization correction.
    polarization_factor : float or None
        pyFAI polarization factor when correction is enabled.
    root_path, data_path, output_path, calibration_path : pathlib.Path or None
        Compatibility aliases retained for older saved sessions.
    selected_run, selected_scan : str or None
        Optional session-level run and scan selections.
    metadata, results : dict
        Extensible shared storage for tab metadata and computed results.
    fluence_* : str
        Shared fluence-plot controls synchronized across Viewer,
        Differential, and Fitting tabs.
    delay_* : str
        Shared delay-scan display controls synchronized across Viewer,
        Differential, and Fitting tabs.
    q_norm_range : str
        Shared q-normalization interval synchronized across integration,
        viewer, differential, and fitting controls.
    """

    # Facility
    facility: Optional[str] = None

    # Legacy Session tab paths
    path_root: Optional[Path] = None
    analysis_subdir: str = "analysis"
    raw_subdir: str = ""

    # FemtoMAX session configuration
    femtomax_ping_reference_path: Optional[Path] = None

    # Calibration and shared geometry
    poni_path: Optional[Path] = None
    mask_edf_path: Optional[Path] = None
    azim_offset_deg: float = -90.0
    polarization_enabled: bool = True
    polarization_factor: Optional[float] = 0.99

    # Compatibility aliases for newer internal naming.
    # These can be removed later if we decide on only one naming convention.
    root_path: Optional[Path] = None
    data_path: Optional[Path] = None
    output_path: Optional[Path] = None
    calibration_path: Optional[Path] = None

    # Selected experiment/run metadata
    selected_run: Optional[str] = None
    selected_scan: Optional[str] = None

    # Shared fluence plotting controls
    fluence_delay_fs: str = "0"
    fluence_values: str = "all"
    fluence_ref_type: str = "dark"
    fluence_ref_value: str = "[1466556]"
    fluence_scale: str = "1.0"
    fluence_offset: str = "0"
    fluence_delay_offset_fs: str = "0"
    fluence_delay_display_unit: str = "ps"
    fluence_delay_digits: str = "2"

    # Shared delay-scan display controls
    delay_offset_fs: str = "0"
    delay_display_unit: str = "ps"
    delay_fluence_scale: str = "1.0"
    delay_fluence_offset: str = "0"

    # Shared azimuthal-integration controls
    q_norm_range: str = "(2.65, 2.75)"

    # Generic storage for later tabs
    metadata: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
