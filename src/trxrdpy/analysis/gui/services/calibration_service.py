"""
Calibration service for the analysis GUI.

This module contains backend-facing calibration helpers that do not construct
Qt widgets.
"""
from __future__ import annotations

try:
    from trxrdpy.analysis import calibration
except Exception:
    calibration = None


import shutil
from pathlib import Path
from typing import Optional


class CalibrationService:
    """Service layer for calibration-related analysis operations."""

    def find_pyfai_calib2(self) -> Optional[str]:
        """Return the path to pyFAI-calib2 if available in the current environment."""

        return shutil.which("pyFAI-calib2")

    def build_pyfai_calib2_command(self, image_path: Optional[str] = None):
        """Build the command used to launch pyFAI-calib2.

        Parameters
        ----------
        image_path:
            Optional 2D image path. If empty, pyFAI-calib2 is launched without
            an input image.
        """

        exe = self.find_pyfai_calib2()

        if exe is None:
            raise FileNotFoundError(
                "Could not find 'pyFAI-calib2' in the current environment.\n"
                "Make sure pyFAI is installed in the same Python environment "
                "used to launch this GUI."
            )

        image_path = (image_path or "").strip()

        if image_path:
            image_path = str(Path(image_path).expanduser())
            args = [image_path]
        else:
            args = []

        return exe, args

    def ensure_backend(self):
        """Import and return the backend module, raising a focused dependency error on failure."""
        if calibration is None:
            raise ImportError("Calibration backend is not available.")

        return calibration

    def compute_xy_files(self, **kwargs):
        """Compute XY pattern files."""
        backend = self.ensure_backend()
        return backend.compute_xy_files(**kwargs)

    def do_peak_fitting(self, **kwargs):
        """Perform peak fitting."""
        backend = self.ensure_backend()
        return backend.do_peak_fitting(**kwargs)

    def plot_caked_1d_patterns(self, **kwargs):
        """Plot caked 1d patterns."""
        backend = self.ensure_backend()
        return backend.plot_caked_1D_patterns(**kwargs)

    def plot_detector_and_cake(self, **kwargs):
        """Plot a bare detector image beside its two-dimensional cake."""
        backend = self.ensure_backend()
        return backend.plot_detector_and_cake(**kwargs)

    def plot_property_vs_azimuth(self, **kwargs):
        """Plot property vs azimuth."""
        backend = self.ensure_backend()
        return backend.plot_property_vs_azimuth(**kwargs)
