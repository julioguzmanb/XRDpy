"""
Fitting service for the analysis GUI.

This module wraps peak-fitting backend calls without constructing Qt widgets.
"""
from __future__ import annotations

try:
    from trxrdpy.analysis import fitting
except Exception:
    fitting = None


class FittingService:
    """
    Service layer for peak fitting and fit-summary plotting workflows.
    """

    def ensure_backend(self):
        if fitting is None:
            raise ImportError("Fitting backend is not available.")

        return fitting

    def run_delay_peak_fitting(self, **kwargs):
        backend = self.ensure_backend()
        return backend.run_delay_peak_fitting(**kwargs)

    def run_fluence_peak_fitting(self, **kwargs):
        backend = self.ensure_backend()
        return backend.run_fluence_peak_fitting(**kwargs)

    def plot_fit_overlay_from_csv(self, **kwargs):
        backend = self.ensure_backend()
        return backend.plot_fit_overlay_from_csv(**kwargs)

    def plot_fit_overlay_from_csv_fluence(self, **kwargs):
        backend = self.ensure_backend()
        return backend.plot_fit_overlay_from_csv_fluence(**kwargs)

    def plot_time_evolution(self, **kwargs):
        backend = self.ensure_backend()
        return backend.plot_time_evolution(**kwargs)

    def plot_fluence_evolution(self, **kwargs):
        backend = self.ensure_backend()
        return backend.plot_fluence_evolution(**kwargs)

    def plot_time_evolution_multi(self, **kwargs):
        backend = self.ensure_backend()
        return backend.plot_time_evolution_multi(**kwargs)

    def plot_fluence_evolution_multi(self, **kwargs):
        backend = self.ensure_backend()
        return backend.plot_fluence_evolution_multi(**kwargs)

    def default_csv_name_for_series(self, series_text: str):
        series = str(series_text)

        if series.strip() == "Fluence scan":
            return "peak_fits_fluence.csv"

        return "peak_fits_delay.csv"

    def normalized_out_csv_name(self, value: str, series_text: str):
        value = (value or "").strip()

        if value in ("", "peak_fits_delay.csv", "peak_fits_fluence.csv"):
            return self.default_csv_name_for_series(series_text)

        return value
