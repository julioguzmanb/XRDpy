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
    """Adapt peak fitting and fitted-property plotting for GUI callbacks.

    The service is stateless and keeps Qt widgets decoupled from direct backend
    imports. Validated keyword arguments are forwarded unchanged.
    """

    def ensure_backend(self):
        """Import and return the backend module, raising a focused dependency error on failure."""
        if fitting is None:
            raise ImportError("Fitting backend is not available.")

        return fitting

    def run_delay_peak_fitting(self, **kwargs):
        """Fit configured peaks across a delay series and write the result CSV."""
        backend = self.ensure_backend()
        return backend.run_delay_peak_fitting(**kwargs)

    def run_fluence_peak_fitting(self, **kwargs):
        """Fit configured peaks across a fixed-delay fluence series."""
        backend = self.ensure_backend()
        return backend.run_fluence_peak_fitting(**kwargs)

    def plot_fit_overlay_from_csv(self, **kwargs):
        """Plot one delay-pattern fit using stored CSV parameters."""
        backend = self.ensure_backend()
        return backend.plot_fit_overlay_from_csv(**kwargs)

    def plot_fit_overlay_from_csv_fluence(self, **kwargs):
        """Plot one fluence-pattern fit using stored CSV parameters."""
        backend = self.ensure_backend()
        return backend.plot_fit_overlay_from_csv_fluence(**kwargs)

    def plot_time_evolution(self, **kwargs):
        """Plot a fitted property versus delay for one experiment."""
        backend = self.ensure_backend()
        return backend.plot_time_evolution(**kwargs)

    def plot_fluence_evolution(self, **kwargs):
        """Plot a fitted property versus fluence for one experiment."""
        backend = self.ensure_backend()
        return backend.plot_fluence_evolution(**kwargs)

    def plot_time_evolution_multi(self, **kwargs):
        """Compare one delay-dependent fitted peak property across multiple experiments."""
        backend = self.ensure_backend()
        return backend.plot_time_evolution_multi(**kwargs)

    def plot_fluence_evolution_multi(self, **kwargs):
        """Compare one fluence-dependent fitted peak property across multiple experiments."""
        backend = self.ensure_backend()
        return backend.plot_fluence_evolution_multi(**kwargs)

    def default_csv_name_for_series(self, series_text: str):
        """Return the conventional fitting-table filename for a series kind."""
        series = str(series_text)

        if series.strip() == "Fluence scan":
            return "peak_fits_fluence.csv"

        return "peak_fits_delay.csv"

    def normalized_out_csv_name(self, value: str, series_text: str):
        """Return normalized out CSV name.

        Parameters
        ----------
        value : str
            Value to validate, convert, or display.
        series_text : str
            GUI label identifying delay- or fluence-series mode.
        """
        value = (value or "").strip()

        if value in ("", "peak_fits_delay.csv", "peak_fits_fluence.csv"):
            return self.default_csv_name_for_series(series_text)

        return value
