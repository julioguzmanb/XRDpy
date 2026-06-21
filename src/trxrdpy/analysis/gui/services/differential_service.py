"""
Differential analysis service for the analysis GUI.

This module wraps differential-analysis backend calls without constructing
Qt widgets.
"""
from __future__ import annotations

try:
    from trxrdpy.analysis import differential_analysis
except Exception:
    differential_analysis = None


class DifferentialService:
    """Service layer for differential-analysis plotting workflows."""

    def ensure_backend(self):
        """Import and return the backend module, raising a focused dependency error on failure."""
        if differential_analysis is None:
            raise ImportError("Differential analysis backend is not available.")

        return differential_analysis

    def plot_differential_integrals(self, **kwargs):
        """Plot differential integrals."""
        backend = self.ensure_backend()
        return backend.plot_differential_integrals(**kwargs)

    def plot_differential_integrals_fluence(self, **kwargs):
        """Plot differential integrals fluence."""
        backend = self.ensure_backend()
        return backend.plot_differential_integrals_fluence(**kwargs)

    def plot_differential_fft(self, **kwargs):
        """Plot differential FFT."""
        backend = self.ensure_backend()
        return backend.plot_differential_fft(**kwargs)

    def plot_differential_integrals_multi(self, **kwargs):
        """Plot differential integrals multi."""
        backend = self.ensure_backend()
        return backend.plot_differential_integrals_multi(**kwargs)

    def plot_differential_integrals_fluence_multi(self, **kwargs):
        """Plot differential integrals fluence multi."""
        backend = self.ensure_backend()
        return backend.plot_differential_integrals_fluence_multi(**kwargs)

    def plot_differential_fft_multi(self, **kwargs):
        """Plot differential FFT multi."""
        backend = self.ensure_backend()
        return backend.plot_differential_fft_multi(**kwargs)

    def validate_multi_experiments(self, experiments, *, required_fields):
        """Validate multi experiments.

        Parameters
        ----------
        experiments : object
            Experiment dictionaries describing datasets, labels, offsets, and optional merged fragments.
        required_fields : object
            Field names that every experiment definition must contain.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        if not experiments:
            raise ValueError("At least one experiment must be defined.")

        for index, experiment in enumerate(experiments, start=1):
            if "merge" in experiment:
                merged_entries = experiment.get("merge", [])

                if not merged_entries:
                    raise ValueError(
                        f"Merged experiment {index} must contain at least one source."
                    )

                for sub_index, sub_experiment in enumerate(merged_entries, start=1):
                    self._validate_experiment_fields(
                        sub_experiment,
                        required_fields=required_fields,
                        label=f"merged experiment {index}, source {sub_index}",
                    )
            else:
                self._validate_experiment_fields(
                    experiment,
                    required_fields=required_fields,
                    label=f"experiment {index}",
                )

        return experiments

    def _validate_experiment_fields(self, experiment, *, required_fields, label):
        """Validate experiment fields."""
        missing = [
            field
            for field in required_fields
            if field not in experiment or experiment[field] in ("", None)
        ]

        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{label} is missing required fields: {joined}")
