"""
Integration service for the analysis GUI.

This module wraps facility-specific azimuthal integration backend calls without
constructing Qt widgets.
"""
from __future__ import annotations
import math
import re

from trxrdpy.analysis.gui.utils import (
    parse_edges,
    parse_float_like,
    parse_int_like,
    parse_python_literal,
    parse_tuple2,
)


try:
    from trxrdpy.analysis.Spring8_SACLA import azimint as sacla_azimint
except Exception:
    sacla_azimint = None

try:
    from trxrdpy.analysis.MaxIV_FemtoMAX import azimint as femto_azimint
except Exception:
    femto_azimint = None

try:
    from trxrdpy.analysis.ESRF_ID09 import azimint as id09_azimint
except Exception:
    id09_azimint = None


class IntegrationService:
    """Parse integration inputs and route them to a facility backend.

    The service normalizes editable GUI text and exposes one stable interface
    to pattern-creation and viewer tabs. It stores no experiment state.
    """

    def get_azimint_module(self, facility: str):
        """Return azimint module.

        Parameters
        ----------
        facility : str
            Facility key selecting the corresponding data-reduction or integration backend.

        Raises
        ------
        ImportError
            If the operation encounters this explicit failure condition.
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        if facility == "SACLA":
            module = sacla_azimint
        elif facility == "FemtoMAX":
            module = femto_azimint
        elif facility == "ID09":
            module = id09_azimint
        else:
            raise ValueError(f"Unsupported facility: {facility}")

        if module is None:
            raise ImportError(
                f"Azimuthal integration backend is not available for {facility}."
            )

        return module

    def build_poni_mask_kwargs(self, *, poni_path=None, mask_edf_path=None):
        """Build PONI mask keyword arguments.

        Parameters
        ----------
        poni_path : object
            pyFAI PONI calibration file. Automatic discovery is used where supported when omitted.
        mask_edf_path : object
            Optional detector-mask EDF file; masked pixels are excluded from integration.

        Returns
        -------
        dict
            Validated keyword-argument mapping accepted by the selected analysis backend.
        """
        def clean_path(value):
            """Convert an optional path-like widget value to a stripped string."""
            if value is None:
                return None

            text = str(value).strip()

            if not text:
                return None

            return text

        return {
            "poni_path": clean_path(poni_path),
            "mask_edf_path": clean_path(mask_edf_path),
        }

    def parse_delays_value(self, delays_text: str):
        """Parse scalar or sequence delay text into backend-compatible values."""
        text = (delays_text or "").strip()

        if not text:
            raise ValueError("Delays field cannot be empty.")

        return parse_python_literal(text)

    def parse_fluences_value(self, fluences_text: str):
        """Parse scalar or sequence fluence text into backend-compatible values."""
        text = (fluences_text or "").strip()

        if not text:
            raise ValueError("Fluences field cannot be empty.")

        return parse_python_literal(text)

    def parse_azim_offset_deg(self, azim_offset_text: str):
        """Validate and return the package-to-pyFAI azimuth offset in degrees."""
        return parse_float_like(azim_offset_text, name="azim_offset_deg")

    def parse_polarization_factor(self, value):
        """Validate an optional pyFAI polarization factor in ``[-1, 1]``."""
        if value is None or not str(value).strip():
            return None
        factor = parse_float_like(value, name="polarization_factor")
        if not math.isfinite(factor) or not -1.0 <= factor <= 1.0:
            raise ValueError("polarization_factor must be between -1 and 1.")
        return float(factor)

    def build_dark_integration_kwargs(
        self,
        *,
        metadata_values: dict,
        poni_path: str,
        mask_edf_path: str,
        dark_tag_text: str,
        azimuthal_edges_text: str,
        include_full: bool,
        overwrite_xy: bool,
        paths,
        polarization_factor=None,
    ):
        """Build dark integration keyword arguments.

        Parameters
        ----------
        metadata_values : dict
            Validated experiment metadata collected from the GUI fields.
        poni_path : str
            pyFAI PONI calibration file. Automatic discovery is used where supported when omitted.
        mask_edf_path : str
            Optional detector-mask EDF file; masked pixels are excluded from integration.
        dark_tag_text : str
            GUI text containing dark tag input.
        azimuthal_edges_text : str
            GUI text containing azimuthal edge values in degrees.
        include_full : bool
            Whether to include an additional pattern integrated over ``full_range``.
        overwrite_xy : bool
            Whether existing XY cache files should be recomputed.
        paths : object
            Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.
        polarization_factor : object
            pyFAI polarization correction in ``[-1, 1]``; ``None`` disables correction.

        Returns
        -------
        dict
            Validated keyword-argument mapping accepted by the selected analysis backend.
        """
        kwargs = self.build_dark_experiment_kwargs(metadata_values)
        kwargs.update(
            self.build_poni_mask_kwargs(
                poni_path=poni_path,
                mask_edf_path=mask_edf_path,
            )
        )
        kwargs.update(
            dark_tag=self.parse_dark_tag_value(dark_tag_text),
            azimuthal_edges=parse_edges(azimuthal_edges_text),
            include_full=include_full,
            overwrite_xy=overwrite_xy,
            polarization_factor=self.parse_polarization_factor(
                polarization_factor
            ),
            paths=paths,
        )
        return kwargs

    def build_delay_integration_kwargs(
        self,
        *,
        metadata_values: dict,
        poni_path: str,
        mask_edf_path: str,
        delays_text: str,
        azimuthal_edges_text: str,
        include_full: bool,
        full_range_text: str,
        npt_text: str,
        q_norm_range_text: str,
        overwrite_xy: bool,
        paths,
        polarization_factor=None,
    ):
        """Build delay integration keyword arguments.

        Parameters
        ----------
        metadata_values : dict
            Validated experiment metadata collected from the GUI fields.
        poni_path : str
            pyFAI PONI calibration file. Automatic discovery is used where supported when omitted.
        mask_edf_path : str
            Optional detector-mask EDF file; masked pixels are excluded from integration.
        delays_text : str
            GUI text containing delays input.
        azimuthal_edges_text : str
            GUI text containing azimuthal edge values in degrees.
        include_full : bool
            Whether to include an additional pattern integrated over ``full_range``.
        full_range_text : str
            GUI text containing the full azimuthal ``(start, stop)`` range in degrees.
        npt_text : str
            GUI text containing the number of radial integration points.
        q_norm_range_text : str
            GUI text containing the q-normalization interval in Å⁻¹.
        overwrite_xy : bool
            Whether existing XY cache files should be recomputed.
        paths : object
            Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.
        polarization_factor : object
            pyFAI polarization correction in ``[-1, 1]``; ``None`` disables correction.

        Returns
        -------
        dict
            Validated keyword-argument mapping accepted by the selected analysis backend.
        """
        kwargs = self.build_experiment_kwargs(metadata_values)
        kwargs.update(
            self.build_poni_mask_kwargs(
                poni_path=poni_path,
                mask_edf_path=mask_edf_path,
            )
        )
        kwargs.update(
            delays_fs=self.parse_delays_value(delays_text),
            azimuthal_edges=parse_edges(azimuthal_edges_text),
            include_full=include_full,
            full_range=parse_tuple2(full_range_text, name="full_range", cast=float),
            npt=parse_int_like(npt_text, name="npt"),
            q_norm_range=parse_tuple2(
                q_norm_range_text,
                name="q_norm_range",
                cast=float,
            ),
            overwrite_xy=overwrite_xy,
            polarization_factor=self.parse_polarization_factor(
                polarization_factor
            ),
            paths=paths,
        )
        return kwargs

    def build_fluence_integration_kwargs(
        self,
        *,
        metadata_values: dict,
        poni_path: str,
        mask_edf_path: str,
        delay_fs_text: str,
        fluences_text: str,
        azimuthal_edges_text: str,
        include_full: bool,
        full_range_text: str,
        npt_text: str,
        q_norm_range_text: str,
        overwrite_xy: bool,
        paths,
        polarization_factor=None,
    ):
        """Build fluence integration keyword arguments.

        Parameters
        ----------
        metadata_values : dict
            Validated experiment metadata collected from the GUI fields.
        poni_path : str
            pyFAI PONI calibration file. Automatic discovery is used where supported when omitted.
        mask_edf_path : str
            Optional detector-mask EDF file; masked pixels are excluded from integration.
        delay_fs_text : str
            GUI text containing delay fs input.
        fluences_text : str
            GUI text containing fluence values in mJ/cm² or ``"all"``.
        azimuthal_edges_text : str
            GUI text containing azimuthal edge values in degrees.
        include_full : bool
            Whether to include an additional pattern integrated over ``full_range``.
        full_range_text : str
            GUI text containing the full azimuthal ``(start, stop)`` range in degrees.
        npt_text : str
            GUI text containing the number of radial integration points.
        q_norm_range_text : str
            GUI text containing the q-normalization interval in Å⁻¹.
        overwrite_xy : bool
            Whether existing XY cache files should be recomputed.
        paths : object
            Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.
        polarization_factor : object
            pyFAI polarization correction in ``[-1, 1]``; ``None`` disables correction.

        Returns
        -------
        dict
            Validated keyword-argument mapping accepted by the selected analysis backend.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        sample_name = metadata_values["sample_name"].strip()
        if not sample_name:
            raise ValueError("sample_name cannot be empty.")

        kwargs = {
            "sample_name": sample_name,
            "temperature_K": parse_int_like(
                metadata_values["temperature_K"],
                name="temperature_K",
            ),
            "excitation_wl_nm": parse_float_like(
                metadata_values["excitation_wl_nm"],
                name="excitation_wl_nm",
            ),
            "time_window_fs": parse_int_like(
                metadata_values["time_window_fs"],
                name="time_window_fs",
            ),
        }
        kwargs.update(
            self.build_poni_mask_kwargs(
                poni_path=poni_path,
                mask_edf_path=mask_edf_path,
            )
        )
        kwargs.update(
            delay_fs=parse_int_like(delay_fs_text, name="delay_fs"),
            fluences_mJ_cm2=self.parse_fluences_value(fluences_text),
            azimuthal_edges=parse_edges(azimuthal_edges_text),
            include_full=include_full,
            full_range=parse_tuple2(full_range_text, name="full_range", cast=float),
            npt=parse_int_like(npt_text, name="npt"),
            q_norm_range=parse_tuple2(
                q_norm_range_text,
                name="q_norm_range",
                cast=float,
            ),
            overwrite_xy=overwrite_xy,
            polarization_factor=self.parse_polarization_factor(
                polarization_factor
            ),
            paths=paths,
        )
        return kwargs

    def build_id09_kwargs(self, metadata_values: dict):
        """Build ID09 keyword arguments.

        Parameters
        ----------
        metadata_values : dict
            Validated experiment metadata collected from the GUI fields.

        Returns
        -------
        dict
            Validated keyword-argument mapping accepted by the selected analysis backend.
        """
        kwargs = {
            "dataset": parse_int_like(metadata_values["dataset"], name="dataset"),
            "scan_nb": parse_int_like(metadata_values["scan_nb"], name="scan_nb"),
        }

        raw_sample_name = metadata_values.get("raw_sample_name", "").strip()

        if raw_sample_name:
            kwargs["raw_sample_name"] = raw_sample_name

        return kwargs

    def build_dark_experiment_kwargs(self, metadata_values: dict):
        """Build dark experiment keyword arguments.

        Parameters
        ----------
        metadata_values : dict
            Validated experiment metadata collected from the GUI fields.

        Returns
        -------
        dict
            Validated keyword-argument mapping accepted by the selected analysis backend.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        sample_name = metadata_values["sample_name"].strip()
        if not sample_name:
            raise ValueError("sample_name cannot be empty.")
        return {
            "sample_name": sample_name,
            "temperature_K": parse_int_like(
                metadata_values["temperature_K"],
                name="temperature_K",
            ),
        }

    def parse_dark_tag_value(self, dark_tag_text: str):
        """Parse a dark scan tag, integer scan, or combined scan specification."""
        text = (dark_tag_text or "").strip()
        if not text:
            return None
        try:
            return parse_python_literal(text)
        except Exception:
            return text

    def build_experiment_kwargs(self, metadata_values: dict):
        """Build experiment keyword arguments.

        Parameters
        ----------
        metadata_values : dict
            Validated experiment metadata collected from the GUI fields.

        Returns
        -------
        dict
            Validated keyword-argument mapping accepted by the selected analysis backend.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        sample_name = metadata_values["sample_name"].strip()

        if not sample_name:
            raise ValueError("sample_name cannot be empty.")

        return {
            "sample_name": sample_name,
            "temperature_K": parse_int_like(
                metadata_values["temperature_K"],
                name="temperature_K",
            ),
            "excitation_wl_nm": parse_float_like(
                metadata_values["excitation_wl_nm"],
                name="excitation_wl_nm",
            ),
            "fluence_mJ_cm2": parse_float_like(
                metadata_values["fluence_mJ_cm2"],
                name="fluence_mJ_cm2",
            ),
            "time_window_fs": parse_int_like(
                metadata_values["time_window_fs"],
                name="time_window_fs",
            ),
        }

    def integrate_dark_1d(self, *, facility: str, **kwargs):
        """Integrate one dark/reference detector image into cached XY patterns."""
        module = self.get_azimint_module(facility)
        return module.integrate_dark_1d(**kwargs)

    def integrate_delay_1d(self, *, facility: str, **kwargs):
        """Integrate requested delay images into cached azimuthal XY patterns."""
        module = self.get_azimint_module(facility)
        return module.integrate_delay_1d(**kwargs)

    def integrate_fluence_1d(self, *, facility: str, **kwargs):
        """Integrate fixed-delay fluence images into cached XY patterns."""
        module = self.get_azimint_module(facility)
        return module.integrate_fluence_1d(**kwargs)

    def create_id09_fluence_scan_from_delay_scans(self, **kwargs):
        """Assemble an ID09 fluence series from selected delay-scan outputs."""
        if id09_azimint is None:
            raise ImportError("ID09 azimuthal integration backend is not available.")

        return id09_azimint.create_fluence_scan_from_delay_scans(**kwargs)

    def parse_azimuthal_edges(self, text: str):
        """Parse and validate ordered azimuthal bin edges in degrees."""
        return parse_edges(text)


    def parse_range_tuple(self, text: str, *, name: str):
        """Parse range tuple.

        Parameters
        ----------
        text : str
            Text entered in the corresponding GUI field.
        name : str
            Field or result name used in validation messages.

        Returns
        -------
        object
            Parsed and validated field value.
        """
        return parse_tuple2(text, name=name, cast=float)

    def parse_ref_value(self, text):
        """Parse a delay or dark reference selector from editable text."""
        from trxrdpy.analysis.gui.utils import parse_ref_value

        return parse_ref_value(text)

    def ensure_id09_fluence_cache(
        self,
        *,
        sample_name: str,
        temperature_K: int,
        excitation_wl_nm: float,
        delay_fs: int,
        time_window_fs: int,
        fluences_mJ_cm2,
        azim_windows,
        copy_2d_image: bool,
        overwrite: bool,
        paths,
    ):
        """Ensure ID09 fluence cache.

        Parameters
        ----------
        sample_name : str
            Sample identifier used in the standardized analysis directory layout.
        temperature_K : int
            Sample temperature in kelvin.
        excitation_wl_nm : float
            Pump wavelength in nanometres.
        delay_fs : int
            Pump-probe delay in femtoseconds.
        time_window_fs : int
            Width of the delay bin or acquisition window in femtoseconds.
        fluences_mJ_cm2 : object
            Fluence selector in mJ/cm²; many workflows also accept ``"all"``.
        azim_windows : object
            Sequence of azimuthal ``(start, stop)`` windows in degrees.
        copy_2d_image : bool
            Whether the source delay image is copied into the synthetic fluence cache.
        overwrite : bool
            Whether existing output artifacts may be replaced.
        paths : object
            Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.

        Raises
        ------
        ImportError
            If the operation encounters this explicit failure condition.

        Notes
        -----
        This operation may create or replace analysis artifacts according to its save and overwrite settings.
        """
        if id09_azimint is None:
            raise ImportError("ESRF-ID09 backend is not available in this environment.")

        if azim_windows is None:
            windows = [(-90.0, 90.0)]
        elif (
            isinstance(azim_windows, tuple)
            and len(azim_windows) == 2
            and not isinstance(azim_windows[0], (list, tuple))
        ):
            windows = [(float(azim_windows[0]), float(azim_windows[1]))]
        else:
            windows = [(float(window[0]), float(window[1])) for window in list(azim_windows)]

        for window in windows:
            id09_azimint.create_fluence_scan_from_delay_scans(
                sample_name=str(sample_name),
                temperature_K=int(temperature_K),
                excitation_wl_nm=float(excitation_wl_nm),
                delay_fs=int(delay_fs),
                time_window_fs=int(time_window_fs),
                fluences_mJ_cm2=fluences_mJ_cm2,
                paths=paths,
                azimuthal_edges=[float(window[0]), float(window[1])],
                include_full=False,
                full_range=(float(window[0]), float(window[1])),
                copy_2d_image=bool(copy_2d_image),
                overwrite=bool(overwrite),
            )


    def plot_1d_abs_and_diffs_delay(self, *, facility: str, **kwargs):
        """Plot absolute delay patterns and differences from a reference."""
        module = self.get_azimint_module(facility)
        return module.plot_1D_abs_and_diffs_delay(**kwargs)


    def plot_1d_abs_and_diffs_fluence(self, *, facility: str, **kwargs):
        """Plot absolute fluence patterns and differences from a reference."""
        module = self.get_azimint_module(facility)
        return module.plot_1D_abs_and_diffs_fluence(**kwargs)
