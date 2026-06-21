"""
2D preparation service for the analysis GUI.

This module wraps facility-specific 2D preparation backend calls without
constructing Qt widgets.
"""
from __future__ import annotations

from trxrdpy.analysis.gui.utils import (
    parse_float_like,
    parse_int_like,
    parse_optional_int_like,
    parse_optional_tuple2,
    parse_python_literal,
    parse_scan_spec,
)


try:
    from trxrdpy.analysis.MaxIV_FemtoMAX import datared as femto_datared
except Exception:
    femto_datared = None

try:
    from trxrdpy.analysis.ESRF_ID09 import datared as id09_datared
except Exception:
    id09_datared = None


class PreparationService:
    """Service layer for facility-specific 2D preparation operations."""

    def ensure_femtomax_backend(self):
        """Ensure femtomax backend."""
        if femto_datared is None:
            raise ImportError("FemtoMAX data reduction backend is not available.")

        return femto_datared

    def parse_femtomax_scans(self, scans_text: str):
        """Parse femtomax scans."""
        return parse_scan_spec(scans_text)

    @staticmethod
    def _scan_list(scans):
        """Return scan list."""
        if isinstance(scans, int):
            return [int(scans)]
        if isinstance(scans, (list, tuple)):
            return [int(scan) for scan in scans]
        raise ValueError("scans must be an integer or a list/tuple of integers.")

    def default_femtomax_ping_reference_path(self) -> str:
        """Return the default femtomax ping reference path."""
        backend = self.ensure_femtomax_backend()
        return str(backend.default_ping_reference_path())

    def validate_femtomax_ping_reference_file(
        self,
        reference_path_text: str,
        *,
        scans_text: str = "",
    ):
        """Validate femtomax ping reference file.

        Parameters
        ----------
        reference_path_text : str
            GUI text containing an optional ping-reference CSV path.
        scans_text : str
            GUI text containing one scan, a Python scan list, or scan ranges.
        """
        backend = self.ensure_femtomax_backend()
        path = (reference_path_text or "").strip()
        if not path:
            path = self.default_femtomax_ping_reference_path()
        table = backend.load_ping_reference_table(path)
        if (scans_text or "").strip():
            scans = self._scan_list(self.parse_femtomax_scans(scans_text))
            table.validate_scans(scans)
        return table

    def plot_femtomax_ping_distribution(
        self,
        *,
        scans_text: str,
        mode: str,
        delay_source: str,
        unit: str,
        view: str,
        bins_text: str,
        hist_range_text: str,
        density: bool,
        show_median: bool,
        require_both: bool,
        reference_path_text: str,
        paths,
    ):
        """Plot femtomax ping distribution.

        Parameters
        ----------
        scans_text : str
            GUI text containing one scan, a Python scan list, or scan ranges.
        mode : str
            Operation mode controlling how the input data are grouped or displayed.
        delay_source : str
            Column or metadata source from which delay values are read.
        unit : str
            Display unit used for the independent variable.
        view : str
            Display representation, such as histogram or delay trace.
        bins_text : str
            GUI text containing bins input.
        hist_range_text : str
            GUI text containing hist range input.
        density : bool
            Whether histogram counts are normalized to probability density.
        show_median : bool
            Whether to display median.
        require_both : bool
            Whether both members of each symmetric azimuthal pair are required.
        reference_path_text : str
            GUI text containing an optional ping-reference CSV path.
        paths : object
            Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.

        Returns
        -------
        object
            Plot result returned by the underlying renderer, including saved-path metadata when enabled.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        backend = self.ensure_femtomax_backend()

        bins = parse_int_like(bins_text, name="bins")
        if bins < 1 or bins > 100_000:
            raise ValueError("bins must be between 1 and 100000.")

        scans = self.parse_femtomax_scans(scans_text)
        table = self.validate_femtomax_ping_reference_file(
            reference_path_text,
            scans_text=scans_text,
        )

        return backend.plot_pings_distribution(
            scans=scans,
            mode=mode,
            delay_source=delay_source,
            unit=unit,
            view=view,
            bins=bins,
            hist_range=parse_optional_tuple2(
                hist_range_text,
                name="hist_range",
                cast=float,
            ),
            density=bool(density),
            show_median=bool(show_median),
            require_both=bool(require_both),
            ping_reference_path=table.path,
            paths=paths,
        )
    
    def parse_femtomax_selected_delays(self, selected_delays_text: str):
        """Parse femtomax selected delays."""
        text = (selected_delays_text or "").strip()

        if text == "":
            return "auto"

        if text.lower() == "auto":
            return "auto"

        return parse_python_literal(text)

    def parse_femtomax_fluences(self, fluences_text: str):
        """Parse femtomax fluences.

        Parameters
        ----------
        fluences_text : str
            GUI text containing fluence values in mJ/cm² or ``"all"``.

        Returns
        -------
        object
            Parsed and validated field value.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        value = parse_python_literal((fluences_text or "").strip())

        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(
                "For a FemtoMAX fluence scan, fluences_mJ_cm2 must be a "
                "non-empty list aligned with scans, e.g. [1.0, 5.0, 15.0]."
            )

        try:
            return [float(fluence) for fluence in value]
        except (TypeError, ValueError) as exc:
            raise ValueError("Every FemtoMAX fluence value must be numeric.") from exc

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

    def build_femtomax_common_kwargs(
        self,
        *,
        metadata_values: dict,
        scans_text: str,
        scan_type: str,
        selected_delays_text: str,
        delay_source: str,
        require_both: bool,
        nb_shot_threshold_text: str,
        overwrite: bool,
        paths,
        fluences_text: str = "",
        reference_path_text: str = "",
    ):
        """Build femtomax common keyword arguments.

        Parameters
        ----------
        metadata_values : dict
            Validated experiment metadata collected from the GUI fields.
        scans_text : str
            GUI text containing one scan, a Python scan list, or scan ranges.
        scan_type : str
            Reduction series to process: delay, fluence, or dark.
        selected_delays_text : str
            GUI text containing selected delay-bin centers or ``"all"``.
        delay_source : str
            Column or metadata source from which delay values are read.
        require_both : bool
            Whether both members of each symmetric azimuthal pair are required.
        nb_shot_threshold_text : str
            GUI text containing nb shot threshold input.
        overwrite : bool
            Whether existing output artifacts may be replaced.
        paths : object
            Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.
        fluences_text : str
            GUI text containing fluence values in mJ/cm² or ``"all"``.
        reference_path_text : str
            GUI text containing an optional ping-reference CSV path.

        Returns
        -------
        dict
            Validated keyword-argument mapping accepted by the selected analysis backend.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.

        Notes
        -----
        This operation may create or replace analysis artifacts according to its save and overwrite settings.
        """
        scan_type = (scan_type or "").strip().lower()
        if scan_type not in {"delay", "fluence", "dark"}:
            raise ValueError("scan_type must be 'delay', 'fluence', or 'dark'.")

        scans = self.parse_femtomax_scans(scans_text)
        selected_delays = self.parse_femtomax_selected_delays(selected_delays_text)

        scan_values = self._scan_list(scans)
        if not scan_values:
            raise ValueError("At least one FemtoMAX scan must be provided.")

        if delay_source not in {"avg", "p2", "p4"}:
            raise ValueError("delay_source must be 'avg', 'p2', or 'p4'.")

        sample_name = metadata_values["sample_name"].strip()
        if not sample_name:
            raise ValueError("sample_name cannot be empty.")

        kwargs = {
            "sample_name": sample_name,
            "temperature_K": parse_int_like(
                metadata_values["temperature_K"],
                name="temperature_K",
            ),
            "excitation_wl_nm": None,
            "fluence_mJ_cm2": None,
            "time_window_fs": None,
        }

        if scan_type != "dark":
            kwargs.update(
                excitation_wl_nm=parse_float_like(
                    metadata_values["excitation_wl_nm"],
                    name="excitation_wl_nm",
                ),
                time_window_fs=parse_int_like(
                    metadata_values["time_window_fs"],
                    name="time_window_fs",
                ),
            )

        if scan_type == "delay":
            kwargs["fluence_mJ_cm2"] = parse_float_like(
                metadata_values["fluence_mJ_cm2"],
                name="fluence_mJ_cm2",
            )

        elif scan_type == "fluence":
            if isinstance(selected_delays, str) and selected_delays.lower() == "auto":
                raise ValueError(
                    "For a FemtoMAX fluence scan, selected_delays must be explicit, "
                    "e.g. [-1000]."
                )

            fluences = self.parse_femtomax_fluences(fluences_text)
            if len(fluences) != len(scan_values):
                raise ValueError(
                    "For a FemtoMAX fluence scan, fluences_mJ_cm2 must have the "
                    f"same length as scans ({len(fluences)} fluences for "
                    f"{len(scan_values)} scans)."
                )
            kwargs["fluence_mJ_cm2"] = fluences

        table = self.validate_femtomax_ping_reference_file(
            reference_path_text,
            scans_text=(scans_text if scan_type != "dark" else ""),
        )

        kwargs.update(
            scans=scans,
            scan_type=scan_type,
            selected_delays=selected_delays,
            delay_source=delay_source,
            require_both=require_both,
            nb_shot_threshold=parse_optional_int_like(nb_shot_threshold_text),
            overwrite=overwrite,
            ping_reference_path=table.path,
            paths=paths,
        )

        return kwargs

    def create_femtomax_metadata_h5(self, **kwargs):
        """Create femtomax metadata HDF5."""
        backend = self.ensure_femtomax_backend()
        backend.create_h5_files(**kwargs)
    
    def generate_femtomax_2d_images(self, **kwargs):
        """Generate femtomax 2D images."""
        backend = self.ensure_femtomax_backend()
        return backend.generate_2D_imgs(**kwargs)

    def ensure_id09_backend(self):
        """Ensure ID09 backend."""
        if id09_datared is None:
            raise ImportError("ID09 data reduction backend is not available.")

        return id09_datared

    def parse_delays_value(self, delays_text: str):
        """Parse delays value."""
        text = (delays_text or "").strip()

        if not text:
            raise ValueError("Delays field cannot be empty.")

        return parse_python_literal(text)

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

    def build_id09_dark_kwargs(self, *, metadata_values: dict, paths):
        """Build ID09 dark keyword arguments.

        Parameters
        ----------
        metadata_values : dict
            Validated experiment metadata collected from the GUI fields.
        paths : object
            Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.

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
            "paths": paths,
        }

        kwargs.update(self.build_id09_kwargs(metadata_values))

        return kwargs

    def create_id09_dark_from_ref_delay(self, **kwargs):
        """Create ID09 dark from ref delay."""
        backend = self.ensure_id09_backend()
        return backend.create_dark_from_ref_delay(**kwargs)

    def create_id09_final_2d_images(self, **kwargs):
        """Create ID09 final 2D images."""
        backend = self.ensure_id09_backend()
        return backend.create_final_2D_images(**kwargs)
