"""
Integration service for the analysis GUI.

This module wraps facility-specific azimuthal integration backend calls without
constructing Qt widgets.
"""
from __future__ import annotations
import math
import os
import re
import subprocess
from pathlib import Path

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

    def build_parallel_integration_kwargs(
        self,
        *,
        use_parallel: bool,
        max_workers_text: str,
    ):
        """Build validated bounded-concurrency options for final integration."""
        workers = parse_int_like(max_workers_text, name="parallel workers")
        if workers < 1:
            raise ValueError("parallel workers must be at least 1.")
        return {
            "use_parallel": bool(use_parallel),
            "max_workers": int(workers),
        }

    def build_dark_integration_kwargs(
        self,
        *,
        metadata_values: dict,
        poni_path: str,
        mask_edf_path: str,
        dark_tag_text: str,
        azimuthal_edges_text: str,
        include_full: bool,
        full_range_text: str,
        npt_text: str,
        q_norm_range_text: str,
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

    def build_single_shot_integration_kwargs(
        self,
        *,
        metadata_h5_path: str,
        poni_path: str,
        mask_edf_path: str,
        azimuthal_edges_text: str,
        include_full: bool,
        full_range_text: str,
        npt_text: str,
        overwrite: bool,
        paths,
        polarization_factor=None,
        azim_offset_deg=-90.0,
        facility=None,
        sacla_beamline_text="",
        sacla_detector_id_text="",
        sacla_background_text="",
        sacla_threshold_counts_text="40",
        sacla_intensity_col_text="",
        femtomax_read_batch_size_text="16",
        femtomax_work_chunk_size_text="64",
        femtomax_use_parallel=True,
        femtomax_max_workers_text="4",
        femtomax_start_method="spawn",
    ):
        """Build facility-specific raw-frame to single-shot 1D arguments."""
        metadata_path = str(metadata_h5_path or "").strip()
        if not metadata_path:
            raise ValueError("Select the metadata HDF5 file before integrating single shots.")
        kwargs = self.build_poni_mask_kwargs(
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
        )
        kwargs.update(
            metadata_h5_path=metadata_path,
            azimuthal_edges=parse_edges(azimuthal_edges_text),
            include_full=bool(include_full),
            full_range=parse_tuple2(full_range_text, name="full_range", cast=float),
            npt=parse_int_like(npt_text, name="npt"),
            overwrite=bool(overwrite),
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=self.parse_polarization_factor(
                polarization_factor
            ),
            paths=paths,
        )
        if facility == "SACLA":
            kwargs.update(
                self.build_sacla_single_shot_kwargs(
                    beamline_text=sacla_beamline_text,
                    detector_id_text=sacla_detector_id_text,
                    background_text=sacla_background_text,
                    threshold_counts_text=sacla_threshold_counts_text,
                    intensity_col_text=sacla_intensity_col_text,
                )
            )
        elif facility == "FemtoMAX":
            read_batch_size = parse_int_like(
                femtomax_read_batch_size_text,
                name="FemtoMAX HDF5 frame batch size",
            )
            if read_batch_size < 1:
                raise ValueError(
                    "FemtoMAX HDF5 frame batch size must be at least 1."
                )
            kwargs["read_batch_size"] = read_batch_size
            work_chunk_size = parse_int_like(
                femtomax_work_chunk_size_text,
                name="FemtoMAX shots per worker task",
            )
            if work_chunk_size < read_batch_size:
                raise ValueError(
                    "FemtoMAX shots per worker task must be at least the "
                    "HDF5 frame batch size."
                )
            max_workers = parse_int_like(
                femtomax_max_workers_text,
                name="FemtoMAX single-shot max workers",
            )
            if max_workers < 1:
                raise ValueError(
                    "FemtoMAX single-shot max workers must be at least 1."
                )
            start_method = str(femtomax_start_method).strip().lower()
            if start_method not in {"spawn", "forkserver", "fork"}:
                raise ValueError(
                    "FemtoMAX multiprocessing start method must be spawn, "
                    "forkserver, or fork."
                )
            kwargs.update(
                use_parallel=bool(femtomax_use_parallel),
                max_workers=max_workers,
                start_method=start_method,
                work_chunk_size=work_chunk_size,
            )
        return kwargs

    def resolve_single_shot_metadata_h5(
        self,
        *,
        facility: str,
        explicit_path: str,
        scan_type: str,
        metadata_values: dict,
        paths,
        delay_fs=None,
        scans=None,
    ) -> str:
        """Resolve an optional metadata path from the active experiment fields."""
        explicit = str(explicit_path or "").strip()
        if explicit:
            return explicit
        if facility != "FemtoMAX":
            raise ValueError(
                "Select the metadata HDF5 file for SACLA single-shot processing."
            )

        module = self.get_azimint_module("FemtoMAX")
        scan_type = str(scan_type).strip().lower()
        sample_name = str(metadata_values.get("sample_name", "")).strip()
        if not sample_name:
            raise ValueError("sample_name cannot be empty.")
        temperature_K = parse_int_like(
            metadata_values.get("temperature_K"),
            name="temperature_K",
        )
        excitation_wl_nm = None
        time_window_fs = None
        fluence_mJ_cm2 = None
        if scan_type != "dark":
            excitation_wl_nm = parse_float_like(
                metadata_values.get("excitation_wl_nm"),
                name="excitation_wl_nm",
            )
            time_window_fs = parse_int_like(
                metadata_values.get("time_window_fs"),
                name="time_window_fs",
            )
        if scan_type == "delay":
            fluence_mJ_cm2 = parse_float_like(
                metadata_values.get("fluence_mJ_cm2"),
                name="fluence_mJ_cm2",
            )
        resolved = module.resolve_metadata_h5_path(
            scan_type=scan_type,
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            fluence_mJ_cm2=fluence_mJ_cm2,
            time_window_fs=time_window_fs,
            delay_fs=delay_fs,
            scans=scans,
            paths=paths,
        )
        return str(resolved)

    def build_sacla_single_shot_kwargs(
        self,
        *,
        beamline_text="",
        detector_id_text="",
        background_text="",
        threshold_counts_text="40",
        intensity_col_text="",
    ):
        """Parse SACLA detector access and per-frame preprocessing controls."""
        beamline_value = str(beamline_text or "").strip()
        background_value = str(background_text or "").strip()
        detector_id = str(detector_id_text or "").strip()
        if not detector_id:
            detector_id = "MPCCD-8N0-3-002"
        threshold_text = str(threshold_counts_text or "").strip()
        if not threshold_text:
            threshold_text = "40"
        return {
            "beamline": (
                None
                if not beamline_value
                else parse_int_like(beamline_value, name="SACLA beamline")
            ),
            "detector_id": detector_id,
            "background": (
                None
                if not background_value
                else parse_int_like(background_value, name="SACLA background run")
            ),
            "threshold_counts": parse_float_like(
                threshold_text,
                name="SACLA threshold_counts",
            ),
            "intensity_col": str(intensity_col_text or "").strip() or None,
        }

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

    def integrate_single_shot_1d(self, *, facility: str, **kwargs):
        """Integrate metadata-selected detector frames into a facility 1D cache."""
        if facility not in {"FemtoMAX", "SACLA"}:
            raise ValueError(
                "Single-shot 1D integration is available for FemtoMAX and SACLA."
            )
        module = self.get_azimint_module(facility)
        return module.integrate_single_shot_1d(**kwargs)

    def submit_sacla_single_shot_1d(
        self,
        *,
        integration_kwargs,
        n_chunks: int = 20,
        qsub_command: str = "qsub",
    ):
        """Submit SACLA single-shot production as a PBS array job."""
        if sacla_azimint is None:
            raise ImportError("SACLA azimuthal integration backend is not available.")

        n_chunks = int(n_chunks)
        if n_chunks < 1:
            raise ValueError("SACLA PBS chunks must be at least 1.")

        kwargs = dict(integration_kwargs)
        paths = kwargs.get("paths")
        if paths is None:
            raise ValueError("SACLA PBS submission requires paths=AnalysisPaths(...).")
        metadata_path = str(kwargs.get("metadata_h5_path") or "").strip()
        poni_path = str(kwargs.get("poni_path") or "").strip()
        if not metadata_path:
            raise ValueError("SACLA PBS submission requires a metadata HDF5 path.")
        if not poni_path:
            raise ValueError("SACLA PBS submission requires a PONI path.")

        sender = (
            Path(sacla_azimint.__file__).resolve().parent
            / "pbs"
            / "single_shot_1d_job_sender.sh"
        )
        if not sender.is_file():
            raise FileNotFoundError(
                "Packaged SACLA PBS sender was not found: {}".format(sender)
            )

        def numeric_sequence(values):
            return " ".join(format(float(value), ".17g") for value in values)

        environment = os.environ.copy()
        environment.update(
            {
                "XRDPY_PATH_ROOT": str(paths.path_root),
                "XRDPY_RAW_SUBDIR": str(paths.raw_subdir),
                "XRDPY_ANALYSIS_SUBDIR": str(paths.analysis_subdir),
                "XRDPY_METADATA_H5": metadata_path,
                "XRDPY_PONI_PATH": poni_path,
                "XRDPY_MASK_PATH": str(kwargs.get("mask_edf_path") or ""),
                "XRDPY_N_CHUNKS": str(n_chunks),
                "XRDPY_AZIMUTHAL_EDGES": numeric_sequence(
                    kwargs["azimuthal_edges"]
                ),
                "XRDPY_INCLUDE_FULL": "1" if kwargs.get("include_full", True) else "0",
                "XRDPY_FULL_RANGE": numeric_sequence(
                    kwargs.get("full_range", (-90.0, 90.0))
                ),
                "XRDPY_NPT": str(int(kwargs.get("npt", 1000))),
                "XRDPY_AZIM_OFFSET_DEG": format(
                    float(kwargs.get("azim_offset_deg", -90.0)), ".17g"
                ),
                "XRDPY_DETECTOR_ID": str(kwargs.get("detector_id") or ""),
                "XRDPY_THRESHOLD_COUNTS": format(
                    float(kwargs.get("threshold_counts", 40.0)), ".17g"
                ),
                "XRDPY_OVERWRITE": "1" if kwargs.get("overwrite", False) else "0",
            }
        )
        optional_values = {
            "XRDPY_BEAMLINE": kwargs.get("beamline"),
            "XRDPY_BACKGROUND": kwargs.get("background"),
            "XRDPY_BACKGROUND_PATH": kwargs.get("background_path"),
            "XRDPY_POLARIZATION_FACTOR": kwargs.get("polarization_factor"),
            "XRDPY_INTENSITY_COL": kwargs.get("intensity_col"),
        }
        for name, value in optional_values.items():
            environment[name] = "" if value is None else str(value)

        try:
            completed = subprocess.run(
                [str(qsub_command), "-J", "1-{}".format(n_chunks), str(sender)],
                cwd=str(paths.path_root),
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "SACLA PBS submission requires the qsub command in PATH."
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                "SACLA PBS submission failed{}".format(
                    ": {}".format(detail) if detail else "."
                )
            )

        output = (completed.stdout or "").strip()
        return {
            "submitted": True,
            "job_id": output.splitlines()[-1] if output else "unknown",
            "n_chunks": n_chunks,
            "sender_path": str(sender),
        }

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
