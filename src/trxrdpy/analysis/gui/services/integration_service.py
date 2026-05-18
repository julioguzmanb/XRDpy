"""
Integration service for the analysis GUI.

This module wraps facility-specific azimuthal integration backend calls without
constructing Qt widgets.
"""
from __future__ import annotations
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
    """
    Service layer for 1D integration and pattern-creation workflows.
    """

    def get_azimint_module(self, facility: str):
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
        def clean_path(value):
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
        text = (delays_text or "").strip()

        if not text:
            raise ValueError("Delays field cannot be empty.")

        return parse_python_literal(text)

    def parse_fluences_value(self, fluences_text: str):
        text = (fluences_text or "").strip()

        if not text:
            raise ValueError("Fluences field cannot be empty.")

        return parse_python_literal(text)

    def parse_azim_offset_deg(self, azim_offset_text: str):
        return parse_float_like(azim_offset_text, name="azim_offset_deg")

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
    ):
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
    ):
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
            paths=paths,
        )
        return kwargs

    def build_id09_kwargs(self, metadata_values: dict):
        kwargs = {
            "dataset": parse_int_like(metadata_values["dataset"], name="dataset"),
            "scan_nb": parse_int_like(metadata_values["scan_nb"], name="scan_nb"),
        }

        raw_sample_name = metadata_values.get("raw_sample_name", "").strip()

        if raw_sample_name:
            kwargs["raw_sample_name"] = raw_sample_name

        return kwargs

    def build_dark_experiment_kwargs(self, metadata_values: dict):
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
        text = (dark_tag_text or "").strip()
        if not text:
            return None
        try:
            return parse_python_literal(text)
        except Exception:
            return text

    def build_experiment_kwargs(self, metadata_values: dict):
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
        module = self.get_azimint_module(facility)
        return module.integrate_dark_1d(**kwargs)

    def integrate_delay_1d(self, *, facility: str, **kwargs):
        module = self.get_azimint_module(facility)
        return module.integrate_delay_1d(**kwargs)

    def create_id09_fluence_scan_from_delay_scans(self, **kwargs):
        if id09_azimint is None:
            raise ImportError("ID09 azimuthal integration backend is not available.")

        return id09_azimint.create_fluence_scan_from_delay_scans(**kwargs)

    def parse_azimuthal_edges(self, text: str):
        return parse_edges(text)


    def parse_range_tuple(self, text: str, *, name: str):
        return parse_tuple2(text, name=name, cast=float)

    def parse_ref_value(self, text):
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
        module = self.get_azimint_module(facility)
        return module.plot_1D_abs_and_diffs_delay(**kwargs)


    def plot_1d_abs_and_diffs_fluence(self, *, facility: str, **kwargs):
        module = self.get_azimint_module(facility)
        return module.plot_1D_abs_and_diffs_fluence(**kwargs)
