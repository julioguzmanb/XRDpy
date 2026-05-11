"""
2D preparation service for the analysis GUI.

This module wraps facility-specific 2D preparation backend calls without
constructing Qt widgets.
"""

from trxrdpy.analysis.gui.utils import (
    parse_float_like,
    parse_int_like,
    parse_optional_int_like,
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
    """
    Service layer for facility-specific 2D preparation operations.
    """

    def ensure_femtomax_backend(self):
        if femto_datared is None:
            raise ImportError("FemtoMAX data reduction backend is not available.")

        return femto_datared

    def parse_femtomax_scans(self, scans_text: str):
        return parse_scan_spec(scans_text)

    def plot_femtomax_ping_distribution(
        self,
        *,
        scans_text: str,
        mode: str,
        delay_source: str,
        unit: str,
        view: str,
        bins_text: str,
        paths,
    ):
        backend = self.ensure_femtomax_backend()

        backend.plot_pings_distribution(
            scans=self.parse_femtomax_scans(scans_text),
            mode=mode,
            delay_source=delay_source,
            unit=unit,
            view=view,
            bins=parse_int_like(bins_text, name="bins"),
            paths=paths,
        )
    
    def parse_femtomax_selected_delays(self, selected_delays_text: str):
        text = (selected_delays_text or "").strip()

        if text == "":
            return "auto"

        if text.lower() == "auto":
            return "auto"

        return parse_python_literal(text)

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
    ):
        kwargs = self.build_experiment_kwargs(metadata_values)

        kwargs.update(
            scans=self.parse_femtomax_scans(scans_text),
            scan_type=(scan_type or "").strip().lower(),
            selected_delays=self.parse_femtomax_selected_delays(selected_delays_text),
            delay_source=delay_source,
            require_both=require_both,
            nb_shot_threshold=parse_optional_int_like(nb_shot_threshold_text),
            overwrite=overwrite,
            paths=paths,
        )

        return kwargs

    def create_femtomax_metadata_h5(self, **kwargs):
        backend = self.ensure_femtomax_backend()
        backend.create_h5_files(**kwargs)
    
    def generate_femtomax_2d_images(self, **kwargs):
        backend = self.ensure_femtomax_backend()
        return backend.generate_2D_imgs(**kwargs)

    def ensure_id09_backend(self):
        if id09_datared is None:
            raise ImportError("ID09 data reduction backend is not available.")

        return id09_datared

    def parse_delays_value(self, delays_text: str):
        text = (delays_text or "").strip()

        if not text:
            raise ValueError("Delays field cannot be empty.")

        return parse_python_literal(text)

    def build_id09_kwargs(self, metadata_values: dict):
        kwargs = {
            "dataset": parse_int_like(metadata_values["dataset"], name="dataset"),
            "scan_nb": parse_int_like(metadata_values["scan_nb"], name="scan_nb"),
        }

        raw_sample_name = metadata_values.get("raw_sample_name", "").strip()

        if raw_sample_name:
            kwargs["raw_sample_name"] = raw_sample_name

        return kwargs

    def build_id09_dark_kwargs(self, *, metadata_values: dict, paths):
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
        backend = self.ensure_id09_backend()
        return backend.create_dark_from_ref_delay(**kwargs)

    def create_id09_final_2d_images(self, **kwargs):
        backend = self.ensure_id09_backend()
        return backend.create_final_2D_images(**kwargs)