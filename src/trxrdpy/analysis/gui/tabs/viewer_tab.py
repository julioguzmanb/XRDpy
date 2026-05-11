"""
1D Viewer tab for the analysis GUI.

This reproduces the legacy 1D Viewer tab layout while keeping backend actions
separated from the main window.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.widgets import ExperimentMetadataWidget
from trxrdpy.analysis.gui.services import IntegrationService, PathService


class ViewerTab(QWidget):
    """
    Legacy-compatible 1D Viewer tab.
    """

    def __init__(
        self,
        state: AnalysisGuiState,
        path_service: PathService,
        integration_service: IntegrationService,
        log: Optional[Callable[[str], None]] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.state = state
        self.path_service = path_service
        self.integration_service = integration_service
        self.log = log or (lambda message: None)

        layout = self._make_scroll_layout()

        self._init_experiment_type_group(layout)

        self.experiment_metadata = ExperimentMetadataWidget(
            title="Experiment Metadata",
            include_id09=True,
        )
        layout.addWidget(self.experiment_metadata)

        self._init_delay_group(layout)
        self._init_fluence_group(layout)
        self._init_plot_settings_group(layout)
        self._init_id09_group(layout)
        self._init_runtime_group(layout)
        self._init_actions(layout)

        self.set_facility(self.state.facility or "SACLA")
        self._refresh_series_widgets()

        layout.addStretch()

    def _make_scroll_layout(self) -> QVBoxLayout:
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout()
        content.setLayout(layout)

        return layout

    def _init_experiment_type_group(self, layout: QVBoxLayout):
        mode_group = QGroupBox("Experiment Type")
        mg = QHBoxLayout()
        mode_group.setLayout(mg)
        layout.addWidget(mode_group)

        mg.addWidget(QLabel("Experiment type:"))

        self.viewer_series_combo = QComboBox()
        self.viewer_series_combo.addItems(["Delay scan", "Fluence scan"])
        self.viewer_series_combo.currentIndexChanged.connect(
            self._refresh_series_widgets
        )
        mg.addWidget(self.viewer_series_combo)

        mg.addStretch()

    def _init_delay_group(self, layout: QVBoxLayout):
        self.viewer_delay_group = QGroupBox("Delay-scan Reference and Selection")
        grid = QGridLayout()
        self.viewer_delay_group.setLayout(grid)
        layout.addWidget(self.viewer_delay_group)

        grid.addWidget(QLabel("delays_fs:"), 0, 0)

        self.viewer_delays = QLineEdit("all")
        grid.addWidget(self.viewer_delays, 0, 1)

        grid.addWidget(QLabel("Reference type:"), 1, 0)

        self.viewer_ref_type = QComboBox()
        self.viewer_ref_type.addItems(["dark", "delay"])
        grid.addWidget(self.viewer_ref_type, 1, 1)

        grid.addWidget(QLabel("Reference value:"), 2, 0)

        self.viewer_ref_value = QLineEdit("[1466556]")
        grid.addWidget(self.viewer_ref_value, 2, 1)

    def _init_fluence_group(self, layout: QVBoxLayout):
        self.viewer_fluence_group = QGroupBox("Fluence-scan Reference and Selection")
        fg = QGridLayout()
        self.viewer_fluence_group.setLayout(fg)
        layout.addWidget(self.viewer_fluence_group)

        fg.addWidget(QLabel("Delay [fs]:"), 0, 0)

        self.viewer_fluence_delay_fs = QLineEdit("0")
        self.viewer_fluence_delay_fs.setValidator(QDoubleValidator())
        fg.addWidget(self.viewer_fluence_delay_fs, 0, 1)

        fg.addWidget(QLabel("Fluences [mJ/cm²]:"), 1, 0)

        self.viewer_fluences = QLineEdit("all")
        fg.addWidget(self.viewer_fluences, 1, 1)

        fg.addWidget(QLabel("Reference type:"), 2, 0)

        self.viewer_fluence_ref_type = QComboBox()
        self.viewer_fluence_ref_type.addItems(["dark", "fluence"])
        fg.addWidget(self.viewer_fluence_ref_type, 2, 1)

        fg.addWidget(QLabel("Reference value:"), 3, 0)

        self.viewer_fluence_ref_value = QLineEdit("[1466556]")
        fg.addWidget(self.viewer_fluence_ref_value, 3, 1)

        self.viewer_fluence_compute_if_missing = QCheckBox("compute_if_missing")
        self.viewer_fluence_compute_if_missing.setChecked(True)
        fg.addWidget(self.viewer_fluence_compute_if_missing, 4, 0, 1, 2)

        self.viewer_fluence_copy_2d = QCheckBox("copy_2d_image_if_missing")
        self.viewer_fluence_copy_2d.setChecked(False)
        fg.addWidget(self.viewer_fluence_copy_2d, 5, 0, 1, 2)

    def _init_plot_settings_group(self, layout: QVBoxLayout):
        common_group = QGroupBox("Plot Settings")
        cg = QGridLayout()
        common_group.setLayout(cg)
        layout.addWidget(common_group)

        cg.addWidget(QLabel("Azimuthal window [deg]:"), 0, 0)

        self.viewer_azim_window = QLineEdit("(-90, 90)")
        cg.addWidget(self.viewer_azim_window, 0, 1)

        cg.addWidget(QLabel("xlim:"), 1, 0)

        self.viewer_xlim = QLineEdit("(1.5, 4.5)")
        cg.addWidget(self.viewer_xlim, 1, 1)

        cg.addWidget(QLabel("digits:"), 2, 0)

        self.viewer_digits = QLineEdit("2")
        self.viewer_digits.setValidator(QDoubleValidator())
        cg.addWidget(self.viewer_digits, 2, 1)

    def _init_id09_group(self, layout: QVBoxLayout):
        self.viewer_id09_group = QGroupBox("ESRF-ID09 Delay-specific Viewer Options")
        vg = QGridLayout()
        self.viewer_id09_group.setLayout(vg)
        layout.addWidget(self.viewer_id09_group)

        vg.addWidget(QLabel("ref_delay:"), 0, 0)

        self.viewer_ref_delay = QLineEdit("-5ns")
        vg.addWidget(self.viewer_ref_delay, 0, 1)

        self.viewer_compute_if_missing = QCheckBox("compute_if_missing")
        self.viewer_compute_if_missing.setChecked(True)
        vg.addWidget(self.viewer_compute_if_missing, 1, 0, 1, 2)

        vg.addWidget(QLabel("fs_or_ps:"), 2, 0)

        self.viewer_fs_or_ps = QComboBox()
        self.viewer_fs_or_ps.addItems(["ps", "fs"])
        vg.addWidget(self.viewer_fs_or_ps, 2, 1)

    def _init_runtime_group(self, layout: QVBoxLayout):
        runtime_group = QGroupBox("Save / Runtime Options")
        rg = QGridLayout()
        runtime_group.setLayout(rg)
        layout.addWidget(runtime_group)

        self.viewer_overwrite_xy = QCheckBox("overwrite_xy")
        rg.addWidget(self.viewer_overwrite_xy, 0, 0, 1, 2)

        self.viewer_from2d_checkbox = QCheckBox("from_2D_imgs")
        rg.addWidget(self.viewer_from2d_checkbox, 1, 0, 1, 2)

        rg.addWidget(QLabel("Number of q points:"), 2, 0)
        self.viewer_npt = QLineEdit("1000")
        self.viewer_npt.setValidator(QDoubleValidator())
        rg.addWidget(self.viewer_npt, 2, 1)

        rg.addWidget(QLabel("Q normalization range:"), 3, 0)
        self.viewer_q_norm_range = QLineEdit("(2.65, 2.75)")
        rg.addWidget(self.viewer_q_norm_range, 3, 1)

        self.viewer_save_plots = QCheckBox("save_plots")
        self.viewer_save_plots.setChecked(True)
        rg.addWidget(self.viewer_save_plots, 4, 0, 1, 2)

        rg.addWidget(QLabel("save_format:"), 5, 0)
        self.viewer_save_format = QComboBox()
        self.viewer_save_format.addItems(["png", "pdf", "svg"])
        rg.addWidget(self.viewer_save_format, 5, 1)

        rg.addWidget(QLabel("Save DPI:"), 6, 0)
        self.viewer_save_dpi = QLineEdit("400")
        self.viewer_save_dpi.setValidator(QDoubleValidator())
        rg.addWidget(self.viewer_save_dpi, 6, 1)

        self.viewer_save_overwrite = QCheckBox("save_overwrite")
        self.viewer_save_overwrite.setChecked(True)
        rg.addWidget(self.viewer_save_overwrite, 7, 0, 1, 2)

    def _init_actions(self, layout: QVBoxLayout):
        btn = QPushButton("Plot 1D Absolute + Differences")
        btn.clicked.connect(self._plot_1d_abs_and_diffs)
        layout.addWidget(btn)

    def set_facility(self, facility: str):
        """
        Apply legacy facility-dependent visibility rules.
        """

        self.state.facility = facility
        self._refresh_series_widgets()

    def _refresh_series_widgets(self):
        """
        Apply legacy experiment-type visibility rules.
        """

        delay_mode = self.viewer_series_combo.currentText() == "Delay scan"
        is_id09 = self.state.facility == "ID09"

        self.viewer_delay_group.setVisible(delay_mode)
        self.viewer_fluence_group.setVisible(not delay_mode)
        self.viewer_id09_group.setVisible(delay_mode and is_id09)
        self.viewer_from2d_checkbox.setVisible(delay_mode and (not is_id09))

        self.experiment_metadata.set_field_visible("fluence_mJ_cm2", delay_mode)
        self.experiment_metadata.set_id09_visible(is_id09 and delay_mode)
    
    def _build_analysis_paths(self):
        return self.path_service.build_analysis_paths(
            path_root=self.state.path_root,
            analysis_subdir=self.state.analysis_subdir,
            raw_subdir=self.state.raw_subdir,
        )


    def _poni_path(self):
        return getattr(self.state, "poni_path", None)


    def _mask_path(self):
        return getattr(self.state, "mask_edf_path", None) or getattr(
            self.state,
            "mask_path",
            None,
        )


    def _azim_offset_deg(self):
        return self.integration_service.parse_azim_offset_deg(
            getattr(self.state, "azim_offset_deg", "-90.0")
        )


    def _base_viewer_kwargs(self):
        kwargs = self.integration_service.build_experiment_kwargs(
            self.experiment_metadata.values()
        )
        kwargs.update(
            self.integration_service.build_poni_mask_kwargs(
                poni_path=self._poni_path(),
                mask_edf_path=self._mask_path(),
            )
        )
        kwargs.update(
            azim_window=self.integration_service.parse_range_tuple(
                self.viewer_azim_window.text(),
                name="azim_window",
            ),
            xlim=self.integration_service.parse_range_tuple(
                self.viewer_xlim.text(),
                name="xlim",
            ),
            save_plots=self.viewer_save_plots.isChecked(),
            save_format=self.viewer_save_format.currentText(),
            save_dpi=int(float(self.viewer_save_dpi.text())),
            save_overwrite=self.viewer_save_overwrite.isChecked(),
            paths=self._build_analysis_paths(),
        )
        return kwargs


    def _plot_1d_abs_and_diffs(self):
        try:
            facility = self.state.facility
            kwargs = self._base_viewer_kwargs()

            if self.viewer_series_combo.currentText() == "Delay scan":
                kwargs.update(
                    delays_fs=self.integration_service.parse_delays_value(
                        self.viewer_delays.text()
                    ),
                    ref_type=self.viewer_ref_type.currentText(),
                    ref_value=self.integration_service.parse_ref_value(
                        self.viewer_ref_value.text()
                    ),
                    digits=int(float(self.viewer_digits.text())),
                )

                if facility == "ID09":
                    kwargs.update(
                        self.integration_service.build_id09_kwargs(
                            self.experiment_metadata.values()
                        )
                    )
                    kwargs.update(
                        npt=int(float(self.viewer_npt.text())),
                        ref_delay=self.viewer_ref_delay.text().strip() or None,
                        q_norm_range=self.integration_service.parse_range_tuple(
                            self.viewer_q_norm_range.text(),
                            name="q_norm_range",
                        ),
                        compute_if_missing=self.viewer_compute_if_missing.isChecked(),
                        overwrite_xy=self.viewer_overwrite_xy.isChecked(),
                        ylim_top=None,
                        ylim_diff=None,
                        vlines_peak=None,
                        vlines_bckg=None,
                        fs_or_ps=self.viewer_fs_or_ps.currentText(),
                        title=None,
                        azim_offset_deg=self._azim_offset_deg(),
                    )
                else:
                    kwargs.update(
                        from_2D_imgs=self.viewer_from2d_checkbox.isChecked(),
                    )

                self.integration_service.plot_1d_abs_and_diffs_delay(
                    facility=facility,
                    **kwargs,
                )

            else:
                delay_fs = int(float(self.viewer_fluence_delay_fs.text()))
                fluences = self.integration_service.parse_fluences_value(
                    self.viewer_fluences.text()
                )
                kwargs.pop("fluence_mJ_cm2", None)

                if facility == "ID09" and self.viewer_fluence_compute_if_missing.isChecked():
                    self.integration_service.ensure_id09_fluence_cache(
                        sample_name=kwargs["sample_name"],
                        temperature_K=kwargs["temperature_K"],
                        excitation_wl_nm=kwargs["excitation_wl_nm"],
                        delay_fs=delay_fs,
                        time_window_fs=kwargs["time_window_fs"],
                        fluences_mJ_cm2=fluences,
                        azim_windows=[kwargs["azim_window"]],
                        copy_2d_image=self.viewer_fluence_copy_2d.isChecked(),
                        overwrite=self.viewer_overwrite_xy.isChecked(),
                        paths=kwargs["paths"],
                    )

                kwargs.update(
                    delay_fs=delay_fs,
                    fluences_mJ_cm2=fluences,
                    ref_type=self.viewer_fluence_ref_type.currentText(),
                    ref_value=self.integration_service.parse_ref_value(
                        self.viewer_fluence_ref_value.text()
                    ),
                    normalize=True,
                    q_norm_range=self.integration_service.parse_range_tuple(
                        self.viewer_q_norm_range.text(),
                        name="q_norm_range",
                    ),
                    compute_if_missing=(
                        self.viewer_fluence_compute_if_missing.isChecked()
                        if facility == "ID09"
                        else True
                    ),
                    overwrite_xy=self.viewer_overwrite_xy.isChecked(),
                    ylim_top=None,
                    ylim_diff=None,
                    vlines_peak=None,
                    vlines_bckg=None,
                    title=None,
                )

                self.integration_service.plot_1d_abs_and_diffs_fluence(
                    facility=facility,
                    **kwargs,
                )

            self.log("1D comparison plot finished.")

        except Exception as exc:
            self.log(f"1D Viewer Error: {exc}")
