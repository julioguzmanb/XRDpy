"""
1D Pattern Creation tab for the analysis GUI.

This reproduces the legacy 1D Pattern Creation tab layout while keeping backend
actions separated from the main window.
"""

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
from trxrdpy.analysis.gui.widgets.task_output_dialog import run_task_with_output_dialog


class PatternCreationTab(QWidget):
    """
    Legacy-compatible 1D Pattern Creation tab.
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
        self._init_fluence_unavailable_group(layout)
        self._init_dark_group(layout)
        self._init_id09_group(layout)
        self._init_azimuthal_group(layout)
        self._init_runtime_group(layout)
        self._init_actions_group(layout)

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

        self.pattern_series_combo = QComboBox()
        self.pattern_series_combo.addItems(["Delay scan", "Fluence scan"])
        self.pattern_series_combo.currentIndexChanged.connect(
            self._refresh_series_widgets
        )
        mg.addWidget(self.pattern_series_combo)

        mg.addStretch()

    def _init_delay_group(self, layout: QVBoxLayout):
        self.pattern_delay_group = QGroupBox("Delay-scan Target")
        grid = QGridLayout()
        self.pattern_delay_group.setLayout(grid)
        layout.addWidget(self.pattern_delay_group)

        grid.addWidget(QLabel("delays_fs:"), 0, 0)

        self.pattern_delays = QLineEdit("all")
        grid.addWidget(self.pattern_delays, 0, 1)

    def _init_fluence_group(self, layout: QVBoxLayout):
        self.pattern_fluence_group = QGroupBox("Fluence Scan from ID09 Delay Scans")
        fg = QGridLayout()
        self.pattern_fluence_group.setLayout(fg)
        layout.addWidget(self.pattern_fluence_group)

        fg.addWidget(QLabel("Delay [fs]:"), 0, 0)

        self.pattern_fluence_delay_fs = QLineEdit("0")
        self.pattern_fluence_delay_fs.setValidator(QDoubleValidator())
        fg.addWidget(self.pattern_fluence_delay_fs, 0, 1)

        fg.addWidget(QLabel("Fluences [mJ/cm²]:"), 1, 0)

        self.pattern_fluences = QLineEdit("all")
        self.pattern_fluences.setPlaceholderText("Examples: all, [1.5, 5, 12], 5")
        fg.addWidget(self.pattern_fluences, 1, 1)

        self.pattern_copy_2d_image = QCheckBox("copy_2d_image")
        self.pattern_copy_2d_image.setChecked(False)
        fg.addWidget(self.pattern_copy_2d_image, 2, 0, 1, 2)

    def _init_dark_group(self, layout: QVBoxLayout):
        self.pattern_dark_group = QGroupBox("Dark Integration (SACLA / FemtoMAX)")
        dark_grid = QGridLayout()
        self.pattern_dark_group.setLayout(dark_grid)
        layout.addWidget(self.pattern_dark_group)

        dark_grid.addWidget(QLabel("dark_tag:"), 0, 0)

        self.pattern_dark_tag = QLineEdit("")
        dark_grid.addWidget(self.pattern_dark_tag, 0, 1)

    def _init_id09_group(self, layout: QVBoxLayout):
        self.pattern_id09_group = QGroupBox("ESRF-ID09 Delay-specific Options")
        id09_grid = QGridLayout()
        self.pattern_id09_group.setLayout(id09_grid)
        layout.addWidget(self.pattern_id09_group)

        id09_grid.addWidget(QLabel("ref_delay:"), 0, 0)

        self.pattern_ref_delay = QLineEdit("-5ns")
        id09_grid.addWidget(self.pattern_ref_delay, 0, 1)

        self.pattern_force_checkbox = QCheckBox("force")
        self.pattern_force_checkbox.setChecked(True)
        id09_grid.addWidget(self.pattern_force_checkbox, 1, 0, 1, 2)

    def _init_azimuthal_group(self, layout: QVBoxLayout):
        az_group = QGroupBox("Azimuthal and Integration Settings")
        az_grid = QGridLayout()
        az_group.setLayout(az_grid)
        layout.addWidget(az_group)

        az_grid.addWidget(QLabel("Azimuthal edges [deg]:"), 0, 0)

        self.pattern_azimuthal_edges = QLineEdit("-90, -60, -30, 0, 30, 60, 90")
        az_grid.addWidget(self.pattern_azimuthal_edges, 0, 1)

        self.pattern_include_full = QCheckBox("include_full")
        self.pattern_include_full.setChecked(True)
        az_grid.addWidget(self.pattern_include_full, 1, 0, 1, 2)

        az_grid.addWidget(QLabel("Full azimuthal range [deg]:"), 2, 0)

        self.pattern_full_range = QLineEdit("(-90, 90)")
        az_grid.addWidget(self.pattern_full_range, 2, 1)

        az_grid.addWidget(QLabel("Number of q points:"), 3, 0)

        self.pattern_npt = QLineEdit("1000")
        self.pattern_npt.setValidator(QDoubleValidator())
        az_grid.addWidget(self.pattern_npt, 3, 1)

        self.pattern_normalize_checkbox = QCheckBox("normalize")
        self.pattern_normalize_checkbox.setChecked(True)
        az_grid.addWidget(self.pattern_normalize_checkbox, 4, 0, 1, 2)

        az_grid.addWidget(QLabel("Q normalization range:"), 5, 0)

        self.pattern_q_norm_range = QLineEdit("(2.65, 2.75)")
        az_grid.addWidget(self.pattern_q_norm_range, 5, 1)

    def _init_runtime_group(self, layout: QVBoxLayout):
        runtime_group = QGroupBox("Runtime Options")
        rg = QGridLayout()
        runtime_group.setLayout(rg)
        layout.addWidget(runtime_group)

        self.pattern_overwrite_xy = QCheckBox("overwrite_xy")
        self.pattern_overwrite_xy.setChecked(True)
        rg.addWidget(self.pattern_overwrite_xy, 0, 0, 1, 2)

    def _init_actions_group(self, layout: QVBoxLayout):
        action_group = QGroupBox("Actions")
        al = QHBoxLayout()
        action_group.setLayout(al)
        layout.addWidget(action_group)

        self.pattern_integrate_dark_btn = QPushButton("Integrate Dark 1D")
        self.pattern_integrate_dark_btn.clicked.connect(self._integrate_dark_1d)
        al.addWidget(self.pattern_integrate_dark_btn)

        self.pattern_integrate_delay_btn = QPushButton("Integrate Delay 1D")
        self.pattern_integrate_delay_btn.clicked.connect(self._integrate_delay_1d)
        al.addWidget(self.pattern_integrate_delay_btn)

        self.pattern_create_fluence_btn = QPushButton(
            "Create Fluence Scan from Delay Scans"
        )
        self.pattern_create_fluence_btn.clicked.connect(
            self._create_id09_fluence_scan
        )
        al.addWidget(self.pattern_create_fluence_btn)

        al.addStretch()

    def set_facility(self, facility: str):
        """
        Apply legacy facility-dependent visibility rules.
        """

        self.state.facility = facility

        is_id09 = facility == "ID09"

        self.pattern_normalize_checkbox.setVisible(not is_id09)
        self._refresh_series_widgets()

    def _refresh_series_widgets(self):
        """
        Apply legacy experiment-type visibility rules.
        """

        delay_mode = self.pattern_series_combo.currentText() == "Delay scan"
        is_id09 = self.state.facility == "ID09"

        self.pattern_delay_group.setVisible(delay_mode)
        self.pattern_fluence_group.setVisible((not delay_mode) and is_id09)
        self.pattern_fluence_unavailable_group.setVisible((not delay_mode) and (not is_id09))
        self.pattern_dark_group.setVisible(delay_mode and (not is_id09))
        self.pattern_id09_group.setVisible(delay_mode and is_id09)

        self.pattern_integrate_dark_btn.setVisible(delay_mode and (not is_id09))
        self.pattern_integrate_delay_btn.setVisible(delay_mode)
        self.pattern_create_fluence_btn.setVisible((not delay_mode) and is_id09)

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


    def _integrate_dark_1d(self):
        try:
            def error_summary(traceback_text):
                lines = [
                    line.strip()
                    for line in str(traceback_text).splitlines()
                    if line.strip()
                ]
                return lines[-1] if lines else "unknown error"

            facility = self.state.facility

            if facility == "ID09":
                raise ValueError("Dark integration is not exposed here for ID09.")

            kwargs = self.integration_service.build_dark_integration_kwargs(
                metadata_values=self.experiment_metadata.values(),
                poni_path=self._poni_path(),
                mask_edf_path=self._mask_path(),
                dark_tag_text=self.pattern_dark_tag.text(),
                azimuthal_edges_text=self.pattern_azimuthal_edges.text(),
                include_full=self.pattern_include_full.isChecked(),
                overwrite_xy=self.pattern_overwrite_xy.isChecked(),
                paths=self._build_analysis_paths(),
            )

            def task():
                return self.integration_service.integrate_dark_1d(
                    facility=facility,
                    **kwargs,
                )

            run_task_with_output_dialog(
                self,
                "Integrate Dark 1D",
                task,
                on_success=lambda _result: self.log("Dark 1D integration finished."),
                on_error=lambda tb: self.log(
                    f"Integrate Dark 1D Error: {error_summary(tb)}"
                ),
            )

        except Exception as exc:
            self.log(f"Integrate Dark 1D Error: {exc}")


    def _integrate_delay_1d(self):
        try:
            def error_summary(traceback_text):
                lines = [
                    line.strip()
                    for line in str(traceback_text).splitlines()
                    if line.strip()
                ]
                return lines[-1] if lines else "unknown error"

            facility = self.state.facility

            kwargs = self.integration_service.build_delay_integration_kwargs(
                metadata_values=self.experiment_metadata.values(),
                poni_path=self._poni_path(),
                mask_edf_path=self._mask_path(),
                delays_text=self.pattern_delays.text(),
                azimuthal_edges_text=self.pattern_azimuthal_edges.text(),
                include_full=self.pattern_include_full.isChecked(),
                full_range_text=self.pattern_full_range.text(),
                npt_text=self.pattern_npt.text(),
                q_norm_range_text=self.pattern_q_norm_range.text(),
                overwrite_xy=self.pattern_overwrite_xy.isChecked(),
                paths=self._build_analysis_paths(),
            )

            if facility == "ID09":
                kwargs.update(
                    self.integration_service.build_id09_kwargs(
                        self.experiment_metadata.values()
                    )
                )
                kwargs.update(
                    force=self.pattern_force_checkbox.isChecked(),
                    ref_delay=self.pattern_ref_delay.text().strip() or None,
                    azim_offset_deg=self._azim_offset_deg(),
                )
            else:
                kwargs.update(
                    normalize=self.pattern_normalize_checkbox.isChecked(),
                )

            def task():
                return self.integration_service.integrate_delay_1d(
                    facility=facility,
                    **kwargs,
                )

            run_task_with_output_dialog(
                self,
                "Integrate Delay 1D",
                task,
                on_success=lambda _result: self.log("Delay 1D integration finished."),
                on_error=lambda tb: self.log(
                    f"Integrate Delay 1D Error: {error_summary(tb)}"
                ),
            )

        except Exception as exc:
            self.log(f"Integrate Delay 1D Error: {exc}")


    def _create_id09_fluence_scan(self):
        try:
            def error_summary(traceback_text):
                lines = [
                    line.strip()
                    for line in str(traceback_text).splitlines()
                    if line.strip()
                ]
                return lines[-1] if lines else "unknown error"

            if self.state.facility != "ID09":
                raise ValueError(
                    "Synthetic fluence-scan creation is currently implemented only for ESRF-ID09."
                )

            metadata_values = self.experiment_metadata.values()

            kwargs = self.integration_service.build_experiment_kwargs(metadata_values)
            kwargs.pop("fluence_mJ_cm2", None)

            kwargs.update(
                delay_fs=int(float(self.pattern_fluence_delay_fs.text())),
                fluences_mJ_cm2=self.integration_service.parse_fluences_value(
                    self.pattern_fluences.text()
                ),
                paths=self._build_analysis_paths(),
                azimuthal_edges=self.integration_service.parse_azimuthal_edges(
                    self.pattern_azimuthal_edges.text()
                ),
                include_full=self.pattern_include_full.isChecked(),
                full_range=self.integration_service.parse_range_tuple(
                    self.pattern_full_range.text(),
                    name="full_range",
                ),
                copy_2d_image=self.pattern_copy_2d_image.isChecked(),
                overwrite=self.pattern_overwrite_xy.isChecked(),
            )

            def task():
                return self.integration_service.create_id09_fluence_scan_from_delay_scans(
                    **kwargs
                )

            def success(result):
                datasets, _copied = result
                self.log(
                    f"Synthetic ESRF-ID09 fluence scan created. Fluence entries: {len(datasets)}"
                )

            run_task_with_output_dialog(
                self,
                "Create ESRF-ID09 Fluence Scan",
                task,
                on_success=success,
                on_error=lambda tb: self.log(
                    f"Create ESRF-ID09 Fluence Scan Error: {error_summary(tb)}"
                ),
            )

        except Exception as exc:
            self.log(f"Create ESRF-ID09 Fluence Scan Error: {exc}")


    def _init_fluence_unavailable_group(self, layout: QVBoxLayout):
        self.pattern_fluence_unavailable_group = QGroupBox("Fluence Scan")
        fluence_unavailable_layout = QVBoxLayout()
        self.pattern_fluence_unavailable_group.setLayout(fluence_unavailable_layout)
        layout.addWidget(self.pattern_fluence_unavailable_group)

        message = QLabel(
            "Fluence-scan creation from delay scans is currently implemented only for ESRF-ID09."
        )
        message.setWordWrap(True)
        fluence_unavailable_layout.addWidget(message)
