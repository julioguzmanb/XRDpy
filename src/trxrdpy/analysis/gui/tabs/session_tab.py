"""
Session tab for the analysis GUI.

This reproduces the legacy Session tab layout while keeping the implementation
separate from the main window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QFileDialog,
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

from trxrdpy.analysis.gui.services import FacilityService, PathService
from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.widgets import FacilitySelector


class SessionTab(QWidget):
    """
    Legacy-compatible Session tab.
    """

    def __init__(
        self,
        state: AnalysisGuiState,
        facility_service: FacilityService,
        path_service: PathService,
        log: Optional[Callable[[str], None]] = None,
        save_state_callback: Optional[Callable[[], None]] = None,
        load_state_callback: Optional[Callable[[], None]] = None,
        load_autosave_callback: Optional[Callable[[], None]] = None,
        facility_changed_callback: Optional[Callable[[str], None]] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.state = state
        self.facility_service = facility_service
        self.path_service = path_service
        self.log = log or (lambda message: None)

        self.save_state_callback = save_state_callback
        self.load_state_callback = load_state_callback
        self.load_autosave_callback = load_autosave_callback
        self.facility_changed_callback = facility_changed_callback

        layout = self._make_scroll_layout()

        self._init_facility_and_paths_group(layout)
        self._init_calibration_group(layout)
        self._init_gui_state_group(layout)

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

    def _init_facility_and_paths_group(self, layout: QVBoxLayout):
        group = QGroupBox("Facility and Paths")
        grid = QGridLayout()
        group.setLayout(grid)
        layout.addWidget(group)

        row = 0

        grid.addWidget(QLabel("Facility:"), row, 0)
        self.session_facility_combo = FacilitySelector(
            facility_service=self.facility_service,
            on_facility_changed=self._on_facility_changed,
        )
        grid.addWidget(self.session_facility_combo, row, 1)
        row += 1

        grid.addWidget(QLabel("Root path:"), row, 0)
        h = QHBoxLayout()
        self.session_path_root = QLineEdit("")
        self.session_path_root.editingFinished.connect(self._sync_state_from_widgets)
        h.addWidget(self.session_path_root)

        b = QPushButton("Browse")
        b.clicked.connect(lambda: self._browse_directory_into(self.session_path_root))
        h.addWidget(b)

        grid.addLayout(h, row, 1)
        row += 1

        grid.addWidget(QLabel("Analysis subdirectory:"), row, 0)
        self.session_analysis_subdir = QLineEdit("analysis")
        self.session_analysis_subdir.editingFinished.connect(self._sync_state_from_widgets)
        grid.addWidget(self.session_analysis_subdir, row, 1)
        row += 1

        grid.addWidget(QLabel("Raw-data subdirectory:"), row, 0)
        self.session_raw_subdir = QLineEdit("")
        self.session_raw_subdir.setPlaceholderText("Optional. Example: RAW_DATA/")
        self.session_raw_subdir.editingFinished.connect(self._sync_state_from_widgets)
        grid.addWidget(self.session_raw_subdir, row, 1)

    def _init_calibration_group(self, layout: QVBoxLayout):
        calib_group = QGroupBox("Calibration and Shared Geometry")
        calib_grid = QGridLayout()
        calib_group.setLayout(calib_grid)
        layout.addWidget(calib_group)

        row = 0

        calib_grid.addWidget(QLabel("PONI path:"), row, 0)
        h = QHBoxLayout()
        self.session_poni_path = QLineEdit("")
        self.session_poni_path.editingFinished.connect(self._sync_state_from_widgets)
        h.addWidget(self.session_poni_path)

        b = QPushButton("Browse")
        b.clicked.connect(
            lambda: self._browse_file_into(
                self.session_poni_path,
                "Select PONI file",
                "PONI Files (*.poni);;All Files (*)",
            )
        )
        h.addWidget(b)

        calib_grid.addLayout(h, row, 1)
        row += 1

        calib_grid.addWidget(QLabel("Mask EDF path:"), row, 0)
        h = QHBoxLayout()
        self.session_mask_path = QLineEdit("")
        self.session_mask_path.editingFinished.connect(self._sync_state_from_widgets)
        h.addWidget(self.session_mask_path)

        b = QPushButton("Browse")
        b.clicked.connect(
            lambda: self._browse_file_into(
                self.session_mask_path,
                "Select mask file",
                "EDF Files (*.edf);;All Files (*)",
            )
        )
        h.addWidget(b)

        calib_grid.addLayout(h, row, 1)
        row += 1

        calib_grid.addWidget(QLabel("Azimuth offset [deg]:"), row, 0)
        self.session_azim_offset_deg = QLineEdit("-90.0")
        self.session_azim_offset_deg.setValidator(QDoubleValidator())
        self.session_azim_offset_deg.editingFinished.connect(self._sync_state_from_widgets)
        calib_grid.addWidget(self.session_azim_offset_deg, row, 1)

    def _init_gui_state_group(self, layout: QVBoxLayout):
        persist_group = QGroupBox("GUI State")
        persist_layout = QHBoxLayout()
        persist_group.setLayout(persist_layout)
        layout.addWidget(persist_group)

        self.btn_save_state = QPushButton("Save GUI State...")
        self.btn_save_state.clicked.connect(self._on_save_state_clicked)
        persist_layout.addWidget(self.btn_save_state)

        self.btn_load_state = QPushButton("Load GUI State...")
        self.btn_load_state.clicked.connect(self._on_load_state_clicked)
        persist_layout.addWidget(self.btn_load_state)

        self.btn_load_autosave = QPushButton("Restore Last Autosave")
        self.btn_load_autosave.clicked.connect(self._on_load_autosave_clicked)
        persist_layout.addWidget(self.btn_load_autosave)

        persist_layout.addStretch()

    def _browse_directory_into(self, line_edit: QLineEdit):
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select directory",
            line_edit.text() or str(Path.home()),
        )

        if selected:
            line_edit.setText(selected)
            self._sync_state_from_widgets()

    def _browse_file_into(self, line_edit: QLineEdit, title: str, file_filter: str):
        selected, _ = QFileDialog.getOpenFileName(
            self,
            title,
            line_edit.text() or str(Path.home()),
            file_filter,
        )

        if selected:
            line_edit.setText(selected)
            self._sync_state_from_widgets()

    def _on_facility_changed(self, facility_key: str):
        self.state.facility = facility_key

        if self.facility_changed_callback is not None:
            self.facility_changed_callback(facility_key)

        self.log(
            f"Facility set to {self.facility_service.label_from_key(facility_key)}."
        )

    def _sync_state_from_widgets(self):
        self.state.path_root = self.path_service.normalize(self.session_path_root.text().strip() or None)
        self.state.root_path = self.state.path_root

        self.state.analysis_subdir = self.session_analysis_subdir.text().strip()
        self.state.raw_subdir = self.session_raw_subdir.text().strip()

        self.state.poni_path = self.path_service.normalize(self.session_poni_path.text().strip() or None)
        self.state.calibration_path = self.state.poni_path

        self.state.mask_edf_path = self.path_service.normalize(self.session_mask_path.text().strip() or None)

        azim_text = self.session_azim_offset_deg.text().strip()
        if azim_text:
            self.state.azim_offset_deg = float(azim_text)

    def _on_save_state_clicked(self):
        self._sync_state_from_widgets()

        if self.save_state_callback is not None:
            self.save_state_callback()
        else:
            self.log("Save GUI State callback is not configured.")

    def _on_load_state_clicked(self):
        if self.load_state_callback is not None:
            self.load_state_callback()
        else:
            self.log("Load GUI State callback is not configured.")

    def _on_load_autosave_clicked(self):
        if self.load_autosave_callback is not None:
            self.load_autosave_callback()
        else:
            self.log("Restore Last Autosave callback is not configured.")
