"""
Session tab for the analysis GUI.

This reproduces the legacy Session tab layout while keeping the implementation
separate from the main window.
"""
from __future__ import annotations

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

from trxrdpy.analysis.gui.services import (
    FacilityService,
    PathService,
    PreparationService,
)
from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.widgets import DropPathLineEdit, FacilitySelector


class SessionTab(QWidget):
    """Edit shared paths, facility settings, and experiment metadata.

    Attributes
    ----------
    state : AnalysisGuiState
        Mutable session object updated from the visible controls.
    facility_service : FacilityService
        Registry translating displayed labels to stable facility keys.
    path_service : PathService
        Normalizes selected paths and manages dialog history.
    preparation_service : PreparationService
        Validates optional FemtoMAX ping-reference tables.
    session_facility_combo : QComboBox
        Active facility selector.
    session_path_root, session_analysis_subdir, session_raw_subdir : QWidget
        Experiment root and relative directory controls.
    session_poni_path, session_mask_path : QWidget
        Shared pyFAI geometry and detector-mask selectors.
    session_azim_offset_deg : QLineEdit
        Package-to-pyFAI azimuthal offset editor.
    save_state_callback, load_state_callback, load_autosave_callback : callable
        Main-window persistence callbacks.
    """

    def __init__(
        self,
        state: AnalysisGuiState,
        facility_service: FacilityService,
        path_service: PathService,
        preparation_service: PreparationService,
        log: Optional[Callable[[str], None]] = None,
        save_state_callback: Optional[Callable[[], None]] = None,
        load_state_callback: Optional[Callable[[], None]] = None,
        load_state_path_callback: Optional[Callable[[str], None]] = None,
        load_autosave_callback: Optional[Callable[[], None]] = None,
        facility_changed_callback: Optional[Callable[[str], None]] = None,
        ping_reference_changed_callback: Optional[Callable[[str], None]] = None,
        parent=None,
    ):
        """Initialize ``SessionTab``, bind shared state and services, and create its controls."""
        super().__init__(parent)

        self.state = state
        self.facility_service = facility_service
        self.path_service = path_service
        self.preparation_service = preparation_service
        self.log = log or (lambda message: None)

        self.save_state_callback = save_state_callback
        self.load_state_callback = load_state_callback
        self.load_state_path_callback = load_state_path_callback
        self.load_autosave_callback = load_autosave_callback
        self.facility_changed_callback = facility_changed_callback
        self.ping_reference_changed_callback = ping_reference_changed_callback

        layout = self._make_scroll_layout()

        self._init_facility_and_paths_group(layout)
        self._init_femtomax_group(layout)
        self._init_calibration_group(layout)
        self._init_gui_state_group(layout)

        self.set_facility(self.state.facility or "SACLA")

        layout.addStretch()

    def _make_scroll_layout(self) -> QVBoxLayout:
        """Create a scrollable content widget and return its vertical layout."""
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
        """Create and connect the controls for facility and paths group."""
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
        self.session_path_root = DropPathLineEdit("", mode="directory")
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

    def _init_femtomax_group(self, layout: QVBoxLayout):
        """Create and connect the controls for FemtoMAX group."""
        self.session_femtomax_group = QGroupBox("FemtoMAX Session")
        grid = QGridLayout()
        self.session_femtomax_group.setLayout(grid)
        layout.addWidget(self.session_femtomax_group)

        current = getattr(self.state, "femtomax_ping_reference_path", None)
        if current is None:
            current = self.preparation_service.default_femtomax_ping_reference_path()

        grid.addWidget(QLabel("Ping references CSV:"), 0, 0)
        path_layout = QHBoxLayout()
        self.session_femtomax_ping_reference_path = DropPathLineEdit(
            str(current),
            mode="file",
        )
        self.session_femtomax_ping_reference_path.setToolTip(
            "Enter, browse, or drag a CSV file here. Columns: "
            "scan_start, scan_end, ping2_ref_s, ping4_ref_s"
        )
        self.session_femtomax_ping_reference_path.editingFinished.connect(
            self._load_femtomax_ping_references
        )
        path_layout.addWidget(self.session_femtomax_ping_reference_path)

        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_femtomax_ping_references)
        path_layout.addWidget(browse)

        use_default = QPushButton("Default")
        use_default.clicked.connect(self._use_default_femtomax_ping_references)
        path_layout.addWidget(use_default)

        load = QPushButton("Load / Validate")
        load.clicked.connect(self._load_femtomax_ping_references)
        path_layout.addWidget(load)
        grid.addLayout(path_layout, 0, 1)

        self.session_femtomax_ping_reference_status = QLabel("")
        self.session_femtomax_ping_reference_status.setWordWrap(True)
        grid.addWidget(
            self.session_femtomax_ping_reference_status,
            1,
            0,
            1,
            2,
        )

    def _init_calibration_group(self, layout: QVBoxLayout):
        """Create and connect the controls for calibration group."""
        calib_group = QGroupBox("Calibration and Shared Geometry")
        calib_grid = QGridLayout()
        calib_group.setLayout(calib_grid)
        layout.addWidget(calib_group)

        row = 0

        calib_grid.addWidget(QLabel("PONI path:"), row, 0)
        h = QHBoxLayout()
        self.session_poni_path = DropPathLineEdit("", mode="file")
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
        self.session_mask_path = DropPathLineEdit("", mode="file")
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
        """Create and connect the controls for gui state group."""
        persist_group = QGroupBox("GUI State")
        persist_layout = QGridLayout()
        persist_group.setLayout(persist_layout)
        layout.addWidget(persist_group)

        self.btn_save_state = QPushButton("Save GUI State...")
        self.btn_save_state.clicked.connect(self._on_save_state_clicked)
        persist_layout.addWidget(self.btn_save_state, 0, 0)

        self.btn_load_state = QPushButton("Load GUI State...")
        self.btn_load_state.clicked.connect(self._on_load_state_clicked)
        persist_layout.addWidget(self.btn_load_state, 0, 1)

        self.btn_load_autosave = QPushButton("Restore Last Autosave")
        self.btn_load_autosave.clicked.connect(self._on_load_autosave_clicked)
        persist_layout.addWidget(self.btn_load_autosave, 0, 2)

        persist_layout.addWidget(QLabel("GUI state path:"), 1, 0)
        self.session_gui_state_path = DropPathLineEdit("", mode="file")
        self.session_gui_state_path.setPlaceholderText("Paste or drop a saved GUI-state JSON path")
        persist_layout.addWidget(self.session_gui_state_path, 1, 1)

        self.btn_load_state_path = QPushButton("Load Path")
        self.btn_load_state_path.clicked.connect(self._on_load_state_path_clicked)
        persist_layout.addWidget(self.btn_load_state_path, 1, 2)

    def _browse_directory_into(self, line_edit: QLineEdit):
        """Choose a directory and place its normalized path in a line edit."""
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select directory",
            str(
                self.path_service.dialog_start_path(
                    current=line_edit.text(),
                    preferred_directory=self.state.path_root,
                )
            ),
        )

        if selected:
            self.path_service.remember_dialog_selection(selected)
            line_edit.setText(selected)
            self._sync_state_from_widgets()

    def _browse_file_into(self, line_edit: QLineEdit, title: str, file_filter: str):
        """Choose a file and place its normalized path in a line edit."""
        selected, _ = QFileDialog.getOpenFileName(
            self,
            title,
            str(
                self.path_service.dialog_start_path(
                    current=line_edit.text(),
                    preferred_directory=self.state.path_root,
                )
            ),
            file_filter,
        )

        if selected:
            self.path_service.remember_dialog_selection(selected)
            line_edit.setText(selected)
            self._sync_state_from_widgets()

    def _on_facility_changed(self, facility_key: str):
        """Persist the selected facility and refresh facility-specific controls."""
        self.state.facility = facility_key
        self.set_facility(facility_key)

        if self.facility_changed_callback is not None:
            self.facility_changed_callback(facility_key)

        self.log(
            f"Facility set to {self.facility_service.label_from_key(facility_key)}."
        )

    def _sync_state_from_widgets(self):
        """Synchronize state from widgets without recursively emitting changes."""
        self.state.path_root = self.path_service.normalize(self.session_path_root.text().strip() or None)
        self.state.root_path = self.state.path_root

        self.state.analysis_subdir = self.session_analysis_subdir.text().strip()
        self.state.raw_subdir = self.session_raw_subdir.text().strip()

        self.state.poni_path = self.path_service.normalize(self.session_poni_path.text().strip() or None)
        self.state.calibration_path = self.state.poni_path

        self.state.mask_edf_path = self.path_service.normalize(self.session_mask_path.text().strip() or None)

        ping_path = self.session_femtomax_ping_reference_path.text().strip()
        self.state.femtomax_ping_reference_path = self.path_service.normalize(
            ping_path or None
        )

        azim_text = self.session_azim_offset_deg.text().strip()
        if azim_text:
            self.state.azim_offset_deg = float(azim_text)

    def set_facility(self, facility_key: str):
        """Store the active facility and refresh facility-dependent controls."""
        is_femto = facility_key == self.facility_service.FEMTOMAX
        if hasattr(self, "session_femtomax_group"):
            self.session_femtomax_group.setVisible(is_femto)
        if is_femto and hasattr(
            self,
            "session_femtomax_ping_reference_path",
        ):
            self._load_femtomax_ping_references(log_success=False)

    def _browse_femtomax_ping_references(self):
        """Open a file chooser for FemtoMAX ping references and store the selected path."""
        current_path = self.session_femtomax_ping_reference_path.text().strip()
        default_path = self.preparation_service.default_femtomax_ping_reference_path()
        if current_path == default_path:
            current_path = None

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select FemtoMAX ping-reference CSV",
            str(
                self.path_service.dialog_start_path(
                    current=current_path,
                    preferred_directory=self.state.path_root,
                )
            ),
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
        )
        if selected:
            self.path_service.remember_dialog_selection(selected)
            self.session_femtomax_ping_reference_path.setText(selected)
            self._load_femtomax_ping_references()

    def _use_default_femtomax_ping_references(self):
        """Select the packaged FemtoMAX ping-reference table and refresh status."""
        self.session_femtomax_ping_reference_path.setText(
            self.preparation_service.default_femtomax_ping_reference_path()
        )
        self._load_femtomax_ping_references()

    def _load_femtomax_ping_references(
        self,
        *_args,
        log_success: bool = True,
    ):
        """Load and validate FemtoMAX ping references, then update the relevant controls."""
        try:
            table = self.preparation_service.validate_femtomax_ping_reference_file(
                self.session_femtomax_ping_reference_path.text()
            )
            self.session_femtomax_ping_reference_path.setText(str(table.path))
            self.state.femtomax_ping_reference_path = PathService.normalize(
                table.path
            )
            self.session_femtomax_ping_reference_status.setText(
                f"Loaded {len(table.ranges)} ranges; scan coverage "
                f"{table.scan_min}-{table.scan_max}; SHA-256 "
                f"{table.sha256[:12]}…"
            )
            self.session_femtomax_ping_reference_status.setStyleSheet(
                "color: #287a3d;"
            )
            if self.ping_reference_changed_callback is not None:
                self.ping_reference_changed_callback(str(table.path))
            if log_success:
                self.log(
                    f"FemtoMAX ping references loaded: {table.path} "
                    f"({len(table.ranges)} ranges)."
                )
            return table
        except Exception as exc:
            self.session_femtomax_ping_reference_status.setText(
                f"Ping references not loaded: {exc}"
            )
            self.session_femtomax_ping_reference_status.setStyleSheet(
                "color: #b33a3a;"
            )
            if log_success:
                self.log(f"FemtoMAX Ping Reference Error: {exc}")
            return None

    def _on_save_state_clicked(self):
        """Synchronize visible controls and invoke the main-window save callback."""
        self._sync_state_from_widgets()

        if self.save_state_callback is not None:
            self.save_state_callback()
        else:
            self.log("Save GUI State callback is not configured.")

    def _on_load_state_clicked(self):
        """Invoke the main-window callback for loading a selected state file."""
        if self.load_state_callback is not None:
            self.load_state_callback()
        else:
            self.log("Load GUI State callback is not configured.")

    def _on_load_state_path_clicked(self):
        """Load a GUI-state JSON file from the pasted path field."""
        path = self.session_gui_state_path.text().strip()
        if not path:
            self.log("GUI state path is empty.")
            return
        if self.load_state_path_callback is not None:
            self.load_state_path_callback(path)
        else:
            self.log("Load GUI State Path callback is not configured.")

    def _on_load_autosave_clicked(self):
        """Invoke the main-window callback for restoring crash-safe autosave state."""
        if self.load_autosave_callback is not None:
            self.load_autosave_callback()
        else:
            self.log("Restore Last Autosave callback is not configured.")
