"""Single-crystal simulation, geometry scan, and inverse-targeting controls."""
from __future__ import annotations

import json
import traceback
import matplotlib.pyplot as plt
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyFAI.detectors

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ... import diffractometers, single_crystal
from ...geometry import DiffractometerGeometry
from ...cif import Cif
from ..services.path_service import SimulationPathService
from ..services.simulation_service import SimulationService
from ..state import GuiState
from ..widgets import DropFileLineEdit, GeometryPanel, MatrixRotationWindow
from ...poni import read_poni_file
from ...results import save_result, load_result, plot_result
from ..widgets.detector_export import update_detector_fields, save_current_detector, current_detector
from ..services.background_simulation import simulation_request, start_simulation
from ..widgets.reciprocal_scan_controls import ReciprocalScanControls
from ..widgets.intensity_controls import IntensityControls
from ..widgets.orientation_fitting_panel import OrientationFittingPanel


def _parse_hkls_string(names_text: str):
    """
    Parse hkls string like: [1,0,2],[0,1,2]
    Returns a list-like array of Miller triplets.
    """
    import numpy as np

    names_text = (names_text or "").strip()
    if not names_text:
        return None

    try:
        parts = names_text.split("],")
        tmp = []
        for p in parts:
            cleaned = p.replace("[", "").replace("]", "").strip()
            if not cleaned:
                continue
            arr = [int(xx.strip()) for xx in cleaned.split(",")]
            if len(arr) != 3:
                raise ValueError("Each hkl must have exactly 3 integers.")
            tmp.append(arr)
        if not tmp:
            return None
        return np.array(tmp, dtype=int)
    except Exception as e:
        raise ValueError("hkls_names must be in the format [h,k,l],[h,k,l],...") from e


def _parse_csv_floats(text: str):
    """Parse comma-separated floating-point values or return ``None``."""
    import numpy as np

    text = (text or "").strip()
    if not text:
        return None
    return np.array([float(x.strip()) for x in text.split(",")], dtype=float)


def _parse_single_hkl_string(text: str):
    """Parse exactly one ``[h,k,l]`` Miller-index triplet."""
    hkls = _parse_hkls_string(text)
    if hkls is None:
        return None
    if hkls.shape[0] != 1:
        raise ValueError("Expected exactly one Miller triplet in the format [h,k,l].")
    return hkls[0]


def _parse_json_value(text: str, name: str, default=None):
    """Decode an optional JSON value and raise a field-specific error."""
    text = (text or "").strip()
    if not text:
        return default

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON.\n{exc}") from exc


def _parse_json_object(text: str, name: str, allow_empty: bool = True):
    """Decode and type-check a JSON object used by geometry controls."""
    obj = _parse_json_value(text, name, default={} if allow_empty else None)
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise ValueError(f"{name} must be a JSON object/dictionary.")
    return obj


def _parse_range_triplet_text(text: str, name: str):
    """Parse an inclusive numeric ``start, stop, step`` scan range."""
    text = (text or "").strip()
    if not text:
        raise ValueError(f"{name} must be provided as start,stop,step.")
    try:
        values = [float(x.strip()) for x in text.split(",")]
    except Exception as exc:
        raise ValueError(f"{name} must be in the form start,stop,step.") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers: start,stop,step.")
    return tuple(values)


def _pretty_json(obj) -> str:
    """Serialize a value as stable, indented JSON for editable controls."""
    return json.dumps(obj, indent=2)


class _SingleFunctionSelector(QComboBox):
    """Accept stable function keys when set programmatically."""

    def setCurrentText(self, text: str) -> None:
        index = self.findData(text)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            super().setCurrentText(text)


class SingleCrystalTab(QWidget):
    """Configure and dispatch single-crystal simulations and searches.

    The tab supports legacy Euler angles, predefined diffractometers, custom
    motor chains, 2D/3D diffraction views, Bragg scans, detector collection
    scans, and fixed-energy inverse targeting near a detector pixel.
    """
    """
    Full single-crystal simulation tab extracted from the legacy GUI.

    The legacy attribute names are intentionally kept very close to the original
    implementation so the migration remains predictable.
    """

    state_changed = pyqtSignal()
    run_completed = pyqtSignal(bool, str)

    def __init__(
        self,
        service: SimulationService,
        state: GuiState,
        path_service: SimulationPathService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build controls, restore state, and initialize geometry visibility."""
        super().__init__(parent)

        self.service = service
        self.state = state
        self.path_service = path_service or SimulationPathService()
        # Widget construction emits signals; keep the supplied state intact
        # until load_from_state has populated every control.
        self._loading_state = True
        self.last_error_message = ""
        self.last_error_traceback = ""

        self.single_cif_file_path: str | None = self.state.paths.single_cif_file_path
        self.matrix_window: MatrixRotationWindow | None = None

        self._build_ui()
        self._connect_stateful_widgets()
        self.load_from_state()

    # ------------------------------------------------------------------
    # External integration
    # ------------------------------------------------------------------
    def set_matrix_rotation_window(self, window: MatrixRotationWindow | None) -> None:
        """Attach the shared orientation-matrix utility used for imports."""
        self.matrix_window = window

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Construct detector, sample, geometry, scan, targeting, and run controls."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        scroll_layout = QVBoxLayout()
        scroll_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        scroll_content.setLayout(scroll_layout)

        func_group = QGroupBox("Function Selection")
        func_layout = QHBoxLayout()
        func_group.setLayout(func_layout)

        self.single_func_combo = _SingleFunctionSelector()
        for label, function in (
            ("Laboratory 3D Simulation", "simulate_3d"),
            ("Detector 2D Simulation", "simulate_2d"),
            ("Ewald Construction Simulation", "simulate_reciprocal"),
            ("Find Rotations for Bragg in Detector", "target_hkl_near_pixel_fixed_energy"),
            ("Detector Rotations Mapping", "detector_rotations_collecting_Braggs"),
            ("Motors Mapping for Catching Braggs", "scan_two_parameters_for_Bragg_condition"),
            ("Find Orientation Matrix", "find_orientation_from_measured_braggs"),
        ):
            self.single_func_combo.addItem(label, function)
        self.single_func_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        longest_label = max(
            self.single_func_combo.fontMetrics().horizontalAdvance(
                self.single_func_combo.itemText(index)
            )
            for index in range(self.single_func_combo.count())
        )
        # macOS leaves room for its menu checkmark and arrow outside the text area.
        popup_width = longest_label + 160
        self.single_func_combo.setMinimumWidth(popup_width)
        self.single_func_combo.view().setMinimumWidth(popup_width)
        func_layout.addWidget(self.single_func_combo)
        func_layout.addStretch()
        scroll_layout.addWidget(func_group)

        self.single_det_group = QGroupBox("Detector Settings")
        det_layout = QGridLayout()
        self.single_det_group.setLayout(det_layout)
        scroll_layout.addWidget(self.single_det_group)

        row = 0
        det_layout.addWidget(QLabel("Detector Type:"), row, 0)
        self.single_combo_det_type = QComboBox()
        detector_list = ["manual", "poni"] + list(pyFAI.detectors.ALL_DETECTORS.keys())
        self.single_combo_det_type.addItems(detector_list)
        det_layout.addWidget(self.single_combo_det_type, row, 1)
        row += 1

        det_layout.addWidget(QLabel("PONI file:"), row, 0)
        poni_hbox = QHBoxLayout()
        self.single_line_poni_file = DropFileLineEdit("", suffixes=(".poni",))
        self.single_line_poni_file.setPlaceholderText("Optional detector calibration file (*.poni)")
        poni_hbox.addWidget(self.single_line_poni_file)
        self.single_btn_browse_poni = QPushButton("Browse")
        poni_hbox.addWidget(self.single_btn_browse_poni)
        det_layout.addLayout(poni_hbox, row, 1)
        row += 1

        self.single_manual_group = QGroupBox("Detector Parameters")
        self.single_manual_group.setFlat(True)
        manual_layout = QGridLayout()
        self.single_manual_group.setLayout(manual_layout)
        det_layout.addWidget(self.single_manual_group, row, 0, 1, 2)
        row += 1

        manual_layout.addWidget(QLabel("Pixel Size H [m]:"), 0, 0)
        self.single_line_pxsize_h = QLineEdit("50e-6")
        self.single_line_pxsize_h.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.single_line_pxsize_h, 0, 1)

        manual_layout.addWidget(QLabel("Pixel Size V [m]:"), 0, 2)
        self.single_line_pxsize_v = QLineEdit("50e-6")
        self.single_line_pxsize_v.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.single_line_pxsize_v, 0, 3)

        manual_layout.addWidget(QLabel("Number of Pixels H:"), 1, 0)
        self.single_line_num_px_h = QLineEdit("2000")
        self.single_line_num_px_h.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.single_line_num_px_h, 1, 1)

        manual_layout.addWidget(QLabel("Number of Pixels V:"), 1, 2)
        self.single_line_num_px_v = QLineEdit("2000")
        self.single_line_num_px_v.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.single_line_num_px_v, 1, 3)

        det_common_group = QGroupBox("Common Detector Parameters")
        det_common_layout = QGridLayout()
        det_common_group.setLayout(det_common_layout)
        det_layout.addWidget(det_common_group, row, 0, 1, 2)
        row += 1

        det_common_layout.addWidget(QLabel("Binning (H, V):"), 0, 0)
        bin_hbox = QHBoxLayout()
        self.single_line_bin_h = QLineEdit("1")
        self.single_line_bin_h.setValidator(QDoubleValidator())
        self.single_line_bin_v = QLineEdit("1")
        self.single_line_bin_v.setValidator(QDoubleValidator())
        bin_hbox.addWidget(QLabel("H:"))
        bin_hbox.addWidget(self.single_line_bin_h)
        bin_hbox.addWidget(QLabel("V:"))
        bin_hbox.addWidget(self.single_line_bin_v)
        det_common_layout.addLayout(bin_hbox, 0, 1)

        det_common_layout.addWidget(QLabel("Distance [m]:"), 1, 0)
        self.single_line_dist = QLineEdit("0.1")
        self.single_line_dist.setValidator(QDoubleValidator())
        det_common_layout.addWidget(self.single_line_dist, 1, 1)

        det_common_layout.addWidget(QLabel("PONI1 vertical [m]:"), 1, 2)
        self.single_line_poni1 = QLineEdit("0")
        self.single_line_poni1.setValidator(QDoubleValidator())
        self.single_line_poni1.setToolTip(
            "Detector axis 1 coordinate, corresponding to the vertical image direction."
        )
        det_common_layout.addWidget(self.single_line_poni1, 1, 3)

        det_common_layout.addWidget(QLabel("PONI2 horizontal [m]:"), 1, 4)
        self.single_line_poni2 = QLineEdit("0")
        self.single_line_poni2.setValidator(QDoubleValidator())
        self.single_line_poni2.setToolTip(
            "Detector axis 2 coordinate, corresponding to the horizontal image direction."
        )
        det_common_layout.addWidget(self.single_line_poni2, 1, 5)

        self.single_detector_euler_group = QGroupBox("Legacy detector Euler rotations [°]")
        detector_euler_layout = QHBoxLayout()
        self.single_detector_euler_group.setLayout(detector_euler_layout)
        det_common_layout.addWidget(self.single_detector_euler_group, 2, 0, 1, 6)

        detector_euler_layout.addWidget(QLabel("rotx:"))
        self.single_line_rotx = QLineEdit("0")
        self.single_line_rotx.setValidator(QDoubleValidator())
        detector_euler_layout.addWidget(self.single_line_rotx)

        detector_euler_layout.addWidget(QLabel("roty:"))
        self.single_line_roty = QLineEdit("0")
        self.single_line_roty.setValidator(QDoubleValidator())
        detector_euler_layout.addWidget(self.single_line_roty)

        detector_euler_layout.addWidget(QLabel("rotz:"))
        self.single_line_rotz = QLineEdit("0")
        self.single_line_rotz.setValidator(QDoubleValidator())
        detector_euler_layout.addWidget(self.single_line_rotz)

        detector_euler_layout.addWidget(QLabel("Detector rotation order:"))
        self.single_combo_det_rotation_order = QComboBox()
        self.single_combo_det_rotation_order.addItems(["zyx", "zxy", "yzx", "yxz", "xzy", "xyz"])
        self.single_combo_det_rotation_order.setMinimumWidth(180)
        self.single_combo_det_rotation_order.view().setMinimumWidth(180)
        self.single_combo_det_rotation_order.setMinimumContentsLength(8)
        detector_euler_layout.addWidget(self.single_combo_det_rotation_order)

        detector_euler_layout.addStretch()
        self.single_save_poni_button = QPushButton("Save PONI...")
        self.single_save_poni_button.clicked.connect(lambda: save_current_detector(self, "single"))
        det_common_layout.addWidget(self.single_save_poni_button, 3, 0, 1, 6)

        self.single_beam_group = QGroupBox("Beam Parameters")
        beam_layout = QGridLayout()
        self.single_beam_group.setLayout(beam_layout)
        scroll_layout.addWidget(self.single_beam_group)

        beam_layout.addWidget(QLabel("Energy [eV]:"), 0, 0)
        self.single_line_energy = QLineEdit("15000")
        self.single_line_energy.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.single_line_energy, 0, 1)

        beam_layout.addWidget(QLabel("∆E/E [%]:"), 0, 2)
        self.single_line_ebw = QLineEdit("1.5")
        self.single_line_ebw.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.single_line_ebw, 0, 3)

        beam_layout.addWidget(QLabel("Title digits:"), 0, 4)
        self.single_line_title_digits = QLineEdit("2")
        self.single_line_title_digits.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.single_line_title_digits, 0, 5)

        sample_group = QGroupBox("Sample Parameters")
        sample_layout = QGridLayout()
        sample_group.setLayout(sample_layout)
        scroll_layout.addWidget(sample_group)

        row = 0
        sample_layout.addWidget(QLabel("Space Group (e.g. 167):"), row, 0)
        self.single_line_space_group = QLineEdit("167")
        self.single_line_space_group.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_space_group, row, 1)
        row += 1

        sample_layout.addWidget(QLabel("qmax [Å^-1]:"), row, 0)
        self.single_line_qmax = QLineEdit("5")
        self.single_line_qmax.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_qmax, row, 1)
        row += 1

        sample_layout.addWidget(QLabel("a [Å]:"), row, 0)
        self.single_line_sam_a = QLineEdit("4.954")
        self.single_line_sam_a.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_a, row, 1)
        sample_layout.addWidget(QLabel("b [Å]:"), row, 2)
        self.single_line_sam_b = QLineEdit("4.954")
        self.single_line_sam_b.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_b, row, 3)
        sample_layout.addWidget(QLabel("c [Å]:"), row, 4)
        self.single_line_sam_c = QLineEdit("14.01")
        self.single_line_sam_c.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_c, row, 5)
        row += 1

        sample_layout.addWidget(QLabel("alpha [°]:"), row, 0)
        self.single_line_sam_alpha = QLineEdit("90")
        self.single_line_sam_alpha.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_alpha, row, 1)
        sample_layout.addWidget(QLabel("beta [°]:"), row, 2)
        self.single_line_sam_beta = QLineEdit("90")
        self.single_line_sam_beta.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_beta, row, 3)
        sample_layout.addWidget(QLabel("gamma [°]:"), row, 4)
        self.single_line_sam_gamma = QLineEdit("120")
        self.single_line_sam_gamma.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_gamma, row, 5)
        row += 1

        row += 1
        self.single_line_cif_path = DropFileLineEdit(suffixes=(".cif",))
        self.single_line_cif_path.setPlaceholderText("CIF file")
        self.single_line_cif_path.fileDropped.connect(self._single_load_cif)
        self.single_line_cif_path.returnPressed.connect(
            lambda: self._single_load_cif(self.single_line_cif_path.text()))
        sample_layout.addWidget(QLabel("CIF file:"), row, 0)
        sample_layout.addWidget(self.single_line_cif_path, row, 1, 1, 4)
        self.load_cif_button = QPushButton("Load CIF...")
        sample_layout.addWidget(self.load_cif_button, row, 5)

        orientation_checkbox_layout = QHBoxLayout()
        self.single_orientation_checkbox = QCheckBox("Use Custom Initial Orientation")
        orientation_checkbox_layout.addWidget(self.single_orientation_checkbox)

        self.import_from_rotation_button = QPushButton("Import from Rotation Tool")
        orientation_checkbox_layout.addWidget(self.import_from_rotation_button)

        orientation_checkbox_layout.addStretch()
        scroll_layout.addLayout(orientation_checkbox_layout)

        self.single_label_sam_init = QLabel("Initial Crystal Orientation Matrix:")
        self.single_label_sam_init.setVisible(False)
        scroll_layout.addWidget(self.single_label_sam_init)

        self.orientation_group = QGroupBox()
        self.orientation_group.setVisible(False)
        self.orientation_layout = QGridLayout()
        self.orientation_group.setLayout(self.orientation_layout)

        col_headers = ["x", "y", "z"]
        for j, header in enumerate(col_headers):
            label = QLabel(header)
            label.setAlignment(Qt.AlignCenter)
            self.orientation_layout.addWidget(label, 0, j + 1)

        self.orientation_entries = [[QLineEdit("0") for _ in range(3)] for _ in range(3)]
        dummy = ["a", "b", "c"]
        for i, dum in enumerate(dummy):
            row_label = QLabel(f"{dum}")
            row_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.orientation_layout.addWidget(row_label, i + 1, 0)
            for j in range(3):
                entry = self.orientation_entries[i][j]
                entry.setValidator(QDoubleValidator())
                entry.setMinimumWidth(170)
                entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                entry.textChanged.connect(
                    lambda value, field=entry: field.setMinimumWidth(
                        max(170, field.fontMetrics().horizontalAdvance(value) + 48)))
                self.orientation_layout.addWidget(entry, i + 1, j + 1)

        scroll_layout.addWidget(self.orientation_group)

        self.single_extra_hkls_group = QGroupBox("Extra HKLs to Force Include (2D / 3D)")
        extra_hkls_layout = QGridLayout()
        self.single_extra_hkls_group.setLayout(extra_hkls_layout)
        scroll_layout.addWidget(self.single_extra_hkls_group)

        extra_hkls_layout.addWidget(QLabel("extra_hkls:"), 0, 0)
        self.single_line_extra_hkls = QLineEdit("")
        self.single_line_extra_hkls.setPlaceholderText("e.g., [0,0,3],[2,2,1]")
        extra_hkls_layout.addWidget(self.single_line_extra_hkls, 0, 1)

        extra_hkls_layout.addWidget(
            QLabel("These reflections are appended to the automatically allowed hkls."),
            1,
            0,
            1,
            2,
        )

        self.single_sample_rotation_group = QGroupBox("Legacy sample Euler rotations [°]")
        rot_layout = QHBoxLayout()
        self.single_sample_rotation_group.setLayout(rot_layout)
        rot_layout.addWidget(QLabel("rotx:"))
        self.single_line_sam_rotx = QLineEdit("0")
        self.single_line_sam_rotx.setValidator(QDoubleValidator())
        rot_layout.addWidget(self.single_line_sam_rotx)
        rot_layout.addWidget(QLabel("roty:"))
        self.single_line_sam_roty = QLineEdit("0")
        self.single_line_sam_roty.setValidator(QDoubleValidator())
        rot_layout.addWidget(self.single_line_sam_roty)
        rot_layout.addWidget(QLabel("rotz:"))
        self.single_line_sam_rotz = QLineEdit("0")
        self.single_line_sam_rotz.setValidator(QDoubleValidator())
        rot_layout.addWidget(self.single_line_sam_rotz)
        rot_layout.addWidget(QLabel("Rotation order:"))
        self.single_combo_sam_rotation_order = QComboBox()
        self.single_combo_sam_rotation_order.addItems(["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
        self.single_combo_sam_rotation_order.setMinimumWidth(180)
        self.single_combo_sam_rotation_order.view().setMinimumWidth(180)
        self.single_combo_sam_rotation_order.setMinimumContentsLength(8)
        self.single_combo_sam_rotation_order.setToolTip(
            "Order of the legacy sample Euler rotations in the 2D, 3D, and reciprocal simulations.")
        rot_layout.addWidget(self.single_combo_sam_rotation_order)
        rot_layout.addStretch()
        scroll_layout.addWidget(self.single_sample_rotation_group)

        self.single_geometry_group = QGroupBox("Experiment Geometry")
        geometry_layout = QVBoxLayout()
        self.single_geometry_group.setLayout(geometry_layout)
        scroll_layout.addWidget(self.single_geometry_group)

        geometry_mode_row = QHBoxLayout()
        geometry_mode_row.addWidget(QLabel("Geometry mode:"))
        self.single_geometry_mode_combo = QComboBox()
        self.single_geometry_mode_combo.addItems([
            "Legacy Euler",
            "Predefined diffractometer",
            "Custom geometry",
        ])
        geometry_mode_row.addWidget(self.single_geometry_mode_combo)
        geometry_mode_row.addStretch()
        geometry_layout.addLayout(geometry_mode_row)

        self.single_geometry_note = QLabel(
            "Use a predefined diffractometer to drive the geometry-aware single-crystal functions."
        )
        self.single_geometry_note.setWordWrap(True)
        geometry_layout.addWidget(self.single_geometry_note)

        self.single_predefined_geometry_group = QGroupBox("Predefined Diffractometer")
        predefined_layout = QGridLayout()
        self.single_predefined_geometry_group.setLayout(predefined_layout)
        geometry_layout.addWidget(self.single_predefined_geometry_group)

        self.geometry_panel = GeometryPanel()
        predefined_geometries = [
            geom for geom in self.service.list_geometries() if geom.name != "legacy_euler"
        ]
        self.geometry_panel.set_geometries(predefined_geometries)
        self.single_geometry_kind_combo = self.geometry_panel.geometry_combo

        predefined_layout.addWidget(self.geometry_panel, 0, 0, 1, 3)

        predefined_layout.addWidget(QLabel("Geometry kwargs (JSON):"), 1, 0)
        self.single_geometry_kwargs = QPlainTextEdit()
        self.single_geometry_kwargs.setPlaceholderText('{\n  "kappa_tilt_deg": 50.0\n}')
        self.single_geometry_kwargs.setFixedHeight(70)
        predefined_layout.addWidget(self.single_geometry_kwargs, 1, 1, 1, 2)

        self.single_geometry_copy_to_custom_btn = QPushButton("Copy preset into custom editors")
        predefined_layout.addWidget(self.single_geometry_copy_to_custom_btn, 2, 0, 1, 3)

        self.single_custom_angle_group = QGroupBox("Custom Geometry Motor Angles")
        custom_angle_layout = QGridLayout()
        self.single_custom_angle_group.setLayout(custom_angle_layout)
        geometry_layout.addWidget(self.single_custom_angle_group)

        custom_angle_layout.addWidget(QLabel("Sample motor angles (JSON):"), 0, 0)
        self.single_geometry_sample_angles = QPlainTextEdit()
        self.single_geometry_sample_angles.setPlaceholderText(
            '{\n  "omega": 0.0,\n  "kappa": 35.0,\n  "phi": 10.0\n}'
        )
        self.single_geometry_sample_angles.setFixedHeight(80)
        custom_angle_layout.addWidget(self.single_geometry_sample_angles, 0, 1)

        custom_angle_layout.addWidget(QLabel("Detector motor angles (JSON):"), 1, 0)
        self.single_geometry_detector_angles = QPlainTextEdit()
        self.single_geometry_detector_angles.setPlaceholderText('{\n  "tth": 25.0\n}')
        self.single_geometry_detector_angles.setFixedHeight(70)
        custom_angle_layout.addWidget(self.single_geometry_detector_angles, 1, 1)

        self.single_custom_geometry_group = QGroupBox("Custom Geometry Definition")
        custom_geometry_layout = QGridLayout()
        self.single_custom_geometry_group.setLayout(custom_geometry_layout)
        scroll_layout.addWidget(self.single_custom_geometry_group)

        custom_note = QLabel(
            "Define the ordered motor chains explicitly as JSON. "
            "The order of the list is the order in which the motors are applied."
        )
        custom_note.setWordWrap(True)
        custom_geometry_layout.addWidget(custom_note, 0, 0, 1, 2)

        custom_geometry_layout.addWidget(QLabel("Custom sample chain (JSON):"), 1, 0)
        self.single_custom_sample_chain = QPlainTextEdit()
        self.single_custom_sample_chain.setPlaceholderText(
            '[\n'
            '  {"name": "omega", "axis": "z", "origin": [0,0,0], "frame": "lab", "default_angle": 0},\n'
            '  {"name": "kappa", "axis": [0.766044, 0, 0.642788], "origin": [0,0,0], "frame": "local", "default_angle": 0},\n'
            '  {"name": "phi", "axis": "z", "origin": [0,0,0], "frame": "local", "default_angle": 0}\n'
            "]"
        )
        self.single_custom_sample_chain.setFixedHeight(140)
        custom_geometry_layout.addWidget(self.single_custom_sample_chain, 1, 1)

        custom_geometry_layout.addWidget(QLabel("Custom detector chain (JSON):"), 2, 0)
        self.single_custom_detector_chain = QPlainTextEdit()
        self.single_custom_detector_chain.setPlaceholderText(
            '[\n  {"name": "tth", "axis": "y", "origin": [0,0,0], "frame": "lab", "default_angle": 0}\n]'
        )
        self.single_custom_detector_chain.setFixedHeight(100)
        custom_geometry_layout.addWidget(self.single_custom_detector_chain, 2, 1)

        self.single_geometry_scan_group = QGroupBox("Geometry Scan Controls")
        geometry_scan_layout = QVBoxLayout()
        self.single_geometry_scan_group.setLayout(geometry_scan_layout)
        scroll_layout.addWidget(self.single_geometry_scan_group)

        geometry_scan_note = QLabel(
            "These controls are used by the geometry-aware scan functions. "
            "For sample-arm scans, define the two motor names and ranges. "
            "For detector-arm scans, provide a JSON mapping like {\"tth\": [-30, 30, 2]}."
        )
        geometry_scan_note.setWordWrap(True)
        geometry_scan_layout.addWidget(geometry_scan_note)

        self.single_geometry_sample_scan_group = QGroupBox("Sample-arm motor scan")
        sample_scan_layout = QGridLayout()
        self.single_geometry_sample_scan_group.setLayout(sample_scan_layout)
        geometry_scan_layout.addWidget(self.single_geometry_sample_scan_group)

        sample_scan_layout.addWidget(QLabel("Motor 1 name:"), 0, 0)
        self.single_geometry_motor1_name = QLineEdit("omega")
        sample_scan_layout.addWidget(self.single_geometry_motor1_name, 0, 1)

        sample_scan_layout.addWidget(QLabel("Motor 1 range:"), 1, 0)
        self.single_geometry_motor1_range = QLineEdit("-90,90,5")
        self.single_geometry_motor1_range.setPlaceholderText("start,stop,step")
        sample_scan_layout.addWidget(self.single_geometry_motor1_range, 1, 1)

        sample_scan_layout.addWidget(QLabel("Motor 2 name:"), 2, 0)
        self.single_geometry_motor2_name = QLineEdit("kappa")
        sample_scan_layout.addWidget(self.single_geometry_motor2_name, 2, 1)

        sample_scan_layout.addWidget(QLabel("Motor 2 range:"), 3, 0)
        self.single_geometry_motor2_range = QLineEdit("-90,90,5")
        self.single_geometry_motor2_range.setPlaceholderText("start,stop,step")
        sample_scan_layout.addWidget(self.single_geometry_motor2_range, 3, 1)

        self.single_geometry_detector_scan_group = QGroupBox("Detector-arm motor scan")
        detector_scan_layout = QGridLayout()
        self.single_geometry_detector_scan_group.setLayout(detector_scan_layout)
        geometry_scan_layout.addWidget(self.single_geometry_detector_scan_group)

        detector_scan_layout.addWidget(QLabel("Detector scan ranges (JSON):"), 0, 0)
        self.single_geometry_detector_scan_ranges = QPlainTextEdit()
        self.single_geometry_detector_scan_ranges.setPlaceholderText('{\n  "tth": [-30, 30, 2]\n}')
        self.single_geometry_detector_scan_ranges.setFixedHeight(70)
        detector_scan_layout.addWidget(self.single_geometry_detector_scan_ranges, 0, 1)

        self.single_refl_group = QGroupBox("Reflection Lists (for Bragg checks)")
        refl_layout = QGridLayout()
        self.single_refl_group.setLayout(refl_layout)
        scroll_layout.addWidget(self.single_refl_group)

        self.single_label_q = QLabel("q_hkls (comma):")
        refl_layout.addWidget(self.single_label_q, 0, 0)
        self.single_line_q = QLineEdit("")
        self.single_line_q.setPlaceholderText(
            "Currently unused by the exposed single-crystal GUI functions"
        )
        refl_layout.addWidget(self.single_line_q, 0, 1)

        self.single_label_d = QLabel("OR d_hkls (comma):")
        refl_layout.addWidget(self.single_label_d, 1, 0)
        self.single_line_d = QLineEdit("")
        self.single_line_d.setPlaceholderText(
            "Currently unused by the exposed single-crystal GUI functions"
        )
        refl_layout.addWidget(self.single_line_d, 1, 1)

        refl_layout.addWidget(QLabel("hkls_names (e.g. [1,0,2],[0,1,2]):"), 2, 0)
        self.single_line_names = QLineEdit("")
        self.single_line_names.setPlaceholderText("e.g., [1,0,2],[0,1,2]")
        refl_layout.addWidget(self.single_line_names, 2, 1)

        self.single_equiv_checkbox = QCheckBox("Consider Equivalent")
        refl_layout.addWidget(self.single_equiv_checkbox, 3, 0, 1, 2)

        self.single_angle_group = QGroupBox("Angle Range (for Detector Rotations)")
        angle_layout = QHBoxLayout()
        self.single_angle_group.setLayout(angle_layout)
        angle_layout.addWidget(QLabel("angle_range (start,stop,step):"))
        self.single_line_angle = QLineEdit("-90,90,5")
        self.single_line_angle.setPlaceholderText("e.g., -90,90,10")
        angle_layout.addWidget(self.single_line_angle)
        angle_layout.addStretch()
        scroll_layout.addWidget(self.single_angle_group)

        self.param_scan_group = QGroupBox("Parameter Scan (for generalized Bragg condition)")
        param_scan_layout = QGridLayout()
        self.param_scan_group.setLayout(param_scan_layout)
        scroll_layout.addWidget(self.param_scan_group)

        param_scan_layout.addWidget(QLabel("Parameter 1:"), 0, 0)
        self.single_param1_combo = QComboBox()
        self.single_param1_combo.addItems(["rotx", "roty", "rotz", "energy"])
        param_scan_layout.addWidget(self.single_param1_combo, 0, 1)

        param_scan_layout.addWidget(QLabel("Parameter 2:"), 1, 0)
        self.single_param2_combo = QComboBox()
        self.single_param2_combo.addItems(["rotx", "roty", "rotz", "energy"])
        param_scan_layout.addWidget(self.single_param2_combo, 1, 1)

        param_scan_layout.addWidget(QLabel("param1_range (start,stop,step):"), 2, 0)
        self.single_line_param1_range = QLineEdit("-90,90,5")
        self.single_line_param1_range.setPlaceholderText("e.g., -90,90,5 or 5000,20000,500")
        param_scan_layout.addWidget(self.single_line_param1_range, 2, 1)

        param_scan_layout.addWidget(QLabel("param2_range (start,stop,step):"), 3, 0)
        self.single_line_param2_range = QLineEdit("-90,90,5")
        self.single_line_param2_range.setPlaceholderText("e.g., -90,90,5 or 5000,20000,500")
        param_scan_layout.addWidget(self.single_line_param2_range, 3, 1)

        self.param_scan_group.setVisible(False)

        self.mapping_extra_group = QGroupBox('Additional Mapping Sliders')
        extra_layout = QGridLayout(self.mapping_extra_group)
        self.mapping_extra_enabled = QCheckBox('Scan additional parameters')
        self.mapping_extra_controls = ReciprocalScanControls()
        extra_layout.addWidget(self.mapping_extra_enabled, 0, 0, 1, 2)
        extra_layout.addWidget(self.mapping_extra_controls, 1, 0, 1, 2)
        self.mapping_extra_enabled.toggled.connect(self._single_func_changed)
        self.mapping_extra_controls.changed.connect(self._on_widget_state_changed)
        scroll_layout.addWidget(self.mapping_extra_group)

        self.reciprocal_group = QGroupBox('Reciprocal Space and Ewald Construction')
        reciprocal_layout = QGridLayout(self.reciprocal_group)
        self.reciprocal_backend = QComboBox()
        self.reciprocal_backend.addItem('Plotly (browser)', 'plotly')
        self.reciprocal_backend.addItem('Matplotlib', 'matplotlib')
        self.reciprocal_non_bragg = QCheckBox('Show non-Bragg reflections')
        self.reciprocal_non_bragg.setChecked(True)
        self.reciprocal_bandwidth = QCheckBox('Show bandwidth boundaries')
        self.reciprocal_bandwidth.setChecked(True)
        reciprocal_layout.addWidget(QLabel('Renderer:'), 0, 0)
        reciprocal_layout.addWidget(self.reciprocal_backend, 0, 1)
        reciprocal_layout.addWidget(self.reciprocal_non_bragg, 1, 0)
        reciprocal_layout.addWidget(self.reciprocal_bandwidth, 1, 1)
        self.reciprocal_scan_enabled = QCheckBox('Scan configurations')
        self.reciprocal_scan_controls = ReciprocalScanControls()
        reciprocal_layout.addWidget(self.reciprocal_scan_enabled, 2, 0, 1, 2)
        reciprocal_layout.addWidget(self.reciprocal_scan_controls, 3, 0, 1, 2)
        self.reciprocal_scan_enabled.toggled.connect(self._single_func_changed)
        self.reciprocal_scan_controls.changed.connect(self._on_widget_state_changed)
        scroll_layout.addWidget(self.reciprocal_group)

        self.single_2d_scan_group = QGroupBox('2D Simulation Sliders')
        single_2d_scan_layout = QVBoxLayout(self.single_2d_scan_group)
        self.single_2d_scan_enabled = QCheckBox('Scan configurations')
        self.single_2d_scan_controls = ReciprocalScanControls()
        single_2d_scan_layout.addWidget(self.single_2d_scan_enabled)
        single_2d_scan_layout.addWidget(self.single_2d_scan_controls)
        self.single_2d_scan_enabled.toggled.connect(self._single_func_changed)
        self.single_2d_scan_controls.changed.connect(self._on_widget_state_changed)
        scroll_layout.addWidget(self.single_2d_scan_group)

        self.single_3d_scan_group = QGroupBox('3D Simulation Sliders')
        single_3d_scan_layout = QVBoxLayout(self.single_3d_scan_group)
        self.single_3d_scan_enabled = QCheckBox('Scan configurations')
        self.single_3d_scan_controls = ReciprocalScanControls()
        single_3d_scan_layout.addWidget(self.single_3d_scan_enabled)
        single_3d_scan_layout.addWidget(self.single_3d_scan_controls)
        self.single_3d_scan_enabled.toggled.connect(self._single_func_changed)
        self.single_3d_scan_controls.changed.connect(self._on_widget_state_changed)
        scroll_layout.addWidget(self.single_3d_scan_group)

        self.single_inverse_target_group = QGroupBox("Fixed-Energy Pixel Targeting")
        inverse_layout = QGridLayout()
        self.single_inverse_target_group.setLayout(inverse_layout)
        scroll_layout.addWidget(self.single_inverse_target_group)

        inverse_layout.addWidget(QLabel("Target hkl ([h,k,l]):"), 0, 0)
        self.single_line_target_hkl = QLineEdit("[1,1,0]")
        self.single_line_target_hkl.setPlaceholderText("e.g., [1,1,0]")
        inverse_layout.addWidget(self.single_line_target_hkl, 0, 1)

        inverse_layout.addWidget(QLabel("Target pixel H:"), 1, 0)
        self.single_line_target_pixel_h = QLineEdit("900")
        self.single_line_target_pixel_h.setValidator(QDoubleValidator())
        inverse_layout.addWidget(self.single_line_target_pixel_h, 1, 1)

        inverse_layout.addWidget(QLabel("Target pixel V:"), 2, 0)
        self.single_line_target_pixel_v = QLineEdit("900")
        self.single_line_target_pixel_v.setValidator(QDoubleValidator())
        inverse_layout.addWidget(self.single_line_target_pixel_v, 2, 1)

        inverse_layout.addWidget(QLabel("Pixel tolerance [px]:"), 3, 0)
        self.single_line_target_pixel_tol = QLineEdit("20")
        self.single_line_target_pixel_tol.setValidator(QDoubleValidator())
        inverse_layout.addWidget(self.single_line_target_pixel_tol, 3, 1)

        inverse_layout.addWidget(QLabel("eta samples:"), 4, 0)
        self.single_line_eta_samples = QLineEdit("1441")
        self.single_line_eta_samples.setValidator(QDoubleValidator())
        inverse_layout.addWidget(self.single_line_eta_samples, 4, 1)

        inverse_layout.addWidget(QLabel("phi samples:"), 5, 0)
        self.single_line_phi_samples = QLineEdit("361")
        self.single_line_phi_samples.setValidator(QDoubleValidator())
        inverse_layout.addWidget(self.single_line_phi_samples, 5, 1)

        self.single_wrap_angles_checkbox = QCheckBox("Wrap legacy Euler angles to [-180, 180)")
        self.single_wrap_angles_checkbox.setChecked(True)
        inverse_layout.addWidget(self.single_wrap_angles_checkbox, 6, 0, 1, 2)

        self.single_inverse_detector_plot_checkbox = QCheckBox("Show detector-space targeting plot")
        self.single_inverse_detector_plot_checkbox.setChecked(True)
        inverse_layout.addWidget(self.single_inverse_detector_plot_checkbox, 7, 0, 1, 2)

        self.single_inverse_2d_plot_checkbox = QCheckBox("Show 2D motor-space projections")
        self.single_inverse_2d_plot_checkbox.setChecked(True)
        inverse_layout.addWidget(self.single_inverse_2d_plot_checkbox, 8, 0, 1, 2)

        self.single_inverse_3d_plot_checkbox = QCheckBox("Show 3D motor-space family")
        self.single_inverse_3d_plot_checkbox.setChecked(False)
        inverse_layout.addWidget(self.single_inverse_3d_plot_checkbox, 9, 0, 1, 2)

        self.single_inverse_target_group.setVisible(False)

        self.single_run_btn = QPushButton("Run Single-Crystal Function")
        self.results_group = QGroupBox("Scan Computation and Results")
        results_layout = QGridLayout(self.results_group)
        self.mapping_display = QComboBox()
        for label, value in [('Points', 'points'), ('Band outlines', 'outlines'), ('Filled bands', 'bands')]:
            self.mapping_display.addItem(label, value)
        self.mapping_display_label = QLabel('Mapping display:')
        self.mapping_backend = QComboBox()
        self.mapping_backend.addItem('Matplotlib', 'matplotlib')
        self.mapping_backend.addItem('Plotly (browser)', 'plotly')
        self.mapping_backend_label = QLabel('Mapping renderer:')
        results_layout.addWidget(self.mapping_backend_label, 4, 0)
        results_layout.addWidget(self.mapping_backend, 4, 1, 1, 2)
        results_layout.addWidget(self.mapping_display_label, 2, 0)
        results_layout.addWidget(self.mapping_display, 2, 1, 1, 2)
        self.require_detector_hit = QCheckBox("Require detector hit")
        self.require_detector_hit.setToolTip(
            "Keep only Bragg rays reaching the declared detector's pixel bounds. "
            "Applied when plotting, including loaded mappings. Saved Bragg points remain unchanged."
        )
        results_layout.addWidget(self.require_detector_hit, 3, 0, 1, 3)
        self.parallel_checkbox = QCheckBox("Parallel processing")
        self.workers_line = QLineEdit("4")
        self.workers_line.setValidator(QIntValidator(1, 1024, self))
        self.workers_line.setMaximumWidth(120)
        results_layout.addWidget(self.parallel_checkbox, 0, 0)
        self.workers_label = QLabel('Workers:')
        results_layout.addWidget(self.workers_label, 0, 1)
        results_layout.addWidget(self.workers_line, 0, 2)
        self.save_result_button = QPushButton("Save Results...")
        self.load_result_button = QPushButton("Load Results...")
        self.plot_result_button = QPushButton("Plot Loaded / Last Results")
        self.save_result_button.setEnabled(False)
        self.plot_result_button.setEnabled(False)
        self.last_scan_result = None
        self.save_result_button.clicked.connect(self._save_scan_result)
        self.load_result_button.clicked.connect(self._load_scan_result)
        self.plot_result_button.clicked.connect(self._plot_scan_result)
        for column, button in enumerate((self.save_result_button, self.load_result_button, self.plot_result_button)):
            results_layout.addWidget(button, 1, column)
            results_layout.setColumnStretch(column, 1)
        scroll_layout.addWidget(self.results_group)
        self.single_intensity_controls = IntensityControls()
        scroll_layout.addWidget(self.single_intensity_controls)
        scroll_layout.addWidget(self.single_run_btn)

        # Keep geometry selection, motor definitions, and detector settings together.
        scroll_layout.removeWidget(self.single_geometry_group)
        scroll_layout.insertWidget(scroll_layout.indexOf(self.single_det_group), self.single_geometry_group)
        for group in (self.single_custom_geometry_group, self.single_geometry_scan_group):
            scroll_layout.removeWidget(group)
            geometry_layout.addWidget(group)
        scroll_layout.removeWidget(self.single_sample_rotation_group)
        scroll_layout.insertWidget(scroll_layout.indexOf(self.single_extra_hkls_group),
                                   self.single_sample_rotation_group)
        self.orientation_fitting_panel = OrientationFittingPanel(self)
        scroll_layout.insertWidget(scroll_layout.indexOf(self.single_sample_rotation_group) + 1,
                                   self.orientation_fitting_panel)

        self.single_combo_det_type.currentIndexChanged.connect(self._single_detector_changed)
        self.single_combo_det_rotation_order.currentIndexChanged.connect(self._write_back_to_state)
        self.single_combo_sam_rotation_order.currentIndexChanged.connect(self._write_back_to_state)
        self.single_btn_browse_poni.clicked.connect(self._single_browse_poni)
        self.single_line_poni_file.editingFinished.connect(self._single_apply_poni_path_from_line)
        self.load_cif_button.clicked.connect(self._single_load_cif)
        self.import_from_rotation_button.clicked.connect(self._import_orientation_from_rotation_tool)
        self.single_orientation_checkbox.stateChanged.connect(self._toggle_orientation_matrix)
        self.single_func_combo.currentIndexChanged.connect(self._single_func_changed)
        self.require_detector_hit.toggled.connect(self._single_func_changed)
        self.single_geometry_mode_combo.currentIndexChanged.connect(self._single_geometry_mode_changed)
        self.single_geometry_copy_to_custom_btn.clicked.connect(self._load_selected_geometry_into_custom)
        self.geometry_panel.geometry_changed.connect(self._on_geometry_panel_changed)
        self.geometry_panel.angles_changed.connect(self._on_geometry_panel_angles_changed)
        self.single_run_btn.clicked.connect(self._single_run_function)

        self._single_detector_changed()
        self._single_geometry_mode_changed()
        self._single_func_changed()

    # ------------------------------------------------------------------
    # State synchronization
    # ------------------------------------------------------------------
    def load_from_state(self) -> None:
        """Apply persisted single-crystal state without triggering autosaves."""
        self._loading_state = True
        try:
            single = self.state.single
            self.parallel_checkbox.setChecked(single.parallel_processing)
            self.workers_line.setText(single.parallel_workers)
            self.mapping_display.setCurrentIndex(max(0, self.mapping_display.findData(single.mapping_display)))
            self.mapping_extra_enabled.setChecked(single.mapping_extra_enabled)
            self.mapping_extra_controls.set_rows(single.mapping_extra_rows)
            self.mapping_backend.setCurrentIndex(max(0,self.mapping_backend.findData(single.mapping_backend)))
            self.require_detector_hit.setChecked(single.require_detector_hit)
            self.reciprocal_backend.setCurrentIndex(max(0, self.reciprocal_backend.findData(single.reciprocal_backend)))
            self.reciprocal_non_bragg.setChecked(single.reciprocal_non_bragg)
            self.reciprocal_bandwidth.setChecked(single.reciprocal_bandwidth)
            self.reciprocal_scan_controls.set_rows(single.reciprocal_scan_rows)
            self.reciprocal_scan_enabled.setChecked(single.reciprocal_scan_enabled)
            self.single_2d_scan_controls.set_rows(single.single_2d_scan_rows)
            self.single_2d_scan_enabled.setChecked(single.single_2d_scan_enabled)
            self.single_3d_scan_controls.set_rows(single.single_3d_scan_rows)
            self.single_3d_scan_enabled.setChecked(single.single_3d_scan_enabled)

            function_index = self.single_func_combo.findData(single.func)
            if function_index >= 0:
                self.single_func_combo.setCurrentIndex(function_index)
            else:
                self._combo_set_text_if_present(self.single_func_combo, single.func)
            self.single_intensity_controls.load(single)
            self._combo_set_text_if_present(self.single_combo_det_type, single.det_type)
            self._set_line_text(self.single_line_poni_file, getattr(single, "poni_file", ""))

            self._set_line_text(self.single_line_pxsize_h, single.pxsize_h)
            self._set_line_text(self.single_line_pxsize_v, single.pxsize_v)
            self._set_line_text(self.single_line_num_px_h, single.num_px_h)
            self._set_line_text(self.single_line_num_px_v, single.num_px_v)
            self._set_line_text(self.single_line_bin_h, single.bin_h)
            self._set_line_text(self.single_line_bin_v, single.bin_v)
            self._set_line_text(self.single_line_dist, single.dist)
            self._set_line_text(self.single_line_poni1, single.poni1)
            self._set_line_text(self.single_line_poni2, single.poni2)
            self._set_line_text(self.single_line_rotx, single.det_rotx)
            self._set_line_text(self.single_line_roty, single.det_roty)
            self._set_line_text(self.single_line_rotz, single.det_rotz)
            self._combo_set_text_if_present(
                self.single_combo_det_rotation_order,
                getattr(single, "det_rotation_order", "zyx"),
            )
            self._set_line_text(self.single_line_energy, single.energy)
            self._set_line_text(self.single_line_ebw, single.ebw)
            self._set_line_text(
                self.single_line_title_digits,
                getattr(single, "title_digits", "2"),
            )
            self._set_line_text(self.single_line_space_group, single.space_group)
            self._set_line_text(self.single_line_qmax, single.qmax)
            self._set_line_text(self.single_line_sam_a, single.a)
            self._set_line_text(self.single_line_sam_b, single.b)
            self._set_line_text(self.single_line_sam_c, single.c)
            self._set_line_text(self.single_line_sam_alpha, single.alpha)
            self._set_line_text(self.single_line_sam_beta, single.beta)
            self._set_line_text(self.single_line_sam_gamma, single.gamma)

            self.single_orientation_checkbox.setChecked(bool(single.use_custom_orientation))
            self._apply_matrix_texts(self.orientation_entries, single.orientation_matrix)

            self._set_line_text(self.single_line_sam_rotx, single.sam_rotx)
            self._set_line_text(self.single_line_sam_roty, single.sam_roty)
            self._set_line_text(self.single_line_sam_rotz, single.sam_rotz)
            rotation_order = single.sam_rotation_order
            if (single.func == 'scan_two_parameters_for_Bragg_condition'
                    and single.mapping_rotation_order != rotation_order):
                rotation_order = single.mapping_rotation_order
            self._combo_set_text_if_present(
                self.single_combo_sam_rotation_order, rotation_order)

            self._set_line_text(self.single_line_q, single.q)
            self._set_line_text(self.single_line_d, single.d)
            self._set_line_text(self.single_line_names, single.names)
            self._set_line_text(self.single_line_extra_hkls, single.extra_hkls)
            self.single_equiv_checkbox.setChecked(bool(single.equiv))
            self._set_line_text(self.single_line_angle, single.angle_range)

            self._combo_set_text_if_present(self.single_param1_combo, single.param1)
            self._combo_set_text_if_present(self.single_param2_combo, single.param2)
            self._set_line_text(self.single_line_param1_range, single.param1_range)
            self._set_line_text(self.single_line_param2_range, single.param2_range)

            self._combo_set_text_if_present(self.single_geometry_mode_combo, single.geometry_mode)

            geometry_name = single.geometry_kind or self.geometry_panel.current_geometry_name()
            if geometry_name:
                try:
                    self.geometry_panel.set_current_geometry(geometry_name)
                except Exception:
                    pass

            self._set_plain_text(self.single_geometry_kwargs, single.geometry_kwargs)
            self._set_plain_text(self.single_geometry_sample_angles, single.geometry_sample_angles)
            self._set_plain_text(self.single_geometry_detector_angles, single.geometry_detector_angles)
            self._set_plain_text(self.single_custom_sample_chain, single.custom_sample_chain)
            self._set_plain_text(self.single_custom_detector_chain, single.custom_detector_chain)
            self._set_line_text(self.single_geometry_motor1_name, single.geometry_motor1_name)
            self._set_line_text(self.single_geometry_motor1_range, single.geometry_motor1_range)
            self._set_line_text(self.single_geometry_motor2_name, single.geometry_motor2_name)
            self._set_line_text(self.single_geometry_motor2_range, single.geometry_motor2_range)
            self._set_plain_text(
                self.single_geometry_detector_scan_ranges,
                single.geometry_detector_scan_ranges,
            )

            self._set_line_text(self.single_line_target_hkl, single.target_hkl)
            self._set_line_text(self.single_line_target_pixel_h, single.target_pixel_h)
            self._set_line_text(self.single_line_target_pixel_v, single.target_pixel_v)
            self._set_line_text(self.single_line_target_pixel_tol, single.target_pixel_tol)
            self._set_line_text(self.single_line_eta_samples, single.eta_samples)
            self._set_line_text(self.single_line_phi_samples, single.phi_samples)

            self.single_wrap_angles_checkbox.setChecked(bool(single.wrap_angles))
            self.single_inverse_detector_plot_checkbox.setChecked(bool(single.inverse_detector_plot))
            self.single_inverse_2d_plot_checkbox.setChecked(bool(single.inverse_2d_plot))
            self.single_inverse_3d_plot_checkbox.setChecked(bool(single.inverse_3d_plot))

            self.single_cif_file_path = self.state.paths.single_cif_file_path
            self._set_line_text(self.single_line_cif_path, self.single_cif_file_path)

            self._restore_predefined_panel_angles_from_state()

            self._single_detector_changed()
            self._single_geometry_mode_changed()
            self._single_func_changed()
            self._toggle_orientation_matrix(
                Qt.Checked if self.single_orientation_checkbox.isChecked() else Qt.Unchecked
            )
        finally:
            self._loading_state = False
            self._write_back_to_state()
            if self._selected_single_function() == 'find_orientation_from_measured_braggs':
                self.orientation_fitting_panel.ensure_project()

    def _apply_fitted_orientation(self, project, candidate, pose_id=None):
        """Install a fitted zero-motor orientation and selected pose conditions."""
        import numpy as np

        from ...diffractometers import make_euler_geometry, motor_chain_to_dict
        from ...orientation_fit.forward_model import zero_motor_direct_lattice
        from ...orientation_fit.observations import resolve_pose
        from ...orientation_fit.setup import normalize_goniometer

        reference = resolve_pose(project, project.reference_pose_id or project.poses[0].id)
        chosen = resolve_pose(project, pose_id or reference.id)
        detector = next(item for item in project.detectors if item.id == chosen.detector_setup_id)
        if not self._single_geometry_enabled():
            from itertools import permutations
            for letters in permutations('xyz'):
                order = ''.join(letters)
                euler = make_euler_geometry(sample_rotation_order=order,
                                            detector_rotation_order=order)
                if motor_chain_to_dict(euler.sample)['motors'] == project.sample_chain['motors']:
                    self.single_combo_sam_rotation_order.setCurrentText(order)
                if (detector.motor_chain is not None and
                        motor_chain_to_dict(euler.detector)['motors'] ==
                        detector.motor_chain['motors']):
                    self.single_combo_det_rotation_order.setCurrentText(order)
        active = self._build_selected_geometry() if self._single_geometry_enabled() else None
        chain = normalize_goniometer(active,
                                     euler_order=self.single_combo_sam_rotation_order.currentText())
        current_detector_chain = (active.detector if active is not None else
                                  make_euler_geometry(detector_rotation_order=
                                      self.single_combo_det_rotation_order.currentText()).detector)
        sample_motors_match = (motor_chain_to_dict(chain)['motors'] ==
                               project.sample_chain['motors'])
        detector_motors_match = (detector.motor_chain is None or
                                 motor_chain_to_dict(current_detector_chain)['motors'] ==
                                 detector.motor_chain['motors'])
        if not sample_motors_match or not detector_motors_match:
            self.single_custom_sample_chain.setPlainText(
                _pretty_json(project.sample_chain['motors']))
            self.single_custom_detector_chain.setPlainText(
                _pretty_json(detector.motor_chain['motors']
                             if detector.motor_chain is not None else []))
            self.single_geometry_mode_combo.setCurrentText('Custom geometry')
            active = self._build_selected_geometry()
        zero_motor_matrix = zero_motor_direct_lattice(project, candidate.reference_matrix)
        if zero_motor_matrix.shape != (3, 3) or not np.isfinite(zero_motor_matrix).all():
            raise ValueError('The fitted orientation matrix is invalid.')
        self.single_orientation_checkbox.setChecked(True)
        for row in range(3):
            for column in range(3):
                self.orientation_entries[row][column].setText(
                    f'{zero_motor_matrix[row, column]:.9g}')
        self.single_line_energy.setText(f'{chosen.energy_eV:.9g}')
        for name, value in project.lattice.items():
            field = getattr(self, 'single_line_sam_' + name, None)
            if field is not None:
                field.setText(str(value))
        if 'space_group' in project.lattice:
            self.single_line_space_group.setText(str(project.lattice['space_group']))
        if active is None:
            for name in ('rotx', 'roty', 'rotz'):
                getattr(self, 'single_line_sam_' + name).setText(
                    f'{chosen.motor_angles.get("sam_" + name, 0):.9g}')
        elif self._single_predefined_geometry_enabled():
            self.geometry_panel.set_sample_angles(chosen.motor_angles)
        else:
            self.single_geometry_sample_angles.setPlainText(_pretty_json(chosen.motor_angles))
        detector_field_names = {
            'pxsize_h': 'pxsize_h', 'pxsize_v': 'pxsize_v',
            'num_pixels_h': 'num_px_h', 'num_pixels_v': 'num_px_v',
            'dist': 'dist', 'poni1': 'poni1', 'poni2': 'poni2',
            'rotx': 'rotx', 'roty': 'roty', 'rotz': 'rotz',
        }
        self.single_combo_det_type.setCurrentText('manual')
        for parameter, field in detector_field_names.items():
            getattr(self, 'single_line_' + field).setText(str(detector.parameters[parameter]))
        self.single_combo_det_rotation_order.setCurrentText(detector.parameters['rotation_order'])
        if active is None:
            for name in ('rotx', 'roty', 'rotz'):
                self._set_line_text(getattr(self, 'single_line_' + name),
                                    chosen.detector_angles.get('det_' + name,
                                                               detector.parameters[name]))
        else:
            if self._single_predefined_geometry_enabled():
                self.geometry_panel.set_detector_angles(chosen.detector_angles)
            else:
                self.single_geometry_detector_angles.setPlainText(_pretty_json(chosen.detector_angles))
        self._write_back_to_state()

    def save_to_state(self) -> None:
        """Copy current controls into the shared GUI state."""
        self._write_back_to_state()

    def _write_back_to_state(self) -> None:
        """Serialize all single-crystal and geometry controls to state."""
        if self._loading_state:
            return
        single = self.state.single
        single.parallel_processing = self.parallel_checkbox.isChecked()
        single.parallel_workers = self.workers_line.text()
        single.mapping_display = self.mapping_display.currentData()
        single.mapping_extra_enabled = self.mapping_extra_enabled.isChecked()
        single.mapping_extra_rows = self.mapping_extra_controls.rows()
        single.mapping_rotation_order = self.single_combo_sam_rotation_order.currentText()
        single.mapping_backend = self.mapping_backend.currentData()
        single.require_detector_hit = self.require_detector_hit.isChecked()
        single.reciprocal_backend = self.reciprocal_backend.currentData()
        single.reciprocal_non_bragg = self.reciprocal_non_bragg.isChecked()
        single.reciprocal_bandwidth = self.reciprocal_bandwidth.isChecked()
        single.reciprocal_scan_enabled = self.reciprocal_scan_enabled.isChecked()
        single.reciprocal_scan_rows = self.reciprocal_scan_controls.rows()
        single.single_2d_scan_enabled = self.single_2d_scan_enabled.isChecked()
        single.single_2d_scan_rows = self.single_2d_scan_controls.rows()
        single.single_3d_scan_enabled = self.single_3d_scan_enabled.isChecked()
        single.single_3d_scan_rows = self.single_3d_scan_controls.rows()

        single.func = self._selected_single_function()
        self.single_intensity_controls.save(single)
        single.det_type = self.single_combo_det_type.currentText()
        single.poni_file = self.single_line_poni_file.text()
        single.pxsize_h = self.single_line_pxsize_h.text()
        single.pxsize_v = self.single_line_pxsize_v.text()
        single.num_px_h = self.single_line_num_px_h.text()
        single.num_px_v = self.single_line_num_px_v.text()
        single.bin_h = self.single_line_bin_h.text()
        single.bin_v = self.single_line_bin_v.text()
        single.dist = self.single_line_dist.text()
        single.poni1 = self.single_line_poni1.text()
        single.poni2 = self.single_line_poni2.text()
        single.det_rotx = self.single_line_rotx.text()
        single.det_roty = self.single_line_roty.text()
        single.det_rotz = self.single_line_rotz.text()
        single.det_rotation_order = self.single_combo_det_rotation_order.currentText()
        single.energy = self.single_line_energy.text()
        single.ebw = self.single_line_ebw.text()
        single.title_digits = self.single_line_title_digits.text()
        single.space_group = self.single_line_space_group.text()
        single.qmax = self.single_line_qmax.text()
        single.a = self.single_line_sam_a.text()
        single.b = self.single_line_sam_b.text()
        single.c = self.single_line_sam_c.text()
        single.alpha = self.single_line_sam_alpha.text()
        single.beta = self.single_line_sam_beta.text()
        single.gamma = self.single_line_sam_gamma.text()
        single.use_custom_orientation = self.single_orientation_checkbox.isChecked()
        single.orientation_matrix = self._matrix_texts(self.orientation_entries)
        single.sam_rotx = self.single_line_sam_rotx.text()
        single.sam_roty = self.single_line_sam_roty.text()
        single.sam_rotz = self.single_line_sam_rotz.text()
        single.sam_rotation_order = self.single_combo_sam_rotation_order.currentText()
        single.q = self.single_line_q.text()
        single.d = self.single_line_d.text()
        single.names = self.single_line_names.text()
        single.extra_hkls = self.single_line_extra_hkls.text()
        single.equiv = self.single_equiv_checkbox.isChecked()
        single.angle_range = self.single_line_angle.text()
        single.param1 = self.single_param1_combo.currentText()
        single.param2 = self.single_param2_combo.currentText()
        single.param1_range = self.single_line_param1_range.text()
        single.param2_range = self.single_line_param2_range.text()
        single.geometry_mode = self.single_geometry_mode_combo.currentText()
        single.geometry_kind = self.geometry_panel.current_geometry_name()
        single.geometry_kwargs = self.single_geometry_kwargs.toPlainText()

        if self._single_predefined_geometry_enabled():
            single.geometry_sample_angles = _pretty_json(self.geometry_panel.current_sample_angles())
            single.geometry_detector_angles = _pretty_json(self.geometry_panel.current_detector_angles())
        else:
            single.geometry_sample_angles = self.single_geometry_sample_angles.toPlainText()
            single.geometry_detector_angles = self.single_geometry_detector_angles.toPlainText()

        single.custom_sample_chain = self.single_custom_sample_chain.toPlainText()
        single.custom_detector_chain = self.single_custom_detector_chain.toPlainText()
        single.geometry_motor1_name = self.single_geometry_motor1_name.text()
        single.geometry_motor1_range = self.single_geometry_motor1_range.text()
        single.geometry_motor2_name = self.single_geometry_motor2_name.text()
        single.geometry_motor2_range = self.single_geometry_motor2_range.text()
        single.geometry_detector_scan_ranges = self.single_geometry_detector_scan_ranges.toPlainText()
        single.target_hkl = self.single_line_target_hkl.text()
        single.target_pixel_h = self.single_line_target_pixel_h.text()
        single.target_pixel_v = self.single_line_target_pixel_v.text()
        single.target_pixel_tol = self.single_line_target_pixel_tol.text()
        single.eta_samples = self.single_line_eta_samples.text()
        single.phi_samples = self.single_line_phi_samples.text()
        single.wrap_angles = self.single_wrap_angles_checkbox.isChecked()
        single.inverse_detector_plot = self.single_inverse_detector_plot_checkbox.isChecked()
        single.inverse_2d_plot = self.single_inverse_2d_plot_checkbox.isChecked()
        single.inverse_3d_plot = self.single_inverse_3d_plot_checkbox.isChecked()

        self.single_cif_file_path = self.single_line_cif_path.text().strip() or None
        self.state.paths.single_cif_file_path = self.single_cif_file_path

        if not self._loading_state:
            self.state_changed.emit()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _toggle_orientation_matrix(self, state) -> None:
        """Enable or disable the custom orientation-matrix editor."""
        on = state == Qt.Checked
        self.single_label_sam_init.setVisible(on)
        self.orientation_group.setVisible(on)
        self._write_back_to_state()

    def _single_geometry_mode(self) -> str:
        """Return the selected geometry mode label."""
        return self.single_geometry_mode_combo.currentText()

    def _single_geometry_enabled(self) -> bool:
        """Return whether any geometry-aware mode is active."""
        return self._single_geometry_mode() in {"Predefined diffractometer", "Custom geometry"}

    def _single_predefined_geometry_enabled(self) -> bool:
        """Return whether predefined geometry controls are active."""
        return self._single_geometry_mode() == "Predefined diffractometer"

    def _single_custom_geometry_enabled(self) -> bool:
        """Return whether custom motor-chain controls are active."""
        return self._single_geometry_mode() == "Custom geometry"

    def _selected_single_function(self) -> str:
        """Use the stable function key where a friendly selector label exists."""
        return self.single_func_combo.currentData() or self.single_func_combo.currentText()

    def _single_func_changed(self) -> None:
        """Show only controls required by the selected backend function."""
        func = self._selected_single_function()
        fitting = func == 'find_orientation_from_measured_braggs'
        self.orientation_fitting_panel.setVisible(fitting)
        self.single_run_btn.setVisible(not fitting)
        if fitting and not self._loading_state:
            self.orientation_fitting_panel.ensure_project()
        self.single_intensity_controls.setVisible(func in ("simulate_2d", "simulate_3d"))
        geometry_enabled = self._single_geometry_enabled()
        self.reciprocal_group.setVisible(func == 'simulate_reciprocal')
        self.single_2d_scan_group.setVisible(func == 'simulate_2d')
        self.single_2d_scan_controls.setVisible(self.single_2d_scan_enabled.isChecked())
        self.single_3d_scan_group.setVisible(func == 'simulate_3d')
        self.single_3d_scan_controls.setVisible(self.single_3d_scan_enabled.isChecked())
        scan = self.reciprocal_scan_enabled.isChecked()
        self.reciprocal_scan_controls.setVisible(scan)
        parameters = ['energy', 'sam_rotx', 'sam_roty', 'sam_rotz', 'det_rotx', 'det_roty', 'det_rotz', 'e_bandwidth']
        if geometry_enabled:
            try:
                geometry = self._build_selected_geometry()
                parameters = ['energy', 'e_bandwidth'] + ['sample:' + name for name in geometry.sample.motor_names] + ['detector:' + name for name in geometry.detector.motor_names]
            except (ValueError, TypeError, KeyError):
                pass
        self.reciprocal_scan_controls.set_parameters(parameters)
        self.single_2d_scan_controls.set_parameters(parameters)
        self.single_3d_scan_controls.set_parameters(parameters)

        needs_detector_info = func in [
            "simulate_2d",
            "simulate_3d",
            "find_orientation_from_measured_braggs",
            "target_hkl_near_pixel_fixed_energy",
            "detector_rotations_collecting_Braggs",
        ]

        needs_bragg = func in [
            "detector_rotations_collecting_Braggs",
            "scan_two_parameters_for_Bragg_condition",
        ]

        needs_param_scan = func == "scan_two_parameters_for_Bragg_condition"
        self.mapping_extra_group.setVisible(needs_param_scan)
        self.mapping_extra_controls.setVisible(self.mapping_extra_enabled.isChecked())
        extra_parameters = ['rotx', 'roty', 'rotz', 'energy', 'e_bandwidth']
        if geometry_enabled:
            try:
                extra_parameters = list(self._build_selected_geometry().sample.motor_names) + ['energy', 'e_bandwidth']
            except (ValueError, TypeError, KeyError):
                pass
        self.mapping_extra_controls.set_parameters(extra_parameters)
        reciprocal = func == 'simulate_reciprocal'
        scans = needs_bragg or func == 'target_hkl_near_pixel_fixed_energy'
        self.results_group.setVisible(scans or reciprocal)
        self.results_group.setTitle('Reciprocal Space Results' if reciprocal else 'Scan Computation and Results')
        for widget in (self.parallel_checkbox, self.workers_line, self.workers_label):
            widget.setVisible(scans or (reciprocal and (scan or 'scan' in getattr(getattr(self, 'last_scan_result', None), 'data', {}))))
        for widget in (self.mapping_display, self.mapping_display_label, self.mapping_backend, self.mapping_backend_label):
            widget.setVisible(needs_param_scan)
        self.require_detector_hit.setVisible(needs_param_scan or reciprocal)
        self.require_detector_hit.setToolTip('Highlight Bragg reflections reaching the detector in green.' if reciprocal
            else 'Plot only Bragg rays reaching the detector; saved points remain unchanged.')
        needs_detector_info = needs_detector_info or ((needs_param_scan or reciprocal) and self.require_detector_hit.isChecked()) or (reciprocal and scan)
        needs_angle_range = (func == "detector_rotations_collecting_Braggs") and (not geometry_enabled)

        needs_extra_hkls = func in [
            "simulate_2d",
            "simulate_3d",
        ]

        needs_inverse_targeting = func == "target_hkl_near_pixel_fixed_energy"

        geometry_supported = func in [
            "simulate_2d",
            "simulate_3d",
            "find_orientation_from_measured_braggs",
            "simulate_reciprocal",
            "target_hkl_near_pixel_fixed_energy",
            "detector_rotations_collecting_Braggs",
            "scan_two_parameters_for_Bragg_condition",
        ]

        self.single_det_group.setVisible(needs_detector_info)
        self.single_manual_group.setVisible(needs_detector_info)

        self.single_extra_hkls_group.setVisible(needs_extra_hkls)
        self.single_refl_group.setVisible(needs_bragg)
        self.single_angle_group.setVisible(needs_angle_range)
        self.param_scan_group.setVisible(needs_param_scan and not geometry_enabled)
        self.single_inverse_target_group.setVisible(needs_inverse_targeting)

        self.single_geometry_group.setVisible(geometry_supported)
        self.single_custom_geometry_group.setVisible(
            geometry_supported and self._single_custom_geometry_enabled()
        )

        self.single_geometry_scan_group.setVisible(
            geometry_supported and geometry_enabled and needs_bragg
        )
        self.single_geometry_sample_scan_group.setVisible(
            geometry_supported
            and geometry_enabled
            and func == "scan_two_parameters_for_Bragg_condition"
        )
        self.single_geometry_detector_scan_group.setVisible(
            geometry_supported
            and geometry_enabled
            and func == "detector_rotations_collecting_Braggs"
        )

        self.single_label_q.setEnabled(False)
        self.single_line_q.setEnabled(False)
        self.single_label_d.setEnabled(False)
        self.single_line_d.setEnabled(False)

        self._write_back_to_state()

    def _single_geometry_mode_changed(self) -> None:
        """Update geometry panels, legacy fields, and state for a mode change."""
        predefined = self._single_predefined_geometry_enabled()
        custom = self._single_custom_geometry_enabled()
        enabled = predefined or custom
        legacy = not enabled

        self.single_predefined_geometry_group.setVisible(predefined)
        self.single_custom_angle_group.setVisible(custom)
        self.single_custom_geometry_group.setEnabled(custom)
        self.single_geometry_scan_group.setEnabled(enabled)

        self.single_sample_rotation_group.setEnabled(legacy)
        self.single_detector_euler_group.setEnabled(legacy)
        self.single_wrap_angles_checkbox.setEnabled(legacy)

        if legacy:
            self.single_geometry_note.setText(
                "Use a predefined diffractometer to drive the geometry-aware single-crystal functions. "
                "Legacy sample and detector Euler rotations are active in this mode."
            )
        elif predefined:
            self.single_geometry_note.setText(
                "Geometry mode is active. Choose a predefined diffractometer and edit its motor "
                "angles directly in the geometry panel below."
            )
        else:
            self.single_geometry_note.setText(
                "Custom geometry mode is active. Define the ordered sample and detector motor chains "
                "as JSON, then provide the fixed motor angles as JSON dictionaries."
            )

        self._single_func_changed()
        self._write_back_to_state()

    def _single_detector_changed(self) -> None:
        """Update manual detector fields for the selected detector source."""
        det_type = self.single_combo_det_type.currentText().lower()
        update_detector_fields(self, "single")
        self._write_back_to_state()

    def _on_geometry_panel_changed(self, _geometry_name: str) -> None:
        """Reset factory options and persist a predefined geometry selection."""
        if self._single_predefined_geometry_enabled():
            default_kwargs = self.service.default_constructor_kwargs(
                self.geometry_panel.current_geometry_name()
            )
            if default_kwargs and not self.single_geometry_kwargs.toPlainText().strip():
                self._set_plain_text(self.single_geometry_kwargs, _pretty_json(default_kwargs))
        self._write_back_to_state()

    def _on_geometry_panel_angles_changed(self) -> None:
        """Persist edits made in dynamic predefined motor controls."""
        self._write_back_to_state()

    # ------------------------------------------------------------------
    # Geometry construction
    # ------------------------------------------------------------------
    def _build_predefined_geometry(self):
        """Instantiate the selected registered geometry from editable JSON options."""
        kind = self.geometry_panel.current_geometry_name().strip()
        if not kind:
            raise ValueError("Select a diffractometer geometry.")
        kwargs = _parse_json_object(self.single_geometry_kwargs.toPlainText(), "Geometry kwargs")
        kwargs = kwargs or {}
        return self.service.build_predefined_geometry(kind, **kwargs)

    def _build_custom_geometry(self):
        """Build a diffractometer from user-supplied sample/detector chain JSON."""
        sample_chain = _parse_json_value(
            self.single_custom_sample_chain.toPlainText(),
            "Custom sample chain",
            default=[],
        )
        detector_chain = _parse_json_value(
            self.single_custom_detector_chain.toPlainText(),
            "Custom detector chain",
            default=[],
        )

        if sample_chain is None:
            sample_chain = []
        if detector_chain is None:
            detector_chain = []

        return DiffractometerGeometry.from_dict({
            "name": "custom_geometry",
            "sample": sample_chain,
            "detector": detector_chain,
        })

    def _build_selected_geometry(self):
        """Return the active geometry object, or ``None`` in legacy mode."""
        if self._single_predefined_geometry_enabled():
            return self._build_predefined_geometry()
        if self._single_custom_geometry_enabled():
            return self._build_custom_geometry()
        return None

    def _single_geometry_sample_angles_dict(self):
        """Return current sample angles from predefined or custom controls."""
        if self._single_predefined_geometry_enabled():
            angles = self.geometry_panel.current_sample_angles()
            return angles or None

        angles = _parse_json_object(
            self.single_geometry_sample_angles.toPlainText(),
            "Sample motor angles",
        )
        return angles or None

    def _single_geometry_detector_angles_dict(self):
        """Return current detector angles from predefined or custom controls."""
        if self._single_predefined_geometry_enabled():
            angles = self.geometry_panel.current_detector_angles()
            return angles or None

        angles = _parse_json_object(
            self.single_geometry_detector_angles.toPlainText(),
            "Detector motor angles",
        )
        return angles or None

    def _single_detector_scan_ranges_dict(self):
        """Decode detector motor scan ranges from the JSON editor."""
        scan_ranges = _parse_json_object(
            self.single_geometry_detector_scan_ranges.toPlainText(),
            "Detector scan ranges",
            allow_empty=False,
        )
        if not scan_ranges:
            raise ValueError("Detector scan ranges cannot be empty in geometry mode.")
        return scan_ranges

    def _load_selected_geometry_into_custom(self) -> None:
        """Serialize the current predefined geometry into custom-chain editors."""
        try:
            geometry = self._build_predefined_geometry()
            geom_dict = diffractometers.diffractometer_to_dict(geometry)
            self.single_custom_sample_chain.setPlainText(_pretty_json(geom_dict["sample"]["motors"]))
            self.single_custom_detector_chain.setPlainText(_pretty_json(geom_dict["detector"]["motors"]))

            sample_defaults = self.geometry_panel.current_sample_angles()
            detector_defaults = self.geometry_panel.current_detector_angles()
            self.single_geometry_sample_angles.setPlainText(_pretty_json(sample_defaults))
            self.single_geometry_detector_angles.setPlainText(_pretty_json(detector_defaults))

            self.single_geometry_mode_combo.setCurrentText("Custom geometry")
        except Exception as e:
            QMessageBox.critical(self, "Geometry Copy Error", str(e))

    def _restore_predefined_panel_angles_from_state(self) -> None:
        """Restore saved dynamic motor values after rendering a geometry panel."""
        sample_text = (self.state.single.geometry_sample_angles or "").strip()
        detector_text = (self.state.single.geometry_detector_angles or "").strip()

        try:
            sample_angles = _parse_json_object(sample_text, "Sample motor angles") if sample_text else None
        except Exception:
            sample_angles = None

        try:
            detector_angles = (
                _parse_json_object(detector_text, "Detector motor angles") if detector_text else None
            )
        except Exception:
            detector_angles = None

        if sample_angles:
            self.geometry_panel.set_sample_angles(sample_angles)
        if detector_angles:
            self.geometry_panel.set_detector_angles(detector_angles)

    # ------------------------------------------------------------------
    # Matrix / CIF helpers
    # ------------------------------------------------------------------
    def collect_matrix(self):
        """Parse and return the custom sample orientation matrix when enabled."""
        import numpy as np

        matrix = np.zeros((3, 3))
        try:
            for i in range(3):
                for j in range(3):
                    value = self.orientation_entries[i][j].text().strip()
                    if value == "":
                        raise ValueError(f"Matrix element at Row {i + 1}, Col {j + 1} is empty.")
                    matrix[i][j] = float(value)
            return matrix
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid matrix entry: {str(e)}")
            return None

    def _format_poni_value(self, value) -> str:
        """Format a parsed PONI scalar compactly for a line edit."""
        return "" if value is None else f"{value:.12g}"

    def _single_apply_poni_file(self, file_name: str) -> None:
        """Parse a PONI file and populate single-crystal detector controls."""
        poni = read_poni_file(file_name)
        detector_kwargs = poni.detector_kwargs(include_rotations=True)

        self.single_line_poni_file.setText(file_name)

        idx = self.single_combo_det_type.findText("poni")
        if idx >= 0:
            self.single_combo_det_type.setCurrentIndex(idx)

        self._set_line_text(self.single_line_pxsize_h, self._format_poni_value(detector_kwargs["pxsize_h"]))
        self._set_line_text(self.single_line_pxsize_v, self._format_poni_value(detector_kwargs["pxsize_v"]))
        self._set_line_text(self.single_line_num_px_h, str(detector_kwargs["num_pixels_h"]))
        self._set_line_text(self.single_line_num_px_v, str(detector_kwargs["num_pixels_v"]))
        self._set_line_text(self.single_line_dist, self._format_poni_value(detector_kwargs["dist"]))
        self._set_line_text(self.single_line_poni1, self._format_poni_value(detector_kwargs["poni1"]))
        self._set_line_text(self.single_line_poni2, self._format_poni_value(detector_kwargs["poni2"]))
        self._set_line_text(self.single_line_rotx, self._format_poni_value(detector_kwargs["rotx"]))
        self._set_line_text(self.single_line_roty, self._format_poni_value(detector_kwargs["roty"]))
        self._set_line_text(self.single_line_rotz, self._format_poni_value(detector_kwargs["rotz"]))
        self._combo_set_text_if_present(self.single_combo_det_rotation_order, "zyx")

        self._write_back_to_state()

    def _single_browse_poni(self) -> None:
        """Select and apply a PONI detector calibration file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open PONI File",
            directory=str(
                self.path_service.dialog_start_path(
                    current=self.single_line_poni_file.text()
                )
            ),
            filter="PONI Files (*.poni);;All Files (*)",
        )
        if file_name:
            self.path_service.remember_dialog_selection(file_name)
            self._single_apply_poni_file(file_name)

    def _single_apply_poni_path_from_line(self) -> None:
        """Apply a typed or dropped PONI path without opening a file dialog."""
        file_name = (self.single_line_poni_file.text() or "").strip()
        if not file_name:
            self._write_back_to_state()
            return

        path = Path(file_name).expanduser()
        if path.suffix.lower() != ".poni" or not path.is_file():
            self._write_back_to_state()
            return

        try:
            resolved = str(path.resolve())
            self.path_service.remember_dialog_selection(resolved)
            self._single_apply_poni_file(resolved)
        except Exception as e:
            QMessageBox.critical(
                self,
                "PONI Error",
                f"Failed to read or parse the PONI file:\n{str(e)}",
            )

    def _single_load_cif(self, file_name=None) -> None:
        """Select or load the current CIF into sample fields."""
        if not isinstance(file_name, str):
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                caption="Open CIF File",
                directory=str(self.path_service.dialog_start_path(current=self.single_cif_file_path)),
                filter="CIF Files (*.cif);;All Files (*)",
            )
        if not file_name:
            return
        try:
            self.path_service.remember_dialog_selection(file_name)
            self._load_cif_into_fields(file_name)
            self.single_cif_file_path = file_name
            self.single_line_cif_path.setText(file_name)
            self.state.paths.single_cif_file_path = file_name
            self._write_back_to_state()
            QMessageBox.information(self, "CIF Loaded", f"Successfully loaded: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "CIF Error", f"Failed to read or parse the CIF file:\n{str(e)}")

    def _load_cif_into_fields(self, file_name: str) -> Cif:
        """Copy unit-cell and space-group values from a CIF into the form."""
        cif_data = Cif(file_path=file_name)

        if cif_data.space_group is not None:
            self.single_line_space_group.setText(str(cif_data.space_group))
        if cif_data.a is not None:
            self.single_line_sam_a.setText(str(cif_data.a))
        if cif_data.b is not None:
            self.single_line_sam_b.setText(str(cif_data.b))
        if cif_data.c is not None:
            self.single_line_sam_c.setText(str(cif_data.c))
        if cif_data.alpha is not None:
            self.single_line_sam_alpha.setText(str(cif_data.alpha))
        if cif_data.beta is not None:
            self.single_line_sam_beta.setText(str(cif_data.beta))
        if cif_data.gamma is not None:
            self.single_line_sam_gamma.setText(str(cif_data.gamma))

        return cif_data

    def _import_orientation_from_rotation_tool(self) -> None:
        """Copy the matrix tool's current valid matrix into sample controls."""
        if self.matrix_window is None:
            QMessageBox.warning(
                self,
                "Rotation Tool Not Available",
                "The matrix rotation tool is not connected yet.",
            )
            return

        try:
            mat = self.matrix_window.get_current_matrix()

            if not self.single_orientation_checkbox.isChecked():
                self.single_orientation_checkbox.setChecked(True)

            for i in range(3):
                for j in range(3):
                    self.orientation_entries[i][j].setText(f"{mat[i, j]:.6f}")

            self._write_back_to_state()

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Could not import matrix:\n{str(e)}")

    # ------------------------------------------------------------------
    # Run logic
    # ------------------------------------------------------------------
    def _single_run_function(self) -> bool:
        if getattr(self, '_simulation_busy', False):
            return False
        try:
            parallel = self.parallel_checkbox.isChecked() and self._selected_single_function() in (
                'scan_two_parameters_for_Bragg_condition', 'target_hkl_near_pixel_fixed_energy',
                'detector_rotations_collecting_Braggs', 'simulate_reciprocal')
            workers = int(self.workers_line.text()) if parallel else None
            if workers is not None and workers < 1:
                raise ValueError('Workers must be positive.')
            request = self._single_run_function_impl()
            if not request:
                return False
            return start_simulation(self, request, self.single_run_btn,
                parallel=parallel, workers=workers,
                metadata={'function': self._selected_single_function(),
                          'display_mode': self.mapping_display.currentData(),
                          'gui_state': self.state.to_dict(include_log=False)},
                on_result=self._receive_scan_result)
        except Exception as exc:
            QMessageBox.critical(self, "Simulation execution", str(exc))
            return False

    def _single_run_function_impl(self) -> bool:
        """Validate inputs and dispatch the selected single-crystal workflow."""
        self._write_back_to_state()

        func = self._selected_single_function()
        intensity_options = (self.single_intensity_controls.options(
            self.single_line_cif_path.text().strip())
            if func in ("simulate_2d", "simulate_3d") else {})
        det_type = self.single_combo_det_type.currentText()
        poni_file = (self.single_line_poni_file.text() or "").strip() or None
        backend_det_type, backend_poni_file = self._backend_detector_source(det_type, poni_file)

        try:
            if func == 'simulate_reciprocal':
                return self._reciprocal_request()
            pxsize_h = float(self.single_line_pxsize_h.text())
            pxsize_v = float(self.single_line_pxsize_v.text())
            num_px_h = int(float(self.single_line_num_px_h.text()))
            num_px_v = int(float(self.single_line_num_px_v.text()))

            bin_h = int(float(self.single_line_bin_h.text()))
            bin_v = int(float(self.single_line_bin_v.text()))
            dist = float(self.single_line_dist.text())
            poni1 = float(self.single_line_poni1.text())
            poni2 = float(self.single_line_poni2.text())
            rotx = float(self.single_line_rotx.text())
            roty = float(self.single_line_roty.text())
            rotz = float(self.single_line_rotz.text())
            det_rotation_order = self.single_combo_det_rotation_order.currentText()

            energy = float(self.single_line_energy.text())
            e_bw = float(self.single_line_ebw.text())
            title_digits = self._parse_title_digits(self.single_line_title_digits.text())

            sam_space_group = int(float(self.single_line_space_group.text().strip()))
            qmax = float(self.single_line_qmax.text().strip())

            sam_a = float(self.single_line_sam_a.text())
            sam_b = float(self.single_line_sam_b.text())
            sam_c = float(self.single_line_sam_c.text())
            sam_alpha = float(self.single_line_sam_alpha.text())
            sam_beta = float(self.single_line_sam_beta.text())
            sam_gamma = float(self.single_line_sam_gamma.text())

            sam_initial_crystal_orientation = None
            if self.single_orientation_checkbox.isChecked():
                sam_initial_crystal_orientation = self.collect_matrix()
                if sam_initial_crystal_orientation is None:
                    return False

            sam_rotx = float(self.single_line_sam_rotx.text())
            sam_roty = float(self.single_line_sam_roty.text())
            sam_rotz = float(self.single_line_sam_rotz.text())
            sam_rotation_order = self.single_combo_sam_rotation_order.currentText()

            angle_str = self.single_line_angle.text().strip()
            if angle_str:
                a_start, a_stop, a_step = [float(x.strip()) for x in angle_str.split(",")]
                angle_range = (a_start, a_stop, a_step)
            else:
                angle_range = None

            _ = _parse_csv_floats(self.single_line_q.text())
            _ = _parse_csv_floats(self.single_line_d.text())
            hkls_names = _parse_hkls_string(self.single_line_names.text())
            extra_hkls = _parse_hkls_string(self.single_line_extra_hkls.text())

            geometry = self._build_selected_geometry() if self._single_geometry_enabled() else None
            sample_angles = (
                self._single_geometry_sample_angles_dict()
                if self._single_geometry_enabled()
                else None
            )
            detector_angles = (
                self._single_geometry_detector_angles_dict()
                if self._single_geometry_enabled()
                else None
            )

            if func == "simulate_2d":
                scan_ranges = None
                if self.single_2d_scan_enabled.isChecked():
                    from ...single_crystal_scan import prepare_axes
                    scan_ranges = self.single_2d_scan_controls.ranges()
                    prepare_axes(scan_ranges, sample_geometry=geometry,
                                 detector_geometry=geometry)
                if geometry is not None:
                    return simulation_request(single_crystal.simulate_2d_with_geometry,
                        det_type=backend_det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
                        det_poni_file=backend_poni_file,
                        det_dist=dist,
                        det_poni1=poni1,
                        det_poni2=poni2,
                        det_rotx=rotx,
                        det_roty=roty,
                        det_rotz=rotz,
                        det_rotation_order=det_rotation_order,
                        energy=energy,
                        e_bandwidth=e_bw,
                        sam_space_group=sam_space_group,
                        sam_a=sam_a,
                        sam_b=sam_b,
                        sam_c=sam_c,
                        sam_alpha=sam_alpha,
                        sam_beta=sam_beta,
                        sam_gamma=sam_gamma,
                        sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                        sam_rotx=sam_rotx,
                        sam_roty=sam_roty,
                        sam_rotz=sam_rotz,
                        sam_rotation_order=sam_rotation_order,
                        qmax=qmax,
                        extra_hkls=extra_hkls,
                        geometry=geometry,
                        sample_angles=sample_angles,
                        detector_angles=detector_angles,
                        title_digits=title_digits,
                        scan_ranges=scan_ranges,
                        **intensity_options,
                    )
                else:
                    return simulation_request(single_crystal.simulate_2d,
                        det_type=backend_det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
                        det_poni_file=backend_poni_file,
                        det_dist=dist,
                        det_poni1=poni1,
                        det_poni2=poni2,
                        det_rotx=rotx,
                        det_roty=roty,
                        det_rotz=rotz,
                        det_rotation_order=det_rotation_order,
                        energy=energy,
                        e_bandwidth=e_bw,
                        sam_space_group=sam_space_group,
                        sam_a=sam_a,
                        sam_b=sam_b,
                        sam_c=sam_c,
                        sam_alpha=sam_alpha,
                        sam_beta=sam_beta,
                        sam_gamma=sam_gamma,
                        sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                        sam_rotx=sam_rotx,
                        sam_roty=sam_roty,
                        sam_rotz=sam_rotz,
                        sam_rotation_order=sam_rotation_order,
                        qmax=qmax,
                        extra_hkls=extra_hkls,
                        title_digits=title_digits,
                        scan_ranges=scan_ranges,
                        **intensity_options,
                    )

            elif func == "simulate_3d":
                scan_ranges = None
                if self.single_3d_scan_enabled.isChecked():
                    from ...single_crystal_scan import prepare_axes
                    scan_ranges = self.single_3d_scan_controls.ranges()
                    prepare_axes(scan_ranges, sample_geometry=geometry,
                                 detector_geometry=geometry)
                if geometry is not None:
                    return simulation_request(single_crystal.simulate_3d_with_geometry,
                        det_type=backend_det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
                        det_poni_file=backend_poni_file,
                        det_dist=dist,
                        det_poni1=poni1,
                        det_poni2=poni2,
                        det_rotx=rotx,
                        det_roty=roty,
                        det_rotz=rotz,
                        det_rotation_order=det_rotation_order,
                        energy=energy,
                        e_bandwidth=e_bw,
                        sam_space_group=sam_space_group,
                        sam_a=sam_a,
                        sam_b=sam_b,
                        sam_c=sam_c,
                        sam_alpha=sam_alpha,
                        sam_beta=sam_beta,
                        sam_gamma=sam_gamma,
                        sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                        sam_rotx=sam_rotx,
                        sam_roty=sam_roty,
                        sam_rotz=sam_rotz,
                        sam_rotation_order=sam_rotation_order,
                        qmax=qmax,
                        extra_hkls=extra_hkls,
                        geometry=geometry,
                        sample_angles=sample_angles,
                        detector_angles=detector_angles,
                        title_digits=title_digits,
                        scan_ranges=scan_ranges,
                        **intensity_options,
                    )
                else:
                    return simulation_request(single_crystal.simulate_3d,
                        det_type=backend_det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
                        det_poni_file=backend_poni_file,
                        det_dist=dist,
                        det_poni1=poni1,
                        det_poni2=poni2,
                        det_rotx=rotx,
                        det_roty=roty,
                        det_rotz=rotz,
                        det_rotation_order=det_rotation_order,
                        energy=energy,
                        e_bandwidth=e_bw,
                        sam_space_group=sam_space_group,
                        sam_a=sam_a,
                        sam_b=sam_b,
                        sam_c=sam_c,
                        sam_alpha=sam_alpha,
                        sam_beta=sam_beta,
                        sam_gamma=sam_gamma,
                        sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                        sam_rotx=sam_rotx,
                        sam_roty=sam_roty,
                        sam_rotz=sam_rotz,
                        sam_rotation_order=sam_rotation_order,
                        qmax=qmax,
                        extra_hkls=extra_hkls,
                        title_digits=title_digits,
                        scan_ranges=scan_ranges,
                        **intensity_options,
                    )

            elif func == "target_hkl_near_pixel_fixed_energy":
                target_hkl = _parse_single_hkl_string(self.single_line_target_hkl.text())
                if target_hkl is None:
                    raise ValueError("You must provide exactly one target hkl in the form [h,k,l].")

                target_pixel = (
                    float(self.single_line_target_pixel_h.text()),
                    float(self.single_line_target_pixel_v.text()),
                )
                pixel_tolerance_px = float(self.single_line_target_pixel_tol.text())
                eta_samples = int(float(self.single_line_eta_samples.text()))
                phi_samples = int(float(self.single_line_phi_samples.text()))
                display_wrapped_angles = self.single_wrap_angles_checkbox.isChecked()
                do_detector_plot = self.single_inverse_detector_plot_checkbox.isChecked()
                do_2d_plot = self.single_inverse_2d_plot_checkbox.isChecked()
                do_3d_plot = self.single_inverse_3d_plot_checkbox.isChecked()

                return simulation_request(single_crystal.target_hkl_near_pixel_fixed_energy,
                    det_type=backend_det_type,
                    det_pxsize_h=pxsize_h,
                    det_pxsize_v=pxsize_v,
                    det_ntum_pixels_h=num_px_h,
                    det_num_pixels_v=num_px_v,
                    det_binning=(bin_h, bin_v),
                    det_poni_file=backend_poni_file,
                    det_dist=dist,
                    det_poni1=poni1,
                    det_poni2=poni2,
                    det_rotx=rotx,
                    det_roty=roty,
                    det_rotz=rotz,
                    det_rotation_order=det_rotation_order,
                    energy=energy,
                    sam_space_group=sam_space_group,
                    sam_a=sam_a,
                    sam_b=sam_b,
                    sam_c=sam_c,
                    sam_alpha=sam_alpha,
                    sam_beta=sam_beta,
                    sam_gamma=sam_gamma,
                    sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                    sam_rotx=sam_rotx,
                    sam_roty=sam_roty,
                    sam_rotz=sam_rotz,
                    target_hkl=target_hkl,
                    target_pixel=target_pixel,
                    pixel_tolerance_px=pixel_tolerance_px,
                    eta_samples=eta_samples,
                    phi_samples=phi_samples,
                    display_wrapped_angles=display_wrapped_angles,
                    do_detector_plot=do_detector_plot,
                    do_2d_plot=do_2d_plot,
                    do_3d_plot=do_3d_plot,
                    geometry=geometry,
                    sample_angles=sample_angles,
                    detector_angles=detector_angles,
                )


            elif func == "detector_rotations_collecting_Braggs":
                if geometry is not None:
                    scan_ranges = self._single_detector_scan_ranges_dict()
                    return simulation_request(single_crystal.detector_rotations_collecting_Braggs_with_geometry,
                        det_type=backend_det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
                        det_poni_file=backend_poni_file,
                        det_dist=dist,
                        det_poni1=poni1,
                        det_poni2=poni2,
                        energy=energy,
                        e_bandwidth=e_bw,
                        sam_space_group=sam_space_group,
                        sam_a=sam_a,
                        sam_b=sam_b,
                        sam_c=sam_c,
                        sam_alpha=sam_alpha,
                        sam_beta=sam_beta,
                        sam_gamma=sam_gamma,
                        sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                        sam_rotx=sam_rotx,
                        sam_roty=sam_roty,
                        sam_rotz=sam_rotz,
                        qmax=qmax,
                        hkls=hkls_names,
                        geometry=geometry,
                        scan_ranges=scan_ranges,
                        fixed_detector_angles=detector_angles,
                    )
                else:
                    if angle_range is None:
                        angle_range = (-90, 90, 10)

                    return simulation_request(single_crystal.detector_rotations_collecting_Braggs,
                        det_type=backend_det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
                        det_poni_file=backend_poni_file,
                        det_dist=dist,
                        det_poni1=poni1,
                        det_poni2=poni2,
                        angle_range=angle_range,
                        energy=energy,
                        e_bandwidth=e_bw,
                        sam_space_group=sam_space_group,
                        sam_a=sam_a,
                        sam_b=sam_b,
                        sam_c=sam_c,
                        sam_alpha=sam_alpha,
                        sam_beta=sam_beta,
                        sam_gamma=sam_gamma,
                        sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                        sam_rotx=sam_rotx,
                        sam_roty=sam_roty,
                        sam_rotz=sam_rotz,
                        qmax=qmax,
                        hkls=hkls_names,
                    )

            elif func == "scan_two_parameters_for_Bragg_condition":
                if hkls_names is None:
                    raise ValueError("You must specify hkls_names for Bragg-condition mapping.")

                hkl_equivalent = self.single_equiv_checkbox.isChecked()
                # Store the complete Bragg map; detector acceptance is a display option.
                require_detector_hit = False
                extra_ranges = self.mapping_extra_controls.ranges() if self.mapping_extra_enabled.isChecked() else None
                if self.mapping_extra_enabled.isChecked() and not extra_ranges:
                    raise ValueError('Add at least one extra mapping parameter.')
                detector_obj = None

                if geometry is not None:
                    motor1_name = self.single_geometry_motor1_name.text().strip()
                    motor2_name = self.single_geometry_motor2_name.text().strip()
                    if not motor1_name or not motor2_name:
                        raise ValueError("Geometry motor scan requires two motor names.")
                    if motor1_name == motor2_name:
                        raise ValueError("Geometry motor scan requires two different motor names.")

                    motor1_range = _parse_range_triplet_text(
                        self.single_geometry_motor1_range.text(),
                        "Motor 1 range",
                    )
                    motor2_range = _parse_range_triplet_text(
                        self.single_geometry_motor2_range.text(),
                        "Motor 2 range",
                    )

                    return simulation_request(single_crystal.scan_two_motors_for_Bragg_condition,
                        extra_scan_ranges=extra_ranges,
                        motor1_name=motor1_name,
                        motor2_name=motor2_name,
                        motor1_range=motor1_range,
                        motor2_range=motor2_range,
                        sam_space_group=sam_space_group,
                        sam_a=sam_a,
                        sam_b=sam_b,
                        sam_c=sam_c,
                        sam_alpha=sam_alpha,
                        sam_beta=sam_beta,
                        sam_gamma=sam_gamma,
                        sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                        sam_rotation_order="xyz",
                        energy=energy,
                        e_bandwidth=e_bw,
                        hkls_names=hkls_names,
                        hkl_equivalent=hkl_equivalent,
                        geometry=geometry,
                        fixed_angles=sample_angles,
                        require_detector_hit=require_detector_hit,
                        detector_obj=detector_obj,
                        detector_angles=detector_angles,
                    )
                else:
                    param1_name = self.single_param1_combo.currentText()
                    param2_name = self.single_param2_combo.currentText()

                    if param1_name == param2_name:
                        raise ValueError("Parameter 1 and Parameter 2 must be different.")

                    p1_start, p1_stop, p1_step = [
                        float(x.strip()) for x in self.single_line_param1_range.text().split(",")
                    ]
                    p2_start, p2_stop, p2_step = [
                        float(x.strip()) for x in self.single_line_param2_range.text().split(",")
                    ]
                    param1_range = (p1_start, p1_stop, p1_step)
                    param2_range = (p2_start, p2_stop, p2_step)

                    return simulation_request(single_crystal.scan_two_parameters_for_Bragg_condition,
                        extra_scan_ranges=extra_ranges,
                        param1_name=param1_name,
                        param2_name=param2_name,
                        param1_range=param1_range,
                        param2_range=param2_range,
                        sam_space_group=sam_space_group,
                        sam_a=sam_a,
                        sam_b=sam_b,
                        sam_c=sam_c,
                        sam_alpha=sam_alpha,
                        sam_beta=sam_beta,
                        sam_gamma=sam_gamma,
                        sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                        sam_rotx=sam_rotx,
                        sam_roty=sam_roty,
                        sam_rotz=sam_rotz,
                        sam_rotation_order=sam_rotation_order,
                        energy=energy,
                        e_bandwidth=e_bw,
                        hkls_names=hkls_names,
                        hkl_equivalent=hkl_equivalent,
                        require_detector_hit=require_detector_hit,
                        detector_obj=detector_obj,
                    )


        except Exception as e:
            self.last_error_message = f"{type(e).__name__}: {e}"
            self.last_error_traceback = traceback.format_exc()
            print(self.last_error_traceback)
            QMessageBox.critical(
                self,
                "Single Crystal Error",
                f"{self.last_error_message}\n\nThe full traceback was written to the run log and terminal.",
            )
            self.run_completed.emit(False, func)
            return False

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _reciprocal_options(self):
        return dict(backend=self.reciprocal_backend.currentData(),
                    show_non_bragg=self.reciprocal_non_bragg.isChecked(),
                    show_bandwidth=self.reciprocal_bandwidth.isChecked(),
                    require_detector_hit=self.require_detector_hit.isChecked(),
                    detector_obj=current_detector(self, 'single') if self.require_detector_hit.isChecked() else None)

    def _reciprocal_request(self):
        orientation = self.collect_matrix() if self.single_orientation_checkbox.isChecked() else None
        if self.single_orientation_checkbox.isChecked() and orientation is None:
            return False
        geometry = self._build_selected_geometry() if self._single_geometry_enabled() else None
        options = self._reciprocal_options()
        ranges = None
        if self.reciprocal_scan_enabled.isChecked():
            from ...reciprocal_scan import scan_axes
            ranges = self.reciprocal_scan_controls.ranges()
            scan_axes(ranges)
            options['detector_obj'] = current_detector(self, 'single')
        return simulation_request(single_crystal.simulate_reciprocal,
            **{'sam_' + name: float(getattr(self, 'single_line_sam_' + name).text())
               for name in ('a', 'b', 'c', 'alpha', 'beta', 'gamma', 'rotx', 'roty', 'rotz')},
            sam_rotation_order=self.single_combo_sam_rotation_order.currentText(),
            sam_space_group=int(float(self.single_line_space_group.text())),
            sam_initial_crystal_orientation=orientation,
            energy=float(self.single_line_energy.text()), e_bandwidth=float(self.single_line_ebw.text()),
            qmax=float(self.single_line_qmax.text()), geometry=geometry,
            sample_angles=self._single_geometry_sample_angles_dict() if geometry is not None else None,
            title_digits=self._parse_title_digits(self.single_line_title_digits.text()),
            scan_ranges=ranges,
            detector_angles=self._single_geometry_detector_angles_dict() if geometry is not None else None,
            **options)

    def _receive_scan_result(self, result):
        self.last_scan_result = result
        self.save_result_button.setEnabled(True)
        self.plot_result_button.setEnabled(True)
        self._single_func_changed()
        if result.kind not in ('targeting', 'reciprocal'):
            self._plot_scan_result()

    def _save_scan_result(self):
        if self.last_scan_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save Simulation Results', 'simulation.h5',
                                             'HDF5 results (*.h5)')
        if not path:
            return
        path = Path(path).expanduser()
        if not path.suffix:
            path = path.with_suffix('.h5')
        try:
            save_result(path, self.last_scan_result, overwrite=path.exists())
        except Exception as exc:
            QMessageBox.critical(self, 'Save results', str(exc))

    def _load_scan_result(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load Simulation Results', '', 'HDF5 results (*.h5 *.hdf5)')
        if not path:
            return
        try:
            result = load_result(path)
            self.last_scan_result = result
            function = {'reciprocal': 'simulate_reciprocal',
                        'parameter_mapping': 'scan_two_parameters_for_Bragg_condition',
                        'targeting': 'target_hkl_near_pixel_fixed_energy',
                        'detector_mapping': 'detector_rotations_collecting_Braggs'}.get(result.kind)
            if function:
                self.single_func_combo.setCurrentText(function)
            self.save_result_button.setEnabled(True)
            self.plot_result_button.setEnabled(True)
            self._single_func_changed()
        except Exception as exc:
            QMessageBox.critical(self, 'Load results', str(exc))

    def _plot_scan_result(self):
        if self.last_scan_result is None or getattr(self, '_mapping_plot_busy', False):
            return
        try:
            if self.last_scan_result.kind in ('parameter_mapping', 'reciprocal'):
                from copy import deepcopy
                from ...mapping_filter import filtered_mapping
                from ...deferred_plotting import capture_plots
                from ...parallel import cancellation, SimulationCancelled
                from threading import Event
                from ....analysis.gui.widgets.task_output_dialog import run_task_with_output_dialog
                result = self.last_scan_result
                reciprocal = result.kind == 'reciprocal'
                reciprocal_options = deepcopy(self._reciprocal_options()) if reciprocal else None
                if reciprocal and 'scan' in result.data:
                    reciprocal_options.update(parallel=self.parallel_checkbox.isChecked(),
                        workers=int(self.workers_line.text()) if self.parallel_checkbox.isChecked() else None)
                filtered = not reciprocal and self.require_detector_hit.isChecked()
                sliced = result.data.get('format') == 'mapping-slices-v1'
                detector_obj = deepcopy(current_detector(self, 'single')) if filtered else None
                display_mode = self.mapping_display.currentData()
                mapping_backend = self.mapping_backend.currentData()
                browser_mapping = not reciprocal and mapping_backend == 'plotly'
                mapping_parallel = self.parallel_checkbox.isChecked()
                mapping_workers = int(self.workers_line.text()) if mapping_parallel else None
                cancel_event = Event()

                def compute():
                    try:
                        with cancellation(cancel_event):
                            if reciprocal or browser_mapping:
                                with capture_plots() as plots:
                                    if reciprocal:
                                        plot_result(result, reciprocal_options=reciprocal_options)
                                    else:
                                        plot_result(result, mapping_backend='plotly',display_mode=display_mode,
                                            require_detector_hit=filtered,detector_obj=detector_obj,
                                            mapping_parallel=mapping_parallel,mapping_workers=mapping_workers)
                                return plots
                            return filtered_mapping(result, detector_obj) if filtered and not sliced else result
                    except SimulationCancelled:
                        return {'cancelled': True}

                def completed(view):
                    if isinstance(view, dict) and view.get('cancelled'):
                        return
                    with plt.ion():
                        if reciprocal or browser_mapping:
                            for function, args, kwargs in view:
                                function(*args, **kwargs)
                        else:
                            plot_result(view, display_mode=display_mode,
                                        require_detector_hit=filtered if sliced else False,
                                        detector_obj=detector_obj)

                def failed(error):
                    self.last_error_traceback = error
                    self.last_error_message = error.strip().splitlines()[-1]
                    print(error)

                self._mapping_plot_busy = True
                self.plot_result_button.setEnabled(False)
                dialog = run_task_with_output_dialog(self, 'Prepare Simulation Plot', compute,
                    on_success=completed, on_error=failed, cancel_callback=cancel_event.set)
                self._mapping_plot_dialog = dialog

                def finished():
                    self._mapping_plot_busy = False
                    self.plot_result_button.setEnabled(True)
                    dialog._worker.func = None

                dialog._thread.finished.connect(finished)
                return
            with plt.ion():
                plot_result(self.last_scan_result,
                            display_mode=self.mapping_display.currentData(),
                            detector_plot=self.single_inverse_detector_plot_checkbox.isChecked(),
                            plot_2d=self.single_inverse_2d_plot_checkbox.isChecked(),
                            plot_3d=self.single_inverse_3d_plot_checkbox.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, 'Plot results', str(exc))

    def _connect_stateful_widgets(self) -> None:
        """Connect editable controls to shared-state change notifications."""
        def maybe_connect(widget, signal_name: str) -> None:
            signal = getattr(widget, signal_name, None)
            if signal is not None:
                signal.connect(self._on_widget_state_changed)

        for line in self.findChildren(QLineEdit):
            maybe_connect(line, "textChanged")
        for combo in self.findChildren(QComboBox):
            maybe_connect(combo, "currentTextChanged")
        for chk in self.findChildren(QCheckBox):
            maybe_connect(chk, "stateChanged")
        for plain in self.findChildren(QPlainTextEdit):
            maybe_connect(plain, "textChanged")

    def _on_widget_state_changed(self, *_args) -> None:
        """Persist a user edit unless state is currently being restored."""
        if self._loading_state:
            return
        self._write_back_to_state()

    @staticmethod
    @contextmanager
    def _blocked(widget):
        """Temporarily block Qt signals for programmatic widget updates."""
        was_blocked = widget.blockSignals(True)
        try:
            yield
        finally:
            widget.blockSignals(was_blocked)

    def _set_line_text(self, widget: QLineEdit, value: str | None) -> None:
        """Set line-edit text without emitting change signals."""
        with self._blocked(widget):
            widget.setText("" if value is None else str(value))

    def _set_plain_text(self, widget: QPlainTextEdit, value: str | None) -> None:
        """Set a plain-text editor without emitting change signals."""
        with self._blocked(widget):
            widget.setPlainText("" if value is None else str(value))

    def _combo_set_text_if_present(self, combo: QComboBox, value: str | None) -> None:
        """Select matching combo text without modifying unavailable values."""
        if value is None:
            return

        idx = combo.findText(str(value))
        with self._blocked(combo):
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif combo.isEditable():
                combo.setEditText(str(value))

    @staticmethod
    def _parse_title_digits(text: str | None) -> int:
        """Return a bounded integer precision for figure-title numbers."""
        try:
            digits = int(float((text or "").strip()))
        except (TypeError, ValueError):
            digits = 2
        return max(0, min(digits, 12))

    @staticmethod
    def _backend_detector_source(det_type: str, poni_file: str | None):
        """
        Return detector source values for backend calls.

        PONI loading fills editable fields in the GUI. Once loaded, those fields
        are sent explicitly so manual edits are not overwritten by re-reading
        the calibration file.
        """
        if str(det_type).lower() == "poni":
            return "poni", None
        return "manual", None

    def _matrix_texts(self, matrix_edits) -> list[list[str]]:
        """Serialize a two-dimensional line-edit matrix to nested strings."""
        return [[cell.text() for cell in row] for row in matrix_edits]

    def _apply_matrix_texts(self, matrix_edits, values) -> None:
        """Apply a bounded nested sequence to the orientation editor."""
        if not values:
            return
        for i, row in enumerate(values[:len(matrix_edits)]):
            for j, value in enumerate(row[:len(matrix_edits[i])]):
                matrix_edits[i][j].setText(str(value))
