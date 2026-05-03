import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyFAI.detectors

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QMessageBox, QGroupBox, QCheckBox, QScrollArea,
    QFileDialog, QPlainTextEdit,
)
from PyQt5.QtCore import Qt, QTimer, QByteArray
from PyQt5.QtGui import QDoubleValidator

# Simulation imports
from . import polycrystalline, single_crystal, diffractometers
from .geometry import DiffractometerGeometry

from .utils import apply_rotation
from .plot import plot_parameter_mapping

# CIF import
from .cif import Cif

# crystallographic helpers for Poly tab auto-reflections
from . import utils as xutils
from . import sample as sample_mod

plt.ion()

GUI_STATE_VERSION = 4
AUTOSAVE_FILENAME = ".xrdpy_simulation_gui_last_session.json"


def compute_lattice_orientation(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    v1 = np.array([a, 0.0, 0.0])
    v2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])

    v3_x = c * np.cos(beta)
    v3_y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)

    term = 1 - np.cos(beta) ** 2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)) ** 2
    term = np.maximum(term, 0)
    v3_z = c * np.sqrt(term)

    v3 = np.array([v3_x, v3_y, v3_z])
    return np.vstack((v1, v2, v3))


def _parse_hkls_string(names_text: str):
    """
    Parse hkls string like: [1,0,2],[0,1,2]
    Returns np.ndarray shape (N,3) dtype=int
    """
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
    text = (text or "").strip()
    if not text:
        return None
    return np.array([float(x.strip()) for x in text.split(",")], dtype=float)


def _parse_single_hkl_string(text: str):
    hkls = _parse_hkls_string(text)
    if hkls is None:
        return None
    if hkls.shape[0] != 1:
        raise ValueError("Expected exactly one Miller triplet in the format [h,k,l].")
    return hkls[0]


def _parse_json_value(text: str, name: str, default=None):
    text = (text or "").strip()
    if not text:
        return default

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON.\n{exc}") from exc


def _parse_json_object(text: str, name: str, allow_empty=True):
    obj = _parse_json_value(text, name, default={} if allow_empty else None)
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise ValueError(f"{name} must be a JSON object/dictionary.")
    return obj


def _parse_range_triplet_text(text: str, name: str):
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


def _pretty_json(obj):
    return json.dumps(obj, indent=2)


def _available_geometry_kinds():
    try:
        return diffractometers.available_diffractometers()
    except Exception:
        registry = getattr(diffractometers, "DIFFRACTOMETER_REGISTRY", {})
        return sorted(registry)


class MatrixRotationWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Matrix Rotation Tool")
        self.resize(450, 450)

        self._result_matrix_valid = False

        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QVBoxLayout()
        container.setLayout(main_layout)

        lattice_box = QGroupBox("Optional: Lattice & Load CIF")
        lattice_layout = QGridLayout()
        lattice_box.setLayout(lattice_layout)
        main_layout.addWidget(lattice_box)

        row = 0
        lattice_layout.addWidget(QLabel("Space Group:"), row, 0)
        self.line_space_group = QLineEdit("1")
        self.line_space_group.setValidator(QDoubleValidator())
        lattice_layout.addWidget(self.line_space_group, row, 1)
        row += 1

        lattice_layout.addWidget(QLabel("a [Å]:"), row, 0)
        self.line_a = QLineEdit("1")
        self.line_a.setValidator(QDoubleValidator())
        lattice_layout.addWidget(self.line_a, row, 1)
        row += 1

        lattice_layout.addWidget(QLabel("b [Å]:"), row, 0)
        self.line_b = QLineEdit("1")
        self.line_b.setValidator(QDoubleValidator())
        lattice_layout.addWidget(self.line_b, row, 1)
        row += 1

        lattice_layout.addWidget(QLabel("c [Å]:"), row, 0)
        self.line_c = QLineEdit("1")
        self.line_c.setValidator(QDoubleValidator())
        lattice_layout.addWidget(self.line_c, row, 1)
        row += 1

        lattice_layout.addWidget(QLabel("alpha [deg]:"), row, 0)
        self.line_alpha = QLineEdit("90")
        self.line_alpha.setValidator(QDoubleValidator())
        lattice_layout.addWidget(self.line_alpha, row, 1)
        row += 1

        lattice_layout.addWidget(QLabel("beta [deg]:"), row, 0)
        self.line_beta = QLineEdit("90")
        self.line_beta.setValidator(QDoubleValidator())
        lattice_layout.addWidget(self.line_beta, row, 1)
        row += 1

        lattice_layout.addWidget(QLabel("gamma [deg]:"), row, 0)
        self.line_gamma = QLineEdit("90")
        self.line_gamma.setValidator(QDoubleValidator())
        lattice_layout.addWidget(self.line_gamma, row, 1)
        row += 1

        row += 1
        load_cif_btn = QPushButton("Load CIF")
        lattice_layout.addWidget(load_cif_btn, row, 0, 1, 2)
        load_cif_btn.clicked.connect(self._load_cif)

        row += 1
        compute_btn = QPushButton("Compute Orientation from Lattice")
        lattice_layout.addWidget(compute_btn, row, 0, 1, 2)
        compute_btn.clicked.connect(self._compute_orientation)

        orientation_group = QGroupBox("Orientation Matrix (manually editable)")
        orientation_layout = QGridLayout()
        orientation_group.setLayout(orientation_layout)
        main_layout.addWidget(orientation_group)

        self.matrix_edits = []
        for i in range(3):
            row_edits = []
            for j in range(3):
                edit = QLineEdit("0.0")
                edit.setValidator(QDoubleValidator())
                edit.setFixedWidth(60)
                edit.textChanged.connect(self._invalidate_result_matrix)
                orientation_layout.addWidget(edit, i, j)
                row_edits.append(edit)
            self.matrix_edits.append(row_edits)

        rotation_box = QGroupBox("Apply Rotation [deg]")
        rotation_layout = QHBoxLayout()
        rotation_box.setLayout(rotation_layout)
        main_layout.addWidget(rotation_box)

        rotation_layout.addWidget(QLabel("rotx:"))
        self.line_rotx = QLineEdit("0")
        self.line_rotx.setValidator(QDoubleValidator())
        self.line_rotx.textChanged.connect(self._invalidate_result_matrix)
        rotation_layout.addWidget(self.line_rotx)

        rotation_layout.addWidget(QLabel("roty:"))
        self.line_roty = QLineEdit("0")
        self.line_roty.setValidator(QDoubleValidator())
        self.line_roty.textChanged.connect(self._invalidate_result_matrix)
        rotation_layout.addWidget(self.line_roty)

        rotation_layout.addWidget(QLabel("rotz:"))
        self.line_rotz = QLineEdit("0")
        self.line_rotz.setValidator(QDoubleValidator())
        self.line_rotz.textChanged.connect(self._invalidate_result_matrix)
        rotation_layout.addWidget(self.line_rotz)

        rotation_layout.addStretch()

        apply_rotation_btn = QPushButton("Apply Rotation")
        main_layout.addWidget(apply_rotation_btn)
        apply_rotation_btn.clicked.connect(self._apply_rotation)

        self.result_group = QGroupBox("Rotated Matrix (Output)")
        self.result_layout = QGridLayout()
        self.result_group.setLayout(self.result_layout)
        main_layout.addWidget(self.result_group)

        self.result_labels = []
        for i in range(3):
            row_labels = []
            for j in range(3):
                lbl = QLabel("")
                lbl.setFixedWidth(60)
                lbl.setAlignment(Qt.AlignCenter)
                self.result_layout.addWidget(lbl, i, j)
                row_labels.append(lbl)
            self.result_labels.append(row_labels)

        self.update_matrix_button = QPushButton("Update Orientation Matrix from Result")
        main_layout.addWidget(self.update_matrix_button)
        self.update_matrix_button.clicked.connect(self._update_orientation_from_result)

    def _invalidate_result_matrix(self):
        self._result_matrix_valid = False
        for row in self.result_labels:
            for lbl in row:
                lbl.setText("")

    def _load_cif(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open CIF File",
            directory="",
            filter="CIF Files (*.cif);;All Files (*)"
        )
        if not file_name:
            return

        try:
            cif_data = Cif(file_path=file_name)

            if cif_data.space_group is not None:
                self.line_space_group.setText(str(cif_data.space_group))
            if cif_data.a is not None:
                self.line_a.setText(str(cif_data.a))
            if cif_data.b is not None:
                self.line_b.setText(str(cif_data.b))
            if cif_data.c is not None:
                self.line_c.setText(str(cif_data.c))
            if cif_data.alpha is not None:
                self.line_alpha.setText(str(cif_data.alpha))
            if cif_data.beta is not None:
                self.line_beta.setText(str(cif_data.beta))
            if cif_data.gamma is not None:
                self.line_gamma.setText(str(cif_data.gamma))

            self._invalidate_result_matrix()
            QMessageBox.information(self, "CIF Loaded", f"Successfully loaded: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "CIF Error", f"Failed to read or parse the CIF file:\n{str(e)}")

    def _compute_orientation(self):
        try:
            a_val = float(self.line_a.text())
            b_val = float(self.line_b.text())
            c_val = float(self.line_c.text())
            alpha_val = float(self.line_alpha.text())
            beta_val = float(self.line_beta.text())
            gamma_val = float(self.line_gamma.text())

            orientation = compute_lattice_orientation(a_val, b_val, c_val, alpha_val, beta_val, gamma_val)
            for i in range(3):
                for j in range(3):
                    self.matrix_edits[i][j].setText(f"{orientation[i, j]:.4f}")
            self._invalidate_result_matrix()
        except ValueError as ex:
            QMessageBox.critical(self, "Input Error", f"Invalid numeric input:\n{str(ex)}")
        except Exception as e:
            QMessageBox.critical(self, "Orientation Error", f"Error computing orientation:\n{str(e)}")

    def _apply_rotation(self):
        try:
            initial_matrix = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    initial_matrix[i, j] = float(self.matrix_edits[i][j].text().strip())

            rx = float(self.line_rotx.text())
            ry = float(self.line_roty.text())
            rz = float(self.line_rotz.text())

            rotated_rows = []
            for row_idx in range(3):
                vec = initial_matrix[row_idx, :]
                rotated_vec = apply_rotation(vec, rx, ry, rz, rotation_order="xyz")
                rotated_rows.append(rotated_vec)

            rotated_matrix = np.vstack(rotated_rows)

            for i in range(3):
                for j in range(3):
                    self.result_labels[i][j].setText(f"{rotated_matrix[i, j]:.4f}")

            self._result_matrix_valid = True

        except ValueError as ex:
            QMessageBox.critical(self, "Input Error", f"Invalid numeric input:\n{str(ex)}")
        except Exception as e:
            QMessageBox.critical(self, "Rotation Error", f"Error applying rotation:\n{str(e)}")

    def _update_orientation_from_result(self):
        if not self._result_matrix_valid:
            QMessageBox.warning(
                self,
                "Update Error",
                "No valid rotated matrix is available yet.\nPlease apply a valid rotation first."
            )
            return

        try:
            for i in range(3):
                for j in range(3):
                    val = float(self.result_labels[i][j].text().strip())
                    self.matrix_edits[i][j].setText(f"{val:.4f}")
        except ValueError:
            QMessageBox.warning(
                self,
                "Update Error",
                "Rotated matrix contains invalid or non-numeric values.\nPlease apply a valid rotation first."
            )

    def get_current_matrix(self):
        if self._result_matrix_valid:
            mat = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    mat[i, j] = float(self.result_labels[i][j].text().strip())
            return mat

        mat = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                text = self.matrix_edits[i][j].text().strip()
                if text == "":
                    raise ValueError(f"Orientation matrix element at Row {i+1}, Col {j+1} is empty.")
                mat[i, j] = float(text)

        return mat


class LegacyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRDpy Simulation GUI")

        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout()
        container.setLayout(layout)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.poly_tab = QWidget()
        self.tabs.addTab(self.poly_tab, "Polycrystalline")
        self._init_poly_tab()

        self.single_tab = QWidget()
        self.tabs.addTab(self.single_tab, "Single Crystal")
        self._init_single_tab()

        bottom_buttons_layout = QHBoxLayout()

        self.open_rotation_button = QPushButton("Open Matrix Rotation Tool")
        bottom_buttons_layout.addWidget(self.open_rotation_button)
        self.open_rotation_button.clicked.connect(self._open_matrix_rotation_window)

        self.close_all_plots_button = QPushButton("Close All Plots")
        bottom_buttons_layout.addWidget(self.close_all_plots_button)
        self.close_all_plots_button.clicked.connect(self._close_all_plots)

        bottom_buttons_layout.addStretch()
        layout.addLayout(bottom_buttons_layout)

        self.matrix_window = MatrixRotationWindow()
        self.resize(750, 850)

        self.poly_cif_file_path = None
        self.single_cif_file_path = None

    def _open_matrix_rotation_window(self):
        self.matrix_window.show()

    def _close_all_plots(self):
        try:
            plt.close("all")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Plot Close Error",
                f"Could not close matplotlib windows:\n{str(e)}"
            )

    def _load_cif_into_fields(self, file_name, line_space_group, line_a, line_b, line_c, line_alpha, line_beta, line_gamma):
        cif_data = Cif(file_path=file_name)

        if cif_data.space_group is not None:
            line_space_group.setText(str(cif_data.space_group))
        if cif_data.a is not None:
            line_a.setText(str(cif_data.a))
        if cif_data.b is not None:
            line_b.setText(str(cif_data.b))
        if cif_data.c is not None:
            line_c.setText(str(cif_data.c))
        if cif_data.alpha is not None:
            line_alpha.setText(str(cif_data.alpha))
        if cif_data.beta is not None:
            line_beta.setText(str(cif_data.beta))
        if cif_data.gamma is not None:
            line_gamma.setText(str(cif_data.gamma))

        return cif_data

    # ==========================================================================
    #   POLYCRYSTALLINE TAB
    # ==========================================================================
    def _init_poly_tab(self):
        main_layout = QVBoxLayout()
        self.poly_tab.setLayout(main_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        scroll_layout = QVBoxLayout()
        scroll_content.setLayout(scroll_layout)

        func_group = QGroupBox("Function Selection")
        func_layout = QHBoxLayout()
        func_group.setLayout(func_layout)
        func_layout.addWidget(QLabel("Select Polycrystalline Function:"))
        self.poly_func_combo = QComboBox()
        self.poly_func_combo.addItems(["simulate_1d", "simulate_2d", "simulate_3d"])
        func_layout.addWidget(self.poly_func_combo)
        func_layout.addStretch()
        scroll_layout.addWidget(func_group)

        self.poly_sample_group = QGroupBox("Sample Parameters (Crystallographic)")
        sam_layout = QGridLayout()
        self.poly_sample_group.setLayout(sam_layout)
        scroll_layout.addWidget(self.poly_sample_group)

        row = 0
        sam_layout.addWidget(QLabel("CIF file:"), row, 0)
        cif_hbox = QHBoxLayout()
        self.poly_line_cif_path = QLineEdit("")
        self.poly_line_cif_path.setPlaceholderText("Optional for 2d/3d (if using manual q/d). Required for simulate_1d.")
        cif_hbox.addWidget(self.poly_line_cif_path)
        self.poly_btn_browse_cif = QPushButton("Browse")
        cif_hbox.addWidget(self.poly_btn_browse_cif)
        sam_layout.addLayout(cif_hbox, row, 1)
        row += 1

        sam_layout.addWidget(QLabel("Space Group (e.g. 167):"), row, 0)
        self.poly_line_space_group = QLineEdit("167")
        self.poly_line_space_group.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_space_group, row, 1)
        row += 1

        sam_layout.addWidget(QLabel("qmax [Å^-1]:"), row, 0)
        self.poly_line_qmax = QLineEdit("10")
        self.poly_line_qmax.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_qmax, row, 1)
        row += 1

        sam_layout.addWidget(QLabel("a [Å]:"), row, 0)
        self.poly_line_sam_a = QLineEdit("4.954")
        self.poly_line_sam_a.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_a, row, 1)
        row += 1

        sam_layout.addWidget(QLabel("b [Å]:"), row, 0)
        self.poly_line_sam_b = QLineEdit("4.954")
        self.poly_line_sam_b.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_b, row, 1)
        row += 1

        sam_layout.addWidget(QLabel("c [Å]:"), row, 0)
        self.poly_line_sam_c = QLineEdit("14.01")
        self.poly_line_sam_c.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_c, row, 1)
        row += 1

        sam_layout.addWidget(QLabel("alpha [deg]:"), row, 0)
        self.poly_line_sam_alpha = QLineEdit("90")
        self.poly_line_sam_alpha.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_alpha, row, 1)
        row += 1

        sam_layout.addWidget(QLabel("beta [deg]:"), row, 0)
        self.poly_line_sam_beta = QLineEdit("90")
        self.poly_line_sam_beta.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_beta, row, 1)
        row += 1

        sam_layout.addWidget(QLabel("gamma [deg]:"), row, 0)
        self.poly_line_sam_gamma = QLineEdit("120")
        self.poly_line_sam_gamma.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_gamma, row, 1)
        row += 1

        row += 1
        self.poly_btn_load_cif = QPushButton("Load CIF → fill lattice params")
        sam_layout.addWidget(self.poly_btn_load_cif, row, 0, 1, 2)

        self.poly_btn_browse_cif.clicked.connect(self._poly_browse_cif)
        self.poly_btn_load_cif.clicked.connect(self._poly_load_cif)

        self.poly_beam_group = QGroupBox("Beam Parameters")
        beam_layout = QGridLayout()
        self.poly_beam_group.setLayout(beam_layout)
        scroll_layout.addWidget(self.poly_beam_group)

        beam_layout.addWidget(QLabel("Energy [eV]:"), 0, 0)
        self.poly_line_energy = QLineEdit("15000")
        self.poly_line_energy.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.poly_line_energy, 0, 1)

        beam_layout.addWidget(QLabel("∆E/E [%]:"), 1, 0)
        self.poly_line_ebw = QLineEdit("1.5")
        self.poly_line_ebw.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.poly_line_ebw, 1, 1)

        self.poly_det_group = QGroupBox("Detector Settings (for 2D/3D)")
        det_layout = QGridLayout()
        self.poly_det_group.setLayout(det_layout)
        scroll_layout.addWidget(self.poly_det_group)

        row = 0
        det_layout.addWidget(QLabel("Detector Type:"), row, 0)
        self.poly_combo_det_type = QComboBox()
        detector_list = ["manual"] + list(pyFAI.detectors.ALL_DETECTORS.keys())
        self.poly_combo_det_type.addItems(detector_list)
        det_layout.addWidget(self.poly_combo_det_type, row, 1)
        row += 1

        self.poly_manual_group = QGroupBox("Manual Detector Parameters")
        self.poly_manual_group.setFlat(True)
        manual_layout = QGridLayout()
        self.poly_manual_group.setLayout(manual_layout)
        det_layout.addWidget(self.poly_manual_group, row, 0, 1, 2)
        row += 1

        manual_layout.addWidget(QLabel("Pixel Size H [m]:"), 0, 0)
        self.poly_line_pxsize_h = QLineEdit("50e-6")
        self.poly_line_pxsize_h.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.poly_line_pxsize_h, 0, 1)

        manual_layout.addWidget(QLabel("Pixel Size V [m]:"), 1, 0)
        self.poly_line_pxsize_v = QLineEdit("50e-6")
        self.poly_line_pxsize_v.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.poly_line_pxsize_v, 1, 1)

        manual_layout.addWidget(QLabel("Number of Pixels H:"), 2, 0)
        self.poly_line_num_px_h = QLineEdit("2000")
        self.poly_line_num_px_h.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.poly_line_num_px_h, 2, 1)

        manual_layout.addWidget(QLabel("Number of Pixels V:"), 3, 0)
        self.poly_line_num_px_v = QLineEdit("2000")
        self.poly_line_num_px_v.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.poly_line_num_px_v, 3, 1)

        det_common_group = QGroupBox("Common Detector Parameters")
        det_common_layout = QGridLayout()
        det_common_group.setLayout(det_common_layout)
        det_layout.addWidget(det_common_group, row, 0, 1, 2)
        row += 1

        det_common_layout.addWidget(QLabel("Binning (H, V):"), 0, 0)
        bin_hbox = QHBoxLayout()
        self.poly_line_bin_h = QLineEdit("1")
        self.poly_line_bin_h.setValidator(QDoubleValidator())
        self.poly_line_bin_v = QLineEdit("1")
        self.poly_line_bin_v.setValidator(QDoubleValidator())
        bin_hbox.addWidget(QLabel("H:"))
        bin_hbox.addWidget(self.poly_line_bin_h)
        bin_hbox.addWidget(QLabel("V:"))
        bin_hbox.addWidget(self.poly_line_bin_v)
        det_common_layout.addLayout(bin_hbox, 0, 1)

        det_common_layout.addWidget(QLabel("Distance [m]:"), 1, 0)
        self.poly_line_dist = QLineEdit("0.1")
        self.poly_line_dist.setValidator(QDoubleValidator())
        det_common_layout.addWidget(self.poly_line_dist, 1, 1)

        det_common_layout.addWidget(QLabel("PONI1 [m]:"), 2, 0)
        self.poly_line_poni1 = QLineEdit("0")
        self.poly_line_poni1.setValidator(QDoubleValidator())
        det_common_layout.addWidget(self.poly_line_poni1, 2, 1)

        det_common_layout.addWidget(QLabel("PONI2 [m]:"), 3, 0)
        self.poly_line_poni2 = QLineEdit("0")
        self.poly_line_poni2.setValidator(QDoubleValidator())
        det_common_layout.addWidget(self.poly_line_poni2, 3, 1)

        det_common_layout.addWidget(QLabel("Detector Rotations [deg]:"), 4, 0)
        rots_hbox = QHBoxLayout()
        self.poly_line_rotx = QLineEdit("0")
        self.poly_line_rotx.setValidator(QDoubleValidator())
        self.poly_line_roty = QLineEdit("0")
        self.poly_line_roty.setValidator(QDoubleValidator())
        self.poly_line_rotz = QLineEdit("0")
        self.poly_line_rotz.setValidator(QDoubleValidator())
        rots_hbox.addWidget(QLabel("rotx:"))
        rots_hbox.addWidget(self.poly_line_rotx)
        rots_hbox.addWidget(QLabel("roty:"))
        rots_hbox.addWidget(self.poly_line_roty)
        rots_hbox.addWidget(QLabel("rotz:"))
        rots_hbox.addWidget(self.poly_line_rotz)
        det_common_layout.addLayout(rots_hbox, 4, 1)

        self.poly_combo_det_type.currentIndexChanged.connect(self._poly_detector_changed)
        self._poly_detector_changed()

        self.poly_refsrc_group = QGroupBox("Reflections Source (for 2D/3D)")
        refsrc_layout = QHBoxLayout()
        self.poly_refsrc_group.setLayout(refsrc_layout)
        refsrc_layout.addWidget(QLabel("Use reflections from:"))
        self.poly_combo_refsrc = QComboBox()
        self.poly_combo_refsrc.addItems([
            "CIF / lattice (auto from qmax)",
            "Manual q/d + hkls_names"
        ])
        refsrc_layout.addWidget(self.poly_combo_refsrc)
        refsrc_layout.addStretch()
        scroll_layout.addWidget(self.poly_refsrc_group)

        self.poly_combo_refsrc.currentIndexChanged.connect(self._poly_refsrc_changed)

        self.poly_refl_group = QGroupBox("Manual Reflection Lists (only if selected above)")
        refl_layout = QVBoxLayout()
        self.poly_refl_group.setLayout(refl_layout)
        scroll_layout.addWidget(self.poly_refl_group)

        qdhkls_layout = QGridLayout()
        refl_layout.addLayout(qdhkls_layout)

        qdhkls_layout.addWidget(QLabel("q_hkls (comma):"), 0, 0)
        self.poly_line_qhkls = QLineEdit("")
        self.poly_line_qhkls.setPlaceholderText("e.g., 1.0,2.0,3.0  (Å^-1)")
        qdhkls_layout.addWidget(self.poly_line_qhkls, 0, 1)

        qdhkls_layout.addWidget(QLabel("OR d_hkls (comma):"), 1, 0)
        self.poly_line_dhkls = QLineEdit("")
        self.poly_line_dhkls.setPlaceholderText("e.g., 3.0,2.0,1.5  (Å)")
        qdhkls_layout.addWidget(self.poly_line_dhkls, 1, 1)

        refl_layout.addWidget(QLabel("hkls_names (e.g. [1,0,2],[0,1,2]):"))
        self.poly_line_hkls = QLineEdit("")
        self.poly_line_hkls.setPlaceholderText("e.g., [1,0,2],[0,1,2]")
        refl_layout.addWidget(self.poly_line_hkls)

        self.poly_func_specific_group = QGroupBox("Function-Specific Parameters (2D/3D cones)")
        func_specific_layout = QGridLayout()
        self.poly_func_specific_group.setLayout(func_specific_layout)
        scroll_layout.addWidget(self.poly_func_specific_group)

        func_specific_layout.addWidget(QLabel("Cones Number of Points:"), 0, 0)
        self.poly_line_cones = QLineEdit("30")
        self.poly_line_cones.setValidator(QDoubleValidator())
        func_specific_layout.addWidget(self.poly_line_cones, 0, 1)

        self.poly_1d_group = QGroupBox("1D Pattern Options (simulate_1d)")
        one_d_layout = QGridLayout()
        self.poly_1d_group.setLayout(one_d_layout)
        scroll_layout.addWidget(self.poly_1d_group)

        one_d_layout.addWidget(QLabel("x_axis:"), 0, 0)
        self.poly_combo_xaxis = QComboBox()
        self.poly_combo_xaxis.addItems(["q", "two_theta"])
        one_d_layout.addWidget(self.poly_combo_xaxis, 0, 1)

        self.poly_chk_lorpol = QCheckBox("include_lorentz_polarization")
        self.poly_chk_lorpol.setChecked(True)
        one_d_layout.addWidget(self.poly_chk_lorpol, 1, 0, 1, 2)

        one_d_layout.addWidget(QLabel("Peak FWHM:"), 2, 0)
        self.poly_line_fwhm = QLineEdit("0.0")
        self.poly_line_fwhm.setValidator(QDoubleValidator())
        self.poly_line_fwhm.setToolTip(
            "Peak broadening FWHM.\n"
            "Units: deg if x_axis='two_theta', Å^-1 if x_axis='q'.\n"
            "Use 0 for a stick pattern (no broadening)."
        )
        one_d_layout.addWidget(self.poly_line_fwhm, 2, 1)

        self.poly_run_btn = QPushButton("Run Polycrystal Function")
        scroll_layout.addWidget(self.poly_run_btn)

        self.poly_func_combo.currentIndexChanged.connect(self._poly_func_changed)
        self.poly_run_btn.clicked.connect(self._poly_run_function)

        self._poly_refsrc_changed()
        self._poly_func_changed()

    def _poly_browse_cif(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open CIF File",
            directory="",
            filter="CIF Files (*.cif);;All Files (*)"
        )
        if file_name:
            self.poly_cif_file_path = file_name
            self.poly_line_cif_path.setText(file_name)

    def _poly_load_cif(self):
        file_name = (self.poly_line_cif_path.text() or "").strip()
        if not file_name:
            self._poly_browse_cif()
            file_name = (self.poly_line_cif_path.text() or "").strip()
            if not file_name:
                return

        try:
            self.poly_cif_file_path = file_name
            self._load_cif_into_fields(
                file_name,
                self.poly_line_space_group,
                self.poly_line_sam_a,
                self.poly_line_sam_b,
                self.poly_line_sam_c,
                self.poly_line_sam_alpha,
                self.poly_line_sam_beta,
                self.poly_line_sam_gamma
            )
            QMessageBox.information(self, "CIF Loaded", f"Successfully loaded: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "CIF Error", f"Failed to read or parse the CIF file:\n{str(e)}")

    def _poly_detector_changed(self):
        det_type = self.poly_combo_det_type.currentText().lower()
        self.poly_manual_group.setVisible(det_type == "manual")

    def _poly_refsrc_changed(self):
        manual = self.poly_combo_refsrc.currentText().startswith("Manual")
        self.poly_refl_group.setVisible(manual)

    def _poly_func_changed(self):
        func = self.poly_func_combo.currentText()

        is_1d = (func == "simulate_1d")
        is_cones = func in ("simulate_2d", "simulate_3d")

        self.poly_det_group.setVisible(is_cones)
        self.poly_refsrc_group.setVisible(is_cones)
        self.poly_func_specific_group.setVisible(is_cones)

        if is_cones:
            self._poly_refsrc_changed()
        else:
            self.poly_refl_group.setVisible(False)

        self.poly_1d_group.setVisible(is_1d)

    def _poly_build_reflections_from_crystal(self, energy_eV, e_bw_pct, qmax):
        """
        Build (q_magnitudes, hkls_names) from CIF/lattice parameters using LatticeStructure.

        IMPORTANT: We de-duplicate by |q| so symmetry-equivalent hkls that produce the same powder ring
        do NOT get plotted multiple times in 2D/3D.
        """
        cif_path = (self.poly_line_cif_path.text() or "").strip()
        if cif_path:
            lattice = sample_mod.LatticeStructure(cif_file_path=cif_path)
        else:
            sam_space_group = int(float(self.poly_line_space_group.text()))
            a = float(self.poly_line_sam_a.text())
            b = float(self.poly_line_sam_b.text())
            c = float(self.poly_line_sam_c.text())
            alpha = float(self.poly_line_sam_alpha.text())
            beta = float(self.poly_line_sam_beta.text())
            gamma = float(self.poly_line_sam_gamma.text())
            lattice = sample_mod.LatticeStructure(
                space_group=sam_space_group,
                a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma
            )

        lattice.calculate_reciprocal_lattice()
        lattice.create_possible_reflections(qmax=qmax)

        hkls_names = lattice.allowed_hkls.astype(int)
        q_vecs = sample_mod.calculate_q_hkl(hkls_names, lattice.reciprocal_lattice)
        q_mags = np.linalg.norm(q_vecs, axis=1)

        e_max = energy_eV * (1.0 + (e_bw_pct / 200.0))
        lam_min_A = xutils.energy_to_wavelength(e_max) * 1e10
        q_ewald_max = 4.0 * np.pi / lam_min_A

        mask = np.isfinite(q_mags) & (q_mags > 0) & (q_mags <= q_ewald_max)
        q_mags = q_mags[mask]
        hkls_names = hkls_names[mask]

        order = np.argsort(q_mags)
        q_mags = q_mags[order]
        hkls_names = hkls_names[order]

        if q_mags.size == 0:
            return q_mags, hkls_names

        q_tol = 1e-5

        q_unique = []
        hkls_unique = []

        i = 0
        n = len(q_mags)
        while i < n:
            q0 = q_mags[i]
            j = i + 1
            while j < n and abs(q_mags[j] - q0) <= q_tol:
                j += 1

            group = hkls_names[i:j]

            best = min(
                group,
                key=lambda v: (
                    abs(v[0]) + abs(v[1]) + abs(v[2]),
                    abs(v[0]), abs(v[1]), abs(v[2]),
                    int(v[0]), int(v[1]), int(v[2]),
                )
            )

            q_unique.append(q0)
            hkls_unique.append(best)
            i = j

        return np.array(q_unique, dtype=float), np.array(hkls_unique, dtype=int)

    def _poly_run_function(self):
        try:
            func = self.poly_func_combo.currentText()

            energy = float(self.poly_line_energy.text())
            e_bw = float(self.poly_line_ebw.text())
            qmax = float(self.poly_line_qmax.text())

            if func == "simulate_1d":
                cif_path = (self.poly_line_cif_path.text() or "").strip()
                if not cif_path:
                    raise ValueError("simulate_1d requires a CIF file path. Load or browse a CIF first.")

                x_axis = self.poly_combo_xaxis.currentText()
                include_lorentz_polarization = self.poly_chk_lorpol.isChecked()

                fwhm_txt = (self.poly_line_fwhm.text() or "").strip()
                fwhm_val = float(fwhm_txt) if fwhm_txt else 0.0
                fwhm = None if (fwhm_val <= 0.0) else fwhm_val

                try:
                    polycrystalline.simulate_1d(
                        cif_file_path=cif_path,
                        qmax=qmax,
                        x_axis=x_axis,
                        include_lorentz_polarization=include_lorentz_polarization,
                        include_multiplicity=False,
                        atom_positions=False,
                        energy=energy,
                        fwhm=fwhm
                    )
                except TypeError:
                    polycrystalline.simulate_1d(
                        cif_file_path=cif_path,
                        qmax=qmax,
                        x_axis=x_axis,
                        include_lorentz_polarization=include_lorentz_polarization,
                        include_multiplicity=False,
                        atom_positions=False,
                        energy=energy
                    )
                    if fwhm is not None:
                        QMessageBox.warning(
                            self,
                            "FWHM not applied",
                            "Your current polycrystalline.simulate_1d() does not accept an 'fwhm' argument.\n"
                            "Update the simulation function to support broadening, or expose it via plotting."
                        )
                return True

            det_type = self.poly_combo_det_type.currentText()

            pxsize_h = pxsize_v = None
            num_px_h = num_px_v = None
            if det_type.lower() == "manual":
                pxsize_h = float(self.poly_line_pxsize_h.text())
                pxsize_v = float(self.poly_line_pxsize_v.text())
                num_px_h = int(float(self.poly_line_num_px_h.text()))
                num_px_v = int(float(self.poly_line_num_px_v.text()))

            bin_h = int(float(self.poly_line_bin_h.text()))
            bin_v = int(float(self.poly_line_bin_v.text()))
            dist = float(self.poly_line_dist.text())
            poni1 = float(self.poly_line_poni1.text())
            poni2 = float(self.poly_line_poni2.text())
            rotx = float(self.poly_line_rotx.text())
            roty = float(self.poly_line_roty.text())
            rotz = float(self.poly_line_rotz.text())
            cones_num = int(float(self.poly_line_cones.text()))

            if self.poly_combo_refsrc.currentText().startswith("CIF / lattice"):
                q_hkls, hkls_names = self._poly_build_reflections_from_crystal(
                    energy_eV=energy,
                    e_bw_pct=e_bw,
                    qmax=qmax
                )
                d_hkls = None
            else:
                q_text = (self.poly_line_qhkls.text() or "").strip()
                d_text = (self.poly_line_dhkls.text() or "").strip()

                if q_text:
                    q_hkls = _parse_csv_floats(q_text)
                    d_hkls = None
                elif d_text:
                    d_hkls = _parse_csv_floats(d_text)
                    q_hkls = None
                else:
                    q_hkls = None
                    d_hkls = None

                hkls_names = _parse_hkls_string(self.poly_line_hkls.text())

                if hkls_names is None:
                    raise ValueError("Manual mode requires hkls_names (e.g. [1,0,2],[0,1,2]).")
                if (q_hkls is None) and (d_hkls is None):
                    raise ValueError("Manual mode requires either q_hkls or d_hkls.")
                if q_hkls is not None and len(q_hkls) != len(hkls_names):
                    raise ValueError("Manual mode: q_hkls length must match hkls_names length.")
                if d_hkls is not None and len(d_hkls) != len(hkls_names):
                    raise ValueError("Manual mode: d_hkls length must match hkls_names length.")

            if func == "simulate_2d":
                polycrystalline.simulate_2d(
                    det_type=det_type,
                    det_pxsize_h=pxsize_h,
                    det_pxsize_v=pxsize_v,
                    det_ntum_pixels_h=num_px_h,
                    det_num_pixels_v=num_px_v,
                    det_binning=(bin_h, bin_v),
                    det_dist=dist,
                    det_poni1=poni1,
                    det_poni2=poni2,
                    det_rotx=rotx,
                    det_roty=roty,
                    det_rotz=rotz,
                    cones_num_of_points=cones_num,
                    energy=energy,
                    e_bandwidth=e_bw,
                    q_hkls=q_hkls,
                    d_hkls=d_hkls,
                    hkls_names=hkls_names
                )
            elif func == "simulate_3d":
                polycrystalline.simulate_3d(
                    det_type=det_type,
                    det_pxsize_h=pxsize_h,
                    det_pxsize_v=pxsize_v,
                    det_ntum_pixels_h=num_px_h,
                    det_num_pixels_v=num_px_v,
                    det_binning=(bin_h, bin_v),
                    det_dist=dist,
                    det_poni1=poni1,
                    det_poni2=poni2,
                    det_rotx=rotx,
                    det_roty=roty,
                    det_rotz=rotz,
                    cones_num_of_points=cones_num,
                    energy=energy,
                    e_bandwidth=e_bw,
                    q_hkls=q_hkls,
                    d_hkls=d_hkls,
                    hkls_names=hkls_names
                )

            return True

        except Exception as e:
            QMessageBox.critical(self, "Polycrystalline Error", str(e))
            return False

    # ==========================================================================
    #   SINGLE CRYSTAL TAB
    # ==========================================================================
    def _init_single_tab(self):
        main_layout = QVBoxLayout()
        self.single_tab.setLayout(main_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        scroll_layout = QVBoxLayout()
        scroll_content.setLayout(scroll_layout)

        func_group = QGroupBox("Function Selection")
        func_layout = QHBoxLayout()
        func_group.setLayout(func_layout)
        func_layout.addWidget(QLabel("Select Single-Crystal Function:"))

        self.single_func_combo = QComboBox()
        self.single_func_combo.addItems([
            "simulate_2d",
            "simulate_3d",
            "target_hkl_near_pixel_fixed_energy",
            "detector_rotations_collecting_Braggs",
            "scan_two_parameters_for_Bragg_condition"
        ])
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
        detector_list = ["manual"] + list(pyFAI.detectors.ALL_DETECTORS.keys())
        self.single_combo_det_type.addItems(detector_list)
        det_layout.addWidget(self.single_combo_det_type, row, 1)
        row += 1

        self.single_manual_group = QGroupBox("Manual Detector Parameters")
        self.single_manual_group.setFlat(True)
        manual_layout = QGridLayout()
        self.single_manual_group.setLayout(manual_layout)
        det_layout.addWidget(self.single_manual_group, row, 0, 1, 2)
        row += 1

        manual_layout.addWidget(QLabel("Pixel Size H [m]:"), 0, 0)
        self.single_line_pxsize_h = QLineEdit("50e-6")
        self.single_line_pxsize_h.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.single_line_pxsize_h, 0, 1)

        manual_layout.addWidget(QLabel("Pixel Size V [m]:"), 1, 0)
        self.single_line_pxsize_v = QLineEdit("50e-6")
        self.single_line_pxsize_v.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.single_line_pxsize_v, 1, 1)

        manual_layout.addWidget(QLabel("Number of Pixels H:"), 2, 0)
        self.single_line_num_px_h = QLineEdit("2000")
        self.single_line_num_px_h.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.single_line_num_px_h, 2, 1)

        manual_layout.addWidget(QLabel("Number of Pixels V:"), 3, 0)
        self.single_line_num_px_v = QLineEdit("2000")
        self.single_line_num_px_v.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.single_line_num_px_v, 3, 1)

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

        det_common_layout.addWidget(QLabel("PONI1 [m]:"), 2, 0)
        self.single_line_poni1 = QLineEdit("0")
        self.single_line_poni1.setValidator(QDoubleValidator())
        det_common_layout.addWidget(self.single_line_poni1, 2, 1)

        det_common_layout.addWidget(QLabel("PONI2 [m]:"), 3, 0)
        self.single_line_poni2 = QLineEdit("0")
        self.single_line_poni2.setValidator(QDoubleValidator())
        det_common_layout.addWidget(self.single_line_poni2, 3, 1)

        self.single_detector_euler_group = QGroupBox("Legacy detector Euler rotations [deg]")
        detector_euler_layout = QHBoxLayout()
        self.single_detector_euler_group.setLayout(detector_euler_layout)
        det_common_layout.addWidget(self.single_detector_euler_group, 4, 0, 1, 2)

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

        detector_euler_layout.addStretch()

        self.single_combo_det_type.currentIndexChanged.connect(self._single_detector_changed)
        self._single_detector_changed()

        self.single_beam_group = QGroupBox("Beam Parameters")
        beam_layout = QGridLayout()
        self.single_beam_group.setLayout(beam_layout)
        scroll_layout.addWidget(self.single_beam_group)

        beam_layout.addWidget(QLabel("Energy [eV]:"), 0, 0)
        self.single_line_energy = QLineEdit("15000")
        self.single_line_energy.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.single_line_energy, 0, 1)

        beam_layout.addWidget(QLabel("∆E/E [%]:"), 1, 0)
        self.single_line_ebw = QLineEdit("1.5")
        self.single_line_ebw.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.single_line_ebw, 1, 1)

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
        self.single_line_qmax = QLineEdit("10")
        self.single_line_qmax.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_qmax, row, 1)
        row += 1

        sample_layout.addWidget(QLabel("a [Å]:"), row, 0)
        self.single_line_sam_a = QLineEdit("4.954")
        self.single_line_sam_a.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_a, row, 1)
        row += 1

        sample_layout.addWidget(QLabel("b [Å]:"), row, 0)
        self.single_line_sam_b = QLineEdit("4.954")
        self.single_line_sam_b.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_b, row, 1)
        row += 1

        sample_layout.addWidget(QLabel("c [Å]:"), row, 0)
        self.single_line_sam_c = QLineEdit("14.01")
        self.single_line_sam_c.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_c, row, 1)
        row += 1

        sample_layout.addWidget(QLabel("alpha [deg]:"), row, 0)
        self.single_line_sam_alpha = QLineEdit("90")
        self.single_line_sam_alpha.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_alpha, row, 1)
        row += 1

        sample_layout.addWidget(QLabel("beta [deg]:"), row, 0)
        self.single_line_sam_beta = QLineEdit("90")
        self.single_line_sam_beta.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_beta, row, 1)
        row += 1

        sample_layout.addWidget(QLabel("gamma [deg]:"), row, 0)
        self.single_line_sam_gamma = QLineEdit("120")
        self.single_line_sam_gamma.setValidator(QDoubleValidator())
        sample_layout.addWidget(self.single_line_sam_gamma, row, 1)
        row += 1

        row += 1
        self.load_cif_button = QPushButton("Load CIF")
        sample_layout.addWidget(self.load_cif_button, row, 0, 1, 2)
        self.load_cif_button.clicked.connect(self._single_load_cif)

        orientation_checkbox_layout = QHBoxLayout()
        self.single_orientation_checkbox = QCheckBox("Use Custom Initial Orientation")
        orientation_checkbox_layout.addWidget(self.single_orientation_checkbox)

        self.import_from_rotation_button = QPushButton("Import from Rotation Tool")
        orientation_checkbox_layout.addWidget(self.import_from_rotation_button)
        self.import_from_rotation_button.clicked.connect(self._import_orientation_from_rotation_tool)

        orientation_checkbox_layout.addStretch()
        scroll_layout.addLayout(orientation_checkbox_layout)
        self.single_orientation_checkbox.stateChanged.connect(self._toggle_orientation_matrix)

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
                entry.setFixedWidth(80)
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
            1, 0, 1, 2
        )

        self.single_sample_rotation_group = QGroupBox("Legacy sample Euler rotations [deg]")
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
        rot_layout.addStretch()
        scroll_layout.addWidget(self.single_sample_rotation_group)

        self.single_geometry_group = QGroupBox("Experiment Geometry")
        geometry_layout = QGridLayout()
        self.single_geometry_group.setLayout(geometry_layout)
        scroll_layout.addWidget(self.single_geometry_group)

        geometry_layout.addWidget(QLabel("Geometry mode:"), 0, 0)
        self.single_geometry_mode_combo = QComboBox()
        self.single_geometry_mode_combo.addItems([
            "Legacy Euler",
            "Predefined diffractometer",
            "Custom geometry",
        ])
        geometry_layout.addWidget(self.single_geometry_mode_combo, 0, 1)

        geometry_layout.addWidget(QLabel("Geometry kind:"), 1, 0)
        self.single_geometry_kind_combo = QComboBox()
        self.single_geometry_kind_combo.addItems(_available_geometry_kinds())
        geometry_layout.addWidget(self.single_geometry_kind_combo, 1, 1)

        self.single_geometry_copy_to_custom_btn = QPushButton("Copy preset into custom editors")
        self.single_geometry_copy_to_custom_btn.clicked.connect(self._load_selected_geometry_into_custom)
        geometry_layout.addWidget(self.single_geometry_copy_to_custom_btn, 1, 2)

        self.single_geometry_note = QLabel(
            "Use a predefined diffractometer to drive the geometry-aware single-crystal functions. "
            "Constructor kwargs, sample motor angles, and detector motor angles are entered as JSON dictionaries."
        )
        self.single_geometry_note.setWordWrap(True)
        geometry_layout.addWidget(self.single_geometry_note, 2, 0, 1, 3)

        geometry_layout.addWidget(QLabel("Geometry kwargs (JSON):"), 3, 0)
        self.single_geometry_kwargs = QPlainTextEdit()
        self.single_geometry_kwargs.setPlaceholderText("{\n  \"kappa_tilt_deg\": 50.0\n}")
        self.single_geometry_kwargs.setFixedHeight(70)
        geometry_layout.addWidget(self.single_geometry_kwargs, 3, 1, 1, 2)

        geometry_layout.addWidget(QLabel("Sample motor angles (JSON):"), 4, 0)
        self.single_geometry_sample_angles = QPlainTextEdit()
        self.single_geometry_sample_angles.setPlaceholderText("{\n  \"omega\": 0.0,\n  \"kappa\": 35.0,\n  \"phi\": 10.0\n}")
        self.single_geometry_sample_angles.setFixedHeight(80)
        geometry_layout.addWidget(self.single_geometry_sample_angles, 4, 1, 1, 2)

        geometry_layout.addWidget(QLabel("Detector motor angles (JSON):"), 5, 0)
        self.single_geometry_detector_angles = QPlainTextEdit()
        self.single_geometry_detector_angles.setPlaceholderText("{\n  \"tth\": 25.0\n}")
        self.single_geometry_detector_angles.setFixedHeight(70)
        geometry_layout.addWidget(self.single_geometry_detector_angles, 5, 1, 1, 2)

        self.single_custom_geometry_group = QGroupBox("Custom Geometry Definition")
        custom_geometry_layout = QGridLayout()
        self.single_custom_geometry_group.setLayout(custom_geometry_layout)
        scroll_layout.addWidget(self.single_custom_geometry_group)

        custom_note = QLabel(
            "Define the ordered motor chains explicitly as JSON. The order of the list is the order in which the motors are applied."
        )
        custom_note.setWordWrap(True)
        custom_geometry_layout.addWidget(custom_note, 0, 0, 1, 2)

        custom_geometry_layout.addWidget(QLabel("Custom sample chain (JSON):"), 1, 0)
        self.single_custom_sample_chain = QPlainTextEdit()
        self.single_custom_sample_chain.setPlaceholderText("[\n  {\"name\": \"omega\", \"axis\": \"z\", \"origin\": [0,0,0], \"frame\": \"lab\", \"default_angle\": 0},\n  {\"name\": \"kappa\", \"axis\": [0.766044, 0, 0.642788], \"origin\": [0,0,0], \"frame\": \"local\", \"default_angle\": 0},\n  {\"name\": \"phi\", \"axis\": \"z\", \"origin\": [0,0,0], \"frame\": \"local\", \"default_angle\": 0}\n]")
        self.single_custom_sample_chain.setFixedHeight(140)
        custom_geometry_layout.addWidget(self.single_custom_sample_chain, 1, 1)

        custom_geometry_layout.addWidget(QLabel("Custom detector chain (JSON):"), 2, 0)
        self.single_custom_detector_chain = QPlainTextEdit()
        self.single_custom_detector_chain.setPlaceholderText("[\n  {\"name\": \"tth\", \"axis\": \"y\", \"origin\": [0,0,0], \"frame\": \"lab\", \"default_angle\": 0}\n]")
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
        self.single_geometry_detector_scan_ranges.setPlaceholderText("{\n  \"tth\": [-30, 30, 2]\n}")
        self.single_geometry_detector_scan_ranges.setFixedHeight(70)
        detector_scan_layout.addWidget(self.single_geometry_detector_scan_ranges, 0, 1)

        self.single_refl_group = QGroupBox("Reflection Lists (for Bragg checks)")
        refl_layout = QGridLayout()
        self.single_refl_group.setLayout(refl_layout)
        scroll_layout.addWidget(self.single_refl_group)

        self.single_label_q = QLabel("q_hkls (comma):")
        refl_layout.addWidget(self.single_label_q, 0, 0)
        self.single_line_q = QLineEdit("")
        self.single_line_q.setPlaceholderText("Currently unused by the exposed single-crystal GUI functions")
        refl_layout.addWidget(self.single_line_q, 0, 1)

        self.single_label_d = QLabel("OR d_hkls (comma):")
        refl_layout.addWidget(self.single_label_d, 1, 0)
        self.single_line_d = QLineEdit("")
        self.single_line_d.setPlaceholderText("Currently unused by the exposed single-crystal GUI functions")
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
        scroll_layout.addWidget(self.single_run_btn)

        self.single_func_combo.currentIndexChanged.connect(self._single_func_changed)
        self.single_geometry_mode_combo.currentIndexChanged.connect(self._single_geometry_mode_changed)
        self.single_run_btn.clicked.connect(self._single_run_function)
        self._single_geometry_mode_changed()
        self._single_func_changed()

    def _toggle_orientation_matrix(self, state):
        on = (state == Qt.Checked)
        self.single_label_sam_init.setVisible(on)
        self.orientation_group.setVisible(on)

    def _single_geometry_mode(self):
        return self.single_geometry_mode_combo.currentText()

    def _single_geometry_enabled(self):
        return self._single_geometry_mode() in {"Predefined diffractometer", "Custom geometry"}

    def _single_predefined_geometry_enabled(self):
        return self._single_geometry_mode() == "Predefined diffractometer"

    def _single_custom_geometry_enabled(self):
        return self._single_geometry_mode() == "Custom geometry"

    def _single_func_changed(self):
        func = self.single_func_combo.currentText()
        geometry_enabled = self._single_geometry_enabled()

        needs_detector_info = func in [
            "simulate_2d",
            "simulate_3d",
            "target_hkl_near_pixel_fixed_energy",
            "detector_rotations_collecting_Braggs",
        ]

        needs_bragg = func in [
            "detector_rotations_collecting_Braggs",
            "scan_two_parameters_for_Bragg_condition",
        ]

        needs_param_scan = (func == "scan_two_parameters_for_Bragg_condition")
        needs_angle_range = (func == "detector_rotations_collecting_Braggs") and (not geometry_enabled)

        needs_extra_hkls = func in [
            "simulate_2d",
            "simulate_3d",
        ]

        needs_inverse_targeting = (func == "target_hkl_near_pixel_fixed_energy")

        geometry_supported = func in [
            "simulate_2d",
            "simulate_3d",
            "target_hkl_near_pixel_fixed_energy",
            "detector_rotations_collecting_Braggs",
            "scan_two_parameters_for_Bragg_condition",
        ]

        self.single_det_group.setVisible(needs_detector_info)
        if not needs_detector_info:
            self.single_manual_group.setVisible(False)

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
            geometry_supported and geometry_enabled and func == "scan_two_parameters_for_Bragg_condition"
        )
        self.single_geometry_detector_scan_group.setVisible(
            geometry_supported and geometry_enabled and func == "detector_rotations_collecting_Braggs"
        )

        self.single_label_q.setEnabled(False)
        self.single_line_q.setEnabled(False)
        self.single_label_d.setEnabled(False)
        self.single_line_d.setEnabled(False)

    def _single_geometry_mode_changed(self):
        predefined = self._single_predefined_geometry_enabled()
        custom = self._single_custom_geometry_enabled()
        enabled = predefined or custom
        legacy = not enabled

        self.single_geometry_kind_combo.setEnabled(predefined)
        self.single_geometry_kwargs.setEnabled(predefined)
        self.single_geometry_copy_to_custom_btn.setEnabled(predefined)

        self.single_geometry_sample_angles.setEnabled(enabled)
        self.single_geometry_detector_angles.setEnabled(enabled)
        self.single_geometry_scan_group.setEnabled(enabled)

        self.single_custom_geometry_group.setEnabled(custom)

        self.single_sample_rotation_group.setEnabled(legacy)
        self.single_detector_euler_group.setEnabled(legacy)
        self.single_wrap_angles_checkbox.setEnabled(legacy)

        if legacy:
            self.single_geometry_note.setText(
                "Use a predefined diffractometer to drive the geometry-aware single-crystal functions. "
                "Constructor kwargs, sample motor angles, and detector motor angles are entered as JSON dictionaries."
            )
        else:
            self.single_geometry_note.setText(
                "Geometry mode is active. The legacy sample Euler rotations, legacy detector Euler rotations, "
                "and Euler-angle wrapping checkbox are ignored by the geometry-aware backend."
            )

        self._single_func_changed()

    def _build_predefined_geometry(self):
        kind = self.single_geometry_kind_combo.currentText().strip()
        if not kind:
            raise ValueError("Select a diffractometer geometry.")
        kwargs = _parse_json_object(self.single_geometry_kwargs.toPlainText(), "Geometry kwargs")
        return diffractometers.make_diffractometer(kind, **kwargs)

    def _build_custom_geometry(self):
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
        if self._single_predefined_geometry_enabled():
            return self._build_predefined_geometry()
        if self._single_custom_geometry_enabled():
            return self._build_custom_geometry()
        return None

    def _single_geometry_sample_angles_dict(self):
        angles = _parse_json_object(self.single_geometry_sample_angles.toPlainText(), "Sample motor angles")
        return angles or None

    def _single_geometry_detector_angles_dict(self):
        angles = _parse_json_object(self.single_geometry_detector_angles.toPlainText(), "Detector motor angles")
        return angles or None

    def _single_detector_scan_ranges_dict(self):
        scan_ranges = _parse_json_object(
            self.single_geometry_detector_scan_ranges.toPlainText(),
            "Detector scan ranges",
            allow_empty=False,
        )
        if not scan_ranges:
            raise ValueError("Detector scan ranges cannot be empty in geometry mode.")
        return scan_ranges

    def _load_selected_geometry_into_custom(self):
        try:
            geometry = self._build_predefined_geometry()
            geom_dict = diffractometers.diffractometer_to_dict(geometry)
            self.single_custom_sample_chain.setPlainText(_pretty_json(geom_dict["sample"]["motors"]))
            self.single_custom_detector_chain.setPlainText(_pretty_json(geom_dict["detector"]["motors"]))
            self.single_geometry_mode_combo.setCurrentText("Custom geometry")
        except Exception as e:
            QMessageBox.critical(self, "Geometry Copy Error", str(e))

    def _single_detector_changed(self):
        det_type = self.single_combo_det_type.currentText().lower()
        self.single_manual_group.setVisible(det_type == "manual")

    def collect_matrix(self):
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

    def _single_load_cif(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open CIF File",
            directory="",
            filter="CIF Files (*.cif);;All Files (*)"
        )
        if not file_name:
            return
        try:
            self.single_cif_file_path = file_name
            self._load_cif_into_fields(
                file_name,
                self.single_line_space_group,
                self.single_line_sam_a,
                self.single_line_sam_b,
                self.single_line_sam_c,
                self.single_line_sam_alpha,
                self.single_line_sam_beta,
                self.single_line_sam_gamma
            )
            QMessageBox.information(self, "CIF Loaded", f"Successfully loaded: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "CIF Error", f"Failed to read or parse the CIF file:\n{str(e)}")

    def _import_orientation_from_rotation_tool(self):
        try:
            mat = self.matrix_window.get_current_matrix()

            if not self.single_orientation_checkbox.isChecked():
                self.single_orientation_checkbox.setChecked(True)

            for i in range(3):
                for j in range(3):
                    self.orientation_entries[i][j].setText(f"{mat[i, j]:.6f}")

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Could not import matrix:\n{str(e)}")

    def _single_run_function(self):
        func = self.single_func_combo.currentText()
        det_type = self.single_combo_det_type.currentText()

        try:
            pxsize_h = pxsize_v = None
            num_px_h = num_px_v = None

            if det_type.lower() == "manual":
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

            energy = float(self.single_line_energy.text())
            e_bw = float(self.single_line_ebw.text())

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
            sample_angles = self._single_geometry_sample_angles_dict() if self._single_geometry_enabled() else None
            detector_angles = self._single_geometry_detector_angles_dict() if self._single_geometry_enabled() else None

            if func == "simulate_2d":
                if geometry is not None:
                    single_crystal.simulate_2d_with_geometry(
                        det_type=det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
                        det_dist=dist,
                        det_poni1=poni1,
                        det_poni2=poni2,
                        det_rotx=rotx,
                        det_roty=roty,
                        det_rotz=rotz,
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
                        extra_hkls=extra_hkls,
                        geometry=geometry,
                        sample_angles=sample_angles,
                        detector_angles=detector_angles,
                    )
                else:
                    single_crystal.simulate_2d(
                        det_type=det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
                        det_dist=dist,
                        det_poni1=poni1,
                        det_poni2=poni2,
                        det_rotx=rotx,
                        det_roty=roty,
                        det_rotz=rotz,
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
                        extra_hkls=extra_hkls,
                    )

            elif func == "simulate_3d":
                if geometry is not None:
                    single_crystal.simulate_3d_with_geometry(
                        det_type=det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
                        det_dist=dist,
                        det_poni1=poni1,
                        det_poni2=poni2,
                        det_rotx=rotx,
                        det_roty=roty,
                        det_rotz=rotz,
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
                        extra_hkls=extra_hkls,
                        geometry=geometry,
                        sample_angles=sample_angles,
                        detector_angles=detector_angles,
                    )
                else:
                    single_crystal.simulate_3d(
                        det_type=det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
                        det_dist=dist,
                        det_poni1=poni1,
                        det_poni2=poni2,
                        det_rotx=rotx,
                        det_roty=roty,
                        det_rotz=rotz,
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
                        extra_hkls=extra_hkls,
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

                result = single_crystal.target_hkl_near_pixel_fixed_energy(
                    det_type=det_type,
                    det_pxsize_h=pxsize_h,
                    det_pxsize_v=pxsize_v,
                    det_ntum_pixels_h=num_px_h,
                    det_num_pixels_v=num_px_v,
                    det_binning=(bin_h, bin_v),
                    det_dist=dist,
                    det_poni1=poni1,
                    det_poni2=poni2,
                    det_rotx=rotx,
                    det_roty=roty,
                    det_rotz=rotz,
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

                if result.solutions is None:
                    QMessageBox.information(
                        self,
                        "No exact solution found",
                        "No exact sample-orientation solution was found for the current target/pixel tolerance.\n"
                        "The detector-space search still ran, but no real inverse-kinematic solution survived."
                    )

            elif func == "detector_rotations_collecting_Braggs":
                if geometry is not None:
                    scan_ranges = self._single_detector_scan_ranges_dict()
                    single_crystal.detector_rotations_collecting_Braggs_with_geometry(
                        det_type=det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
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

                    single_crystal.detector_rotations_collecting_Braggs(
                        det_type=det_type,
                        det_pxsize_h=pxsize_h,
                        det_pxsize_v=pxsize_v,
                        det_ntum_pixels_h=num_px_h,
                        det_num_pixels_v=num_px_v,
                        det_binning=(bin_h, bin_v),
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

                if geometry is not None:
                    motor1_name = self.single_geometry_motor1_name.text().strip()
                    motor2_name = self.single_geometry_motor2_name.text().strip()
                    if not motor1_name or not motor2_name:
                        raise ValueError("Geometry motor scan requires two motor names.")
                    if motor1_name == motor2_name:
                        raise ValueError("Geometry motor scan requires two different motor names.")

                    motor1_range = _parse_range_triplet_text(self.single_geometry_motor1_range.text(), "Motor 1 range")
                    motor2_range = _parse_range_triplet_text(self.single_geometry_motor2_range.text(), "Motor 2 range")

                    valid_points = single_crystal.scan_two_motors_for_Bragg_condition(
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
                    )
                    plot_parameter_mapping(valid_points, motor1_name, motor2_name)
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

                    valid_points = single_crystal.scan_two_parameters_for_Bragg_condition(
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
                        sam_rotation_order="xyz",
                        energy=energy,
                        e_bandwidth=e_bw,
                        hkls_names=hkls_names,
                        hkl_equivalent=hkl_equivalent,
                    )
                    plot_parameter_mapping(valid_points, param1_name, param2_name)

            return True

        except Exception as e:
            QMessageBox.critical(self, "Single Crystal Error", str(e))
            return False


class MainWindow(LegacyMainWindow):
    """
    Enhanced simulation GUI that preserves all legacy functionality while adding
    session persistence, autosave, configuration summaries, and a run log.
    """

    def __init__(self):
        self._loading_gui_state = False
        self._run_counter = 0
        super().__init__()
        self.setWindowTitle("XRDpy Simulation GUI")
        self.resize(750, 850)

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._autosave_now)

        self._summary_timer = QTimer(self)
        self._summary_timer.setSingleShot(True)
        self._summary_timer.timeout.connect(self._refresh_summary)

        self._build_session_controls()
        self._build_session_tab()
        self._connect_stateful_widgets()
        self.tabs.currentChanged.connect(lambda *_: self._schedule_summary_refresh())

        self._default_state = self._gather_gui_state(include_log=False)
        self._refresh_summary()
        self._maybe_restore_autosave()
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # UI augmentation
    # ------------------------------------------------------------------
    def _build_session_controls(self):
        root_layout = self.centralWidget().layout()

        self.session_group = QGroupBox("Session Persistence")
        session_layout = QGridLayout()
        self.session_group.setLayout(session_layout)

        self.btn_save_state = QPushButton("Save GUI State...")
        self.btn_save_state.clicked.connect(self._save_gui_state_to_file)
        session_layout.addWidget(self.btn_save_state, 0, 0)

        self.btn_load_state = QPushButton("Load GUI State...")
        self.btn_load_state.clicked.connect(self._load_gui_state_from_file)
        session_layout.addWidget(self.btn_load_state, 0, 1)

        self.btn_restore_autosave = QPushButton("Restore Last Autosave")
        self.btn_restore_autosave.clicked.connect(self._load_autosave_from_disk)
        session_layout.addWidget(self.btn_restore_autosave, 0, 2)

        self.btn_reset_defaults = QPushButton("Reset to Defaults")
        self.btn_reset_defaults.clicked.connect(self._reset_to_defaults)
        session_layout.addWidget(self.btn_reset_defaults, 0, 3)

        session_layout.addWidget(QLabel("Session Name:"), 1, 0)
        self.session_name_line = QLineEdit("")
        self.session_name_line.setPlaceholderText("Optional label for this simulation session")
        session_layout.addWidget(self.session_name_line, 1, 1, 1, 3)

        self.autosave_path_label = QLabel(str(self._autosave_path()))
        self.autosave_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.autosave_status_label = QLabel("Autosave: idle")
        self.autosave_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        session_layout.addWidget(QLabel("Autosave file:"), 2, 0)
        session_layout.addWidget(self.autosave_path_label, 2, 1, 1, 3)
        session_layout.addWidget(self.autosave_status_label, 3, 0, 1, 4)

        root_layout.insertWidget(0, self.session_group)

    def _build_session_tab(self):
        self.session_tab = QWidget()
        self.tabs.addTab(self.session_tab, "Session / Log")

        main_layout = QVBoxLayout()
        self.session_tab.setLayout(main_layout)

        notes_group = QGroupBox("Session Notes")
        notes_layout = QVBoxLayout()
        notes_group.setLayout(notes_layout)
        self.session_notes = QPlainTextEdit()
        self.session_notes.setPlaceholderText(
            "Write experimental context, detector notes, reflection choices, or run intentions here."
        )
        notes_layout.addWidget(self.session_notes)
        main_layout.addWidget(notes_group, stretch=2)

        summary_group = QGroupBox("Current Configuration Summary")
        summary_layout = QVBoxLayout()
        summary_group.setLayout(summary_layout)
        summary_btn_row = QHBoxLayout()
        self.refresh_summary_btn = QPushButton("Refresh Summary")
        self.refresh_summary_btn.clicked.connect(self._refresh_summary)
        summary_btn_row.addWidget(self.refresh_summary_btn)
        summary_btn_row.addStretch()
        summary_layout.addLayout(summary_btn_row)
        self.summary_view = QPlainTextEdit()
        self.summary_view.setReadOnly(True)
        summary_layout.addWidget(self.summary_view)
        main_layout.addWidget(summary_group, stretch=2)

        log_group = QGroupBox("Run Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        log_btn_row = QHBoxLayout()
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self._clear_log)
        log_btn_row.addWidget(self.clear_log_btn)
        log_btn_row.addStretch()
        log_layout.addLayout(log_btn_row)
        self.run_log = QPlainTextEdit()
        self.run_log.setReadOnly(True)
        log_layout.addWidget(self.run_log)
        main_layout.addWidget(log_group, stretch=2)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _autosave_path(self) -> Path:
        return Path.home() / AUTOSAVE_FILENAME

    def _save_state_dict_to_path(self, state: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_state_dict_from_path(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def _set_line_text(self, widget, value):
        widget.setText("" if value is None else str(value))

    def _set_plain_text(self, widget, value):
        widget.setPlainText("" if value is None else str(value))

    def _combo_set_text_if_present(self, combo, value):
        if value is None:
            return
        idx = combo.findText(str(value), Qt.MatchFixedString)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            if combo.isEditable():
                combo.setEditText(str(value))

    def _matrix_texts(self, matrix_edits):
        return [[cell.text() for cell in row] for row in matrix_edits]

    def _apply_matrix_texts(self, matrix_edits, values):
        if not values:
            return
        for i, row in enumerate(values[:len(matrix_edits)]):
            for j, value in enumerate(row[:len(matrix_edits[i])]):
                matrix_edits[i][j].setText(str(value))

    def _encode_geometry(self):
        return bytes(self.saveGeometry().toBase64()).decode("ascii")

    def _restore_geometry(self, value):
        if not value:
            return
        try:
            self.restoreGeometry(QByteArray.fromBase64(value.encode("ascii")))
        except Exception:
            pass

    def _gather_gui_state(self, *, include_log=True):
        state = {
            "state_version": GUI_STATE_VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "geometry": self._encode_geometry(),
            "ui": {
                "current_tab_index": self.tabs.currentIndex(),
                "session_name": self.session_name_line.text(),
                "session_notes": self.session_notes.toPlainText(),
            },
            "paths": {
                "poly_cif_file_path": self.poly_cif_file_path,
                "single_cif_file_path": self.single_cif_file_path,
            },
            "poly": {
                "func": self.poly_func_combo.currentText(),
                "cif_path": self.poly_line_cif_path.text(),
                "space_group": self.poly_line_space_group.text(),
                "qmax": self.poly_line_qmax.text(),
                "a": self.poly_line_sam_a.text(),
                "b": self.poly_line_sam_b.text(),
                "c": self.poly_line_sam_c.text(),
                "alpha": self.poly_line_sam_alpha.text(),
                "beta": self.poly_line_sam_beta.text(),
                "gamma": self.poly_line_sam_gamma.text(),
                "energy": self.poly_line_energy.text(),
                "ebw": self.poly_line_ebw.text(),
                "det_type": self.poly_combo_det_type.currentText(),
                "pxsize_h": self.poly_line_pxsize_h.text(),
                "pxsize_v": self.poly_line_pxsize_v.text(),
                "num_px_h": self.poly_line_num_px_h.text(),
                "num_px_v": self.poly_line_num_px_v.text(),
                "bin_h": self.poly_line_bin_h.text(),
                "bin_v": self.poly_line_bin_v.text(),
                "dist": self.poly_line_dist.text(),
                "poni1": self.poly_line_poni1.text(),
                "poni2": self.poly_line_poni2.text(),
                "rotx": self.poly_line_rotx.text(),
                "roty": self.poly_line_roty.text(),
                "rotz": self.poly_line_rotz.text(),
                "ref_source": self.poly_combo_refsrc.currentText(),
                "q_hkls": self.poly_line_qhkls.text(),
                "d_hkls": self.poly_line_dhkls.text(),
                "hkls": self.poly_line_hkls.text(),
                "cones": self.poly_line_cones.text(),
                "x_axis": self.poly_combo_xaxis.currentText(),
                "lorpol": self.poly_chk_lorpol.isChecked(),
                "fwhm": self.poly_line_fwhm.text(),
            },
            "single": {
                "func": self.single_func_combo.currentText(),
                "det_type": self.single_combo_det_type.currentText(),
                "pxsize_h": self.single_line_pxsize_h.text(),
                "pxsize_v": self.single_line_pxsize_v.text(),
                "num_px_h": self.single_line_num_px_h.text(),
                "num_px_v": self.single_line_num_px_v.text(),
                "bin_h": self.single_line_bin_h.text(),
                "bin_v": self.single_line_bin_v.text(),
                "dist": self.single_line_dist.text(),
                "poni1": self.single_line_poni1.text(),
                "poni2": self.single_line_poni2.text(),
                "det_rotx": self.single_line_rotx.text(),
                "det_roty": self.single_line_roty.text(),
                "det_rotz": self.single_line_rotz.text(),
                "energy": self.single_line_energy.text(),
                "ebw": self.single_line_ebw.text(),
                "space_group": self.single_line_space_group.text(),
                "qmax": self.single_line_qmax.text(),
                "a": self.single_line_sam_a.text(),
                "b": self.single_line_sam_b.text(),
                "c": self.single_line_sam_c.text(),
                "alpha": self.single_line_sam_alpha.text(),
                "beta": self.single_line_sam_beta.text(),
                "gamma": self.single_line_sam_gamma.text(),
                "use_custom_orientation": self.single_orientation_checkbox.isChecked(),
                "orientation_matrix": self._matrix_texts(self.orientation_entries),
                "sam_rotx": self.single_line_sam_rotx.text(),
                "sam_roty": self.single_line_sam_roty.text(),
                "sam_rotz": self.single_line_sam_rotz.text(),
                "q": self.single_line_q.text(),
                "d": self.single_line_d.text(),
                "names": self.single_line_names.text(),
                "extra_hkls": self.single_line_extra_hkls.text(),
                "equiv": self.single_equiv_checkbox.isChecked(),
                "angle_range": self.single_line_angle.text(),
                "param1": self.single_param1_combo.currentText(),
                "param2": self.single_param2_combo.currentText(),
                "param1_range": self.single_line_param1_range.text(),
                "param2_range": self.single_line_param2_range.text(),
                "geometry_mode": self.single_geometry_mode_combo.currentText(),
                "geometry_kind": self.single_geometry_kind_combo.currentText(),
                "geometry_kwargs": self.single_geometry_kwargs.toPlainText(),
                "geometry_sample_angles": self.single_geometry_sample_angles.toPlainText(),
                "geometry_detector_angles": self.single_geometry_detector_angles.toPlainText(),
                "custom_sample_chain": self.single_custom_sample_chain.toPlainText(),
                "custom_detector_chain": self.single_custom_detector_chain.toPlainText(),
                "geometry_motor1_name": self.single_geometry_motor1_name.text(),
                "geometry_motor1_range": self.single_geometry_motor1_range.text(),
                "geometry_motor2_name": self.single_geometry_motor2_name.text(),
                "geometry_motor2_range": self.single_geometry_motor2_range.text(),
                "geometry_detector_scan_ranges": self.single_geometry_detector_scan_ranges.toPlainText(),
                "target_hkl": self.single_line_target_hkl.text(),
                "target_pixel_h": self.single_line_target_pixel_h.text(),
                "target_pixel_v": self.single_line_target_pixel_v.text(),
                "target_pixel_tol": self.single_line_target_pixel_tol.text(),
                "eta_samples": self.single_line_eta_samples.text(),
                "phi_samples": self.single_line_phi_samples.text(),
                "wrap_angles": self.single_wrap_angles_checkbox.isChecked(),
                "inverse_detector_plot": self.single_inverse_detector_plot_checkbox.isChecked(),
                "inverse_2d_plot": self.single_inverse_2d_plot_checkbox.isChecked(),
                "inverse_3d_plot": self.single_inverse_3d_plot_checkbox.isChecked(),
            },
            "matrix_tool": {
                "space_group": self.matrix_window.line_space_group.text(),
                "a": self.matrix_window.line_a.text(),
                "b": self.matrix_window.line_b.text(),
                "c": self.matrix_window.line_c.text(),
                "alpha": self.matrix_window.line_alpha.text(),
                "beta": self.matrix_window.line_beta.text(),
                "gamma": self.matrix_window.line_gamma.text(),
                "orientation_matrix": self._matrix_texts(self.matrix_window.matrix_edits),
                "rotx": self.matrix_window.line_rotx.text(),
                "roty": self.matrix_window.line_roty.text(),
                "rotz": self.matrix_window.line_rotz.text(),
            },
        }
        if include_log:
            state["log"] = self.run_log.toPlainText()
        return state

    def _apply_gui_state(self, state: dict):
        self._loading_gui_state = True
        try:
            ui = state.get("ui", {})
            self._set_line_text(self.session_name_line, ui.get("session_name", ""))
            self._set_plain_text(self.session_notes, ui.get("session_notes", ""))

            paths = state.get("paths", {})
            self.poly_cif_file_path = paths.get("poly_cif_file_path")
            self.single_cif_file_path = paths.get("single_cif_file_path")

            poly = state.get("poly", {})
            self._combo_set_text_if_present(self.poly_func_combo, poly.get("func", "simulate_2d"))
            self._set_line_text(self.poly_line_cif_path, poly.get("cif_path", ""))
            self._set_line_text(self.poly_line_space_group, poly.get("space_group", ""))
            self._set_line_text(self.poly_line_qmax, poly.get("qmax", ""))
            self._set_line_text(self.poly_line_sam_a, poly.get("a", ""))
            self._set_line_text(self.poly_line_sam_b, poly.get("b", ""))
            self._set_line_text(self.poly_line_sam_c, poly.get("c", ""))
            self._set_line_text(self.poly_line_sam_alpha, poly.get("alpha", ""))
            self._set_line_text(self.poly_line_sam_beta, poly.get("beta", ""))
            self._set_line_text(self.poly_line_sam_gamma, poly.get("gamma", ""))
            self._set_line_text(self.poly_line_energy, poly.get("energy", ""))
            self._set_line_text(self.poly_line_ebw, poly.get("ebw", ""))
            self._combo_set_text_if_present(self.poly_combo_det_type, poly.get("det_type", "manual"))
            self._set_line_text(self.poly_line_pxsize_h, poly.get("pxsize_h", ""))
            self._set_line_text(self.poly_line_pxsize_v, poly.get("pxsize_v", ""))
            self._set_line_text(self.poly_line_num_px_h, poly.get("num_px_h", ""))
            self._set_line_text(self.poly_line_num_px_v, poly.get("num_px_v", ""))
            self._set_line_text(self.poly_line_bin_h, poly.get("bin_h", ""))
            self._set_line_text(self.poly_line_bin_v, poly.get("bin_v", ""))
            self._set_line_text(self.poly_line_dist, poly.get("dist", ""))
            self._set_line_text(self.poly_line_poni1, poly.get("poni1", ""))
            self._set_line_text(self.poly_line_poni2, poly.get("poni2", ""))
            self._set_line_text(self.poly_line_rotx, poly.get("rotx", ""))
            self._set_line_text(self.poly_line_roty, poly.get("roty", ""))
            self._set_line_text(self.poly_line_rotz, poly.get("rotz", ""))
            self._combo_set_text_if_present(self.poly_combo_refsrc, poly.get("ref_source", "Manual q/d + hkls"))
            self._set_line_text(self.poly_line_qhkls, poly.get("q_hkls", ""))
            self._set_line_text(self.poly_line_dhkls, poly.get("d_hkls", ""))
            self._set_line_text(self.poly_line_hkls, poly.get("hkls", ""))
            self._set_line_text(self.poly_line_cones, poly.get("cones", ""))
            self._combo_set_text_if_present(self.poly_combo_xaxis, poly.get("x_axis", "q"))
            self.poly_chk_lorpol.setChecked(bool(poly.get("lorpol", True)))
            self._set_line_text(self.poly_line_fwhm, poly.get("fwhm", "0.0"))

            single = state.get("single", {})
            self._combo_set_text_if_present(self.single_func_combo, single.get("func", "simulate_2d"))
            self._combo_set_text_if_present(self.single_combo_det_type, single.get("det_type", "manual"))
            self._set_line_text(self.single_line_pxsize_h, single.get("pxsize_h", ""))
            self._set_line_text(self.single_line_pxsize_v, single.get("pxsize_v", ""))
            self._set_line_text(self.single_line_num_px_h, single.get("num_px_h", ""))
            self._set_line_text(self.single_line_num_px_v, single.get("num_px_v", ""))
            self._set_line_text(self.single_line_bin_h, single.get("bin_h", ""))
            self._set_line_text(self.single_line_bin_v, single.get("bin_v", ""))
            self._set_line_text(self.single_line_dist, single.get("dist", ""))
            self._set_line_text(self.single_line_poni1, single.get("poni1", ""))
            self._set_line_text(self.single_line_poni2, single.get("poni2", ""))
            self._set_line_text(self.single_line_rotx, single.get("det_rotx", ""))
            self._set_line_text(self.single_line_roty, single.get("det_roty", ""))
            self._set_line_text(self.single_line_rotz, single.get("det_rotz", ""))
            self._set_line_text(self.single_line_energy, single.get("energy", ""))
            self._set_line_text(self.single_line_ebw, single.get("ebw", ""))
            self._set_line_text(self.single_line_space_group, single.get("space_group", ""))
            self._set_line_text(self.single_line_qmax, single.get("qmax", ""))
            self._set_line_text(self.single_line_sam_a, single.get("a", ""))
            self._set_line_text(self.single_line_sam_b, single.get("b", ""))
            self._set_line_text(self.single_line_sam_c, single.get("c", ""))
            self._set_line_text(self.single_line_sam_alpha, single.get("alpha", ""))
            self._set_line_text(self.single_line_sam_beta, single.get("beta", ""))
            self._set_line_text(self.single_line_sam_gamma, single.get("gamma", ""))
            self.single_orientation_checkbox.setChecked(bool(single.get("use_custom_orientation", False)))
            self._apply_matrix_texts(self.orientation_entries, single.get("orientation_matrix"))
            self._set_line_text(self.single_line_sam_rotx, single.get("sam_rotx", ""))
            self._set_line_text(self.single_line_sam_roty, single.get("sam_roty", ""))
            self._set_line_text(self.single_line_sam_rotz, single.get("sam_rotz", ""))
            self._set_line_text(self.single_line_q, single.get("q", ""))
            self._set_line_text(self.single_line_d, single.get("d", ""))
            self._set_line_text(self.single_line_names, single.get("names", ""))
            self._set_line_text(self.single_line_extra_hkls, single.get("extra_hkls", ""))
            self.single_equiv_checkbox.setChecked(bool(single.get("equiv", False)))
            self._set_line_text(self.single_line_angle, single.get("angle_range", ""))
            self._combo_set_text_if_present(self.single_param1_combo, single.get("param1", "rotx"))
            self._combo_set_text_if_present(self.single_param2_combo, single.get("param2", "roty"))
            self._set_line_text(self.single_line_param1_range, single.get("param1_range", ""))
            self._set_line_text(self.single_line_param2_range, single.get("param2_range", ""))
            self._combo_set_text_if_present(self.single_geometry_mode_combo, single.get("geometry_mode", "Legacy Euler"))
            self._combo_set_text_if_present(self.single_geometry_kind_combo, single.get("geometry_kind"))
            self._set_plain_text(self.single_geometry_kwargs, single.get("geometry_kwargs", ""))
            self._set_plain_text(self.single_geometry_sample_angles, single.get("geometry_sample_angles", ""))
            self._set_plain_text(self.single_geometry_detector_angles, single.get("geometry_detector_angles", ""))
            self._set_plain_text(self.single_custom_sample_chain, single.get("custom_sample_chain", ""))
            self._set_plain_text(self.single_custom_detector_chain, single.get("custom_detector_chain", ""))
            self._set_line_text(self.single_geometry_motor1_name, single.get("geometry_motor1_name", "omega"))
            self._set_line_text(self.single_geometry_motor1_range, single.get("geometry_motor1_range", "-90,90,5"))
            self._set_line_text(self.single_geometry_motor2_name, single.get("geometry_motor2_name", "kappa"))
            self._set_line_text(self.single_geometry_motor2_range, single.get("geometry_motor2_range", "-90,90,5"))
            self._set_plain_text(self.single_geometry_detector_scan_ranges, single.get("geometry_detector_scan_ranges", ""))
            self._set_line_text(self.single_line_target_hkl, single.get("target_hkl", "[1,1,0]"))
            self._set_line_text(self.single_line_target_pixel_h, single.get("target_pixel_h", "900"))
            self._set_line_text(self.single_line_target_pixel_v, single.get("target_pixel_v", "900"))
            self._set_line_text(self.single_line_target_pixel_tol, single.get("target_pixel_tol", "20"))
            self._set_line_text(self.single_line_eta_samples, single.get("eta_samples", "1441"))
            self._set_line_text(self.single_line_phi_samples, single.get("phi_samples", "361"))
            self.single_wrap_angles_checkbox.setChecked(bool(single.get("wrap_angles", True)))
            self.single_inverse_detector_plot_checkbox.setChecked(bool(single.get("inverse_detector_plot", True)))
            self.single_inverse_2d_plot_checkbox.setChecked(bool(single.get("inverse_2d_plot", True)))
            self.single_inverse_3d_plot_checkbox.setChecked(bool(single.get("inverse_3d_plot", False)))

            matrix = state.get("matrix_tool", {})
            self._set_line_text(self.matrix_window.line_space_group, matrix.get("space_group", ""))
            self._set_line_text(self.matrix_window.line_a, matrix.get("a", ""))
            self._set_line_text(self.matrix_window.line_b, matrix.get("b", ""))
            self._set_line_text(self.matrix_window.line_c, matrix.get("c", ""))
            self._set_line_text(self.matrix_window.line_alpha, matrix.get("alpha", ""))
            self._set_line_text(self.matrix_window.line_beta, matrix.get("beta", ""))
            self._set_line_text(self.matrix_window.line_gamma, matrix.get("gamma", ""))
            self._apply_matrix_texts(self.matrix_window.matrix_edits, matrix.get("orientation_matrix"))
            self._set_line_text(self.matrix_window.line_rotx, matrix.get("rotx", ""))
            self._set_line_text(self.matrix_window.line_roty, matrix.get("roty", ""))
            self._set_line_text(self.matrix_window.line_rotz, matrix.get("rotz", ""))
            self.matrix_window._invalidate_result_matrix()

            self._restore_geometry(state.get("geometry"))
            self.run_log.setPlainText(state.get("log", ""))

            self._poly_detector_changed()
            self._poly_refsrc_changed()
            self._poly_func_changed()
            self._single_detector_changed()
            self._single_geometry_mode_changed()
            self._single_func_changed()
            self._toggle_orientation_matrix(Qt.Checked if self.single_orientation_checkbox.isChecked() else Qt.Unchecked)
            self.tabs.setCurrentIndex(int(ui.get("current_tab_index", 0)))
        finally:
            self._loading_gui_state = False
            self._refresh_summary()
            self._schedule_autosave()

    def _save_gui_state_to_file(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save Simulation GUI State",
            directory=str(Path.home() / "simulation_gui_state.json"),
            filter="JSON Files (*.json);;All Files (*)",
        )
        if not file_name:
            return
        try:
            state = self._gather_gui_state(include_log=True)
            self._save_state_dict_to_path(state, Path(file_name))
            self._log(f"Saved GUI state to: {file_name}")
            self.statusBar().showMessage(f"Saved GUI state to {file_name}", 4000)
        except Exception as exc:
            QMessageBox.critical(self, "Save GUI State Error", str(exc))

    def _load_gui_state_from_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Load Simulation GUI State",
            directory=str(Path.home()),
            filter="JSON Files (*.json);;All Files (*)",
        )
        if not file_name:
            return
        try:
            state = self._load_state_dict_from_path(Path(file_name))
            self._apply_gui_state(state)
            self._log(f"Loaded GUI state from: {file_name}")
            self.statusBar().showMessage(f"Loaded GUI state from {file_name}", 4000)
        except Exception as exc:
            QMessageBox.critical(self, "Load GUI State Error", str(exc))

    def _load_autosave_from_disk(self):
        path = self._autosave_path()
        if not path.exists():
            QMessageBox.information(self, "No Autosave", f"No autosave file found at:\n{path}")
            return
        try:
            state = self._load_state_dict_from_path(path)
            self._apply_gui_state(state)
            self._log(f"Restored autosaved state from: {path}")
            self.statusBar().showMessage(f"Restored autosave from {path}", 4000)
        except Exception as exc:
            QMessageBox.critical(self, "Load Autosave Error", str(exc))

    def _reset_to_defaults(self):
        answer = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Restore the GUI to its initial default state?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._apply_gui_state(self._default_state)
        self.run_log.clear()
        self._log("Reset GUI to default state.")

    def _maybe_restore_autosave(self):
        path = self._autosave_path()
        if not path.exists():
            self.autosave_status_label.setText("Autosave: no previous autosave found")
            return
        answer = QMessageBox.question(
            self,
            "Restore Previous Session",
            f"A previous autosaved simulation GUI state was found:\n{path}\n\nRestore it now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            try:
                state = self._load_state_dict_from_path(path)
                self._apply_gui_state(state)
                self._log(f"Restored autosaved GUI state from: {path}")
                self.autosave_status_label.setText(f"Autosave: restored from {path}")
            except Exception as exc:
                QMessageBox.warning(self, "Autosave Restore Failed", str(exc))
        else:
            self.autosave_status_label.setText(f"Autosave available at: {path}")

    # ------------------------------------------------------------------
    # Logging, summary, autosave
    # ------------------------------------------------------------------
    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.run_log.appendPlainText(f"[{timestamp}] {message}")

    def _clear_log(self):
        self.run_log.clear()
        self._log("Cleared run log.")

    def _schedule_autosave(self):
        if self._loading_gui_state:
            return
        self._autosave_timer.start(900)
        self._schedule_summary_refresh()

    def _schedule_summary_refresh(self):
        if self._loading_gui_state:
            return
        self._summary_timer.start(250)

    def _autosave_now(self):
        try:
            path = self._autosave_path()
            self._save_state_dict_to_path(self._gather_gui_state(include_log=True), path)
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.autosave_status_label.setText(f"Autosave: saved at {stamp}")
        except Exception as exc:
            self.autosave_status_label.setText(f"Autosave failed: {exc}")

    def _connect_stateful_widgets(self):
        def maybe_connect(widget, signal_name):
            if widget in {self.summary_view, self.run_log}:
                return
            signal = getattr(widget, signal_name, None)
            if signal is not None:
                signal.connect(self._schedule_autosave)

        for line in self.findChildren(QLineEdit):
            maybe_connect(line, "textChanged")
        for combo in self.findChildren(QComboBox):
            maybe_connect(combo, "currentTextChanged")
        for chk in self.findChildren(QCheckBox):
            maybe_connect(chk, "stateChanged")
        for plain in self.findChildren(QPlainTextEdit):
            maybe_connect(plain, "textChanged")
        for line in self.matrix_window.findChildren(QLineEdit):
            maybe_connect(line, "textChanged")
        for plain in self.matrix_window.findChildren(QPlainTextEdit):
            maybe_connect(plain, "textChanged")

    def _refresh_summary(self):
        current_tab = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else ""
        geometry_mode = self.single_geometry_mode_combo.currentText()
        geometry_active = self._single_geometry_enabled()
        legacy_suffix = " (ignored in geometry mode)" if geometry_active else ""

        lines = [
            f"Session name: {self.session_name_line.text().strip() or '(unnamed)'}",
            f"Active tab: {current_tab}",
            "",
            "Polycrystalline",
            f"  Function: {self.poly_func_combo.currentText()}",
            f"  Reflection source: {self.poly_combo_refsrc.currentText()}",
            f"  CIF path: {self.poly_line_cif_path.text().strip() or '(none)'}",
            (
                "  Lattice: SG={sg}, a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}".format(
                    sg=self.poly_line_space_group.text(),
                    a=self.poly_line_sam_a.text(),
                    b=self.poly_line_sam_b.text(),
                    c=self.poly_line_sam_c.text(),
                    alpha=self.poly_line_sam_alpha.text(),
                    beta=self.poly_line_sam_beta.text(),
                    gamma=self.poly_line_sam_gamma.text(),
                )
            ),
            f"  Beam: E={self.poly_line_energy.text()} eV, ΔE/E={self.poly_line_ebw.text()} %",
            f"  Detector: {self.poly_combo_det_type.currentText()}, dist={self.poly_line_dist.text()} m, binning=({self.poly_line_bin_h.text()}, {self.poly_line_bin_v.text()})",
            f"  qmax={self.poly_line_qmax.text()} Å^-1, cones={self.poly_line_cones.text()}, x_axis={self.poly_combo_xaxis.currentText()}, lorentz/polarization={self.poly_chk_lorpol.isChecked()}, FWHM={self.poly_line_fwhm.text()}",
            "",
            "Single Crystal",
            f"  Function: {self.single_func_combo.currentText()}",
            f"  Detector: {self.single_combo_det_type.currentText()}, dist={self.single_line_dist.text()} m, binning=({self.single_line_bin_h.text()}, {self.single_line_bin_v.text()})",
            f"  Beam: E={self.single_line_energy.text()} eV, ΔE/E={self.single_line_ebw.text()} %",
            (
                "  Lattice: SG={sg}, a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}".format(
                    sg=self.single_line_space_group.text(),
                    a=self.single_line_sam_a.text(),
                    b=self.single_line_sam_b.text(),
                    c=self.single_line_sam_c.text(),
                    alpha=self.single_line_sam_alpha.text(),
                    beta=self.single_line_sam_beta.text(),
                    gamma=self.single_line_sam_gamma.text(),
                )
            ),
            f"  qmax={self.single_line_qmax.text()} Å^-1, custom orientation={self.single_orientation_checkbox.isChecked()}",
            f"  Geometry mode: {geometry_mode}",
            f"  Legacy sample rotations: ({self.single_line_sam_rotx.text()}, {self.single_line_sam_roty.text()}, {self.single_line_sam_rotz.text()}) deg{legacy_suffix}",
            f"  Legacy detector Euler rotations: ({self.single_line_rotx.text()}, {self.single_line_roty.text()}, {self.single_line_rotz.text()}) deg{legacy_suffix}",
            f"  Geometry kind: {self.single_geometry_kind_combo.currentText() if self._single_predefined_geometry_enabled() else '(custom/legacy)'}",
            f"  Geometry kwargs: {self.single_geometry_kwargs.toPlainText().strip() or '(none)'}",
            f"  Sample motor angles: {self.single_geometry_sample_angles.toPlainText().strip() or '(none)'}",
            f"  Detector motor angles: {self.single_geometry_detector_angles.toPlainText().strip() or '(none)'}",
            f"  Custom sample chain: {self.single_custom_sample_chain.toPlainText().strip() or '(none)'}",
            f"  Custom detector chain: {self.single_custom_detector_chain.toPlainText().strip() or '(none)'}",
            f"  Bragg hkls: {self.single_line_names.text().strip() or '(none)'}",
            f"  Forced extra HKLs: {self.single_line_extra_hkls.text().strip() or '(none)'}",
            f"  Legacy detector angle range: {self.single_line_angle.text().strip() or '(default)'}{' (ignored in geometry mode)' if geometry_active else ''}",
            f"  Legacy parameter scan: {self.single_param1_combo.currentText()} in {self.single_line_param1_range.text().strip() or '(none)'}, {self.single_param2_combo.currentText()} in {self.single_line_param2_range.text().strip() or '(none)'}{' (ignored in geometry mode)' if geometry_active else ''}",
            f"  Geometry sample scan: {self.single_geometry_motor1_name.text().strip() or '(motor1)'} in {self.single_geometry_motor1_range.text().strip() or '(none)'}, {self.single_geometry_motor2_name.text().strip() or '(motor2)'} in {self.single_geometry_motor2_range.text().strip() or '(none)'}",
            f"  Geometry detector scan ranges: {self.single_geometry_detector_scan_ranges.toPlainText().strip() or '(none)'}",
            f"  Fixed-energy target: hkl={self.single_line_target_hkl.text().strip() or '(none)'}, pixel=({self.single_line_target_pixel_h.text().strip() or '?'}, {self.single_line_target_pixel_v.text().strip() or '?'}) ± {self.single_line_target_pixel_tol.text().strip() or '?'} px",
            "",
            "Matrix Rotation Tool",
            (
                "  Lattice: SG={sg}, a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}".format(
                    sg=self.matrix_window.line_space_group.text(),
                    a=self.matrix_window.line_a.text(),
                    b=self.matrix_window.line_b.text(),
                    c=self.matrix_window.line_c.text(),
                    alpha=self.matrix_window.line_alpha.text(),
                    beta=self.matrix_window.line_beta.text(),
                    gamma=self.matrix_window.line_gamma.text(),
                )
            ),
            f"  Pending rotation: ({self.matrix_window.line_rotx.text()}, {self.matrix_window.line_roty.text()}, {self.matrix_window.line_rotz.text()}) deg",
        ]
        self.summary_view.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Legacy action wrappers
    # ------------------------------------------------------------------
    def _poly_run_function(self):
        self._run_counter += 1
        func_name = self.poly_func_combo.currentText()
        self._log(f"Run #{self._run_counter}: requested polycrystalline function '{func_name}'.")
        ok = LegacyMainWindow._poly_run_function(self)
        self._refresh_summary()
        self._schedule_autosave()
        if ok:
            self.statusBar().showMessage(f"Ran polycrystalline function: {func_name}", 4000)
        else:
            self._log(f"Run #{self._run_counter}: polycrystalline function '{func_name}' failed.")
            self.statusBar().showMessage(f"Polycrystalline function failed: {func_name}", 4000)

    def _single_run_function(self):
        self._run_counter += 1
        func_name = self.single_func_combo.currentText()
        self._log(f"Run #{self._run_counter}: requested single-crystal function '{func_name}'.")
        ok = LegacyMainWindow._single_run_function(self)
        self._refresh_summary()
        self._schedule_autosave()
        if ok:
            self.statusBar().showMessage(f"Ran single-crystal function: {func_name}", 4000)
        else:
            self._log(f"Run #{self._run_counter}: single-crystal function '{func_name}' failed.")
            self.statusBar().showMessage(f"Single-crystal function failed: {func_name}", 4000)

    def _poly_load_cif(self):
        before = self.poly_line_cif_path.text()
        LegacyMainWindow._poly_load_cif(self)
        after = self.poly_line_cif_path.text()
        if after and after != before:
            self._log("Updated polycrystalline lattice fields from CIF.")
            self._refresh_summary()
            self._schedule_autosave()

    def _single_load_cif(self):
        before = self.single_cif_file_path
        LegacyMainWindow._single_load_cif(self)
        after = self.single_cif_file_path
        if after and after != before:
            self._log("Updated single-crystal lattice fields from CIF.")
            self._refresh_summary()
            self._schedule_autosave()

    def _import_orientation_from_rotation_tool(self):
        before = self._matrix_texts(self.orientation_entries)
        LegacyMainWindow._import_orientation_from_rotation_tool(self)
        after = self._matrix_texts(self.orientation_entries)
        if after != before:
            self._log("Imported orientation matrix from the rotation tool into the single-crystal tab.")
            self._refresh_summary()
            self._schedule_autosave()

    def closeEvent(self, event):
        try:
            self._autosave_now()
        finally:
            super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
