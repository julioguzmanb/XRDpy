import sys
import numpy as np
import matplotlib.pyplot as plt
import pyFAI.detectors

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QMessageBox, QGroupBox, QCheckBox, QScrollArea,
    QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

# Simulation imports
from XRDpy.simulation import polycrystalline, single_crystal

from XRDpy.utils import apply_rotation
from XRDpy.plot import plot_parameter_mapping


# CIF import
from XRDpy.cif import Cif


plt.ion()


def compute_lattice_orientation(a, b, c, alpha_deg, beta_deg, gamma_deg):

    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    # v1 along x
    v1 = np.array([a, 0., 0.])
    # v2 in the xy-plane
    v2 = np.array([
        b * np.cos(gamma),
        b * np.sin(gamma),
        0.
    ])
    # v3 general
    v3_x = c * np.cos(beta)
    v3_y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    # Corrected formula for v3_z
    term = 1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2
    # Ensure numerical stability by handling potential negative values inside the sqrt
    term = np.maximum(term, 0)
    v3_z = c * np.sqrt(term)

    v3 = np.array([v3_x, v3_y, v3_z])

    return np.vstack((v1, v2, v3))


class MatrixRotationWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Matrix Rotation Tool")
        self.resize(800, 600)

        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QVBoxLayout()
        container.setLayout(main_layout)

        # (A) Lattice + CIF group
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

        # Load CIF button
        row += 1
        load_cif_btn = QPushButton("Load CIF")
        lattice_layout.addWidget(load_cif_btn, row, 0, 1, 2)
        load_cif_btn.clicked.connect(self._load_cif)

        # Compute orientation from lattice
        row += 1
        compute_btn = QPushButton("Compute Orientation from Lattice")
        lattice_layout.addWidget(compute_btn, row, 0, 1, 2)
        compute_btn.clicked.connect(self._compute_orientation)

        # (B) Orientation Matrix group
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
                orientation_layout.addWidget(edit, i, j)
                row_edits.append(edit)
            self.matrix_edits.append(row_edits)

        # (C) Rotation Angles
        rotation_box = QGroupBox("Apply Rotation [deg]")
        rotation_layout = QHBoxLayout()
        rotation_box.setLayout(rotation_layout)
        main_layout.addWidget(rotation_box)

        rotation_layout.addWidget(QLabel("rotx:"))
        self.line_rotx = QLineEdit("0")
        self.line_rotx.setValidator(QDoubleValidator())
        rotation_layout.addWidget(self.line_rotx)

        rotation_layout.addWidget(QLabel("roty:"))
        self.line_roty = QLineEdit("0")
        self.line_roty.setValidator(QDoubleValidator())
        rotation_layout.addWidget(self.line_roty)

        rotation_layout.addWidget(QLabel("rotz:"))
        self.line_rotz = QLineEdit("0")
        self.line_rotz.setValidator(QDoubleValidator())
        rotation_layout.addWidget(self.line_rotz)

        rotation_layout.addStretch()

        apply_rotation_btn = QPushButton("Apply Rotation")
        main_layout.addWidget(apply_rotation_btn)
        apply_rotation_btn.clicked.connect(self._apply_rotation)

        # (D) Rotated Result
        self.result_group = QGroupBox("Rotated Matrix (Output)")
        self.result_layout = QGridLayout()
        self.result_group.setLayout(self.result_layout)
        main_layout.addWidget(self.result_group)

        self.result_labels = []
        for i in range(3):
            row_labels = []
            for j in range(3):
                lbl = QLabel("0.0")
                lbl.setFixedWidth(60)
                lbl.setAlignment(Qt.AlignCenter)
                self.result_layout.addWidget(lbl, i, j)
                row_labels.append(lbl)
            self.result_labels.append(row_labels)

        # (E) NEW: Button to update the orientation matrix from the rotated result
        self.update_matrix_button = QPushButton("Update Orientation Matrix from Result")
        main_layout.addWidget(self.update_matrix_button)
        self.update_matrix_button.clicked.connect(self._update_orientation_from_result)

    # -----------------------
    #   EVENT HANDLERS
    # -----------------------
    def _load_cif(self):
        """Load a CIF file to fill space group & lattice parameters."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open CIF File",
            directory="",
            filter="CIF Files (*.cif);;All Files (*)"
        )
        if file_name:
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

                QMessageBox.information(self, "CIF Loaded",
                                        f"Successfully loaded: {file_name}")

            except Exception as e:
                QMessageBox.critical(self, "CIF Error",
                                     f"Failed to read or parse the CIF file:\n{str(e)}")

    def _compute_orientation(self):
        try:
            a_val = float(self.line_a.text())
            b_val = float(self.line_b.text())
            c_val = float(self.line_c.text())
            alpha_val = float(self.line_alpha.text())
            beta_val = float(self.line_beta.text())
            gamma_val = float(self.line_gamma.text())

            orientation = compute_lattice_orientation(
                a_val, b_val, c_val,
                alpha_val, beta_val, gamma_val
            )
            # Populate the matrix fields
            for i in range(3):
                for j in range(3):
                    self.matrix_edits[i][j].setText(f"{orientation[i, j]:.4f}")

        except ValueError as ex:
            QMessageBox.critical(
                self, "Input Error", f"Invalid numeric input:\n{str(ex)}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Orientation Error", f"Error computing orientation:\n{str(e)}"
            )

    def _apply_rotation(self):

        try:
            initial_matrix = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    val_str = self.matrix_edits[i][j].text().strip()
                    initial_matrix[i, j] = float(val_str)

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

        except ValueError as ex:
            QMessageBox.critical(
                self, "Input Error", f"Invalid numeric input:\n{str(ex)}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Rotation Error", f"Error applying rotation:\n{str(e)}"
            )

    def _update_orientation_from_result(self):
        """
        Copy the values from the Rotated Matrix (Output) back into
        the editable Orientation Matrix, so you can chain rotations.
        """
        try:
            for i in range(3):
                for j in range(3):
                    text = self.result_labels[i][j].text().strip()
                    # Ensure it's a valid number
                    val = float(text)
                    self.matrix_edits[i][j].setText(f"{val:.4f}")
        except ValueError:
            QMessageBox.warning(
                self,
                "Update Error",
                "Rotated matrix contains invalid or non-numeric values.\n"
                "Please apply a valid rotation first."
            )

    def get_current_matrix(self):
        """
        Return a 3x3 matrix representing the 'current' orientation:

        - Prefer the Rotated Matrix (Output) if it contains valid numbers.
        - Otherwise, fall back to the editable Orientation Matrix.

        Raises an exception if neither contains valid numeric data.
        """
        mat = np.zeros((3, 3))

        # First try the rotated result
        use_result = True
        for i in range(3):
            for j in range(3):
                text = self.result_labels[i][j].text().strip()
                try:
                    val = float(text)
                except ValueError:
                    use_result = False
                    break
                mat[i, j] = val
            if not use_result:
                break

        if use_result:
            return mat

        # Fallback: orientation matrix edits
        for i in range(3):
            for j in range(3):
                text = self.matrix_edits[i][j].text().strip()
                if text == "":
                    raise ValueError(
                        f"Orientation matrix element at Row {i+1}, Col {j+1} is empty."
                    )
                mat[i, j] = float(text)

        return mat


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Greetings Vanadorians. Julio is supposed to be writing (probably he's not...) "
            "but you can use my facilities and I'll do better than he would :D"
        )

        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout()
        container.setLayout(layout)

        # Create main tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # 1) Polycrystalline tab
        self.poly_tab = QWidget()
        self.tabs.addTab(self.poly_tab, "Polycrystalline")
        self._init_poly_tab()

        # 2) Single Crystal tab
        self.single_tab = QWidget()
        self.tabs.addTab(self.single_tab, "Single Crystal")
        self._init_single_tab()

        # Button that opens the separate MatrixRotationWindow
        self.open_rotation_button = QPushButton("Open Matrix Rotation Tool")
        layout.addWidget(self.open_rotation_button)
        self.open_rotation_button.clicked.connect(self._open_matrix_rotation_window)

        self.matrix_window = MatrixRotationWindow()

        self.resize(1000, 900)

    def _open_matrix_rotation_window(self):
        """Open the Matrix Rotation tool in a separate window."""
        self.matrix_window.show()

    #   POLYCRYSTALLINE TAB
    def _init_poly_tab(self):
        """
        Here is your existing polycrystalline tab code, directly pasted from
        your provided script.
        """
        main_layout = QVBoxLayout()
        self.poly_tab.setLayout(main_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        scroll_layout = QVBoxLayout()
        scroll_content.setLayout(scroll_layout)

        # (A) Function selection
        func_group = QGroupBox("Function Selection")
        func_layout = QHBoxLayout()
        func_group.setLayout(func_layout)
        func_layout.addWidget(QLabel("Select Polycrystalline Function:"))
        self.poly_func_combo = QComboBox()
        self.poly_func_combo.addItems(["simulate_2d", "simulate_3d"])
        func_layout.addWidget(self.poly_func_combo)
        func_layout.addStretch()
        scroll_layout.addWidget(func_group)

        # (B) Detector Settings
        det_group = QGroupBox("Detector Settings")
        det_layout = QGridLayout()
        det_group.setLayout(det_layout)
        scroll_layout.addWidget(det_group)

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

        # (C) Beam Parameters
        beam_group = QGroupBox("Beam Parameters")
        beam_layout = QGridLayout()
        beam_group.setLayout(beam_layout)
        scroll_layout.addWidget(beam_group)

        beam_layout.addWidget(QLabel("Energy [eV]:"), 0, 0)
        self.poly_line_energy = QLineEdit("15000")
        self.poly_line_energy.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.poly_line_energy, 0, 1)

        beam_layout.addWidget(QLabel("∆E/E [%]:"), 1, 0)
        self.poly_line_ebw = QLineEdit("1.5")
        self.poly_line_ebw.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.poly_line_ebw, 1, 1)

        # (D) Reflection Lists
        refl_group = QGroupBox("Reflection Lists")
        refl_layout = QVBoxLayout()
        refl_group.setLayout(refl_layout)
        scroll_layout.addWidget(refl_group)

        qdhkls_layout = QGridLayout()
        refl_layout.addLayout(qdhkls_layout)

        qdhkls_layout.addWidget(QLabel("q_hkls (comma):"), 0, 0)
        self.poly_line_qhkls = QLineEdit("")
        self.poly_line_qhkls.setPlaceholderText("e.g., 1.0,2.0,3.0")
        qdhkls_layout.addWidget(self.poly_line_qhkls, 0, 1)

        qdhkls_layout.addWidget(QLabel("OR d_hkls (comma):"), 1, 0)
        self.poly_line_dhkls = QLineEdit("")
        self.poly_line_dhkls.setPlaceholderText("e.g., 1.0,2.0,3.0")
        qdhkls_layout.addWidget(self.poly_line_dhkls, 1, 1)

        refl_layout.addWidget(QLabel("hkls_names (e.g. [1,0,2],[0,1,2]):"))
        self.poly_line_hkls = QLineEdit("")
        self.poly_line_hkls.setPlaceholderText("e.g., [1,0,2],[0,1,2]")
        refl_layout.addWidget(self.poly_line_hkls)

        # (E) Function-Specific
        func_specific_group = QGroupBox("Function-Specific Parameters")
        func_specific_layout = QGridLayout()
        func_specific_group.setLayout(func_specific_layout)
        scroll_layout.addWidget(func_specific_group)

        func_specific_layout.addWidget(QLabel("Cones Number of Points:"), 0, 0)
        self.poly_line_cones = QLineEdit("30")
        self.poly_line_cones.setValidator(QDoubleValidator())
        func_specific_layout.addWidget(self.poly_line_cones, 0, 1)

        self.poly_run_btn = QPushButton("Run Polycrystal Function")
        scroll_layout.addWidget(self.poly_run_btn)

        self.poly_func_combo.currentIndexChanged.connect(self._poly_func_changed)
        self.poly_run_btn.clicked.connect(self._poly_run_function)
        self._poly_func_changed()

    def _poly_detector_changed(self):
        det_type = self.poly_combo_det_type.currentText().lower()
        manual = (det_type == "manual")
        self.poly_manual_group.setVisible(manual)

    def _poly_func_changed(self):
        pass

    def _poly_run_function(self):
        try:
            func = self.poly_func_combo.currentText()
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
            energy = float(self.poly_line_energy.text())
            e_bw = float(self.poly_line_ebw.text())

            q_text = self.poly_line_qhkls.text().strip()
            d_text = self.poly_line_dhkls.text().strip()

            if q_text:
                q_hkls = np.array([float(x.strip()) for x in q_text.split(",")])
                d_hkls = None
            elif d_text:
                d_hkls = np.array([float(x.strip()) for x in d_text.split(",")])
                q_hkls = None
            else:
                q_hkls = None
                d_hkls = None

            names_text = self.poly_line_hkls.text().strip()
            hkls_names = None
            if names_text:
                hkl_groups = names_text.split("],")
                tmp_list = []
                for grp in hkl_groups:
                    cleaned = grp.replace("[", "").replace("]", "")
                    arr = [int(xx.strip()) for xx in cleaned.split(",")]
                    tmp_list.append(arr)
                hkls_names = np.array(tmp_list)

            cones_num = int(float(self.poly_line_cones.text()))

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
        except Exception as e:
            QMessageBox.critical(self, "Polycrystalline Error", str(e))

    # --------------------------------------------------------------------------
    #   SINGLE CRYSTAL TAB
    # --------------------------------------------------------------------------
    def _init_single_tab(self):
        main_layout = QVBoxLayout()
        self.single_tab.setLayout(main_layout)

        # Add scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        scroll_layout = QVBoxLayout()
        scroll_content.setLayout(scroll_layout)

        # (A) Function selection
        func_group = QGroupBox("Function Selection")
        func_layout = QHBoxLayout()
        func_group.setLayout(func_layout)
        func_layout.addWidget(QLabel("Select Single-Crystal Function:"))


        # self.single_func_combo = QComboBox()
        # self.single_func_combo.addItems([
        #     "simulate_2d",
        #     "simulate_3d",
        #     "sample_rotations_for_Bragg_condition",
        #     "detector_rotations_collecting_Braggs"
        # ])
        # self.single_func_combo = QComboBox()
        # self.single_func_combo.addItems([
        #     "simulate_2d",
        #     "simulate_3d",
        #     "sample_rotations_for_Bragg_condition",
        #     "detector_rotations_collecting_Braggs",
        #     "scan_two_parameters_for_Bragg_condition"
        # ])

        self.single_func_combo = QComboBox()
        self.single_func_combo.addItems([
            "simulate_2d",
            "simulate_3d",
            "detector_rotations_collecting_Braggs",
            "scan_two_parameters_for_Bragg_condition"
        ])



        func_layout.addWidget(self.single_func_combo)
        func_layout.addStretch()
        scroll_layout.addWidget(func_group)

        # (B) Detector Settings
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

        det_common_layout.addWidget(QLabel("Detector Rotations [deg]:"), 4, 0)
        rots_hbox = QHBoxLayout()
        self.single_line_rotx = QLineEdit("0")
        self.single_line_rotx.setValidator(QDoubleValidator())
        self.single_line_roty = QLineEdit("0")
        self.single_line_roty.setValidator(QDoubleValidator())
        self.single_line_rotz = QLineEdit("0")
        self.single_line_rotz.setValidator(QDoubleValidator())
        rots_hbox.addWidget(QLabel("rotx:"))
        rots_hbox.addWidget(self.single_line_rotx)
        rots_hbox.addWidget(QLabel("roty:"))
        rots_hbox.addWidget(self.single_line_roty)
        rots_hbox.addWidget(QLabel("rotz:"))
        rots_hbox.addWidget(self.single_line_rotz)
        det_common_layout.addLayout(rots_hbox, 4, 1)

        self.single_combo_det_type.currentIndexChanged.connect(self._single_detector_changed)
        self._single_detector_changed()

        # # (C) Beam Parameters
        # beam_group = QGroupBox("Beam Parameters")
        # beam_layout = QGridLayout()
        # beam_group.setLayout(beam_layout)
        # scroll_layout.addWidget(beam_group)

        # (C) Beam Parameters
        beam_group = QGroupBox("Beam Parameters")
        beam_layout = QGridLayout()
        beam_group.setLayout(beam_layout)
        scroll_layout.addWidget(beam_group)
        self.single_beam_group = beam_group  # <-- store reference


        beam_layout.addWidget(QLabel("Energy [eV]:"), 0, 0)
        self.single_line_energy = QLineEdit("15000")
        self.single_line_energy.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.single_line_energy, 0, 1)

        beam_layout.addWidget(QLabel("∆E/E [%]:"), 1, 0)
        self.single_line_ebw = QLineEdit("1.5")
        self.single_line_ebw.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.single_line_ebw, 1, 1)

        # (D) Sample Parameters
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

        # [NEW] Load CIF button
        row += 1
        self.load_cif_button = QPushButton("Load CIF")
        sample_layout.addWidget(self.load_cif_button, row, 0, 1, 2)
        self.load_cif_button.clicked.connect(self._load_cif)

        # (E) Optional Initial Orientation Matrix
        orientation_checkbox_layout = QHBoxLayout()
        self.single_orientation_checkbox = QCheckBox("Use Custom Initial Orientation")
        orientation_checkbox_layout.addWidget(self.single_orientation_checkbox)

        # NEW: Import button to pull matrix from Matrix Rotation Tool
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

        # (F) Sample Rotations
        rot_group = QGroupBox("Sample Rotations [deg]")
        rot_layout = QHBoxLayout()
        rot_group.setLayout(rot_layout)
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
        scroll_layout.addWidget(rot_group)

        # (G) Reflection Lists
        self.single_refl_group = QGroupBox("Reflection Lists (for Bragg checks)")
        refl_layout = QGridLayout()
        self.single_refl_group.setLayout(refl_layout)
        scroll_layout.addWidget(self.single_refl_group)

        refl_layout.addWidget(QLabel("q_hkls (comma):"), 0, 0)
        self.single_line_q = QLineEdit("")
        self.single_line_q.setPlaceholderText("e.g., 1.0,2.0,3.0")
        refl_layout.addWidget(self.single_line_q, 0, 1)

        refl_layout.addWidget(QLabel("OR d_hkls (comma):"), 1, 0)
        self.single_line_d = QLineEdit("")
        self.single_line_d.setPlaceholderText("e.g., 1.0,2.0,3.0")
        refl_layout.addWidget(self.single_line_d, 1, 1)

        refl_layout.addWidget(QLabel("hkls_names (e.g. [1,0,2],[0,1,2]):"), 2, 0)
        self.single_line_names = QLineEdit("")
        self.single_line_names.setPlaceholderText("e.g., [1,0,2],[0,1,2]")
        refl_layout.addWidget(self.single_line_names, 2, 1)
        # Checkbox: consider equivalent hkl's
        self.single_equiv_checkbox = QCheckBox("Consider Equivalent")
        refl_layout.addWidget(self.single_equiv_checkbox, 3, 0, 1, 2)


        # # (H) Angle Range
        # angle_group = QGroupBox("Angle Range (for Bragg or Detector Rotations)")
        # angle_layout = QHBoxLayout()
        # angle_group.setLayout(angle_layout)
        # angle_layout.addWidget(QLabel("angle_range (start,stop,step):"))
        # self.single_line_angle = QLineEdit("-90,90,5")
        # self.single_line_angle.setPlaceholderText("e.g., -90,90,1")
        # angle_layout.addWidget(self.single_line_angle)
        # angle_layout.addStretch()
        # scroll_layout.addWidget(angle_group)
        # (H) Angle Range (for Bragg or Detector Rotations)
        angle_group = QGroupBox("Angle Range (for Bragg or Detector Rotations)")
        angle_layout = QHBoxLayout()
        angle_group.setLayout(angle_layout)
        angle_layout.addWidget(QLabel("angle_range (start,stop,step):"))
        self.single_line_angle = QLineEdit("-90,90,5")
        self.single_line_angle.setPlaceholderText("e.g., -90,90,1")
        angle_layout.addWidget(self.single_line_angle)
        angle_layout.addStretch()
        scroll_layout.addWidget(angle_group)
        self.single_angle_group = angle_group  # <-- store reference



        # (H2) Parameter Scan (for generalized Bragg mapping)
        param_scan_group = QGroupBox("Parameter Scan (for generalized Bragg condition)")
        param_scan_layout = QGridLayout()
        param_scan_group.setLayout(param_scan_layout)
        scroll_layout.addWidget(param_scan_group)

        # Parameter names
        param_scan_layout.addWidget(QLabel("Parameter 1:"), 0, 0)
        self.single_param1_combo = QComboBox()
        self.single_param1_combo.addItems(["rotx", "roty", "rotz", "energy"])
        param_scan_layout.addWidget(self.single_param1_combo, 0, 1)

        param_scan_layout.addWidget(QLabel("Parameter 2:"), 1, 0)
        self.single_param2_combo = QComboBox()
        self.single_param2_combo.addItems(["rotx", "roty", "rotz", "energy"])
        param_scan_layout.addWidget(self.single_param2_combo, 1, 1)

        # Parameter ranges
        param_scan_layout.addWidget(QLabel("param1_range (start,stop,step):"), 2, 0)
        self.single_line_param1_range = QLineEdit("-90,90,5")
        self.single_line_param1_range.setPlaceholderText("e.g., -90,90,5 or 5000,20000,500")
        param_scan_layout.addWidget(self.single_line_param1_range, 2, 1)

        param_scan_layout.addWidget(QLabel("param2_range (start,stop,step):"), 3, 0)
        self.single_line_param2_range = QLineEdit("-90,90,5")
        self.single_line_param2_range.setPlaceholderText("e.g., -90,90,5 or 5000,20000,500")
        param_scan_layout.addWidget(self.single_line_param2_range, 3, 1)

        # Hide by default; only show for the new function
        self.param_scan_group = param_scan_group
        self.param_scan_group.setVisible(False)

        # # (I) Run Button
        # self.single_run_btn = QPushButton("Run Single-Crystal Function")
        # scroll_layout.addWidget(self.single_run_btn)



        # (I) Run Button
        self.single_run_btn = QPushButton("Run Single-Crystal Function")
        scroll_layout.addWidget(self.single_run_btn)

        self.single_func_combo.currentIndexChanged.connect(self._single_func_changed)
        self.single_run_btn.clicked.connect(self._single_run_function)
        self._single_func_changed()

    def _toggle_orientation_matrix(self, state):
        if state == Qt.Checked:
            self.single_label_sam_init.setVisible(True)
            self.orientation_group.setVisible(True)
        else:
            self.single_label_sam_init.setVisible(False)
            self.orientation_group.setVisible(False)



    # def _single_func_changed(self):
    #     func = self.single_func_combo.currentText()
    #     needs_detector_info = func in ["simulate_2d", "simulate_3d", "detector_rotations_collecting_Braggs"]
    #     needs_bragg = func in ["sample_rotations_for_Bragg_condition", "detector_rotations_collecting_Braggs"]

    #     self.single_det_group.setVisible(needs_detector_info)
    #     if not needs_detector_info:
    #         self.single_manual_group.setVisible(False)

    #     self.single_refl_group.setVisible(needs_bragg)
    #     self.single_line_angle.setVisible(needs_bragg or needs_detector_info)
    def _single_func_changed(self):
        func = self.single_func_combo.currentText()
        needs_detector_info = func in [
            "simulate_2d",
            "simulate_3d",
            "detector_rotations_collecting_Braggs"
        ]
        needs_bragg = func in [
            "sample_rotations_for_Bragg_condition",
            "detector_rotations_collecting_Braggs",
            "scan_two_parameters_for_Bragg_condition"
        ]
        needs_param_scan = func == "scan_two_parameters_for_Bragg_condition"
    
    # def _single_func_changed(self):
    #     func = self.single_func_combo.currentText()
    #     needs_detector_info = func in [
    #         "simulate_2d",
    #         "simulate_3d",
    #         "detector_rotations_collecting_Braggs"
    #     ]
    #     needs_bragg = func in [
    #         "detector_rotations_collecting_Braggs",
    #         "scan_two_parameters_for_Bragg_condition"
    #     ]
    #     needs_param_scan = func == "scan_two_parameters_for_Bragg_condition"

    #     self.single_det_group.setVisible(needs_detector_info)
    #     if not needs_detector_info:
    #         self.single_manual_group.setVisible(False)

    #     self.single_refl_group.setVisible(needs_bragg)
    #     self.single_line_angle.setVisible(needs_bragg or needs_detector_info)

    #     # Show parameter-scan controls only for the new function
    #     if hasattr(self, "param_scan_group"):
    #         self.param_scan_group.setVisible(needs_param_scan)
    
    def _single_func_changed(self):
        func = self.single_func_combo.currentText()

        needs_detector_info = func in [
            "simulate_2d",
            "simulate_3d",
            "detector_rotations_collecting_Braggs"
        ]
        needs_bragg = func in [
            "detector_rotations_collecting_Braggs",
            "scan_two_parameters_for_Bragg_condition"
        ]
        needs_param_scan = (func == "scan_two_parameters_for_Bragg_condition")

        # Detector settings
        self.single_det_group.setVisible(needs_detector_info)
        if not needs_detector_info:
            self.single_manual_group.setVisible(False)

        # Reflection lists
        self.single_refl_group.setVisible(needs_bragg)

        # Angle Range group: hide ONLY for the generalized scan
        if hasattr(self, "single_angle_group"):
            self.single_angle_group.setVisible(not needs_param_scan)

        # Parameter-scan controls only for the new function
        if hasattr(self, "param_scan_group"):
            self.param_scan_group.setVisible(needs_param_scan)


    # def _single_func_changed(self):
    #     func = self.single_func_combo.currentText()

    #     needs_detector_info = func in [
    #         "simulate_2d",
    #         "simulate_3d",
    #         "detector_rotations_collecting_Braggs"
    #     ]
    #     needs_bragg = func in [
    #         "detector_rotations_collecting_Braggs",
    #         "scan_two_parameters_for_Bragg_condition"
    #     ]
    #     needs_param_scan = func == "scan_two_parameters_for_Bragg_condition"

    #     # Beam & angle groups: hide ONLY for the generalized scan function
    #     hide_for_scan = (func == "scan_two_parameters_for_Bragg_condition")

    #     # Detector settings
    #     self.single_det_group.setVisible(needs_detector_info)
    #     if not needs_detector_info:
    #         self.single_manual_group.setVisible(False)

    #     # Reflection lists
    #     self.single_refl_group.setVisible(needs_bragg)

    #     # Beam Parameters group
    #     if hasattr(self, "single_beam_group"):
    #         self.single_beam_group.setVisible(not hide_for_scan)

    #     # Angle Range group
    #     if hasattr(self, "single_angle_group"):
    #         self.single_angle_group.setVisible(not hide_for_scan)

    #     # Parameter-scan controls only for the new function
    #     if hasattr(self, "param_scan_group"):
    #         self.param_scan_group.setVisible(needs_param_scan)


    def _single_detector_changed(self):
        det_type = self.single_combo_det_type.currentText().lower()
        manual = (det_type == "manual")
        self.single_manual_group.setVisible(manual)

    def collect_matrix(self):
        matrix = np.zeros((3, 3))
        try:
            for i in range(3):
                for j in range(3):
                    value = self.orientation_entries[i][j].text().strip()
                    if value == "":
                        raise ValueError(
                            f"Matrix element at Row {i + 1}, Col {j + 1} is empty."
                        )
                    matrix[i][j] = float(value)
            return matrix
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid matrix entry: {str(e)}")
            return None

    def _load_cif(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open CIF File",
            directory="",
            filter="CIF Files (*.cif);;All Files (*)"
        )
        if file_name:
            try:
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

                QMessageBox.information(self, "CIF Loaded", f"Successfully loaded: {file_name}")

            except Exception as e:
                QMessageBox.critical(self, "CIF Error", f"Failed to read or parse the CIF file:\n{str(e)}")

    def _import_orientation_from_rotation_tool(self):
        """
        Import a 3x3 orientation matrix from the Matrix Rotation Tool window
        into the Single Crystal 'Custom Initial Orientation' matrix.
        """
        try:
            if not hasattr(self, "matrix_window") or self.matrix_window is None:
                QMessageBox.warning(
                    self,
                    "Matrix Rotation Tool",
                    "Matrix Rotation Tool window is not available."
                )
                return

            mat = self.matrix_window.get_current_matrix()

            # Ensure the custom orientation is enabled
            if not self.single_orientation_checkbox.isChecked():
                self.single_orientation_checkbox.setChecked(True)

            # Fill the orientation_entries with the imported matrix
            for i in range(3):
                for j in range(3):
                    self.orientation_entries[i][j].setText(f"{mat[i, j]:.6f}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Could not import matrix from Matrix Rotation Tool:\n{str(e)}"
            )

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

            sg_txt = self.single_line_space_group.text().strip()
            if not sg_txt:
                raise ValueError("Space Group cannot be empty.")
            sam_space_group = int(float(sg_txt))

            qmax_txt = self.single_line_qmax.text().strip()
            if not qmax_txt:
                raise ValueError("qmax cannot be empty.")
            qmax = float(qmax_txt)

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
                    return

            sam_rotx = float(self.single_line_sam_rotx.text())
            sam_roty = float(self.single_line_sam_roty.text())
            sam_rotz = float(self.single_line_sam_rotz.text())

            angle_str = self.single_line_angle.text().strip()
            if angle_str:
                try:
                    a_start, a_stop, a_step = [float(x.strip()) for x in angle_str.split(",")]
                    angle_range = (a_start, a_stop, a_step)
                except ValueError:
                    QMessageBox.critical(
                        self,
                        "Input Error",
                        "angle_range must be three comma-separated numbers (start,stop,step)."
                    )
                    return
            else:
                angle_range = None

            q_str = self.single_line_q.text().strip()
            d_str = self.single_line_d.text().strip()
            n_str = self.single_line_names.text().strip()

            if q_str:
                q_hkls = np.array([float(x.strip()) for x in q_str.split(",")])
                d_hkls = None
            elif d_str:
                d_hkls = np.array([float(x.strip()) for x in d_str.split(",")])
                q_hkls = None
            else:
                q_hkls = None
                d_hkls = None

            hkls_names = None
            if n_str:
                try:
                    items = n_str.split("],")
                    tmpn = []
                    for it in items:
                        cleaned = it.replace("[", "").replace("]", "")
                        arr = [int(xx.strip()) for xx in cleaned.split(",")]
                        tmpn.append(arr)
                    hkls_names = np.array(tmpn)
                except ValueError:
                    QMessageBox.critical(
                        self,
                        "Input Error",
                        "hkls_names must be in the format [h,k,l],[h,k,l],..."
                    )
                    return

            if func == "simulate_2d":
                single_crystal.simulate_2d(
                    det_type=det_type,
                    det_pxsize_h=pxsize_h,
                    det_pxsize_v=pxsize_v,
                    det_ntum_pixels_h=num_px_h,
                    det_num_pixels_v=num_px_v,
                    det_binning=(bin_h, bin_v),
                    det_dist=dist, det_poni1=poni1, det_poni2=poni2,
                    det_rotx=rotx, det_roty=roty, det_rotz=rotz,
                    energy=energy, e_bandwidth=e_bw,
                    sam_space_group=sam_space_group,
                    sam_a=sam_a, sam_b=sam_b, sam_c=sam_c,
                    sam_alpha=sam_alpha, sam_beta=sam_beta, sam_gamma=sam_gamma,
                    sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                    sam_rotx=sam_rotx, sam_roty=sam_roty, sam_rotz=sam_rotz,
                    qmax=qmax
                )
            elif func == "simulate_3d":
                single_crystal.simulate_3d(
                    det_type=det_type,
                    det_pxsize_h=pxsize_h,
                    det_pxsize_v=pxsize_v,
                    det_ntum_pixels_h=num_px_h,
                    det_num_pixels_v=num_px_v,
                    det_binning=(bin_h, bin_v),
                    det_dist=dist, det_poni1=poni1, det_poni2=poni2,
                    det_rotx=rotx, det_roty=roty, det_rotz=rotz,
                    energy=energy, e_bandwidth=e_bw,
                    sam_space_group=sam_space_group,
                    sam_a=sam_a, sam_b=sam_b, sam_c=sam_c,
                    sam_alpha=sam_alpha, sam_beta=sam_beta, sam_gamma=sam_gamma,
                    sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                    sam_rotx=sam_rotx, sam_roty=sam_roty, sam_rotz=sam_rotz,
                    qmax=qmax
                )
            # elif func == "sample_rotations_for_Bragg_condition":
            #     if angle_range is None:
            #         angle_range = (-90, 90, 5)

            #     single_crystal.sample_rotations_for_Bragg_condition(
            #         sam_space_group=sam_space_group,
            #         sam_a=sam_a, sam_b=sam_b, sam_c=sam_c,
            #         sam_alpha=sam_alpha, sam_beta=sam_beta, sam_gamma=sam_gamma,
            #         sam_initial_crystal_orientation=sam_initial_crystal_orientation,
            #         sam_rotx=sam_rotx, sam_roty=sam_roty, sam_rotz=sam_rotz,
            #         angle_range=angle_range,
            #         energy=energy, e_bandwidth=e_bw,
            #         q_hkls=q_hkls, d_hkls=d_hkls,
            #         hkls_names=hkls_names
            #     )
            elif func == "detector_rotations_collecting_Braggs":
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
                    hkls=hkls_names
                )
            
            # elif func == "scan_two_parameters_for_Bragg_condition":
            #     # Parse param1_name, param2_name
            #     param1_name = self.single_param1_combo.currentText()
            #     param2_name = self.single_param2_combo.currentText()

            #     if param1_name == param2_name:
            #         QMessageBox.critical(
            #             self,
            #             "Input Error",
            #             "Parameter 1 and Parameter 2 must be different."
            #         )
            #         return

            #     # Parse param1_range
            #     p1_str = self.single_line_param1_range.text().strip()
            #     p2_str = self.single_line_param2_range.text().strip()
            #     try:
            #         p1_start, p1_stop, p1_step = [float(x.strip()) for x in p1_str.split(",")]
            #         p2_start, p2_stop, p2_step = [float(x.strip()) for x in p2_str.split(",")]
            #         param1_range = (p1_start, p1_stop, p1_step)
            #         param2_range = (p2_start, p2_stop, p2_step)
            #     except Exception:
            #         QMessageBox.critical(
            #             self,
            #             "Input Error",
            #             "param1_range and param2_range must be of the form start,stop,step."
            #         )
            #         return

            #     if hkls_names is None:
            #         QMessageBox.critical(
            #             self,
            #             "Input Error",
            #             "You must specify hkls_names for Bragg-condition mapping."
            #         )
            #         return

            #     # Call the new scan function from XRDpy.simulation.single_crystal
            #     valid_points = single_crystal.scan_two_parameters_for_Bragg_condition(
            #         param1_name=param1_name,
            #         param2_name=param2_name,
            #         param1_range=param1_range,
            #         param2_range=param2_range,
            #         sam_space_group=sam_space_group,
            #         sam_a=sam_a, sam_b=sam_b, sam_c=sam_c,
            #         sam_alpha=sam_alpha, sam_beta=sam_beta, sam_gamma=sam_gamma,
            #         sam_initial_crystal_orientation=sam_initial_crystal_orientation,
            #         sam_rotx=sam_rotx, sam_roty=sam_roty, sam_rotz=sam_rotz,
            #         sam_rotation_order="xyz",
            #         energy=energy,
            #         e_bandwidth=e_bw,
            #         hkls_names=hkls_names
            #     )
        
            elif func == "scan_two_parameters_for_Bragg_condition":
                # Parse param1_name, param2_name
                param1_name = self.single_param1_combo.currentText()
                param2_name = self.single_param2_combo.currentText()

                if param1_name == param2_name:
                    QMessageBox.critical(
                        self,
                        "Input Error",
                        "Parameter 1 and Parameter 2 must be different."
                    )
                    return

                # Parse param1_range
                p1_str = self.single_line_param1_range.text().strip()
                p2_str = self.single_line_param2_range.text().strip()
                try:
                    p1_start, p1_stop, p1_step = [float(x.strip()) for x in p1_str.split(",")]
                    p2_start, p2_stop, p2_step = [float(x.strip()) for x in p2_str.split(",")]
                    param1_range = (p1_start, p1_stop, p1_step)
                    param2_range = (p2_start, p2_stop, p2_step)
                except Exception:
                    QMessageBox.critical(
                        self,
                        "Input Error",
                        "param1_range and param2_range must be of the form start,stop,step."
                    )
                    return

                if hkls_names is None:
                    QMessageBox.critical(
                        self,
                        "Input Error",
                        "You must specify hkls_names for Bragg-condition mapping."
                    )
                    return

                # Equivalent hkls flag from GUI
                hkl_equivalent = (
                    hasattr(self, "single_equiv_checkbox")
                    and self.single_equiv_checkbox.isChecked()
                )

                # Call the new scan function from XRDpy.simulation.single_crystal
                valid_points = single_crystal.scan_two_parameters_for_Bragg_condition(
                    param1_name=param1_name,
                    param2_name=param2_name,
                    param1_range=param1_range,
                    param2_range=param2_range,
                    sam_space_group=sam_space_group,
                    sam_a=sam_a, sam_b=sam_b, sam_c=sam_c,
                    sam_alpha=sam_alpha, sam_beta=sam_beta, sam_gamma=sam_gamma,
                    sam_initial_crystal_orientation=sam_initial_crystal_orientation,
                    sam_rotx=sam_rotx, sam_roty=sam_roty, sam_rotz=sam_rotz,
                    sam_rotation_order="xyz",
                    energy=energy,
                    e_bandwidth=e_bw,
                    hkls_names=hkls_names,
                    hkl_equivalent=hkl_equivalent     
                )


                # Plot the mapping with clickable legend
                try:
                    plot_parameter_mapping(valid_points, param1_name, param2_name)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Plot Warning",
                        f"Parameter mapping computed but plotting failed:\n{str(e)}"
                    )


        except Exception as e:
            QMessageBox.critical(self, "Single Crystal Error", str(e))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
