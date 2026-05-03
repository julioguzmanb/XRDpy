from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...cif import Cif
from ...utils import apply_rotation


def compute_lattice_orientation(
    a: float,
    b: float,
    c: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> np.ndarray:
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    v1 = np.array([a, 0.0, 0.0])
    v2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])

    v3_x = c * np.cos(beta)
    v3_y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)

    term = (
        1
        - np.cos(beta) ** 2
        - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)) ** 2
    )
    term = np.maximum(term, 0)
    v3_z = c * np.sqrt(term)

    v3 = np.array([v3_x, v3_y, v3_z])
    return np.vstack((v1, v2, v3))


class MatrixRotationWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
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

        self.matrix_edits: list[list[QLineEdit]] = []
        for i in range(3):
            row_edits: list[QLineEdit] = []
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

        self.result_labels: list[list[QLabel]] = []
        for i in range(3):
            row_labels: list[QLabel] = []
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

    def _invalidate_result_matrix(self) -> None:
        self._result_matrix_valid = False
        for row in self.result_labels:
            for lbl in row:
                lbl.setText("")

    def _load_cif(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open CIF File",
            directory="",
            filter="CIF Files (*.cif);;All Files (*)",
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
            QMessageBox.critical(
                self,
                "CIF Error",
                f"Failed to read or parse the CIF file:\n{str(e)}",
            )

    def _compute_orientation(self) -> None:
        try:
            a_val = float(self.line_a.text())
            b_val = float(self.line_b.text())
            c_val = float(self.line_c.text())
            alpha_val = float(self.line_alpha.text())
            beta_val = float(self.line_beta.text())
            gamma_val = float(self.line_gamma.text())

            orientation = compute_lattice_orientation(
                a_val,
                b_val,
                c_val,
                alpha_val,
                beta_val,
                gamma_val,
            )
            for i in range(3):
                for j in range(3):
                    self.matrix_edits[i][j].setText(f"{orientation[i, j]:.4f}")
            self._invalidate_result_matrix()
        except ValueError as ex:
            QMessageBox.critical(self, "Input Error", f"Invalid numeric input:\n{str(ex)}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Orientation Error",
                f"Error computing orientation:\n{str(e)}",
            )

    def _apply_rotation(self) -> None:
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

    def _update_orientation_from_result(self) -> None:
        if not self._result_matrix_valid:
            QMessageBox.warning(
                self,
                "Update Error",
                "No valid rotated matrix is available yet.\nPlease apply a valid rotation first.",
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
                "Rotated matrix contains invalid or non-numeric values.\nPlease apply a valid rotation first.",
            )

    def get_current_matrix(self) -> np.ndarray:
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
                    raise ValueError(
                        f"Orientation matrix element at Row {i + 1}, Col {j + 1} is empty."
                    )
                mat[i, j] = float(text)

        return mat