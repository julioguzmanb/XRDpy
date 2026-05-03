from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pyFAI.detectors

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ... import polycrystalline
from ... import sample as sample_mod
from ... import utils as xutils
from ...cif import Cif
from ..services.simulation_service import SimulationService
from ..state import GuiState


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


class PolycrystallineTab(QWidget):
    """
    Full polycrystalline simulation tab extracted from the legacy GUI.

    The attribute names intentionally stay close to the legacy version so the
    remaining migration steps are easier and less error-prone.
    """

    state_changed = pyqtSignal()
    run_completed = pyqtSignal(bool, str)

    def __init__(
        self,
        service: SimulationService,
        state: GuiState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.service = service
        self.state = state
        self._loading_state = False

        self._build_ui()
        self._connect_stateful_widgets()
        self.load_from_state()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

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
        self.poly_line_cif_path.setPlaceholderText(
            "Optional for 2d/3d (if using manual q/d). Required for simulate_1d."
        )
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

        self.poly_refsrc_group = QGroupBox("Reflections Source (for 2D/3D)")
        refsrc_layout = QHBoxLayout()
        self.poly_refsrc_group.setLayout(refsrc_layout)
        refsrc_layout.addWidget(QLabel("Use reflections from:"))
        self.poly_combo_refsrc = QComboBox()
        self.poly_combo_refsrc.addItems([
            "CIF / lattice (auto from qmax)",
            "Manual q/d + hkls_names",
        ])
        refsrc_layout.addWidget(self.poly_combo_refsrc)
        refsrc_layout.addStretch()
        scroll_layout.addWidget(self.poly_refsrc_group)

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

        self.poly_btn_browse_cif.clicked.connect(self._poly_browse_cif)
        self.poly_btn_load_cif.clicked.connect(self._poly_load_cif)
        self.poly_combo_det_type.currentIndexChanged.connect(self._poly_detector_changed)
        self.poly_combo_refsrc.currentIndexChanged.connect(self._poly_refsrc_changed)
        self.poly_func_combo.currentIndexChanged.connect(self._poly_func_changed)
        self.poly_run_btn.clicked.connect(self._poly_run_function)

        self._poly_detector_changed()
        self._poly_refsrc_changed()
        self._poly_func_changed()

    # ------------------------------------------------------------------
    # State synchronization
    # ------------------------------------------------------------------
    def load_from_state(self) -> None:
        self._loading_state = True
        try:
            poly = self.state.poly

            self._set_line_text(self.poly_line_cif_path, poly.cif_path)
            self._set_line_text(self.poly_line_space_group, poly.space_group)
            self._set_line_text(self.poly_line_qmax, poly.qmax)
            self._set_line_text(self.poly_line_sam_a, poly.a)
            self._set_line_text(self.poly_line_sam_b, poly.b)
            self._set_line_text(self.poly_line_sam_c, poly.c)
            self._set_line_text(self.poly_line_sam_alpha, poly.alpha)
            self._set_line_text(self.poly_line_sam_beta, poly.beta)
            self._set_line_text(self.poly_line_sam_gamma, poly.gamma)
            self._set_line_text(self.poly_line_energy, poly.energy)
            self._set_line_text(self.poly_line_ebw, poly.ebw)
            self._set_line_text(self.poly_line_pxsize_h, poly.pxsize_h)
            self._set_line_text(self.poly_line_pxsize_v, poly.pxsize_v)
            self._set_line_text(self.poly_line_num_px_h, poly.num_px_h)
            self._set_line_text(self.poly_line_num_px_v, poly.num_px_v)
            self._set_line_text(self.poly_line_bin_h, poly.bin_h)
            self._set_line_text(self.poly_line_bin_v, poly.bin_v)
            self._set_line_text(self.poly_line_dist, poly.dist)
            self._set_line_text(self.poly_line_poni1, poly.poni1)
            self._set_line_text(self.poly_line_poni2, poly.poni2)
            self._set_line_text(self.poly_line_rotx, poly.rotx)
            self._set_line_text(self.poly_line_roty, poly.roty)
            self._set_line_text(self.poly_line_rotz, poly.rotz)
            self._set_line_text(self.poly_line_qhkls, poly.q_hkls)
            self._set_line_text(self.poly_line_dhkls, poly.d_hkls)
            self._set_line_text(self.poly_line_hkls, poly.hkls)
            self._set_line_text(self.poly_line_cones, poly.cones)
            self._set_line_text(self.poly_line_fwhm, poly.fwhm)

            self._combo_set_text_if_present(self.poly_func_combo, poly.func)
            self._combo_set_text_if_present(self.poly_combo_det_type, poly.det_type)
            self._combo_set_text_if_present(self.poly_combo_refsrc, poly.ref_source)
            self._combo_set_text_if_present(self.poly_combo_xaxis, poly.x_axis)
            self.poly_chk_lorpol.setChecked(bool(poly.lorpol))

            self._poly_detector_changed()
            self._poly_refsrc_changed()
            self._poly_func_changed()
        finally:
            self._loading_state = False
            self._write_back_to_state()

    def save_to_state(self) -> None:
        self._write_back_to_state()

    def _write_back_to_state(self) -> None:
        self.state.poly.func = self.poly_func_combo.currentText()
        self.state.poly.cif_path = self.poly_line_cif_path.text()
        self.state.poly.space_group = self.poly_line_space_group.text()
        self.state.poly.qmax = self.poly_line_qmax.text()
        self.state.poly.a = self.poly_line_sam_a.text()
        self.state.poly.b = self.poly_line_sam_b.text()
        self.state.poly.c = self.poly_line_sam_c.text()
        self.state.poly.alpha = self.poly_line_sam_alpha.text()
        self.state.poly.beta = self.poly_line_sam_beta.text()
        self.state.poly.gamma = self.poly_line_sam_gamma.text()
        self.state.poly.energy = self.poly_line_energy.text()
        self.state.poly.ebw = self.poly_line_ebw.text()
        self.state.poly.det_type = self.poly_combo_det_type.currentText()
        self.state.poly.pxsize_h = self.poly_line_pxsize_h.text()
        self.state.poly.pxsize_v = self.poly_line_pxsize_v.text()
        self.state.poly.num_px_h = self.poly_line_num_px_h.text()
        self.state.poly.num_px_v = self.poly_line_num_px_v.text()
        self.state.poly.bin_h = self.poly_line_bin_h.text()
        self.state.poly.bin_v = self.poly_line_bin_v.text()
        self.state.poly.dist = self.poly_line_dist.text()
        self.state.poly.poni1 = self.poly_line_poni1.text()
        self.state.poly.poni2 = self.poly_line_poni2.text()
        self.state.poly.rotx = self.poly_line_rotx.text()
        self.state.poly.roty = self.poly_line_roty.text()
        self.state.poly.rotz = self.poly_line_rotz.text()
        self.state.poly.ref_source = self.poly_combo_refsrc.currentText()
        self.state.poly.q_hkls = self.poly_line_qhkls.text()
        self.state.poly.d_hkls = self.poly_line_dhkls.text()
        self.state.poly.hkls = self.poly_line_hkls.text()
        self.state.poly.cones = self.poly_line_cones.text()
        self.state.poly.x_axis = self.poly_combo_xaxis.currentText()
        self.state.poly.lorpol = self.poly_chk_lorpol.isChecked()
        self.state.poly.fwhm = self.poly_line_fwhm.text()
        self.state.paths.poly_cif_file_path = self.poly_line_cif_path.text() or None

        if not self._loading_state:
            self.state_changed.emit()

    # ------------------------------------------------------------------
    # CIF helpers
    # ------------------------------------------------------------------
    def _poly_browse_cif(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open CIF File",
            directory="",
            filter="CIF Files (*.cif);;All Files (*)",
        )
        if file_name:
            self.poly_line_cif_path.setText(file_name)
            self.state.paths.poly_cif_file_path = file_name
            self._write_back_to_state()

    def _poly_load_cif(self) -> None:
        file_name = (self.poly_line_cif_path.text() or "").strip()
        if not file_name:
            self._poly_browse_cif()
            file_name = (self.poly_line_cif_path.text() or "").strip()
            if not file_name:
                return

        try:
            self._load_cif_into_fields(file_name)
            self.state.paths.poly_cif_file_path = file_name
            self._write_back_to_state()
            QMessageBox.information(self, "CIF Loaded", f"Successfully loaded: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "CIF Error", f"Failed to read or parse the CIF file:\n{str(e)}")

    def _load_cif_into_fields(self, file_name: str) -> Cif:
        cif_data = Cif(file_path=file_name)

        if cif_data.space_group is not None:
            self.poly_line_space_group.setText(str(cif_data.space_group))
        if cif_data.a is not None:
            self.poly_line_sam_a.setText(str(cif_data.a))
        if cif_data.b is not None:
            self.poly_line_sam_b.setText(str(cif_data.b))
        if cif_data.c is not None:
            self.poly_line_sam_c.setText(str(cif_data.c))
        if cif_data.alpha is not None:
            self.poly_line_sam_alpha.setText(str(cif_data.alpha))
        if cif_data.beta is not None:
            self.poly_line_sam_beta.setText(str(cif_data.beta))
        if cif_data.gamma is not None:
            self.poly_line_sam_gamma.setText(str(cif_data.gamma))

        return cif_data

    # ------------------------------------------------------------------
    # UI visibility logic
    # ------------------------------------------------------------------
    def _poly_detector_changed(self) -> None:
        det_type = self.poly_combo_det_type.currentText().lower()
        self.poly_manual_group.setVisible(det_type == "manual")
        self._write_back_to_state()

    def _poly_refsrc_changed(self) -> None:
        manual = self.poly_combo_refsrc.currentText().startswith("Manual")
        self.poly_refl_group.setVisible(manual)
        self._write_back_to_state()

    def _poly_func_changed(self) -> None:
        func = self.poly_func_combo.currentText()

        is_1d = func == "simulate_1d"
        is_cones = func in ("simulate_2d", "simulate_3d")

        self.poly_det_group.setVisible(is_cones)
        self.poly_refsrc_group.setVisible(is_cones)
        self.poly_func_specific_group.setVisible(is_cones)

        if is_cones:
            self._poly_refsrc_changed()
        else:
            self.poly_refl_group.setVisible(False)

        self.poly_1d_group.setVisible(is_1d)
        self._write_back_to_state()

    # ------------------------------------------------------------------
    # Reflection construction
    # ------------------------------------------------------------------
    def _poly_build_reflections_from_crystal(
        self,
        energy_eV: float,
        e_bw_pct: float,
        qmax: float,
    ):
        """
        Build (q_magnitudes, hkls_names) from CIF/lattice parameters using
        LatticeStructure.

        IMPORTANT: we de-duplicate by |q| so symmetry-equivalent hkls that
        produce the same powder ring do NOT get plotted multiple times in 2D/3D.
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
                a=a,
                b=b,
                c=c,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
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
                    abs(v[0]),
                    abs(v[1]),
                    abs(v[2]),
                    int(v[0]),
                    int(v[1]),
                    int(v[2]),
                ),
            )

            q_unique.append(q0)
            hkls_unique.append(best)
            i = j

        return np.array(q_unique, dtype=float), np.array(hkls_unique, dtype=int)

    # ------------------------------------------------------------------
    # Run logic
    # ------------------------------------------------------------------
    def _poly_run_function(self) -> bool:
        self._write_back_to_state()

        try:
            func = self.poly_func_combo.currentText()

            energy = float(self.poly_line_energy.text())
            e_bw = float(self.poly_line_ebw.text())
            qmax = float(self.poly_line_qmax.text())

            if func == "simulate_1d":
                cif_path = (self.poly_line_cif_path.text() or "").strip()
                if not cif_path:
                    raise ValueError(
                        "simulate_1d requires a CIF file path. Load or browse a CIF first."
                    )

                x_axis = self.poly_combo_xaxis.currentText()
                include_lorentz_polarization = self.poly_chk_lorpol.isChecked()

                fwhm_txt = (self.poly_line_fwhm.text() or "").strip()
                fwhm_val = float(fwhm_txt) if fwhm_txt else 0.0
                fwhm = None if fwhm_val <= 0.0 else fwhm_val

                try:
                    polycrystalline.simulate_1d(
                        cif_file_path=cif_path,
                        qmax=qmax,
                        x_axis=x_axis,
                        include_lorentz_polarization=include_lorentz_polarization,
                        include_multiplicity=False,
                        atom_positions=False,
                        energy=energy,
                        fwhm=fwhm,
                    )
                except TypeError:
                    polycrystalline.simulate_1d(
                        cif_file_path=cif_path,
                        qmax=qmax,
                        x_axis=x_axis,
                        include_lorentz_polarization=include_lorentz_polarization,
                        include_multiplicity=False,
                        atom_positions=False,
                        energy=energy,
                    )
                    if fwhm is not None:
                        QMessageBox.warning(
                            self,
                            "FWHM not applied",
                            "Your current polycrystalline.simulate_1d() does not accept an "
                            "'fwhm' argument.\nUpdate the simulation function to support "
                            "broadening, or expose it via plotting.",
                        )

                self.run_completed.emit(True, func)
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
                    qmax=qmax,
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
                    raise ValueError(
                        "Manual mode requires hkls_names (e.g. [1,0,2],[0,1,2])."
                    )
                if (q_hkls is None) and (d_hkls is None):
                    raise ValueError("Manual mode requires either q_hkls or d_hkls.")
                if q_hkls is not None and len(q_hkls) != len(hkls_names):
                    raise ValueError(
                        "Manual mode: q_hkls length must match hkls_names length."
                    )
                if d_hkls is not None and len(d_hkls) != len(hkls_names):
                    raise ValueError(
                        "Manual mode: d_hkls length must match hkls_names length."
                    )

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
                    hkls_names=hkls_names,
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
                    hkls_names=hkls_names,
                )

            self.run_completed.emit(True, func)
            return True

        except Exception as e:
            QMessageBox.critical(self, "Polycrystalline Error", str(e))
            self.run_completed.emit(False, self.poly_func_combo.currentText())
            return False

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _connect_stateful_widgets(self) -> None:
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

    def _on_widget_state_changed(self, *_args) -> None:
        if self._loading_state:
            return
        self._write_back_to_state()

    @staticmethod
    @contextmanager
    def _blocked(widget):
        was_blocked = widget.blockSignals(True)
        try:
            yield
        finally:
            widget.blockSignals(was_blocked)

    def _set_line_text(self, widget: QLineEdit, value: str | None) -> None:
        with self._blocked(widget):
            widget.setText("" if value is None else str(value))

    def _combo_set_text_if_present(self, combo: QComboBox, value: str | None) -> None:
        if value is None:
            return

        idx = combo.findText(str(value))
        with self._blocked(combo):
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif combo.isEditable():
                combo.setEditText(str(value))