"""Polycrystalline simulation controls and backend dispatch."""
from __future__ import annotations

from ast import literal_eval
from contextlib import contextmanager
from pathlib import Path

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
from ... import detector as detector_module
from ...poni import read_poni_file
from ..widgets.detector_export import update_detector_fields, save_current_detector
from ..services.background_simulation import simulation_request, start_simulation
from ...powder_preparation import run_powder, powder_reflections
from functools import partial
from ...cif import Cif
from ..services.path_service import SimulationPathService
from ..services.simulation_service import SimulationService
from ..state import GuiState
from ..widgets import DropFileLineEdit
from ..widgets.intensity_controls import IntensityControls


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
    """Parse a comma-separated numeric field, returning ``None`` when empty."""
    text = (text or "").strip()
    if not text:
        return None
    return np.array([float(x.strip()) for x in text.split(",")], dtype=float)


def _parse_azimuth_range(text: str):
    """Accept a pair entered as ``-15,15``, ``(-15, 15)``, or ``[-15, 15]``."""
    message = "Azimuth range must contain two angles, e.g. (-15, 15)."
    try:
        values = literal_eval(text)
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError(message)
        angles = tuple(float(value) for value in values)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.all(np.isfinite(angles)) or not 0 < angles[1] - angles[0] <= 360:
        raise ValueError("Azimuth range must increase and span at most 360 degrees.")
    return angles


class PolycrystallineTab(QWidget):
    """Configure and run 1D, 2D, and 3D powder simulations.

    The tab supports reflections derived from a CIF/lattice or explicit q/d
    values, manual/pyFAI/PONI detectors, state persistence, and conditional
    controls for the selected simulation function.
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
        """Build controls, load persisted values, and connect state signals."""
        super().__init__(parent)

        self.service = service
        self.state = state
        self.path_service = path_service or SimulationPathService()
        # Building the widgets emits change signals; preserve the supplied
        # session until load_from_state has applied it to the controls.
        self._loading_state = True

        self._build_ui()
        self._connect_stateful_widgets()
        self.load_from_state()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Construct crystal, beam, detector, reflection, and run controls."""
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
        self.poly_func_combo.setMinimumContentsLength(28)
        self.poly_func_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
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
        self.poly_line_cif_path = DropFileLineEdit(suffixes=(".cif",))
        self.poly_line_cif_path.fileDropped.connect(lambda _path: self._poly_load_cif())
        self.poly_line_cif_path.setPlaceholderText(
            "Optional for 2d/3d (if using manual q/d). Required for simulate_1d."
        )
        cif_hbox.addWidget(self.poly_line_cif_path)
        self.poly_btn_browse_cif = QPushButton("Browse")
        cif_hbox.addWidget(self.poly_btn_browse_cif)
        sam_layout.addLayout(cif_hbox, row, 1, 1, 5)
        row += 1

        sam_layout.addWidget(QLabel("Space Group (e.g. 167):"), row, 0)
        self.poly_line_space_group = QLineEdit("167")
        self.poly_line_space_group.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_space_group, row, 1)
        row += 1

        sam_layout.addWidget(QLabel("qmax [Å^-1]:"), row, 0)
        self.poly_line_qmax = QLineEdit("5")
        self.poly_line_qmax.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_qmax, row, 1)
        row += 1

        sam_layout.addWidget(QLabel("a [Å]:"), row, 0)
        self.poly_line_sam_a = QLineEdit("4.954")
        self.poly_line_sam_a.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_a, row, 1)
        sam_layout.addWidget(QLabel("b [Å]:"), row, 2)
        self.poly_line_sam_b = QLineEdit("4.954")
        self.poly_line_sam_b.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_b, row, 3)
        sam_layout.addWidget(QLabel("c [Å]:"), row, 4)
        self.poly_line_sam_c = QLineEdit("14.01")
        self.poly_line_sam_c.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_c, row, 5)
        row += 1

        sam_layout.addWidget(QLabel("alpha [°]:"), row, 0)
        self.poly_line_sam_alpha = QLineEdit("90")
        self.poly_line_sam_alpha.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_alpha, row, 1)
        sam_layout.addWidget(QLabel("beta [°]:"), row, 2)
        self.poly_line_sam_beta = QLineEdit("90")
        self.poly_line_sam_beta.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_beta, row, 3)
        sam_layout.addWidget(QLabel("gamma [°]:"), row, 4)
        self.poly_line_sam_gamma = QLineEdit("120")
        self.poly_line_sam_gamma.setValidator(QDoubleValidator())
        sam_layout.addWidget(self.poly_line_sam_gamma, row, 5)
        row += 1

        row += 1
        self.poly_btn_load_cif = QPushButton("Load CIF → fill lattice params")
        sam_layout.addWidget(self.poly_btn_load_cif, row, 0, 1, 6)

        self.poly_beam_group = QGroupBox("Beam Parameters")
        beam_layout = QGridLayout()
        self.poly_beam_group.setLayout(beam_layout)
        scroll_layout.addWidget(self.poly_beam_group)

        beam_layout.addWidget(QLabel("Energy [eV]:"), 0, 0)
        self.poly_line_energy = QLineEdit("15000")
        self.poly_line_energy.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.poly_line_energy, 0, 1)

        beam_layout.addWidget(QLabel("∆E/E [%]:"), 0, 2)
        self.poly_line_ebw = QLineEdit("1.5")
        self.poly_line_ebw.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.poly_line_ebw, 0, 3)

        beam_layout.addWidget(QLabel("Title digits:"), 0, 4)
        self.poly_line_title_digits = QLineEdit("2")
        self.poly_line_title_digits.setValidator(QDoubleValidator())
        beam_layout.addWidget(self.poly_line_title_digits, 0, 5)

        self.poly_det_group = QGroupBox("Detector Settings")
        det_layout = QGridLayout()
        self.poly_det_group.setLayout(det_layout)
        scroll_layout.addWidget(self.poly_det_group)

        row = 0
        det_layout.addWidget(QLabel("Detector Type:"), row, 0)
        self.poly_combo_det_type = QComboBox()
        detector_list = ["manual", "poni"] + list(pyFAI.detectors.ALL_DETECTORS.keys())
        self.poly_combo_det_type.addItems(detector_list)
        det_layout.addWidget(self.poly_combo_det_type, row, 1)
        row += 1

        det_layout.addWidget(QLabel("PONI file:"), row, 0)
        poni_hbox = QHBoxLayout()
        self.poly_line_poni_file = DropFileLineEdit("", suffixes=(".poni",))
        self.poly_line_poni_file.setPlaceholderText("Optional detector calibration file (*.poni)")
        poni_hbox.addWidget(self.poly_line_poni_file)
        self.poly_btn_browse_poni = QPushButton("Browse")
        poni_hbox.addWidget(self.poly_btn_browse_poni)
        det_layout.addLayout(poni_hbox, row, 1)
        row += 1

        self.poly_manual_group = QGroupBox("Detector Parameters")
        self.poly_manual_group.setFlat(True)
        manual_layout = QGridLayout()
        self.poly_manual_group.setLayout(manual_layout)
        det_layout.addWidget(self.poly_manual_group, row, 0, 1, 2)
        row += 1

        manual_layout.addWidget(QLabel("Pixel Size H [m]:"), 0, 0)
        self.poly_line_pxsize_h = QLineEdit("50e-6")
        self.poly_line_pxsize_h.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.poly_line_pxsize_h, 0, 1)

        manual_layout.addWidget(QLabel("Pixel Size V [m]:"), 0, 2)
        self.poly_line_pxsize_v = QLineEdit("50e-6")
        self.poly_line_pxsize_v.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.poly_line_pxsize_v, 0, 3)

        manual_layout.addWidget(QLabel("Number of Pixels H:"), 1, 0)
        self.poly_line_num_px_h = QLineEdit("2000")
        self.poly_line_num_px_h.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.poly_line_num_px_h, 1, 1)

        manual_layout.addWidget(QLabel("Number of Pixels V:"), 1, 2)
        self.poly_line_num_px_v = QLineEdit("2000")
        self.poly_line_num_px_v.setValidator(QDoubleValidator())
        manual_layout.addWidget(self.poly_line_num_px_v, 1, 3)

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

        det_common_layout.addWidget(QLabel("PONI1 vertical [m]:"), 1, 2)
        self.poly_line_poni1 = QLineEdit("0")
        self.poly_line_poni1.setValidator(QDoubleValidator())
        self.poly_line_poni1.setToolTip(
            "Detector axis 1 coordinate, corresponding to the vertical image direction."
        )
        det_common_layout.addWidget(self.poly_line_poni1, 1, 3)

        det_common_layout.addWidget(QLabel("PONI2 horizontal [m]:"), 1, 4)
        self.poly_line_poni2 = QLineEdit("0")
        self.poly_line_poni2.setValidator(QDoubleValidator())
        self.poly_line_poni2.setToolTip(
            "Detector axis 2 coordinate, corresponding to the horizontal image direction."
        )
        det_common_layout.addWidget(self.poly_line_poni2, 1, 5)

        det_common_layout.addWidget(QLabel("Detector Rotations [°]:"), 2, 0)
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
        det_common_layout.addLayout(rots_hbox, 2, 1, 1, 5)

        det_common_layout.addWidget(QLabel("Detector rotation order:"), 3, 0)
        self.poly_combo_rotation_order = QComboBox()
        self.poly_combo_rotation_order.addItems(["zyx", "zxy", "yzx", "yxz", "xzy", "xyz"])
        self.poly_combo_rotation_order.setMinimumWidth(120)
        det_common_layout.addWidget(self.poly_combo_rotation_order, 3, 1)
        self.poly_save_poni_button = QPushButton("Save PONI...")
        self.poly_save_poni_button.clicked.connect(lambda: save_current_detector(self, "poly"))
        det_common_layout.addWidget(self.poly_save_poni_button, 4, 0, 1, 6)

        self.poly_refsrc_group = QGroupBox("Reflections Source (for 2D/3D)")
        refsrc_layout = QHBoxLayout()
        self.poly_refsrc_group.setLayout(refsrc_layout)
        refsrc_layout.addWidget(QLabel("Use reflections from:"))
        self.poly_combo_refsrc = QComboBox()
        self.poly_combo_refsrc.addItems([
            "CIF / lattice (auto from qmax)",
            "Manual q/d + hkls_names",
        ])
        self.poly_combo_refsrc.setMinimumContentsLength(30)
        self.poly_combo_refsrc.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
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

        self.poly_1d_group = QGroupBox("1D Pattern Options (simulate_1d)")
        one_d_layout = QGridLayout()
        self.poly_1d_group.setLayout(one_d_layout)
        scroll_layout.addWidget(self.poly_1d_group)

        one_d_layout.addWidget(QLabel("x_axis:"), 0, 0)
        self.poly_combo_xaxis = QComboBox()
        self.poly_combo_xaxis.addItems(["q", "two_theta", "d"])
        one_d_layout.addWidget(self.poly_combo_xaxis, 0, 1)

        self.poly_chk_lorpol = QCheckBox("include_lorentz_polarization")
        self.poly_chk_lorpol.setChecked(True)
        one_d_layout.addWidget(self.poly_chk_lorpol, 1, 0, 1, 2)

        one_d_layout.addWidget(QLabel("Peak FWHM:"), 2, 0)
        self.poly_line_fwhm = QLineEdit("0.0")
        self.poly_line_fwhm.setValidator(QDoubleValidator())
        self.poly_line_fwhm.setToolTip(
            "Peak broadening FWHM.\n"
            "Units: Å^-1 for q, degrees for two_theta, Å for d.\n"
            "Use 0 for a stick pattern (no broadening)."
        )
        one_d_layout.addWidget(self.poly_line_fwhm, 2, 1)

        self.poly_chk_use_detector_1d = QCheckBox("Use detector coverage")
        one_d_layout.addWidget(self.poly_chk_use_detector_1d, 3, 0, 1, 2)
        self.poly_chk_use_detector_1d.toggled.connect(self._poly_func_changed)

        one_d_layout.addWidget(QLabel("Azimuth range [°]:"), 4, 0)
        self.poly_line_azimuth_range_1d = QLineEdit("")
        self.poly_line_azimuth_range_1d.setPlaceholderText("Full 360° (or (-15, 15))")
        self.poly_line_azimuth_range_1d.setToolTip(
            "Lab frame: 0° up, -90° left, +90° right, looking along the beam. "
            "Leave blank for the full azimuth."
        )
        one_d_layout.addWidget(self.poly_line_azimuth_range_1d, 4, 1)
        one_d_layout.addWidget(QLabel("Azimuth offset [°]:"), 5, 0)
        self.poly_line_azimuth_offset_1d = QLineEdit("0")
        self.poly_line_azimuth_offset_1d.setValidator(QDoubleValidator())
        one_d_layout.addWidget(self.poly_line_azimuth_offset_1d, 5, 1)

        self.poly_intensity_controls = IntensityControls()
        scroll_layout.addWidget(self.poly_intensity_controls)
        self.poly_run_btn = QPushButton("Run Polycrystal Function")
        scroll_layout.addWidget(self.poly_run_btn)

        self.poly_btn_browse_cif.clicked.connect(self._poly_browse_cif)
        self.poly_btn_browse_poni.clicked.connect(self._poly_browse_poni)
        self.poly_line_poni_file.editingFinished.connect(self._poly_apply_poni_path_from_line)
        self.poly_btn_load_cif.clicked.connect(self._poly_load_cif)
        self.poly_combo_det_type.currentIndexChanged.connect(self._poly_detector_changed)
        self.poly_combo_rotation_order.currentIndexChanged.connect(self._write_back_to_state)
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
        """Apply the persisted polycrystalline state to all controls."""
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
            self._set_line_text(
                self.poly_line_title_digits,
                getattr(poly, "title_digits", "2"),
            )
            self._set_line_text(self.poly_line_poni_file, getattr(poly, "poni_file", ""))
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
            self._combo_set_text_if_present(
                self.poly_combo_rotation_order,
                getattr(poly, "rotation_order", "zyx"),
            )
            self._set_line_text(self.poly_line_qhkls, poly.q_hkls)
            self._set_line_text(self.poly_line_dhkls, poly.d_hkls)
            self._set_line_text(self.poly_line_hkls, poly.hkls)
            self._set_line_text(self.poly_line_fwhm, poly.fwhm)
            self.poly_chk_use_detector_1d.setChecked(bool(poly.use_detector_1d))
            self._set_line_text(self.poly_line_azimuth_range_1d, poly.azimuthal_range_1d)
            self._set_line_text(self.poly_line_azimuth_offset_1d, poly.azimuth_offset_1d)

            self._combo_set_text_if_present(self.poly_func_combo, poly.func)
            self._combo_set_text_if_present(self.poly_combo_det_type, poly.det_type)
            self._combo_set_text_if_present(self.poly_combo_refsrc, poly.ref_source)
            self._combo_set_text_if_present(self.poly_combo_xaxis, poly.x_axis)
            self.poly_chk_lorpol.setChecked(bool(poly.lorpol))
            self.poly_intensity_controls.load(poly)

            self._poly_detector_changed()
            self._poly_refsrc_changed()
            self._poly_func_changed()
        finally:
            self._loading_state = False
            self._write_back_to_state()

    def save_to_state(self) -> None:
        """Copy current controls into the shared GUI state."""
        self._write_back_to_state()

    def _write_back_to_state(self) -> None:
        """Serialize every polycrystalline control into ``state.poly``."""
        if self._loading_state:
            return
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
        self.state.poly.title_digits = self.poly_line_title_digits.text()
        self.state.poly.det_type = self.poly_combo_det_type.currentText()
        self.state.poly.poni_file = self.poly_line_poni_file.text()
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
        self.state.poly.rotation_order = self.poly_combo_rotation_order.currentText()
        self.state.poly.ref_source = self.poly_combo_refsrc.currentText()
        self.state.poly.q_hkls = self.poly_line_qhkls.text()
        self.state.poly.d_hkls = self.poly_line_dhkls.text()
        self.state.poly.hkls = self.poly_line_hkls.text()
        self.state.poly.x_axis = self.poly_combo_xaxis.currentText()
        self.state.poly.lorpol = self.poly_chk_lorpol.isChecked()
        self.state.poly.fwhm = self.poly_line_fwhm.text()
        self.state.poly.use_detector_1d = self.poly_chk_use_detector_1d.isChecked()
        self.state.poly.azimuthal_range_1d = self.poly_line_azimuth_range_1d.text()
        self.state.poly.azimuth_offset_1d = self.poly_line_azimuth_offset_1d.text()
        self.poly_intensity_controls.save(self.state.poly)
        self.state.paths.poly_cif_file_path = self.poly_line_cif_path.text() or None

        if not self._loading_state:
            self.state_changed.emit()

    # ------------------------------------------------------------------
    # CIF helpers
    # ------------------------------------------------------------------
    def _poly_browse_cif(self) -> None:
        """Select a CIF path for later loading and simulation."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open CIF File",
            directory=str(
                self.path_service.dialog_start_path(
                    current=self.poly_line_cif_path.text()
                )
            ),
            filter="CIF Files (*.cif);;All Files (*)",
        )
        if file_name:
            self.path_service.remember_dialog_selection(file_name)
            self.poly_line_cif_path.setText(file_name)
            self.state.paths.poly_cif_file_path = file_name
            self._write_back_to_state()

    def _format_poni_value(self, value) -> str:
        """Format a parsed PONI scalar compactly for a line edit."""
        return "" if value is None else f"{value:.12g}"

    def _poly_apply_poni_file(self, file_name: str) -> None:
        """Parse a PONI file and populate detector geometry controls."""
        poni = read_poni_file(file_name)
        detector_kwargs = poni.detector_kwargs(include_rotations=True)

        self.poly_line_poni_file.setText(file_name)

        idx = self.poly_combo_det_type.findText("poni")
        if idx >= 0:
            self.poly_combo_det_type.setCurrentIndex(idx)

        self._set_line_text(self.poly_line_pxsize_h, self._format_poni_value(detector_kwargs["pxsize_h"]))
        self._set_line_text(self.poly_line_pxsize_v, self._format_poni_value(detector_kwargs["pxsize_v"]))
        self._set_line_text(self.poly_line_num_px_h, str(detector_kwargs["num_pixels_h"]))
        self._set_line_text(self.poly_line_num_px_v, str(detector_kwargs["num_pixels_v"]))
        self._set_line_text(self.poly_line_dist, self._format_poni_value(detector_kwargs["dist"]))
        self._set_line_text(self.poly_line_poni1, self._format_poni_value(detector_kwargs["poni1"]))
        self._set_line_text(self.poly_line_poni2, self._format_poni_value(detector_kwargs["poni2"]))
        self._set_line_text(self.poly_line_rotx, self._format_poni_value(detector_kwargs["rotx"]))
        self._set_line_text(self.poly_line_roty, self._format_poni_value(detector_kwargs["roty"]))
        self._set_line_text(self.poly_line_rotz, self._format_poni_value(detector_kwargs["rotz"]))
        self._combo_set_text_if_present(self.poly_combo_rotation_order, "zyx")

        self._write_back_to_state()

    def _poly_browse_poni(self) -> None:
        """Select and apply a PONI detector calibration file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open PONI File",
            directory=str(
                self.path_service.dialog_start_path(
                    current=self.poly_line_poni_file.text()
                )
            ),
            filter="PONI Files (*.poni);;All Files (*)",
        )
        if file_name:
            self.path_service.remember_dialog_selection(file_name)
            self._poly_apply_poni_file(file_name)

    def _poly_apply_poni_path_from_line(self) -> None:
        """Apply a typed or dropped PONI path without forcing a file dialog."""
        file_name = (self.poly_line_poni_file.text() or "").strip()
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
            self._poly_apply_poni_file(resolved)
        except Exception as e:
            QMessageBox.critical(
                self,
                "PONI Error",
                f"Failed to read or parse the PONI file:\n{str(e)}",
            )

    def _poly_load_cif(self) -> None:
        """Load the CIF currently entered in the path control."""
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
        """Copy unit-cell and space-group data from a CIF into the form."""
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
        """Update detector field availability for the selected source type."""
        det_type = self.poly_combo_det_type.currentText().lower()
        update_detector_fields(self, "poly")
        self._write_back_to_state()

    def _poly_refsrc_changed(self) -> None:
        """Toggle automatic versus explicit reflection input controls."""
        manual = self.poly_combo_refsrc.currentText().startswith("Manual")
        self.poly_refl_group.setVisible(manual)
        self._write_back_to_state()

    def _poly_func_changed(self) -> None:
        """Show controls relevant to the selected powder simulation function."""
        func = self.poly_func_combo.currentText()

        is_1d = func == "simulate_1d"
        is_cones = func in ("simulate_2d", "simulate_3d")

        self.poly_det_group.setVisible(is_cones or (is_1d and self.poly_chk_use_detector_1d.isChecked()))
        self.poly_line_qmax.setEnabled(not (is_1d and self.poly_chk_use_detector_1d.isChecked()))
        self.poly_line_azimuth_range_1d.setEnabled(self.poly_chk_use_detector_1d.isChecked())
        self.poly_line_azimuth_offset_1d.setEnabled(self.poly_chk_use_detector_1d.isChecked())
        self.poly_refsrc_group.setVisible(is_cones)

        if is_cones:
            self._poly_refsrc_changed()
        else:
            self.poly_refl_group.setVisible(False)

        self.poly_1d_group.setVisible(is_1d)
        self.poly_intensity_controls.setVisible(is_cones)
        self._write_back_to_state()

    # ------------------------------------------------------------------
    # Reflection construction
    # ------------------------------------------------------------------
    def _powder_lattice_kwargs(self):
        path = self.poly_line_cif_path.text().strip()
        if path:
            return {'cif_file_path': path}
        return {'space_group': int(float(self.poly_line_space_group.text())),
                **{name: float(getattr(self, 'poly_line_sam_' + name).text())
                   for name in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')}}

    def _poly_build_reflections_from_crystal(self, energy_eV, e_bw_pct, qmax):
        """Return powder families using the same preparation as background runs."""
        return powder_reflections(self._powder_lattice_kwargs(), energy_eV, e_bw_pct, qmax)

    # ------------------------------------------------------------------
    # Run logic
    # ------------------------------------------------------------------
    def _poly_run_function(self) -> bool:
        """Validate inputs, dispatch the selected backend call, and report status."""
        if getattr(self, '_simulation_busy', False):
            return False
        self._write_back_to_state()

        try:
            func = self.poly_func_combo.currentText()

            energy = float(self.poly_line_energy.text())
            e_bw = float(self.poly_line_ebw.text())
            use_detector_1d = (func == "simulate_1d" and
                               self.poly_chk_use_detector_1d.isChecked())
            qmax = None if use_detector_1d else float(self.poly_line_qmax.text())
            title_digits = self._parse_title_digits(self.poly_line_title_digits.text())

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

                detector_obj = None
                azimuthal_range = None
                azimuth_offset = 0.0
                if use_detector_1d:
                    detector_obj = self._detector_from_fields()
                    range_text = self.poly_line_azimuth_range_1d.text().strip()
                    if range_text:
                        azimuthal_range = _parse_azimuth_range(range_text)
                    azimuth_offset = float(self.poly_line_azimuth_offset_1d.text())

                return self._start_powder(polycrystalline.simulate_1d,
                        cif_file_path=cif_path,
                        qmax=qmax,
                        x_axis=x_axis,
                        include_lorentz_polarization=include_lorentz_polarization,
                        include_multiplicity=False,
                        atom_positions=False,
                        energy=energy,
                        fwhm=fwhm,
                        title_digits=title_digits,
                        detector=detector_obj,
                        azimuthal_range=azimuthal_range,
                        azimuth_offset_deg=azimuth_offset,
                    )

            det_type = self.poly_combo_det_type.currentText()
            poni_file = (self.poly_line_poni_file.text() or "").strip() or None
            backend_det_type, backend_poni_file = self._backend_detector_source(
                det_type,
                poni_file,
            )

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
            rotation_order = self.poly_combo_rotation_order.currentText()
            if self.poly_combo_refsrc.currentText().startswith("CIF / lattice"):
                q_hkls = hkls_names = d_hkls = None
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
                intensity_options = self.poly_intensity_controls.options(
                    self.poly_line_cif_path.text().strip())
                return self._start_powder(polycrystalline.simulate_2d,
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
                    det_rotation_order=rotation_order,
                    energy=energy,
                    e_bandwidth=e_bw,
                    q_hkls=q_hkls,
                    d_hkls=d_hkls,
                    hkls_names=hkls_names,
                    title_digits=title_digits,
                    qmax=qmax,
                    **intensity_options,
                )
            elif func == "simulate_3d":
                intensity_options = self.poly_intensity_controls.options(
                    self.poly_line_cif_path.text().strip())
                return self._start_powder(polycrystalline.simulate_3d,
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
                    det_rotation_order=rotation_order,
                    energy=energy,
                    e_bandwidth=e_bw,
                    q_hkls=q_hkls,
                    d_hkls=d_hkls,
                    hkls_names=hkls_names,
                    title_digits=title_digits,
                    qmax=qmax,
                    **intensity_options,
                )


        except Exception as e:
            QMessageBox.critical(self, "Polycrystalline Error", str(e))
            self.run_completed.emit(False, self.poly_func_combo.currentText())
            return False

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _detector_from_fields(self):
        """Build the same editable detector geometry used by the cone views."""
        det_type, poni_file = self._backend_detector_source(
            self.poly_combo_det_type.currentText(),
            self.poly_line_poni_file.text().strip() or None,
        )
        return detector_module.Detector(
            detector_type=det_type,
            pxsize_h=float(self.poly_line_pxsize_h.text()),
            pxsize_v=float(self.poly_line_pxsize_v.text()),
            num_pixels_h=int(float(self.poly_line_num_px_h.text())),
            num_pixels_v=int(float(self.poly_line_num_px_v.text())),
            binning=(int(float(self.poly_line_bin_h.text())),
                     int(float(self.poly_line_bin_v.text()))),
            poni_file=poni_file,
            dist=float(self.poly_line_dist.text()),
            poni1=float(self.poly_line_poni1.text()),
            poni2=float(self.poly_line_poni2.text()),
            rotx=float(self.poly_line_rotx.text()),
            roty=float(self.poly_line_roty.text()),
            rotz=float(self.poly_line_rotz.text()),
            rotation_order=self.poly_combo_rotation_order.currentText(),
        )

    def _start_powder(self, function, **kwargs):
        if self.poly_func_combo.currentText() != 'simulate_1d' and self.poly_combo_refsrc.currentText().startswith('CIF / lattice'):
            function = partial(run_powder, function, self._powder_lattice_kwargs())
        return start_simulation(self, simulation_request(function, **kwargs), self.poly_run_btn,
            metadata={'function': self.poly_func_combo.currentText(),
                      'gui_state': self.state.to_dict(include_log=False)})

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

        The GUI uses a PONI file to populate editable detector fields. Once
        populated, the edited fields are authoritative, so the backend should
        not re-read the file and overwrite manual changes.
        """
        if str(det_type).lower() == "poni":
            return "poni", None
        return "manual", None
