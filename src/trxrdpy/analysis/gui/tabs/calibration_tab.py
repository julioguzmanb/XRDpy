"""
Calibration tab for the analysis GUI.

This reproduces the legacy Calibration tab layout while keeping backend actions
separated from the main window.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QProcess
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
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from trxrdpy.analysis.gui.services import CalibrationService, PathService
from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.utils import (
    parse_edges,
    parse_float_like,
    parse_int_like,
    parse_optional_tuple2,
    parse_python_literal,
    parse_scan_spec,
    parse_tuple2,
)
from trxrdpy.analysis.gui.widgets import (
    CalibrationContextWidget,
    DropPathLineEdit,
    PolarizationControlWidget,
)
from trxrdpy.analysis.gui.widgets.task_output_dialog import run_task_with_output_dialog


DEFAULT_CALIBRATION_AZIMUTHAL_EDGES = [-75, -45, -15, 15, 45, 75]
DEFAULT_CALIBRATION_FIGURES_SUBDIR = "figures/calibration/"


def pretty_literal(value):
    """Return an editable Python-literal representation of a value."""
    return repr(value)


class CalibrationTab(QWidget):
    """Configure calibration integration, peak fitting, and diagnostic plots.

    Attributes
    ----------
    state : AnalysisGuiState
        Shared facility, path, geometry, and polarization configuration.
    path_service : PathService
        Builds ``AnalysisPaths`` and supplies file-dialog starting locations.
    calibration_service : CalibrationService
        Stateless adapter for the public calibration backend.
    calibration_context : CalibrationContextWidget
        Sample, temperature, and scan selector shared by all tab actions.
    calib_azimuthal_edges, calib_full_range, calib_npt : QLineEdit
        One-dimensional integration binning controls.
    calib_polarization_control : PolarizationControlWidget
        Optional pyFAI polarization-correction control.
    calib_q_fit_range, calib_eta, calib_fit_method : QLineEdit
        Peak-model range and optimizer settings.
    calib_detector_cake_use_mask : QCheckBox
        Whether the EDF detector mask is applied to the 2D cake.
    calib_detector_cake_invert_x, calib_detector_cake_invert_y : QCheckBox
        Display-only detector-axis flips; integration arrays are unchanged.
    external_processes : list
        Live ``pyFAI-calib2`` processes retained to prevent premature cleanup.
    log : callable
        Callback receiving user-facing status messages.
    """

    def __init__(
        self,
        state: AnalysisGuiState,
        path_service: PathService,
        calibration_service: CalibrationService,
        log: Optional[Callable[[str], None]] = None,
        polarization_changed_callback: Optional[Callable[[bool, float], None]] = None,
        parent=None,
    ):
        """Initialize ``CalibrationTab``, bind shared state and services, and create its controls."""
        super().__init__(parent)

        self.state = state
        self.path_service = path_service
        self.calibration_service = calibration_service
        self.external_processes = []
        self.log = log or (lambda message: None)
        self.polarization_changed_callback = polarization_changed_callback

        layout = self._make_scroll_layout()

        note = QLabel(
            "Calibration is facility agnostic at this stage.\n"
            "It assumes the homogeneous dark 2D image already exists in the analysis structure."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.calibration_context = CalibrationContextWidget(
            title="Calibration Context",
            defaults=dict(sample_name="DET70", temperature_K=110, scan_spec="7"),
        )
        layout.addWidget(self.calibration_context)

        self._init_pyfai_group(layout)
        self._init_integration_group(layout)
        self._init_peak_fitting_group(layout)
        self._init_caked_plot_group(layout)
        self._init_detector_cake_group(layout)
        self._init_cake_azimuthal_distribution_group(layout)
        self._init_property_plot_group(layout)
        self._init_save_group(layout)

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

    def _init_pyfai_group(self, layout: QVBoxLayout):
        """Create and connect the controls for pyfai group."""
        pyfai_group = QGroupBox("pyFAI Calibration GUI")
        pgui = QGridLayout()
        pyfai_group.setLayout(pgui)
        layout.addWidget(pyfai_group)

        pgui.addWidget(QLabel("2D image path (optional):"), 0, 0)

        pyfai_path_box = QHBoxLayout()
        self.calib_pyfai_image_path = DropPathLineEdit("", mode="file")
        self.calib_pyfai_image_path.setPlaceholderText(
            "Optional. Leave empty to open pyFAI-calib2 without a file."
        )
        pyfai_path_box.addWidget(self.calib_pyfai_image_path)

        pyfai_browse_btn = QPushButton("Browse")
        pyfai_browse_btn.clicked.connect(
            lambda: self._browse_file_into(
                self.calib_pyfai_image_path,
                caption="Select 2D image",
                file_filter="Images / Data Files (*.edf *.tif *.tiff *.cbf *.npy *.h5 *.hdf5);;All Files (*)",
            )
        )
        pyfai_path_box.addWidget(pyfai_browse_btn)

        pgui.addLayout(pyfai_path_box, 0, 1)

        pyfai_info = QLabel(
            "Launch pyFAI-calib2 as an external GUI. "
            "Your XRDpy analysis GUI remains usable while the pyFAI window is open."
        )
        pyfai_info.setWordWrap(True)
        pgui.addWidget(pyfai_info, 1, 0, 1, 2)

        self.calib_launch_pyfai_btn = QPushButton("Open pyFAI-calib2")
        self.calib_launch_pyfai_btn.clicked.connect(self._launch_pyfai_calib2)
        pgui.addWidget(self.calib_launch_pyfai_btn, 2, 0, 1, 2)

    def _init_integration_group(self, layout: QVBoxLayout):
        """Create and connect the controls for integration group."""
        integration_group = QGroupBox("Azimuthal Integration Settings")
        ig = QGridLayout()
        integration_group.setLayout(ig)
        layout.addWidget(integration_group)

        ig.addWidget(QLabel("Azimuthal edges [deg]:"), 0, 0)
        self.calib_azimuthal_edges = QLineEdit(
            pretty_literal(DEFAULT_CALIBRATION_AZIMUTHAL_EDGES)
        )
        ig.addWidget(self.calib_azimuthal_edges, 0, 1)

        self.calib_include_full = QCheckBox("include_full")
        self.calib_include_full.setChecked(True)
        ig.addWidget(self.calib_include_full, 1, 0, 1, 2)

        ig.addWidget(QLabel("Full azimuthal range [deg]:"), 2, 0)
        self.calib_full_range = QLineEdit("(-90, 90)")
        ig.addWidget(self.calib_full_range, 2, 1)

        ig.addWidget(QLabel("Number of q points:"), 3, 0)
        self.calib_npt = QLineEdit("1000")
        self.calib_npt.setValidator(QDoubleValidator())
        ig.addWidget(self.calib_npt, 3, 1)

        self.calib_normalize = QCheckBox("normalize")
        self.calib_normalize.setChecked(True)
        ig.addWidget(self.calib_normalize, 4, 0, 1, 2)

        ig.addWidget(QLabel("Q normalization range:"), 5, 0)
        self.calib_q_norm_range = QLineEdit("(2.65, 2.75)")
        ig.addWidget(self.calib_q_norm_range, 5, 1)

        self.calib_overwrite_xy = QCheckBox("overwrite_xy")
        self.calib_overwrite_xy.setChecked(False)
        ig.addWidget(self.calib_overwrite_xy, 6, 0, 1, 2)

        self.calib_polarization_control = PolarizationControlWidget(
            enabled=getattr(self.state, "polarization_enabled", True),
            factor=(
                0.99
                if getattr(self.state, "polarization_factor", 0.99) is None
                else getattr(self.state, "polarization_factor", 0.99)
            ),
        )
        self.calib_polarization_control.valueChanged.connect(
            self._on_polarization_changed
        )
        ig.addWidget(self.calib_polarization_control, 7, 0, 1, 2)

        self.calib_compute_xy_btn = QPushButton("Compute XY Files")
        self.calib_compute_xy_btn.clicked.connect(self._run_calibration_compute_xy)
        ig.addWidget(self.calib_compute_xy_btn, 8, 0, 1, 2)

    def _init_peak_fitting_group(self, layout: QVBoxLayout):
        """Create and connect the controls for peak fitting group."""
        fit_group = QGroupBox("Peak Fitting Settings")
        fg = QGridLayout()
        fit_group.setLayout(fg)
        layout.addWidget(fit_group)

        fg.addWidget(QLabel("Q fit range:"), 0, 0)
        self.calib_q_fit_range = QLineEdit("(2.4, 2.65)")
        fg.addWidget(self.calib_q_fit_range, 0, 1)

        fg.addWidget(QLabel("eta:"), 0, 2)
        self.calib_eta = QLineEdit("0.3")
        self.calib_eta.setValidator(QDoubleValidator())
        fg.addWidget(self.calib_eta, 0, 3)

        fg.addWidget(QLabel("fit_method:"), 1, 0)
        self.calib_fit_method = QComboBox()
        self.calib_fit_method.addItems(["leastsq", "least_squares", "nelder", "powell", "lbfgsb"])
        self.calib_fit_method.setEditable(True)
        fg.addWidget(self.calib_fit_method, 1, 1)

        self.calib_eta_vary = QCheckBox("refine eta")
        self.calib_eta_vary.setChecked(False)
        fg.addWidget(self.calib_eta_vary, 1, 2)

        self.calib_force_refit = QCheckBox("force_refit")
        self.calib_force_refit.setChecked(True)
        fg.addWidget(self.calib_force_refit, 1, 3)

        fg.addWidget(QLabel("out_csv_name:"), 2, 0)
        self.calib_out_csv_name = QLineEdit("peak_fits.csv")
        fg.addWidget(self.calib_out_csv_name, 2, 1, 1, 3)

        self.calib_peak_fitting_btn = QPushButton("Run Peak Fitting")
        self.calib_peak_fitting_btn.clicked.connect(self._run_calibration_peak_fitting)
        fg.addWidget(self.calib_peak_fitting_btn, 3, 0, 1, 4)

    def _init_caked_plot_group(self, layout: QVBoxLayout):
        """Create and connect the controls for caked plot group."""
        caked_group = QGroupBox("Caked 1D Pattern Plot")
        cg = QGridLayout()
        caked_group.setLayout(cg)
        layout.addWidget(caked_group)

        cg.addWidget(QLabel("xlim:"), 0, 0)
        self.calib_caked_xlim = QLineEdit("(2.45, 2.60)")
        cg.addWidget(self.calib_caked_xlim, 0, 1)

        cg.addWidget(QLabel("ylim:"), 0, 2)
        self.calib_caked_ylim = QLineEdit("")
        self.calib_caked_ylim.setPlaceholderText("Optional")
        cg.addWidget(self.calib_caked_ylim, 0, 3)

        cg.addWidget(QLabel("figure_title:"), 1, 0)
        self.calib_caked_figure_title = QLineEdit("")
        self.calib_caked_figure_title.setPlaceholderText("Optional")
        cg.addWidget(self.calib_caked_figure_title, 1, 1, 1, 2)

        self.calib_caked_save = QCheckBox("save")
        self.calib_caked_save.setChecked(True)
        cg.addWidget(self.calib_caked_save, 1, 3)

        self.calib_plot_caked_btn = QPushButton("Plot Caked 1D Patterns")
        self.calib_plot_caked_btn.clicked.connect(self._run_calibration_plot_caked)
        cg.addWidget(self.calib_plot_caked_btn, 2, 0, 1, 4)

    def _init_detector_cake_group(self, layout: QVBoxLayout):
        """Create controls for the side-by-side detector and 2D cake plot."""
        cake_group = QGroupBox("Detector and 2D Cake Plot")
        cg = QGridLayout()
        cake_group.setLayout(cg)
        layout.addWidget(cake_group)

        cg.addWidget(QLabel("Number of azimuth points:"), 0, 0)
        self.calib_detector_cake_npt_azim = QLineEdit("360")
        self.calib_detector_cake_npt_azim.setValidator(QDoubleValidator())
        cg.addWidget(self.calib_detector_cake_npt_azim, 0, 1)

        cg.addWidget(QLabel("Q range:"), 0, 2)
        self.calib_detector_cake_radial_range = QLineEdit("")
        self.calib_detector_cake_radial_range.setPlaceholderText("Optional")
        cg.addWidget(self.calib_detector_cake_radial_range, 0, 3)

        cg.addWidget(QLabel("Detector color limits:"), 1, 0)
        self.calib_detector_cake_detector_clim = QLineEdit("")
        self.calib_detector_cake_detector_clim.setPlaceholderText("Optional")
        cg.addWidget(self.calib_detector_cake_detector_clim, 1, 1)

        cg.addWidget(QLabel("Cake color limits:"), 1, 2)
        self.calib_detector_cake_cake_clim = QLineEdit("")
        self.calib_detector_cake_cake_clim.setPlaceholderText("Optional")
        cg.addWidget(self.calib_detector_cake_cake_clim, 1, 3)

        self.calib_detector_cake_use_mask = QCheckBox("Use detector mask")
        self.calib_detector_cake_use_mask.setChecked(True)
        cg.addWidget(self.calib_detector_cake_use_mask, 4, 0, 1, 2)

        self.calib_detector_cake_normalize = QCheckBox("Normalize cake rows")
        self.calib_detector_cake_normalize.setChecked(False)
        cg.addWidget(self.calib_detector_cake_normalize, 4, 2, 1, 2)

        self.calib_detector_cake_invert_x = QCheckBox("Flip detector X axis")
        self.calib_detector_cake_invert_x.setChecked(False)
        cg.addWidget(self.calib_detector_cake_invert_x, 5, 0)

        self.calib_detector_cake_invert_y = QCheckBox("Flip detector Y axis")
        self.calib_detector_cake_invert_y.setChecked(False)
        cg.addWidget(self.calib_detector_cake_invert_y, 5, 1)

        self.calib_detector_cake_detector_log = QCheckBox("Detector log scale")
        self.calib_detector_cake_detector_log.setChecked(False)
        cg.addWidget(self.calib_detector_cake_detector_log, 6, 0)

        self.calib_detector_cake_cake_log = QCheckBox("Cake log scale")
        self.calib_detector_cake_cake_log.setChecked(False)
        cg.addWidget(self.calib_detector_cake_cake_log, 6, 1)

        cg.addWidget(QLabel("figure_title:"), 7, 0)
        self.calib_detector_cake_figure_title = QLineEdit("")
        self.calib_detector_cake_figure_title.setPlaceholderText("Optional")
        cg.addWidget(self.calib_detector_cake_figure_title, 7, 1, 1, 2)

        self.calib_detector_cake_save = QCheckBox("save")
        self.calib_detector_cake_save.setChecked(True)
        cg.addWidget(self.calib_detector_cake_save, 7, 3)

        self.calib_plot_detector_cake_btn = QPushButton("Plot Detector + 2D Cake")
        self.calib_plot_detector_cake_btn.clicked.connect(
            self._run_calibration_plot_detector_cake
        )
        cg.addWidget(self.calib_plot_detector_cake_btn, 8, 0, 1, 4)

    def _init_cake_azimuthal_distribution_group(self, layout: QVBoxLayout):
        """Create controls for q-band azimuthal distribution analysis."""
        dist_group = QGroupBox("Cake Azimuthal Distribution")
        dg = QGridLayout()
        dist_group.setLayout(dg)
        layout.addWidget(dist_group)

        dg.addWidget(QLabel("Peak q center:"), 0, 0)
        self.calib_cake_dist_q_value = QLineEdit("2.55")
        self.calib_cake_dist_q_value.setValidator(QDoubleValidator())
        dg.addWidget(self.calib_cake_dist_q_value, 0, 1)

        dg.addWidget(QLabel("Peak q full width:"), 0, 2)
        self.calib_cake_dist_q_width = QLineEdit("0.04")
        self.calib_cake_dist_q_width.setValidator(QDoubleValidator())
        dg.addWidget(self.calib_cake_dist_q_width, 0, 3)

        dg.addWidget(QLabel("Background mode:"), 1, 0)
        self.calib_cake_dist_bg_mode = QComboBox()
        self.calib_cake_dist_bg_mode.addItems(["left", "right", "average", "manual"])
        dg.addWidget(self.calib_cake_dist_bg_mode, 1, 1)

        dg.addWidget(QLabel("Manual bg q range:"), 1, 2)
        self.calib_cake_dist_bg_q_range = QLineEdit("")
        self.calib_cake_dist_bg_q_range.setPlaceholderText("Optional, e.g. (2.70, 2.74)")
        dg.addWidget(self.calib_cake_dist_bg_q_range, 1, 3)

        dg.addWidget(QLabel("Phi windows:"), 2, 0)
        self.calib_cake_dist_phi_windows = QLineEdit("[(40, 10)]")
        self.calib_cake_dist_phi_windows.setPlaceholderText("[(center_deg, half_width_deg)]")
        dg.addWidget(self.calib_cake_dist_phi_windows, 2, 1)

        dg.addWidget(QLabel("Mirror mode:"), 2, 2)
        self.calib_cake_dist_mirror_mode = QComboBox()
        self.calib_cake_dist_mirror_mode.addItems(["none", "separate", "together"])
        dg.addWidget(self.calib_cake_dist_mirror_mode, 2, 3)

        dg.addWidget(QLabel("Profile ylim:"), 3, 0)
        self.calib_cake_dist_profile_ylim = QLineEdit("")
        self.calib_cake_dist_profile_ylim.setPlaceholderText("Optional")
        dg.addWidget(self.calib_cake_dist_profile_ylim, 3, 1)

        dg.addWidget(QLabel("Fraction ylim:"), 3, 2)
        self.calib_cake_dist_fraction_ylim = QLineEdit("")
        self.calib_cake_dist_fraction_ylim.setPlaceholderText("Optional")
        dg.addWidget(self.calib_cake_dist_fraction_ylim, 3, 3)

        self.calib_cake_dist_percent = QCheckBox("Plot normalized profile as percent")
        self.calib_cake_dist_percent.setChecked(True)
        dg.addWidget(self.calib_cake_dist_percent, 4, 0, 1, 2)

        self.calib_cake_dist_save = QCheckBox("save")
        self.calib_cake_dist_save.setChecked(True)
        dg.addWidget(self.calib_cake_dist_save, 4, 2, 1, 2)

        self.calib_plot_cake_dist_btn = QPushButton("Analyze Cake Azimuthal Distribution")
        self.calib_plot_cake_dist_btn.clicked.connect(
            self._run_calibration_cake_azimuthal_distribution
        )
        dg.addWidget(self.calib_plot_cake_dist_btn, 5, 0, 1, 4)

    def _init_property_plot_group(self, layout: QVBoxLayout):
        """Create and connect the controls for property plot group."""
        property_group = QGroupBox("Property vs Azimuth Plot")
        pg = QGridLayout()
        property_group.setLayout(pg)
        layout.addWidget(property_group)

        pg.addWidget(QLabel("property:"), 0, 0)
        self.calib_property_name = QComboBox()
        self.calib_property_name.addItems(["pv_center", "pv_sigma", "pv_amplitude", "pv_fraction", "pv_height", "pv_fwhm", "r2"])
        self.calib_property_name.setEditable(True)
        pg.addWidget(self.calib_property_name, 0, 1)

        self.calib_property_only_success = QCheckBox("only_success")
        self.calib_property_only_success.setChecked(True)
        pg.addWidget(self.calib_property_only_success, 0, 2)

        pg.addWidget(QLabel("ylim:"), 0, 3)
        self.calib_property_ylim = QLineEdit("")
        self.calib_property_ylim.setPlaceholderText("Optional")
        pg.addWidget(self.calib_property_ylim, 0, 4)

        pg.addWidget(QLabel("Peak name:"), 1, 0)
        self.calib_peak_name = QLineEdit("")
        self.calib_peak_name.setPlaceholderText("Optional")
        pg.addWidget(self.calib_peak_name, 1, 1)

        pg.addWidget(QLabel("figure_title:"), 1, 2)
        self.calib_property_figure_title = QLineEdit("")
        self.calib_property_figure_title.setPlaceholderText("Optional")
        pg.addWidget(self.calib_property_figure_title, 1, 3)

        self.calib_property_save = QCheckBox("save")
        self.calib_property_save.setChecked(True)
        pg.addWidget(self.calib_property_save, 1, 4)

        self.calib_plot_property_btn = QPushButton("Plot Property vs Azimuth")
        self.calib_plot_property_btn.clicked.connect(self._run_calibration_plot_property)
        pg.addWidget(self.calib_plot_property_btn, 2, 0, 1, 5)

    def _init_save_group(self, layout: QVBoxLayout):
        """Create and connect the controls for save group."""
        save_group = QGroupBox("Save Settings")
        sg = QGridLayout()
        save_group.setLayout(sg)
        layout.addWidget(save_group)

        sg.addWidget(QLabel("figures_subdir:"), 0, 0)
        self.calib_figures_subdir = QLineEdit(DEFAULT_CALIBRATION_FIGURES_SUBDIR)
        sg.addWidget(self.calib_figures_subdir, 0, 1)

        sg.addWidget(QLabel("save_format:"), 1, 0)
        self.calib_save_format = QComboBox()
        self.calib_save_format.addItems(["png", "pdf", "svg"])
        sg.addWidget(self.calib_save_format, 1, 1)

        sg.addWidget(QLabel("Save DPI:"), 1, 2)
        self.calib_save_dpi = QLineEdit("400")
        self.calib_save_dpi.setValidator(QDoubleValidator())
        sg.addWidget(self.calib_save_dpi, 1, 3)

    def _build_analysis_paths(self):
        """Build normalized raw and analysis paths from the shared GUI state."""
        return self.path_service.build_analysis_paths(
            path_root=self.state.path_root,
            analysis_subdir=self.state.analysis_subdir,
            raw_subdir=self.state.raw_subdir,
        )

    def _poni_path(self):
        """Return the shared optional pyFAI geometry path from GUI state."""
        return getattr(self.state, "poni_path", None)

    def _mask_path(self):
        """Return the shared optional detector-mask path from GUI state."""
        return getattr(self.state, "mask_edf_path", None) or getattr(
            self.state,
            "mask_path",
            None,
        )

    def _azim_offset_deg(self):
        """Return the validated package-to-pyFAI azimuthal offset in degrees."""
        return parse_float_like(
            getattr(self.state, "azim_offset_deg", "-90.0"),
            name="azim_offset_deg",
        )

    def _polarization_factor(self):
        """Return the enabled polarization factor, or None when correction is disabled."""
        return self.calib_polarization_control.effective_factor()

    def _on_polarization_changed(self, enabled: bool, factor: float):
        """Persist a changed polarization setting and notify the synchronization callback."""
        self.state.polarization_enabled = bool(enabled)
        self.state.polarization_factor = float(factor)
        if self.polarization_changed_callback is not None:
            self.polarization_changed_callback(bool(enabled), float(factor))

    def _poni_mask_kwargs(self):
        """Return cleaned optional PONI and mask paths for backend calls."""
        def clean_path(value):
            """Normalize an optional path widget value to a stripped string."""
            if value is None:
                return None

            text = str(value).strip()

            if not text:
                return None

            return text

        return {
            "poni_path": clean_path(self._poni_path()),
            "mask_edf_path": clean_path(self._mask_path()),
        }

    def _calibration_context_kwargs(self):
        """Validate calibration metadata and build shared context arguments."""
        values = self.calibration_context.values()
        sample_name = values["sample_name"].strip()

        if not sample_name:
            raise ValueError("sample_name cannot be empty.")

        return {
            "sample_name": sample_name,
            "scan": parse_scan_spec(values["scan_spec"]),
            "temperature_K": parse_int_like(
                values["temperature_K"],
                name="temperature_K",
            ),
            "paths": self._build_analysis_paths(),
        }

    def _calibration_integration_kwargs(self):
        """Validate current fields and assemble keyword arguments for calibration integration."""
        kwargs = self._calibration_context_kwargs()
        kwargs.update(self._poni_mask_kwargs())
        kwargs.update(
            azimuthal_ranges=parse_edges(self.calib_azimuthal_edges.text()),
            include_full=self.calib_include_full.isChecked(),
            full_range=parse_tuple2(
                self.calib_full_range.text(),
                name="full_range",
                cast=float,
            ),
            npt=parse_int_like(self.calib_npt.text(), name="npt"),
            normalize=self.calib_normalize.isChecked(),
            q_norm_range=parse_tuple2(
                self.calib_q_norm_range.text(),
                name="q_norm_range",
                cast=float,
            ),
            overwrite_xy=self.calib_overwrite_xy.isChecked(),
            azim_offset_deg=self._azim_offset_deg(),
            polarization_factor=self._polarization_factor(),
        )
        return kwargs

    def _run_calibration_compute_xy(self):
        """Validate integration controls and create calibration XY cache files."""
        try:
            kwargs = self._calibration_integration_kwargs()
            self.calibration_service.compute_xy_files(**kwargs)
            self.log("Calibration XY computation finished.")

        except Exception as exc:
            self.log(f"Calibration Compute XY Error: {exc}")

    def _run_calibration_peak_fitting(self):
        """Parse the calibration peak fitting controls, invoke the service workflow, and log completion or errors."""
        try:
            def error_summary(traceback_text):
                """Extract the final informative message from a captured worker traceback."""
                lines = [
                    line.strip()
                    for line in str(traceback_text).splitlines()
                    if line.strip()
                ]
                return lines[-1] if lines else "unknown error"

            kwargs = self._calibration_integration_kwargs()
            kwargs.update(
                q_fit_range=parse_tuple2(
                    self.calib_q_fit_range.text(),
                    name="q_fit_range",
                    cast=float,
                ),
                eta=parse_float_like(self.calib_eta.text(), name="eta"),
                eta_vary=self.calib_eta_vary.isChecked(),
                fit_method=self.calib_fit_method.currentText() or "leastsq",
                force_refit=self.calib_force_refit.isChecked(),
                out_csv_name=self.calib_out_csv_name.text().strip() or "peak_fits.csv",
            )

            def task():
                """Execute the validated backend operation inside the background worker thread."""
                return self.calibration_service.do_peak_fitting(**kwargs)

            def success(_df):
                """Summarize the completed background operation and update the GUI log."""
                self.log("Calibration peak fitting finished.")
                if _df is not None and not getattr(_df, "empty", True):
                    cols = [
                        c
                        for c in ("azim_range_str", "success", "pv_center", "pv_fraction", "pv_fwhm", "r2")
                        if c in _df.columns
                    ]
                    if cols:
                        for row in _df[cols].itertuples(index=False):
                            values = row._asdict()
                            label = values.pop("azim_range_str", "azim")
                            details = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in values.items())
                            self.log(f"  {label}: {details}")

            run_task_with_output_dialog(
                self,
                "Calibration Peak Fitting",
                task,
                on_success=success,
                on_error=lambda tb: self.log(
                    f"Calibration Peak Fitting Error: {error_summary(tb)}"
                ),
            )

        except Exception as exc:
            self.log(f"Calibration Peak Fitting Error: {exc}")

    def _run_calibration_plot_caked(self):
        """Plot calibration patterns for the configured azimuthal slices."""
        try:
            kwargs = self._calibration_integration_kwargs()
            kwargs.update(
                xlim=parse_optional_tuple2(
                    self.calib_caked_xlim.text(),
                    name="xlim",
                    cast=float,
                ),
                ylim=parse_optional_tuple2(
                    self.calib_caked_ylim.text(),
                    name="ylim",
                    cast=float,
                ),
                figure_title=self.calib_caked_figure_title.text().strip() or None,
                save=self.calib_caked_save.isChecked(),
                figures_subdir=self.calib_figures_subdir.text().strip()
                or DEFAULT_CALIBRATION_FIGURES_SUBDIR,
                save_format=self.calib_save_format.currentText(),
                save_dpi=parse_int_like(self.calib_save_dpi.text(), name="save_dpi"),
            )

            self.calibration_service.plot_caked_1d_patterns(**kwargs)
            self.log("Calibration caked 1D pattern plot finished.")

        except Exception as exc:
            self.log(f"Calibration Caked Pattern Plot Error: {exc}")

    def _run_calibration_plot_detector_cake(self):
        """Plot the calibration detector image and pyFAI 2D cake side by side."""
        try:
            kwargs = self._calibration_context_kwargs()
            kwargs.update(self._poni_mask_kwargs())
            kwargs.update(
                npt_rad=parse_int_like(self.calib_npt.text(), name="npt_rad"),
                npt_azim=parse_int_like(
                    self.calib_detector_cake_npt_azim.text(),
                    name="npt_azim",
                ),
                radial_range=parse_optional_tuple2(
                    self.calib_detector_cake_radial_range.text(),
                    name="radial_range",
                    cast=float,
                ),
                azimuthal_range=parse_tuple2(
                    self.calib_full_range.text(),
                    name="azimuthal_range",
                    cast=float,
                ),
                normalize=self.calib_detector_cake_normalize.isChecked(),
                q_norm_range=parse_tuple2(
                    self.calib_q_norm_range.text(),
                    name="q_norm_range",
                    cast=float,
                ),
                use_mask=self.calib_detector_cake_use_mask.isChecked(),
                azim_offset_deg=self._azim_offset_deg(),
                polarization_factor=self._polarization_factor(),
                detector_clim=parse_optional_tuple2(
                    self.calib_detector_cake_detector_clim.text(),
                    name="detector_clim",
                    cast=float,
                ),
                cake_clim=parse_optional_tuple2(
                    self.calib_detector_cake_cake_clim.text(),
                    name="cake_clim",
                    cast=float,
                ),
                detector_log_scale=self.calib_detector_cake_detector_log.isChecked(),
                cake_log_scale=self.calib_detector_cake_cake_log.isChecked(),
                invert_detector_x=self.calib_detector_cake_invert_x.isChecked(),
                invert_detector_y=self.calib_detector_cake_invert_y.isChecked(),
                figure_title=self.calib_detector_cake_figure_title.text().strip()
                or None,
                save=self.calib_detector_cake_save.isChecked(),
                figures_subdir=self.calib_figures_subdir.text().strip()
                or DEFAULT_CALIBRATION_FIGURES_SUBDIR,
                save_format=self.calib_save_format.currentText(),
                save_dpi=parse_int_like(self.calib_save_dpi.text(), name="save_dpi"),
            )

            self.calibration_service.plot_detector_and_cake(**kwargs)
            self.log("Calibration detector and 2D cake plot finished.")

        except Exception as exc:
            self.log(f"Calibration Detector/Cake Plot Error: {exc}")

    def _parse_phi_center_halfwidth_windows(self, text: str):
        """Parse optional phi selectors encoded as ``(center, half_width)`` pairs."""
        value = parse_python_literal(text, empty=None)
        if value is None:
            return None

        if isinstance(value, tuple) and len(value) == 2:
            return [(float(value[0]), float(value[1]))]

        if not isinstance(value, (list, tuple)):
            raise ValueError(
                "Phi windows must be a (center, half_width) pair or a list of pairs."
            )

        out = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each phi window must contain center and half_width.")
            out.append((float(item[0]), float(item[1])))
        return out

    def _run_calibration_cake_azimuthal_distribution(self):
        """Analyze q-band intensity distributions from the calibration 2D cake."""
        try:
            def error_summary(traceback_text):
                """Extract the final informative message from a captured worker traceback."""
                lines = [
                    line.strip()
                    for line in str(traceback_text).splitlines()
                    if line.strip()
                ]
                return lines[-1] if lines else "unknown error"

            kwargs = self._calibration_context_kwargs()
            kwargs.update(self._poni_mask_kwargs())
            kwargs.update(
                q_value=parse_float_like(
                    self.calib_cake_dist_q_value.text(),
                    name="q_value",
                ),
                q_width=parse_float_like(
                    self.calib_cake_dist_q_width.text(),
                    name="q_width",
                ),
                bg_mode=self.calib_cake_dist_bg_mode.currentText(),
                bg_q_range=parse_optional_tuple2(
                    self.calib_cake_dist_bg_q_range.text(),
                    name="bg_q_range",
                    cast=float,
                ),
                phi_windows=self._parse_phi_center_halfwidth_windows(
                    self.calib_cake_dist_phi_windows.text()
                ),
                mirror_mode=self.calib_cake_dist_mirror_mode.currentText(),
                npt_rad=parse_int_like(self.calib_npt.text(), name="npt_rad"),
                npt_azim=parse_int_like(
                    self.calib_detector_cake_npt_azim.text(),
                    name="npt_azim",
                ),
                radial_range=parse_optional_tuple2(
                    self.calib_detector_cake_radial_range.text(),
                    name="radial_range",
                    cast=float,
                ),
                azimuthal_range=parse_tuple2(
                    self.calib_full_range.text(),
                    name="azimuthal_range",
                    cast=float,
                ),
                normalize=self.calib_normalize.isChecked(),
                q_norm_range=parse_tuple2(
                    self.calib_q_norm_range.text(),
                    name="q_norm_range",
                    cast=float,
                ),
                use_mask=self.calib_detector_cake_use_mask.isChecked(),
                azim_offset_deg=self._azim_offset_deg(),
                polarization_factor=self._polarization_factor(),
                profile_ylim=parse_optional_tuple2(
                    self.calib_cake_dist_profile_ylim.text(),
                    name="profile_ylim",
                    cast=float,
                ),
                fraction_ylim=parse_optional_tuple2(
                    self.calib_cake_dist_fraction_ylim.text(),
                    name="fraction_ylim",
                    cast=float,
                ),
                fraction_as_percent=self.calib_cake_dist_percent.isChecked(),
                make_plots=False,
                save=self.calib_cake_dist_save.isChecked(),
                figures_subdir=self.calib_figures_subdir.text().strip()
                or DEFAULT_CALIBRATION_FIGURES_SUBDIR,
                save_format=self.calib_save_format.currentText(),
                save_dpi=parse_int_like(self.calib_save_dpi.text(), name="save_dpi"),
            )

            def task():
                """Execute the validated backend operation inside the background worker thread."""
                return self.calibration_service.analyze_cake_azimuthal_distribution(
                    **kwargs
                )

            def success(result):
                """Log the normalized phi-window summary returned by the backend."""
                self.calibration_service.plot_cake_azimuthal_distribution_profiles(
                    profile_df=result.get("profile_df"),
                    summary_df=result.get("summary_df"),
                    title=result.get("figure_title"),
                    profile_ylim=kwargs.get("profile_ylim"),
                    fraction_ylim=kwargs.get("fraction_ylim"),
                    fraction_as_percent=kwargs.get("fraction_as_percent", True),
                    save=kwargs.get("save", False),
                    base_dir=result.get("base_dir"),
                    figures_subdir=kwargs.get(
                        "figures_subdir",
                        DEFAULT_CALIBRATION_FIGURES_SUBDIR,
                    ),
                    save_name_prefix=result.get("save_base", "cake_phi_distribution"),
                    save_format=kwargs.get("save_format", "png"),
                    save_dpi=kwargs.get("save_dpi", 400),
                )
                summary = result.get("summary_df")
                profile_csv = result.get("profile_csv_path")
                summary_csv = result.get("summary_csv_path")
                self.log("Calibration cake azimuthal distribution analysis finished.")
                if profile_csv is not None:
                    self.log(f"Profile CSV: {profile_csv}")
                if summary_csv is not None:
                    self.log(f"Summary CSV: {summary_csv}")
                if summary is not None and not summary.empty:
                    for row in summary.itertuples(index=False):
                        self.log(f"{row.label}: {row.percent:.3f}%")

            run_task_with_output_dialog(
                self,
                "Cake Azimuthal Distribution",
                task,
                on_success=success,
                on_error=lambda tb: self.log(
                    f"Calibration Cake Azimuthal Distribution Error: {error_summary(tb)}"
                ),
            )

        except Exception as exc:
            self.log(f"Calibration Cake Azimuthal Distribution Error: {exc}")

    def _run_calibration_plot_property(self):
        """Plot a selected calibration fit property against azimuth."""
        try:
            kwargs = self._calibration_context_kwargs()
            kwargs.update(
                _property=self.calib_property_name.currentText() or "pv_center",
                peak_name=self.calib_peak_name.text().strip(),
                figure_title=self.calib_property_figure_title.text().strip() or None,
                only_success=self.calib_property_only_success.isChecked(),
                out_csv_name=self.calib_out_csv_name.text().strip() or "peak_fits.csv",
                ylim=parse_optional_tuple2(
                    self.calib_property_ylim.text(),
                    name="ylim",
                    cast=float,
                ),
                save=self.calib_property_save.isChecked(),
                figures_subdir=self.calib_figures_subdir.text().strip()
                or DEFAULT_CALIBRATION_FIGURES_SUBDIR,
                save_format=self.calib_save_format.currentText(),
                save_dpi=parse_int_like(self.calib_save_dpi.text(), name="save_dpi"),
            )

            self.calibration_service.plot_property_vs_azimuth(**kwargs)
            self.log("Calibration property-vs-azimuth plot finished.")

        except Exception as exc:
            self.log(f"Calibration Property Plot Error: {exc}")


    def _browse_file_into(
        self,
        line_edit: QLineEdit,
        caption: str,
        file_filter: str,
    ):
        """Open a file chooser and write the selected path into a line edit."""
        selected, _ = QFileDialog.getOpenFileName(
            self,
            caption,
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

    def _launch_pyfai_calib2(self):
        """Launch ``pyFAI-calib2`` asynchronously with the optional detector image."""
        try:
            image_path = self.calib_pyfai_image_path.text().strip()
            exe, args = self.calibration_service.build_pyfai_calib2_command(image_path)

            process = QProcess(self)
            process.setProgram(exe)
            process.setArguments(args)

            process.errorOccurred.connect(
                lambda _err, p=process: self.log(
                    f"pyFAI-calib2 process error: {p.errorString()}"
                )
            )
            process.finished.connect(
                lambda exit_code, _exit_status, p=process: self._cleanup_process(
                    "pyFAI-calib2",
                    p,
                    exit_code,
                )
            )

            process.start()

            if not process.waitForStarted(2000):
                raise RuntimeError(
                    f"Failed to launch pyFAI-calib2.\nProcess error: {process.errorString()}"
                )

            self.external_processes.append(process)

            if image_path:
                self.log(f"pyFAI-calib2 launched with image: {image_path}")
            else:
                self.log("pyFAI-calib2 launched.")

        except Exception as exc:
            self.log(f"Launch pyFAI-calib2 Error: {exc}")

    def _cleanup_process(self, name: str, process: QProcess, exit_code: int):
        """Release a completed external process and report its exit code."""
        if process in self.external_processes:
            self.external_processes.remove(process)

        self.log(f"{name} finished with exit code {exit_code}.")
