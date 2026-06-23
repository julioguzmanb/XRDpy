"""
Fitting tab for the analysis GUI.

This reproduces the legacy Fitting tab layout while keeping backend actions
separated from the main window.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from trxrdpy.analysis.gui.defaults import (
    DEFAULT_AZIM_WINDOWS,
    DEFAULT_FIT_PEAK_SPECS,
    DEFAULT_MULTI_EXPERIMENTS_FIT,
)
from trxrdpy.analysis.gui.services import FittingService, IntegrationService, PathService
from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.utils import (
    parse_float_like,
    parse_groups,
    parse_int_like,
    parse_optional_float_like,
    parse_optional_int_like,
    parse_python_literal,
    parse_windows,
    pretty_literal,
)
from trxrdpy.analysis.gui.widgets import ExperimentMetadataWidget, MultiExperimentEditor
from trxrdpy.analysis.gui.widgets.task_output_dialog import run_task_with_output_dialog


class FittingTab(QWidget):
    """Configure peak fitting, fit overlays, and parameter-evolution plots.

    Attributes
    ----------
    state : AnalysisGuiState
        Shared facility, path, geometry, and polarization configuration.
    path_service : PathService
        Builds normalized analysis paths for fitting backends.
    integration_service : IntegrationService
        Parses series and reference selectors.
    fitting_service : FittingService
        Stateless adapter for fitting and evolution-plot APIs.
    fit_mode_combo : QComboBox
        Selects single- or multi-experiment mode.
    fit_series_combo, fit_multi_series_combo : QComboBox
        Select delay or fluence fitting for each mode.
    fit_single_metadata : ExperimentMetadataWidget
        Metadata editor for the single-experiment workflow.
    fit_peak_specs, fit_azim_windows : QPlainTextEdit
        Editable peak-model and azimuthal-window definitions.
    fit_default_eta, fit_npt, fit_q_norm_range : QWidget
        Profile and integration settings.
    fit_out_csv_name : QLineEdit
        Output fitting-table filename synchronized with the selected series.
    fit_overlay_* : QWidget
        Controls selecting and rendering one stored delay fit.
    fit_fluence_overlay_* : QWidget
        Controls selecting and rendering one stored fluence fit.
    fit_time_* , fit_fluence_time_* : QWidget
        Single-experiment property-evolution controls.
    fit_multi_editor, fit_multi_editor_fluence : MultiExperimentEditor
        Experiment collections for multi-series evolution plots.
    log : callable
        Callback receiving fit task status and validation errors.
    """

    def __init__(
        self,
        state: AnalysisGuiState,
        path_service: PathService,
        integration_service: IntegrationService,
        fitting_service: FittingService,
        log: Optional[Callable[[str], None]] = None,
        parent=None,
    ):
        """Initialize ``FittingTab``, bind shared state and services, and create its controls."""
        super().__init__(parent)

        self.state = state
        self.path_service = path_service
        self.integration_service = integration_service
        self.fitting_service = fitting_service
        self.log = log or (lambda message: None)

        layout = self._make_scroll_layout()

        self._init_mode_group(layout)
        self._init_single_experiment_widget(layout)
        self._init_multi_experiment_widget(layout)

        self.set_facility(self.state.facility or "SACLA")
        self._refresh_mode_widgets()
        self._refresh_series_widgets()

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

    def _init_mode_group(self, layout: QVBoxLayout):
        """Create the selector that switches single- and multi-experiment controls."""
        mode_group = QGroupBox("Analysis Mode")
        ml = QHBoxLayout()
        mode_group.setLayout(ml)
        layout.addWidget(mode_group)

        ml.addWidget(QLabel("Fitting mode:"))

        self.fit_mode_combo = QComboBox()
        self.fit_mode_combo.addItems(["Single experiment", "Multiple experiments"])
        self.fit_mode_combo.currentIndexChanged.connect(self._refresh_mode_widgets)
        ml.addWidget(self.fit_mode_combo)

        ml.addStretch()

    def _init_single_experiment_widget(self, layout: QVBoxLayout):
        """Create and connect the controls for single experiment widget."""
        self.fit_single_widget = QWidget()
        fsl = QVBoxLayout()
        self.fit_single_widget.setLayout(fsl)
        layout.addWidget(self.fit_single_widget)

        series_group = QGroupBox("Experiment Type")
        sgl = QHBoxLayout()
        series_group.setLayout(sgl)
        fsl.addWidget(series_group)

        sgl.addWidget(QLabel("Experiment type:"))

        self.fit_series_combo = QComboBox()
        self.fit_series_combo.addItems(["Delay scan", "Fluence scan"])
        self.fit_series_combo.currentIndexChanged.connect(self._refresh_series_widgets)
        sgl.addWidget(self.fit_series_combo)

        sgl.addStretch()

        self.fit_single_metadata = ExperimentMetadataWidget(
            title="Experiment Metadata",
            include_id09=True,
        )
        fsl.addWidget(self.fit_single_metadata)

        self._init_single_delay_selector_group(fsl)
        self._init_single_fluence_selector_group(fsl)
        self._init_single_peak_fitting_group(fsl)
        self._init_single_runtime_group(fsl)
        self._init_single_overlay_groups(fsl)
        self._init_single_evolution_groups(fsl)

        fsl.addStretch()

    def _init_single_delay_selector_group(self, layout: QVBoxLayout):
        """Create and connect the controls for single delay selector group."""
        self.fit_delay_selector_group = QGroupBox("Delay-series Selection")
        grid = QGridLayout()
        self.fit_delay_selector_group.setLayout(grid)
        layout.addWidget(self.fit_delay_selector_group)

        grid.addWidget(QLabel("delays_fs:"), 0, 0)
        self.fit_delays = QLineEdit("all")
        grid.addWidget(self.fit_delays, 0, 1)

        grid.addWidget(QLabel("Reference type:"), 1, 0)
        self.fit_ref_type = QComboBox()
        self.fit_ref_type.addItems(["dark", "delay"])
        grid.addWidget(self.fit_ref_type, 1, 1)

        grid.addWidget(QLabel("Reference value:"), 2, 0)
        self.fit_ref_value = QLineEdit("[1466556]")
        self.fit_ref_value.setPlaceholderText(
            "Examples: [1466556], [[1466556],[1466588]], -95000"
        )
        grid.addWidget(self.fit_ref_value, 2, 1)

        grid.addWidget(QLabel("Reference mode:"), 3, 0)
        self.fit_ref_values_mode = QComboBox()
        self.fit_ref_values_mode.addItems(["combine", "separate"])
        grid.addWidget(self.fit_ref_values_mode, 3, 1)

    def _init_single_fluence_selector_group(self, layout: QVBoxLayout):
        """Create and connect the controls for single fluence selector group."""
        self.fit_fluence_selector_group = QGroupBox("Fluence-series Selection")
        fg = QGridLayout()
        self.fit_fluence_selector_group.setLayout(fg)
        layout.addWidget(self.fit_fluence_selector_group)

        fg.addWidget(QLabel("Delay [fs]:"), 0, 0)
        self.fit_fluence_delay_fs = QLineEdit("0")
        self.fit_fluence_delay_fs.setValidator(QDoubleValidator())
        fg.addWidget(self.fit_fluence_delay_fs, 0, 1)

        fg.addWidget(QLabel("Fluences [mJ/cm²]:"), 1, 0)
        self.fit_fluences = QLineEdit("all")
        fg.addWidget(self.fit_fluences, 1, 1)

        fg.addWidget(QLabel("Reference type:"), 2, 0)
        self.fit_fluence_ref_type = QComboBox()
        self.fit_fluence_ref_type.addItems(["dark", "fluence"])
        fg.addWidget(self.fit_fluence_ref_type, 2, 1)

        fg.addWidget(QLabel("Reference value:"), 3, 0)
        self.fit_fluence_ref_value = QLineEdit("[1466556]")
        self.fit_fluence_ref_value.setPlaceholderText(
            "Examples: [1466556], [[167246,167285],[167300,167310]], [1.5, 5.0]"
        )
        fg.addWidget(self.fit_fluence_ref_value, 3, 1)

        fg.addWidget(QLabel("Reference mode:"), 4, 0)
        self.fit_fluence_ref_values_mode = QComboBox()
        self.fit_fluence_ref_values_mode.addItems(["combine", "separate"])
        fg.addWidget(self.fit_fluence_ref_values_mode, 4, 1)

    def _init_single_peak_fitting_group(self, layout: QVBoxLayout):
        """Create and connect the controls for single peak fitting group."""
        common_group = QGroupBox("Peak-fitting Settings")
        grid = QGridLayout()
        common_group.setLayout(grid)
        layout.addWidget(common_group)

        row = 0

        grid.addWidget(QLabel("Peak definitions:"), row, 0)
        self.fit_peak_specs = QPlainTextEdit(pretty_literal(DEFAULT_FIT_PEAK_SPECS))
        grid.addWidget(self.fit_peak_specs, row, 1)

        grid.setRowMinimumHeight(row, 100)
        row += 1

        grid.addWidget(QLabel("Azimuthal windows [deg]:"), row, 0)
        self.fit_azim_windows = QPlainTextEdit(pretty_literal(DEFAULT_AZIM_WINDOWS))
        self.fit_azim_windows.setMinimumHeight(90)
        grid.addWidget(self.fit_azim_windows, row, 1)
        row += 1

        grid.addWidget(QLabel("Azimuthal mode:"), row, 0)
        self.fit_phi_mode = QComboBox()
        self.fit_phi_mode.addItems(["phi_avg", "separate_phi"])
        grid.addWidget(self.fit_phi_mode, row, 1)
        row += 1

        grid.addWidget(QLabel("Azimuthal reduction:"), row, 0)
        self.fit_phi_reduce = QComboBox()
        self.fit_phi_reduce.addItems(["sum", "mean"])
        grid.addWidget(self.fit_phi_reduce, row, 1)
        row += 1

        grid.addWidget(QLabel("default_eta:"), row, 0)
        self.fit_default_eta = QLineEdit("0.3")
        self.fit_default_eta.setValidator(QDoubleValidator())
        grid.addWidget(self.fit_default_eta, row, 1)
        row += 1

        grid.addWidget(QLabel("Number of q points:"), row, 0)
        self.fit_npt = QLineEdit("1000")
        self.fit_npt.setValidator(QDoubleValidator())
        grid.addWidget(self.fit_npt, row, 1)
        row += 1

        grid.addWidget(QLabel("Q normalization range:"), row, 0)
        self.fit_q_norm_range = QLineEdit("(2.65, 2.75)")
        grid.addWidget(self.fit_q_norm_range, row, 1)
        row += 1

        grid.addWidget(QLabel("out_csv_name:"), row, 0)
        self.fit_out_csv_name = QLineEdit(
            self._default_fit_csv_name_for_series("Delay scan")
        )
        grid.addWidget(self.fit_out_csv_name, row, 1)

    def _init_single_runtime_group(self, layout: QVBoxLayout):
        """Create and connect the controls for single runtime group."""
        options_group = QGroupBox("Fit Runtime Options")
        og = QGridLayout()
        options_group.setLayout(og)
        layout.addWidget(options_group)

        self.fit_normalize_xy = QCheckBox("normalize_xy")
        self.fit_normalize_xy.setChecked(True)
        og.addWidget(self.fit_normalize_xy, 0, 0, 1, 2)

        self.fit_compute_if_missing = QCheckBox("compute_if_missing")
        self.fit_compute_if_missing.setChecked(True)
        og.addWidget(self.fit_compute_if_missing, 1, 0, 1, 2)

        self.fit_overwrite_xy = QCheckBox("overwrite_xy")
        og.addWidget(self.fit_overwrite_xy, 2, 0, 1, 2)

        self.fit_include_reference = QCheckBox("include_reference_in_output")
        self.fit_include_reference.setChecked(True)
        og.addWidget(self.fit_include_reference, 3, 0, 1, 2)

        self.fit_show_fit_figures = QCheckBox("show_fit_figures")
        og.addWidget(self.fit_show_fit_figures, 4, 0, 1, 2)

        self.fit_save_fit_figures = QCheckBox("save_fit_figures")
        og.addWidget(self.fit_save_fit_figures, 5, 0, 1, 2)

        og.addWidget(QLabel("fit_figures_format:"), 6, 0)
        self.fit_fig_format = QComboBox()
        self.fit_fig_format.addItems(["png", "pdf", "svg"])
        og.addWidget(self.fit_fig_format, 6, 1)

        og.addWidget(QLabel("fit_figures_dpi:"), 7, 0)
        self.fit_fig_dpi = QLineEdit("300")
        self.fit_fig_dpi.setValidator(QDoubleValidator())
        og.addWidget(self.fit_fig_dpi, 7, 1)

        self.fit_plot_only_success = QCheckBox("plot_only_success")
        self.fit_plot_only_success.setChecked(True)
        og.addWidget(self.fit_plot_only_success, 8, 0, 1, 2)

        og.addWidget(QLabel("fit_oversample:"), 9, 0)
        self.fit_oversample = QLineEdit("10")
        self.fit_oversample.setValidator(QDoubleValidator())
        og.addWidget(self.fit_oversample, 9, 1)

        self.fit_run_btn = QPushButton("Run Peak Fitting")
        self.fit_run_btn.clicked.connect(self._run_delay_peak_fitting)
        layout.addWidget(self.fit_run_btn)

    def _init_single_overlay_groups(self, layout: QVBoxLayout):
        """Create and connect the controls for single overlay groups."""
        self.fit_delay_overlay_group = QGroupBox("Delay Overlay Plot from CSV")
        ov = QGridLayout()
        self.fit_delay_overlay_group.setLayout(ov)
        layout.addWidget(self.fit_delay_overlay_group)

        ov.addWidget(QLabel("peak:"), 0, 0)
        self.fit_overlay_peak = QLineEdit("110")
        ov.addWidget(self.fit_overlay_peak, 0, 1)

        ov.addWidget(QLabel("Delay [fs]:"), 1, 0)
        self.fit_overlay_delay = QLineEdit("0")
        ov.addWidget(self.fit_overlay_delay, 1, 1)

        ov.addWidget(QLabel("group:"), 2, 0)
        self.fit_overlay_group = QLineEdit("Full")
        ov.addWidget(self.fit_overlay_group, 2, 1)

        ov.addWidget(QLabel("reference_index:"), 3, 0)
        self.fit_overlay_reference_index = QLineEdit("")
        self.fit_overlay_reference_index.setPlaceholderText(
            "Optional. 1-based. Blank = first reference"
        )
        self.fit_overlay_reference_index.setValidator(QIntValidator())
        ov.addWidget(self.fit_overlay_reference_index, 3, 1)

        self.fit_overlay_is_reference = QCheckBox("is_reference")
        ov.addWidget(self.fit_overlay_is_reference, 4, 0, 1, 2)

        self.fit_overlay_ensure_csv = QCheckBox("ensure_csv")
        self.fit_overlay_ensure_csv.setChecked(True)
        ov.addWidget(self.fit_overlay_ensure_csv, 5, 0, 1, 2)

        self.fit_overlay_show = QCheckBox("show")
        self.fit_overlay_show.setChecked(True)
        ov.addWidget(self.fit_overlay_show, 6, 0, 1, 2)

        self.fit_overlay_save = QCheckBox("save")
        self.fit_overlay_save.setChecked(True)
        ov.addWidget(self.fit_overlay_save, 7, 0, 1, 2)

        self.fit_fluence_overlay_group = QGroupBox("Fluence Overlay Plot from CSV")
        fov = QGridLayout()
        self.fit_fluence_overlay_group.setLayout(fov)
        layout.addWidget(self.fit_fluence_overlay_group)

        fov.addWidget(QLabel("peak:"), 0, 0)
        self.fit_fluence_overlay_peak = QLineEdit("110")
        fov.addWidget(self.fit_fluence_overlay_peak, 0, 1)

        fov.addWidget(QLabel("Fluence [mJ/cm²]:"), 1, 0)
        self.fit_fluence_overlay_fluence = QLineEdit("1.5")
        self.fit_fluence_overlay_fluence.setValidator(QDoubleValidator())
        fov.addWidget(self.fit_fluence_overlay_fluence, 1, 1)

        fov.addWidget(QLabel("group:"), 2, 0)
        self.fit_fluence_overlay_group_name = QLineEdit("Full")
        fov.addWidget(self.fit_fluence_overlay_group_name, 2, 1)

        fov.addWidget(QLabel("reference_index:"), 3, 0)
        self.fit_fluence_overlay_reference_index = QLineEdit("")
        self.fit_fluence_overlay_reference_index.setPlaceholderText(
            "Optional. 1-based. Blank = first reference"
        )
        self.fit_fluence_overlay_reference_index.setValidator(QIntValidator())
        fov.addWidget(self.fit_fluence_overlay_reference_index, 3, 1)

        self.fit_fluence_overlay_is_reference = QCheckBox("is_reference")
        fov.addWidget(self.fit_fluence_overlay_is_reference, 4, 0, 1, 2)

        self.fit_fluence_overlay_ensure_csv = QCheckBox("ensure_csv")
        self.fit_fluence_overlay_ensure_csv.setChecked(True)
        fov.addWidget(self.fit_fluence_overlay_ensure_csv, 5, 0, 1, 2)

        self.fit_fluence_overlay_show = QCheckBox("show")
        self.fit_fluence_overlay_show.setChecked(True)
        fov.addWidget(self.fit_fluence_overlay_show, 6, 0, 1, 2)

        self.fit_fluence_overlay_save = QCheckBox("save")
        self.fit_fluence_overlay_save.setChecked(True)
        fov.addWidget(self.fit_fluence_overlay_save, 7, 0, 1, 2)

        self.fit_overlay_btn = QPushButton("Plot Fit Overlay")
        self.fit_overlay_btn.clicked.connect(self._run_fit_overlay)
        layout.addWidget(self.fit_overlay_btn)

    def _init_single_evolution_groups(self, layout: QVBoxLayout):
        """Create and connect the controls for single evolution groups."""
        self.fit_delay_time_group = QGroupBox("Delay Evolution Plot")
        tg = QGridLayout()
        self.fit_delay_time_group.setLayout(tg)
        layout.addWidget(self.fit_delay_time_group)

        tg.addWidget(QLabel("peak:"), 0, 0)
        self.fit_time_peak = QLineEdit("110")
        tg.addWidget(self.fit_time_peak, 0, 1)

        tg.addWidget(QLabel("_property:"), 1, 0)
        self.fit_property = QComboBox()
        self.fit_property.addItems(["hkl_pos", "hkl_fwhm", "hkl_i", "hkl_area"])
        tg.addWidget(self.fit_property, 1, 1)

        tg.addWidget(QLabel("unit:"), 2, 0)
        self.fit_time_unit = QComboBox()
        self.fit_time_unit.addItems(["ps", "fs", "ns", "µs", "ms", "s"])
        tg.addWidget(self.fit_time_unit, 2, 1)

        tg.addWidget(QLabel("groups:"), 3, 0)
        self.fit_groups = QLineEdit("['Full', 60, 30, 0]")
        tg.addWidget(self.fit_groups, 3, 1)

        tg.addWidget(QLabel("title:"), 4, 0)
        self.fit_time_title = QLineEdit("")
        self.fit_time_title.setPlaceholderText("Optional")
        tg.addWidget(self.fit_time_title, 4, 1)

        tg.addWidget(QLabel("delay_offset:"), 5, 0)
        self.fit_delay_offset = QLineEdit("0")
        self.fit_delay_offset.setValidator(QDoubleValidator())
        tg.addWidget(self.fit_delay_offset, 5, 1)

        self.fit_as_lines = QCheckBox("as_lines")
        tg.addWidget(self.fit_as_lines, 6, 0, 1, 2)

        self.fit_show_baseline_sigma = QCheckBox("show_baseline_sigma")
        self.fit_show_baseline_sigma.setChecked(True)
        tg.addWidget(self.fit_show_baseline_sigma, 7, 0, 1, 2)

        tg.addWidget(QLabel("baseline_sigma:"), 8, 0)
        self.fit_baseline_sigma = QLineEdit("1")
        self.fit_baseline_sigma.setValidator(QDoubleValidator())
        tg.addWidget(self.fit_baseline_sigma, 8, 1)

        tg.addWidget(QLabel("baseline_alpha:"), 9, 0)
        self.fit_baseline_alpha = QLineEdit("1")
        self.fit_baseline_alpha.setValidator(QDoubleValidator())
        tg.addWidget(self.fit_baseline_alpha, 9, 1)

        tg.addWidget(QLabel("baseline_mode:"), 10, 0)
        self.fit_baseline_mode = QComboBox()
        self.fit_baseline_mode.addItems(["errorbar", "band"])
        tg.addWidget(self.fit_baseline_mode, 10, 1)

        self.fit_time_save = QCheckBox("save")
        self.fit_time_save.setChecked(True)
        tg.addWidget(self.fit_time_save, 11, 0, 1, 2)

        tg.addWidget(QLabel("save_fmt:"), 12, 0)
        self.fit_time_save_fmt = QComboBox()
        self.fit_time_save_fmt.addItems(["png", "pdf", "svg"])
        tg.addWidget(self.fit_time_save_fmt, 12, 1)

        tg.addWidget(QLabel("Save DPI:"), 13, 0)
        self.fit_time_save_dpi = QLineEdit("300")
        self.fit_time_save_dpi.setValidator(QDoubleValidator())
        tg.addWidget(self.fit_time_save_dpi, 13, 1)

        self.fit_fluence_time_group = QGroupBox("Fluence Evolution Plot")
        ftg = QGridLayout()
        self.fit_fluence_time_group.setLayout(ftg)
        layout.addWidget(self.fit_fluence_time_group)

        ftg.addWidget(QLabel("peak:"), 0, 0)
        self.fit_fluence_time_peak = QLineEdit("110")
        ftg.addWidget(self.fit_fluence_time_peak, 0, 1)

        ftg.addWidget(QLabel("_property:"), 1, 0)
        self.fit_fluence_property = QComboBox()
        self.fit_fluence_property.addItems(["hkl_pos", "hkl_fwhm", "hkl_i", "hkl_area"])
        ftg.addWidget(self.fit_fluence_property, 1, 1)

        ftg.addWidget(QLabel("unit:"), 2, 0)
        self.fit_fluence_unit = QLineEdit("mJ/cm$^2$")
        ftg.addWidget(self.fit_fluence_unit, 2, 1)

        ftg.addWidget(QLabel("groups:"), 3, 0)
        self.fit_fluence_groups = QLineEdit("['Full', 60, 30, 0]")
        ftg.addWidget(self.fit_fluence_groups, 3, 1)

        ftg.addWidget(QLabel("title:"), 4, 0)
        self.fit_fluence_time_title = QLineEdit("")
        self.fit_fluence_time_title.setPlaceholderText("Optional")
        ftg.addWidget(self.fit_fluence_time_title, 4, 1)

        ftg.addWidget(QLabel("Fluence offset:"), 5, 0)
        self.fit_fluence_offset = QLineEdit("0")
        self.fit_fluence_offset.setValidator(QDoubleValidator())
        ftg.addWidget(self.fit_fluence_offset, 5, 1)

        self.fit_fluence_as_lines = QCheckBox("as_lines")
        ftg.addWidget(self.fit_fluence_as_lines, 6, 0, 1, 2)

        self.fit_fluence_show_baseline_sigma = QCheckBox("show_baseline_sigma")
        self.fit_fluence_show_baseline_sigma.setChecked(True)
        ftg.addWidget(self.fit_fluence_show_baseline_sigma, 7, 0, 1, 2)

        ftg.addWidget(QLabel("baseline_sigma:"), 8, 0)
        self.fit_fluence_baseline_sigma = QLineEdit("1")
        self.fit_fluence_baseline_sigma.setValidator(QDoubleValidator())
        ftg.addWidget(self.fit_fluence_baseline_sigma, 8, 1)

        ftg.addWidget(QLabel("baseline_alpha:"), 9, 0)
        self.fit_fluence_baseline_alpha = QLineEdit("0.18")
        self.fit_fluence_baseline_alpha.setValidator(QDoubleValidator())
        ftg.addWidget(self.fit_fluence_baseline_alpha, 9, 1)

        ftg.addWidget(QLabel("baseline_mode:"), 10, 0)
        self.fit_fluence_baseline_mode = QComboBox()
        self.fit_fluence_baseline_mode.addItems(["errorbar", "band"])
        ftg.addWidget(self.fit_fluence_baseline_mode, 10, 1)

        self.fit_fluence_time_save = QCheckBox("save")
        self.fit_fluence_time_save.setChecked(True)
        ftg.addWidget(self.fit_fluence_time_save, 11, 0, 1, 2)

        ftg.addWidget(QLabel("save_fmt:"), 12, 0)
        self.fit_fluence_time_save_fmt = QComboBox()
        self.fit_fluence_time_save_fmt.addItems(["png", "pdf", "svg"])
        ftg.addWidget(self.fit_fluence_time_save_fmt, 12, 1)

        ftg.addWidget(QLabel("Save DPI:"), 13, 0)
        self.fit_fluence_time_save_dpi = QLineEdit("300")
        self.fit_fluence_time_save_dpi.setValidator(QDoubleValidator())
        ftg.addWidget(self.fit_fluence_time_save_dpi, 13, 1)

        self.fit_evolution_btn = QPushButton("Plot Evolution")
        self.fit_evolution_btn.clicked.connect(self._run_time_evolution)
        layout.addWidget(self.fit_evolution_btn)

    def _init_multi_experiment_widget(self, layout: QVBoxLayout):
        """Create and connect the controls for multi experiment widget."""
        self.fit_multi_widget = QWidget()
        fml = QVBoxLayout()
        self.fit_multi_widget.setLayout(fml)
        layout.addWidget(self.fit_multi_widget)

        series_group = QGroupBox("Experiment Type")
        sm = QHBoxLayout()
        series_group.setLayout(sm)
        fml.addWidget(series_group)

        sm.addWidget(QLabel("Experiment type:"))

        self.fit_multi_series_combo = QComboBox()
        self.fit_multi_series_combo.addItems(["Delay scan", "Fluence scan"])
        self.fit_multi_series_combo.currentIndexChanged.connect(
            self._refresh_series_widgets
        )
        sm.addWidget(self.fit_multi_series_combo)

        sm.addStretch()

        note = QLabel(
            "In multiple-experiment mode, only plotting/comparison workflows are exposed here.\n"
            "Peak fitting itself remains a single-experiment action."
        )
        note.setWordWrap(True)
        fml.addWidget(note)

        self.fit_multi_editor = MultiExperimentEditor(
            "Multiple-experiment Definitions",
            allow_merge=True,
            defaults=DEFAULT_MULTI_EXPERIMENTS_FIT,
            series_kind="delay",
        )
        fml.addWidget(self.fit_multi_editor)

        self.fit_multi_editor_fluence = MultiExperimentEditor(
            "Multiple-experiment Fluence Definitions",
            allow_merge=True,
            defaults=[],
            series_kind="fluence",
        )
        fml.addWidget(self.fit_multi_editor_fluence)

        self._init_multi_delay_group(fml)
        self._init_multi_fluence_group(fml)
        self._init_multi_actions(fml)

        fml.addStretch()
    
    def _init_multi_delay_group(self, layout: QVBoxLayout):
        """Create and connect the controls for multi delay group."""
        self.fit_multi_delay_group = QGroupBox("Multiple-experiment Delay Evolution")
        grid = QGridLayout()
        self.fit_multi_delay_group.setLayout(grid)
        layout.addWidget(self.fit_multi_delay_group)

        row = 0

        grid.addWidget(QLabel("peak:"), row, 0)
        self.fit_multi_peak = QLineEdit("110")
        grid.addWidget(self.fit_multi_peak, row, 1)
        row += 1

        grid.addWidget(QLabel("_property:"), row, 0)
        self.fit_multi_property = QComboBox()
        self.fit_multi_property.addItems(["hkl_pos", "hkl_fwhm", "hkl_i", "hkl_area"])
        grid.addWidget(self.fit_multi_property, row, 1)
        row += 1

        grid.addWidget(QLabel("out_csv_name:"), row, 0)
        self.fit_multi_out_csv_name = QLineEdit("peak_fits_delay.csv")
        grid.addWidget(self.fit_multi_out_csv_name, row, 1)
        row += 1

        grid.addWidget(QLabel("unit:"), row, 0)
        self.fit_multi_unit = QComboBox()
        self.fit_multi_unit.addItems(["ps", "fs", "ns", "µs", "ms", "s"])
        grid.addWidget(self.fit_multi_unit, row, 1)
        row += 1

        grid.addWidget(QLabel("phi_mode override:"), row, 0)
        self.fit_multi_phi_mode = QComboBox()
        self.fit_multi_phi_mode.addItems(["auto", "phi_avg", "separate_phi"])
        grid.addWidget(self.fit_multi_phi_mode, row, 1)
        row += 1

        grid.addWidget(QLabel("Azimuthal reduction:"), row, 0)
        self.fit_multi_phi_reduce = QComboBox()
        self.fit_multi_phi_reduce.addItems(["sum", "mean"])
        grid.addWidget(self.fit_multi_phi_reduce, row, 1)
        row += 1

        grid.addWidget(QLabel("phi_window:"), row, 0)
        self.fit_multi_phi_window = QLineEdit("Full")
        grid.addWidget(self.fit_multi_phi_window, row, 1)
        row += 1

        grid.addWidget(QLabel("title:"), row, 0)
        self.fit_multi_title = QLineEdit("")
        self.fit_multi_title.setPlaceholderText("Optional")
        grid.addWidget(self.fit_multi_title, row, 1)
        row += 1

        self.fit_multi_only_success = QCheckBox("only_success")
        self.fit_multi_only_success.setChecked(True)
        grid.addWidget(self.fit_multi_only_success, row, 0, 1, 2)
        row += 1

        self.fit_multi_include_reference = QCheckBox("include_reference")
        self.fit_multi_include_reference.setChecked(True)
        grid.addWidget(self.fit_multi_include_reference, row, 0, 1, 2)
        row += 1

        self.fit_multi_as_lines = QCheckBox("as_lines")
        grid.addWidget(self.fit_multi_as_lines, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("delay_offset override:"), row, 0)
        self.fit_multi_delay_offset = QLineEdit("")
        self.fit_multi_delay_offset.setPlaceholderText("Optional global override")
        grid.addWidget(self.fit_multi_delay_offset, row, 1)
        row += 1

        self.fit_multi_show_baseline_sigma = QCheckBox("show_baseline_sigma")
        self.fit_multi_show_baseline_sigma.setChecked(True)
        grid.addWidget(self.fit_multi_show_baseline_sigma, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("baseline_sigma:"), row, 0)
        self.fit_multi_baseline_sigma = QLineEdit("1")
        self.fit_multi_baseline_sigma.setValidator(QDoubleValidator())
        grid.addWidget(self.fit_multi_baseline_sigma, row, 1)
        row += 1

        grid.addWidget(QLabel("baseline_alpha:"), row, 0)
        self.fit_multi_baseline_alpha = QLineEdit("0.18")
        self.fit_multi_baseline_alpha.setValidator(QDoubleValidator())
        grid.addWidget(self.fit_multi_baseline_alpha, row, 1)
        row += 1

        grid.addWidget(QLabel("baseline_mode:"), row, 0)
        self.fit_multi_baseline_mode = QComboBox()
        self.fit_multi_baseline_mode.addItems(["errorbar", "band"])
        grid.addWidget(self.fit_multi_baseline_mode, row, 1)
        row += 1

        self.fit_multi_norm_min_max = QCheckBox("norm_min_max")
        grid.addWidget(self.fit_multi_norm_min_max, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("delay_for_norm_max override:"), row, 0)
        self.fit_multi_delay_for_norm_max = QLineEdit("")
        self.fit_multi_delay_for_norm_max.setPlaceholderText("Optional global override")
        self.fit_multi_delay_for_norm_max.setValidator(QDoubleValidator())
        grid.addWidget(self.fit_multi_delay_for_norm_max, row, 1)
        row += 1

        grid.addWidget(QLabel("cmap:"), row, 0)
        self.fit_multi_cmap = QLineEdit("jet")
        grid.addWidget(self.fit_multi_cmap, row, 1)
        row += 1

        self.fit_multi_save = QCheckBox("save")
        self.fit_multi_save.setChecked(True)
        grid.addWidget(self.fit_multi_save, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("save_fmt:"), row, 0)
        self.fit_multi_save_fmt = QComboBox()
        self.fit_multi_save_fmt.addItems(["png", "pdf", "svg"])
        grid.addWidget(self.fit_multi_save_fmt, row, 1)
        row += 1

        grid.addWidget(QLabel("Save DPI:"), row, 0)
        self.fit_multi_save_dpi = QLineEdit("300")
        self.fit_multi_save_dpi.setValidator(QDoubleValidator())
        grid.addWidget(self.fit_multi_save_dpi, row, 1)
        row += 1

        self.fit_multi_save_overwrite = QCheckBox("save_overwrite")
        self.fit_multi_save_overwrite.setChecked(True)
        grid.addWidget(self.fit_multi_save_overwrite, row, 0, 1, 2)


    def _init_multi_fluence_group(self, layout: QVBoxLayout):
        """Create and connect the controls for multi fluence group."""
        self.fit_multi_fluence_group = QGroupBox("Multiple-experiment Fluence Evolution")
        grid = QGridLayout()
        self.fit_multi_fluence_group.setLayout(grid)
        layout.addWidget(self.fit_multi_fluence_group)

        row = 0

        grid.addWidget(QLabel("peak:"), row, 0)
        self.fit_multi_fluence_peak = QLineEdit("110")
        grid.addWidget(self.fit_multi_fluence_peak, row, 1)
        row += 1

        grid.addWidget(QLabel("_property:"), row, 0)
        self.fit_multi_fluence_property = QComboBox()
        self.fit_multi_fluence_property.addItems(
            ["hkl_pos", "hkl_fwhm", "hkl_i", "hkl_area"]
        )
        grid.addWidget(self.fit_multi_fluence_property, row, 1)
        row += 1

        grid.addWidget(QLabel("group_by:"), row, 0)
        self.fit_multi_fluence_group_by = QComboBox()
        self.fit_multi_fluence_group_by.addItems(["azim_range_str", "phi_label"])
        grid.addWidget(self.fit_multi_fluence_group_by, row, 1)
        row += 1

        grid.addWidget(QLabel("group:"), row, 0)
        self.fit_multi_fluence_group_name = QLineEdit("Full")
        grid.addWidget(self.fit_multi_fluence_group_name, row, 1)
        row += 1

        grid.addWidget(QLabel("fluence_unit:"), row, 0)
        self.fit_multi_fluence_unit = QLineEdit("mJ/cm$^2$")
        grid.addWidget(self.fit_multi_fluence_unit, row, 1)
        row += 1

        grid.addWidget(QLabel("title:"), row, 0)
        self.fit_multi_fluence_title = QLineEdit("")
        self.fit_multi_fluence_title.setPlaceholderText("Optional")
        grid.addWidget(self.fit_multi_fluence_title, row, 1)
        row += 1

        self.fit_multi_fluence_only_success = QCheckBox("only_success")
        self.fit_multi_fluence_only_success.setChecked(True)
        grid.addWidget(self.fit_multi_fluence_only_success, row, 0, 1, 2)
        row += 1

        self.fit_multi_fluence_include_reference = QCheckBox("include_reference")
        self.fit_multi_fluence_include_reference.setChecked(True)
        grid.addWidget(self.fit_multi_fluence_include_reference, row, 0, 1, 2)
        row += 1

        self.fit_multi_fluence_as_lines = QCheckBox("as_lines")
        grid.addWidget(self.fit_multi_fluence_as_lines, row, 0, 1, 2)
        row += 1

        self.fit_multi_fluence_show_baseline_sigma = QCheckBox("show_baseline_sigma")
        self.fit_multi_fluence_show_baseline_sigma.setChecked(True)
        grid.addWidget(self.fit_multi_fluence_show_baseline_sigma, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("baseline_sigma_scale:"), row, 0)
        self.fit_multi_fluence_baseline_sigma = QLineEdit("1")
        self.fit_multi_fluence_baseline_sigma.setValidator(QDoubleValidator())
        grid.addWidget(self.fit_multi_fluence_baseline_sigma, row, 1)
        row += 1

        grid.addWidget(QLabel("baseline_mode:"), row, 0)
        self.fit_multi_fluence_baseline_mode = QComboBox()
        self.fit_multi_fluence_baseline_mode.addItems(["errorbar", "band"])
        grid.addWidget(self.fit_multi_fluence_baseline_mode, row, 1)
        row += 1

        self.fit_multi_fluence_save = QCheckBox("save")
        self.fit_multi_fluence_save.setChecked(True)
        grid.addWidget(self.fit_multi_fluence_save, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("save_fmt:"), row, 0)
        self.fit_multi_fluence_save_fmt = QComboBox()
        self.fit_multi_fluence_save_fmt.addItems(["png", "pdf", "svg"])
        grid.addWidget(self.fit_multi_fluence_save_fmt, row, 1)
        row += 1

        grid.addWidget(QLabel("Save DPI:"), row, 0)
        self.fit_multi_fluence_save_dpi = QLineEdit("300")
        self.fit_multi_fluence_save_dpi.setValidator(QDoubleValidator())
        grid.addWidget(self.fit_multi_fluence_save_dpi, row, 1)
        row += 1

        self.fit_multi_fluence_save_overwrite = QCheckBox("save_overwrite")
        self.fit_multi_fluence_save_overwrite.setChecked(True)
        grid.addWidget(self.fit_multi_fluence_save_overwrite, row, 0, 1, 2)


    def _init_multi_actions(self, layout: QVBoxLayout):
        """Create and connect buttons for multi-experiment plotting operations."""
        self.fit_multi_evolution_btn = QPushButton("Plot Multi Evolution")
        self.fit_multi_evolution_btn.clicked.connect(self._run_time_evolution_multi)
        layout.addWidget(self.fit_multi_evolution_btn)



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
        return self.integration_service.parse_azim_offset_deg(
            getattr(self.state, "azim_offset_deg", "-90.0")
        )

    def _polarization_factor(self):
        """Return the enabled polarization factor, or None when correction is disabled."""
        if not getattr(self.state, "polarization_enabled", True):
            return None
        return self.integration_service.parse_polarization_factor(
            getattr(self.state, "polarization_factor", 0.99)
        )

    def _fit_peak_specs(self):
        """Parse and validate the editable peak-model specification mapping."""
        value = parse_python_literal(self.fit_peak_specs.toPlainText())

        if not isinstance(value, dict) or not value:
            raise ValueError("Fitting peak specs must be a non-empty dict.")

        return value

    def _out_csv_name(self):
        """Return the explicit or series-specific default fitting CSV filename."""
        return self.fitting_service.normalized_out_csv_name(
            self.fit_out_csv_name.text(),
            self.fit_series_combo.currentText(),
        )

    def _fit_base_kwargs(self):
        """Validate shared fitting fields and assemble backend keyword arguments."""
        kwargs = self.integration_service.build_experiment_kwargs(
            self.fit_single_metadata.values()
        )
        kwargs.update(
            self.integration_service.build_poni_mask_kwargs(
                poni_path=self._poni_path(),
                mask_edf_path=self._mask_path(),
            )
        )
        kwargs.update(
            peak_specs=self._fit_peak_specs(),
            azim_windows=parse_windows(self.fit_azim_windows.toPlainText()),
            azim_offset_deg=self._azim_offset_deg(),
            polarization_factor=self._polarization_factor(),
            npt=parse_int_like(self.fit_npt.text(), name="npt"),
            normalize_xy=self.fit_normalize_xy.isChecked(),
            q_norm_range=self.integration_service.parse_range_tuple(
                self.fit_q_norm_range.text(),
                name="q_norm_range",
            ),
            compute_if_missing=self.fit_compute_if_missing.isChecked(),
            overwrite_xy=self.fit_overwrite_xy.isChecked(),
            default_eta=parse_float_like(
                self.fit_default_eta.text(),
                name="default_eta",
            ),
            phi_mode=self.fit_phi_mode.currentText(),
            phi_reduce=self.fit_phi_reduce.currentText(),
            paths=self._build_analysis_paths(),
        )
        return kwargs

    def _ensure_id09_fluence_cache_if_needed(
        self,
        *,
        kwargs: dict,
        delay_fs: int,
        fluences,
        azim_windows,
        overwrite: bool,
    ):
        """Create the synthetic ID09 fluence cache required by downstream fitting."""
        if self.state.facility != "ID09" or not self.fit_compute_if_missing.isChecked():
            return

        self.integration_service.ensure_id09_fluence_cache(
            sample_name=kwargs["sample_name"],
            temperature_K=kwargs["temperature_K"],
            excitation_wl_nm=kwargs["excitation_wl_nm"],
            delay_fs=delay_fs,
            time_window_fs=kwargs["time_window_fs"],
            fluences_mJ_cm2=fluences,
            azim_windows=azim_windows,
            copy_2d_image=False,
            overwrite=overwrite,
            paths=kwargs["paths"],
        )

    def _run_delay_peak_fitting(self):
        """Parse the delay peak fitting controls, invoke the service workflow, and log completion or errors."""
        try:
            def error_summary(traceback_text):
                """Extract the final informative message from a captured worker traceback."""
                lines = [
                    line.strip()
                    for line in str(traceback_text).splitlines()
                    if line.strip()
                ]
                return lines[-1] if lines else "unknown error"

            kwargs = self._fit_base_kwargs()

            out_csv_name = self.fit_out_csv_name.text().strip()
            if out_csv_name in ("", "peak_fits_delay.csv", "peak_fits_fluence.csv"):
                out_csv_name = self._default_fit_csv_name_for_series(
                    self.fit_series_combo.currentText()
                )

            if self.fit_series_combo.currentText() == "Delay scan":
                kwargs.update(
                    delays_fs=self.integration_service.parse_delays_value(
                        self.fit_delays.text()
                    ),
                    ref_type=self.fit_ref_type.currentText(),
                    ref_value=self.integration_service.parse_ref_value(
                        self.fit_ref_value.text()
                    ),
                    ref_values_mode=self.fit_ref_values_mode.currentText(),
                    include_reference_in_output=self.fit_include_reference.isChecked(),
                    out_csv_name=out_csv_name,
                    show_fit_figures=self.fit_show_fit_figures.isChecked(),
                    save_fit_figures=self.fit_save_fit_figures.isChecked(),
                    fit_figures_format=self.fit_fig_format.currentText(),
                    fit_figures_dpi=parse_int_like(
                        self.fit_fig_dpi.text(),
                        name="fit_figures_dpi",
                    ),
                    plot_only_success=self.fit_plot_only_success.isChecked(),
                    fit_oversample=parse_int_like(
                        self.fit_oversample.text(),
                        name="fit_oversample",
                    ),
                )

                def task():
                    """Execute the validated backend operation inside the background worker thread."""
                    return self.fitting_service.run_delay_peak_fitting(**kwargs)

                def success(result):
                    """Summarize the completed background operation and update the GUI log."""
                    _df, csv_path = result
                    self.log(f"Peak fitting finished. CSV: {csv_path}")

                run_task_with_output_dialog(
                    self,
                    "Peak Fitting",
                    task,
                    on_success=success,
                    on_error=lambda tb: self.log(
                        f"Peak Fitting Error: {error_summary(tb)}"
                    ),
                )

            else:
                delay_fs = parse_int_like(
                    self.fit_fluence_delay_fs.text(),
                    name="delay_fs",
                )
                fluences = self.integration_service.parse_fluences_value(
                    self.fit_fluences.text()
                )
                azim_windows = parse_windows(self.fit_azim_windows.toPlainText())

                kwargs.pop("fluence_mJ_cm2", None)

                ensure_kwargs = None
                if self.state.facility == "ID09" and self.fit_compute_if_missing.isChecked():
                    ensure_kwargs = dict(
                        sample_name=kwargs["sample_name"],
                        temperature_K=kwargs["temperature_K"],
                        excitation_wl_nm=kwargs["excitation_wl_nm"],
                        delay_fs=delay_fs,
                        time_window_fs=kwargs["time_window_fs"],
                        fluences_mJ_cm2=fluences,
                        azim_windows=azim_windows,
                        copy_2d_image=False,
                        overwrite=self.fit_overwrite_xy.isChecked(),
                        paths=self._build_analysis_paths(),
                    )

                kwargs.update(
                    delay_fs=delay_fs,
                    fluences_mJ_cm2=fluences,
                    ref_type=self.fit_fluence_ref_type.currentText(),
                    ref_value=self.integration_service.parse_ref_value(
                        self.fit_fluence_ref_value.text()
                    ),
                    ref_values_mode=self.fit_fluence_ref_values_mode.currentText(),
                    include_reference_in_output=self.fit_include_reference.isChecked(),
                    out_csv_name=out_csv_name,
                    show_fit_figures=self.fit_show_fit_figures.isChecked(),
                    save_fit_figures=self.fit_save_fit_figures.isChecked(),
                    fit_figures_format=self.fit_fig_format.currentText(),
                    fit_figures_dpi=parse_int_like(
                        self.fit_fig_dpi.text(),
                        name="fit_figures_dpi",
                    ),
                    plot_only_success=self.fit_plot_only_success.isChecked(),
                    fit_oversample=parse_int_like(
                        self.fit_oversample.text(),
                        name="fit_oversample",
                    ),
                )

                def task():
                    """Execute the validated backend operation inside the background worker thread."""
                    if ensure_kwargs is not None:
                        self.integration_service.ensure_id09_fluence_cache(**ensure_kwargs)
                    return self.fitting_service.run_fluence_peak_fitting(**kwargs)

                def success(result):
                    """Summarize the completed background operation and update the GUI log."""
                    _df, csv_path = result
                    self.log(f"Peak fitting finished. CSV: {csv_path}")

                run_task_with_output_dialog(
                    self,
                    "Peak Fitting",
                    task,
                    on_success=success,
                    on_error=lambda tb: self.log(
                        f"Peak Fitting Error: {error_summary(tb)}"
                    ),
                )

        except Exception as exc:
            self.log(f"Peak Fitting Error: {exc}")


    def _run_fit_overlay(self):
        """Parse the fit overlay controls, invoke the service workflow, and log completion or errors."""
        try:
            kwargs = self._fit_base_kwargs()
            out_csv_name = self._out_csv_name()

            if self.fit_series_combo.currentText() == "Delay scan":
                kwargs.update(
                    peak=self.fit_overlay_peak.text().strip(),
                    delay_fs=parse_int_like(
                        self.fit_overlay_delay.text(),
                        name="delay_fs",
                    ),
                    is_reference=self.fit_overlay_is_reference.isChecked(),
                    reference_index=parse_optional_int_like(
                        self.fit_overlay_reference_index.text()
                    ),
                    group=parse_python_literal(
                        self.fit_overlay_group.text(),
                        empty=None,
                    ),
                    out_csv_name=out_csv_name,
                    ensure_csv=self.fit_overlay_ensure_csv.isChecked(),
                    delays_fs=self.integration_service.parse_delays_value(
                        self.fit_delays.text()
                    ),
                    ref_type=self.fit_ref_type.currentText(),
                    ref_value=self.integration_service.parse_ref_value(
                        self.fit_ref_value.text()
                    ),
                    ref_values_mode=self.fit_ref_values_mode.currentText(),
                    show=self.fit_overlay_show.isChecked(),
                    save=self.fit_overlay_save.isChecked(),
                    save_format=self.fit_fig_format.currentText(),
                    save_dpi=parse_int_like(self.fit_fig_dpi.text(), name="save_dpi"),
                    fit_oversample=parse_int_like(
                        self.fit_oversample.text(),
                        name="fit_oversample",
                    ),
                    only_success=self.fit_plot_only_success.isChecked(),
                )
                out = self.fitting_service.plot_fit_overlay_from_csv(**kwargs)

            else:
                delay_fs = parse_int_like(
                    self.fit_fluence_delay_fs.text(),
                    name="delay_fs",
                )
                fluences = self.integration_service.parse_fluences_value(
                    self.fit_fluences.text()
                )
                azim_windows = parse_windows(self.fit_azim_windows.toPlainText())

                kwargs.pop("fluence_mJ_cm2", None)

                self._ensure_id09_fluence_cache_if_needed(
                    kwargs=kwargs,
                    delay_fs=delay_fs,
                    fluences=fluences,
                    azim_windows=azim_windows,
                    overwrite=self.fit_overwrite_xy.isChecked(),
                )

                kwargs.update(
                    peak=self.fit_fluence_overlay_peak.text().strip(),
                    delay_fs=delay_fs,
                    fluence_mJ_cm2=parse_optional_float_like(
                        self.fit_fluence_overlay_fluence.text()
                    ),
                    is_reference=self.fit_fluence_overlay_is_reference.isChecked(),
                    reference_index=parse_optional_int_like(
                        self.fit_fluence_overlay_reference_index.text()
                    ),
                    group=parse_python_literal(
                        self.fit_fluence_overlay_group_name.text(),
                        empty=None,
                    ),
                    out_csv_name=out_csv_name,
                    ensure_csv=self.fit_fluence_overlay_ensure_csv.isChecked(),
                    fluences_mJ_cm2=fluences,
                    ref_type=self.fit_fluence_ref_type.currentText(),
                    ref_value=self.integration_service.parse_ref_value(
                        self.fit_fluence_ref_value.text()
                    ),
                    ref_values_mode=self.fit_fluence_ref_values_mode.currentText(),
                    show=self.fit_fluence_overlay_show.isChecked(),
                    save=self.fit_fluence_overlay_save.isChecked(),
                    save_format=self.fit_fig_format.currentText(),
                    save_dpi=parse_int_like(self.fit_fig_dpi.text(), name="save_dpi"),
                    fit_oversample=parse_int_like(
                        self.fit_oversample.text(),
                        name="fit_oversample",
                    ),
                    only_success=self.fit_plot_only_success.isChecked(),
                )
                out = self.fitting_service.plot_fit_overlay_from_csv_fluence(**kwargs)

            saved_path = out.get("saved_path", None) if isinstance(out, dict) else None
            self.log(f"Fit overlay finished. Saved path: {saved_path}")

        except Exception as exc:
            self.log(f"Fit Overlay Error: {exc}")

    def _run_time_evolution(self):
        """Parse the time evolution controls, invoke the service workflow, and log completion or errors."""
        try:
            kwargs = self.integration_service.build_experiment_kwargs(
                self.fit_single_metadata.values()
            )
            out_csv_name = self._out_csv_name()

            if self.fit_series_combo.currentText() == "Delay scan":
                kwargs.update(
                    peak=self.fit_time_peak.text().strip(),
                    _property=self.fit_property.currentText(),
                    out_csv_name=out_csv_name,
                    unit=self.fit_time_unit.currentText(),
                    groups=parse_groups(self.fit_groups.text()),
                    title=self.fit_time_title.text().strip() or None,
                    phi_mode=self.fit_phi_mode.currentText(),
                    phi_reduce=self.fit_phi_reduce.currentText(),
                    as_lines=self.fit_as_lines.isChecked(),
                    delay_offset=parse_float_like(
                        self.fit_delay_offset.text(),
                        name="delay_offset",
                    ),
                    show_baseline_sigma=self.fit_show_baseline_sigma.isChecked(),
                    baseline_sigma=parse_float_like(
                        self.fit_baseline_sigma.text(),
                        name="baseline_sigma",
                    ),
                    baseline_alpha=parse_float_like(
                        self.fit_baseline_alpha.text(),
                        name="baseline_alpha",
                    ),
                    baseline_mode=self.fit_baseline_mode.currentText(),
                    save=self.fit_time_save.isChecked(),
                    save_fmt=self.fit_time_save_fmt.currentText(),
                    save_dpi=parse_int_like(
                        self.fit_time_save_dpi.text(),
                        name="save_dpi",
                    ),
                    paths=self._build_analysis_paths(),
                )
                self.fitting_service.plot_time_evolution(**kwargs)

            else:
                kwargs.pop("fluence_mJ_cm2", None)
                kwargs.update(
                    delay_fs=parse_int_like(
                        self.fit_fluence_delay_fs.text(),
                        name="delay_fs",
                    ),
                    peak=self.fit_fluence_time_peak.text().strip(),
                    _property=self.fit_fluence_property.currentText(),
                    out_csv_name=out_csv_name,
                    unit=self.fit_fluence_unit.text().strip() or "mJ/cm$^2$",
                    groups=parse_groups(self.fit_fluence_groups.text()),
                    title=self.fit_fluence_time_title.text().strip() or None,
                    phi_mode=self.fit_phi_mode.currentText(),
                    phi_reduce=self.fit_phi_reduce.currentText(),
                    as_lines=self.fit_fluence_as_lines.isChecked(),
                    fluence_offset=parse_float_like(
                        self.fit_fluence_offset.text(),
                        name="fluence_offset",
                    ),
                    show_baseline_sigma=self.fit_fluence_show_baseline_sigma.isChecked(),
                    baseline_sigma=parse_float_like(
                        self.fit_fluence_baseline_sigma.text(),
                        name="baseline_sigma",
                    ),
                    baseline_alpha=parse_float_like(
                        self.fit_fluence_baseline_alpha.text(),
                        name="baseline_alpha",
                    ),
                    baseline_mode=self.fit_fluence_baseline_mode.currentText(),
                    save=self.fit_fluence_time_save.isChecked(),
                    save_fmt=self.fit_fluence_time_save_fmt.currentText(),
                    save_dpi=parse_int_like(
                        self.fit_fluence_time_save_dpi.text(),
                        name="save_dpi",
                    ),
                    paths=self._build_analysis_paths(),
                )
                self.fitting_service.plot_fluence_evolution(**kwargs)

            self.log("Evolution plot finished.")

        except Exception as exc:
            self.log(f"Time Evolution Error: {exc}")

    def _run_time_evolution_multi(self):
        """Plot delay or fluence evolution across the configured experiments."""
        try:
            if self.fit_multi_series_combo.currentText() == "Delay scan":
                phi_mode = self.fit_multi_phi_mode.currentText()

                if phi_mode == "auto":
                    phi_mode = None

                phi_window_text = self.fit_multi_phi_window.text().strip()
                phi_window = (
                    None
                    if phi_window_text == ""
                    else parse_python_literal(phi_window_text)
                )

                delay_offset = parse_optional_float_like(
                    self.fit_multi_delay_offset.text()
                )
                delay_for_norm_max = parse_optional_float_like(
                    self.fit_multi_delay_for_norm_max.text()
                )

                kwargs = dict(
                    experiments=self.fit_multi_editor.get_experiments(),
                    peak=self.fit_multi_peak.text().strip(),
                    _property=self.fit_multi_property.currentText(),
                    out_csv_name=self.fit_multi_out_csv_name.text().strip()
                    or "peak_fits_delay.csv",
                    unit=self.fit_multi_unit.currentText(),
                    phi_mode=phi_mode,
                    phi_reduce=self.fit_multi_phi_reduce.currentText(),
                    phi_window=phi_window,
                    title=self.fit_multi_title.text().strip() or None,
                    only_success=self.fit_multi_only_success.isChecked(),
                    include_reference=self.fit_multi_include_reference.isChecked(),
                    as_lines=self.fit_multi_as_lines.isChecked(),
                    delay_offset=delay_offset,
                    show_baseline_sigma=self.fit_multi_show_baseline_sigma.isChecked(),
                    baseline_sigma=parse_float_like(
                        self.fit_multi_baseline_sigma.text(),
                        name="baseline_sigma",
                    ),
                    baseline_alpha=parse_float_like(
                        self.fit_multi_baseline_alpha.text(),
                        name="baseline_alpha",
                    ),
                    baseline_mode=self.fit_multi_baseline_mode.currentText(),
                    norm_min_max=self.fit_multi_norm_min_max.isChecked(),
                    delay_for_norm_max=delay_for_norm_max,
                    cmap=self.fit_multi_cmap.text().strip() or None,
                    save=self.fit_multi_save.isChecked(),
                    save_fmt=self.fit_multi_save_fmt.currentText(),
                    save_dpi=parse_int_like(
                        self.fit_multi_save_dpi.text(),
                        name="save_dpi",
                    ),
                    save_overwrite=self.fit_multi_save_overwrite.isChecked(),
                    paths=self._build_analysis_paths(),
                )
                out = self.fitting_service.plot_time_evolution_multi(**kwargs)

            else:
                kwargs = dict(
                    experiments=self.fit_multi_editor_fluence.get_experiments(),
                    peak=self.fit_multi_fluence_peak.text().strip(),
                    prop=self.fit_multi_fluence_property.currentText(),
                    group_by=self.fit_multi_fluence_group_by.currentText(),
                    group=self.fit_multi_fluence_group_name.text().strip() or "Full",
                    fluence_unit=self.fit_multi_fluence_unit.text().strip()
                    or "mJ/cm$^2$",
                    title=self.fit_multi_fluence_title.text().strip() or None,
                    only_success=self.fit_multi_fluence_only_success.isChecked(),
                    include_reference=self.fit_multi_fluence_include_reference.isChecked(),
                    as_lines=self.fit_multi_fluence_as_lines.isChecked(),
                    show_baseline_sigma=self.fit_multi_fluence_show_baseline_sigma.isChecked(),
                    baseline_mode=self.fit_multi_fluence_baseline_mode.currentText(),
                    baseline_sigma_scale=parse_float_like(
                        self.fit_multi_fluence_baseline_sigma.text(),
                        name="baseline_sigma_scale",
                    ),
                    save=self.fit_multi_fluence_save.isChecked(),
                    save_format=self.fit_multi_fluence_save_fmt.currentText(),
                    save_dpi=parse_int_like(
                        self.fit_multi_fluence_save_dpi.text(),
                        name="save_dpi",
                    ),
                    save_overwrite=self.fit_multi_fluence_save_overwrite.isChecked(),
                    paths=self._build_analysis_paths(),
                )
                out = self.fitting_service.plot_fluence_evolution_multi(**kwargs)

            saved_path = out.get("saved_path", None) if isinstance(out, dict) else None
            self.log(f"Multiple-experiment evolution finished. Saved path: {saved_path}")

        except Exception as exc:
            self.log(f"Multi Time Evolution Error: {exc}")


    def set_facility(self, facility: str):
        """Store the active facility and refresh facility-dependent controls."""
        self.state.facility = facility
        self._refresh_series_widgets()

    def _refresh_mode_widgets(self):
        """Update mode-dependent widget visibility and synchronize related defaults."""
        single_mode = self.fit_mode_combo.currentText() == "Single experiment"

        self.fit_single_widget.setVisible(single_mode)
        self.fit_multi_widget.setVisible(not single_mode)

    def _refresh_series_widgets(self):
        """Update series widgets visibility and defaults from the active mode."""
        delay_mode_single = self.fit_series_combo.currentText() == "Delay scan"
        is_id09 = self.state.facility == "ID09"

        self.fit_delay_selector_group.setVisible(delay_mode_single)
        self.fit_fluence_selector_group.setVisible(not delay_mode_single)
        self.fit_delay_overlay_group.setVisible(delay_mode_single)
        self.fit_fluence_overlay_group.setVisible(not delay_mode_single)
        self.fit_delay_time_group.setVisible(delay_mode_single)
        self.fit_fluence_time_group.setVisible(not delay_mode_single)

        self.fit_single_metadata.set_field_visible(
            "fluence_mJ_cm2",
            delay_mode_single,
        )
        self.fit_single_metadata.set_id09_visible(is_id09 and delay_mode_single)

        self._sync_fit_out_csv_name_default()

        delay_mode_multi = self.fit_multi_series_combo.currentText() == "Delay scan"

        self.fit_multi_editor.setVisible(delay_mode_multi)
        self.fit_multi_editor_fluence.setVisible(not delay_mode_multi)
        self.fit_multi_delay_group.setVisible(delay_mode_multi)
        self.fit_multi_fluence_group.setVisible(not delay_mode_multi)

    def _default_fit_csv_name_for_series(self, series_text=None):
        """Return the default fit CSV name for series."""
        series = (
            self.fit_series_combo.currentText()
            if series_text is None
            else str(series_text)
        )
        return (
            "peak_fits_fluence.csv"
            if str(series).strip() == "Fluence scan"
            else "peak_fits_delay.csv"
        )

    def _sync_fit_out_csv_name_default(self):
        """Synchronize the default fitting CSV filename with the selected series."""
        current = self.fit_out_csv_name.text().strip()

        if current in ("", "peak_fits_delay.csv", "peak_fits_fluence.csv"):
            self.fit_out_csv_name.setText(self._default_fit_csv_name_for_series())
