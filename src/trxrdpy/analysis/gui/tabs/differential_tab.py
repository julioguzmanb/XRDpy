"""
Differential tab for the analysis GUI.

This reproduces the legacy Differential tab layout while keeping backend actions
separated from the main window.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtGui import QDoubleValidator
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
    DEFAULT_DIFF_PEAK_SPECS,
    DEFAULT_MULTI_EXPERIMENTS_DIFF,
)
from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.widgets import ExperimentMetadataWidget, MultiExperimentEditor
from trxrdpy.analysis.gui.services import (
    DifferentialService,
    IntegrationService,
    PathService,
)
from trxrdpy.analysis.gui.utils import (
    pretty_literal,
    parse_float_like,
    parse_int_like,
    parse_python_literal,
)

class DifferentialTab(QWidget):
    """Configure single- and multi-experiment differential-analysis workflows."""

    def __init__(
        self,
        state: AnalysisGuiState,
        path_service: PathService,
        integration_service: IntegrationService,
        differential_service: DifferentialService,
        log: Optional[Callable[[str], None]] = None,
        parent=None,
    ):
        """Initialize ``DifferentialTab``, bind shared state and services, and create its controls."""
        super().__init__(parent)

        self.state = state
        self.path_service = path_service
        self.integration_service = integration_service
        self.differential_service = differential_service
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
        """Create scroll layout."""
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
        """Create the mode group controls."""
        mode_group = QGroupBox("Analysis Mode")
        ml = QHBoxLayout()
        mode_group.setLayout(ml)
        layout.addWidget(mode_group)

        ml.addWidget(QLabel("Differential mode:"))

        self.diff_mode_combo = QComboBox()
        self.diff_mode_combo.addItems(["Single experiment", "Multiple experiments"])
        self.diff_mode_combo.currentIndexChanged.connect(self._refresh_mode_widgets)
        ml.addWidget(self.diff_mode_combo)

        ml.addStretch()

    def _init_single_experiment_widget(self, layout: QVBoxLayout):
        """Create and connect the controls for single experiment widget."""
        self.diff_single_widget = QWidget()
        dsl = QVBoxLayout()
        self.diff_single_widget.setLayout(dsl)
        layout.addWidget(self.diff_single_widget)

        series_group = QGroupBox("Experiment Type")
        sgl = QHBoxLayout()
        series_group.setLayout(sgl)
        dsl.addWidget(series_group)

        sgl.addWidget(QLabel("Experiment type:"))

        self.diff_series_combo = QComboBox()
        self.diff_series_combo.addItems(["Delay scan", "Fluence scan"])
        self.diff_series_combo.currentIndexChanged.connect(self._refresh_series_widgets)
        sgl.addWidget(self.diff_series_combo)

        sgl.addStretch()

        self.diff_single_metadata = ExperimentMetadataWidget(
            title="Experiment Metadata",
            include_id09=True,
        )
        dsl.addWidget(self.diff_single_metadata)

        self._init_single_delay_group(dsl)
        self._init_single_fluence_group(dsl)
        self._init_single_delay_integral_group(dsl)
        self._init_single_fluence_integral_group(dsl)
        self._init_single_fft_group(dsl)
        self._init_single_runtime_group(dsl)
        self._init_single_actions(dsl)

        dsl.addStretch()

    def _init_single_delay_group(self, layout: QVBoxLayout):
        """Create and connect the controls for single delay group."""
        self.diff_delay_primary_group = QGroupBox("Single-experiment Delay Analysis")
        grid = QGridLayout()
        self.diff_delay_primary_group.setLayout(grid)
        layout.addWidget(self.diff_delay_primary_group)

        row = 0

        grid.addWidget(QLabel("delays_fs:"), row, 0)
        self.diff_delays = QLineEdit("all")
        grid.addWidget(self.diff_delays, row, 1)
        row += 1

        grid.addWidget(QLabel("Reference type:"), row, 0)
        self.diff_ref_type = QComboBox()
        self.diff_ref_type.addItems(["dark", "delay"])
        grid.addWidget(self.diff_ref_type, row, 1)
        row += 1

        grid.addWidget(QLabel("Reference value:"), row, 0)
        self.diff_ref_value = QLineEdit("[1466556]")
        grid.addWidget(self.diff_ref_value, row, 1)
        row += 1

        grid.addWidget(QLabel("Azimuthal window [deg]:"), row, 0)
        self.diff_azim_window = QLineEdit("(-90, 90)")
        grid.addWidget(self.diff_azim_window, row, 1)
        row += 1

        grid.addWidget(QLabel("peak:"), row, 0)
        self.diff_peak = QLineEdit("110")
        grid.addWidget(self.diff_peak, row, 1)
        row += 1

        grid.addWidget(QLabel("Peak definitions:"), row, 0)
        self.diff_peak_specs = QPlainTextEdit(pretty_literal(DEFAULT_DIFF_PEAK_SPECS))
        grid.addWidget(self.diff_peak_specs, row, 1)

        grid.setRowMinimumHeight(row, 100)

    def _init_single_fluence_group(self, layout: QVBoxLayout):
        """Create and connect the controls for single fluence group."""
        self.diff_fluence_primary_group = QGroupBox("Single-experiment Fluence Analysis")
        fg = QGridLayout()
        self.diff_fluence_primary_group.setLayout(fg)
        layout.addWidget(self.diff_fluence_primary_group)

        row = 0

        fg.addWidget(QLabel("Delay [fs]:"), row, 0)
        self.diff_fluence_delay_fs = QLineEdit("0")
        self.diff_fluence_delay_fs.setValidator(QDoubleValidator())
        fg.addWidget(self.diff_fluence_delay_fs, row, 1)
        row += 1

        fg.addWidget(QLabel("Fluences [mJ/cm²]:"), row, 0)
        self.diff_fluences = QLineEdit("all")
        fg.addWidget(self.diff_fluences, row, 1)
        row += 1

        fg.addWidget(QLabel("Reference type:"), row, 0)
        self.diff_fluence_ref_type = QComboBox()
        self.diff_fluence_ref_type.addItems(["dark", "fluence"])
        fg.addWidget(self.diff_fluence_ref_type, row, 1)
        row += 1

        fg.addWidget(QLabel("Reference value:"), row, 0)
        self.diff_fluence_ref_value = QLineEdit("[1466556]")
        fg.addWidget(self.diff_fluence_ref_value, row, 1)
        row += 1

        fg.addWidget(QLabel("Azimuthal window [deg]:"), row, 0)
        self.diff_fluence_azim_window = QLineEdit("(-90, 90)")
        fg.addWidget(self.diff_fluence_azim_window, row, 1)
        row += 1

        fg.addWidget(QLabel("peak:"), row, 0)
        self.diff_fluence_peak = QLineEdit("110")
        fg.addWidget(self.diff_fluence_peak, row, 1)
        row += 1

        fg.addWidget(QLabel("Peak definitions:"), row, 0)
        self.diff_fluence_peak_specs = QPlainTextEdit(
            pretty_literal(DEFAULT_DIFF_PEAK_SPECS)
        )
        fg.addWidget(self.diff_fluence_peak_specs, row, 1)

        fg.setRowMinimumHeight(row, 100)

    def _init_single_delay_integral_group(self, layout: QVBoxLayout):
        """Create and connect the controls for single delay integral group."""
        self.diff_delay_integral_group = QGroupBox("Integral Plot Settings")
        ig = QGridLayout()
        self.diff_delay_integral_group.setLayout(ig)
        layout.addWidget(self.diff_delay_integral_group)

        ig.addWidget(QLabel("unit:"), 0, 0)
        self.diff_unit = QComboBox()
        self.diff_unit.addItems(["ps", "fs", "ns", "µs", "ms", "s"])
        ig.addWidget(self.diff_unit, 0, 1)

        ig.addWidget(QLabel("delay_offset:"), 1, 0)
        self.diff_delay_offset = QLineEdit("0")
        self.diff_delay_offset.setValidator(QDoubleValidator())
        ig.addWidget(self.diff_delay_offset, 1, 1)

        self.diff_plot_abs_and_diffs = QCheckBox("plot_abs_and_diffs")
        self.diff_plot_abs_and_diffs.setChecked(True)
        ig.addWidget(self.diff_plot_abs_and_diffs, 2, 0, 1, 2)

        self.diff_show_errorbars = QCheckBox("show_errorbars")
        self.diff_show_errorbars.setChecked(True)
        ig.addWidget(self.diff_show_errorbars, 3, 0, 1, 2)

        ig.addWidget(QLabel("errorbar_scale:"), 4, 0)
        self.diff_errorbar_scale = QLineEdit("1.0")
        self.diff_errorbar_scale.setValidator(QDoubleValidator())
        ig.addWidget(self.diff_errorbar_scale, 4, 1)

    def _init_single_fluence_integral_group(self, layout: QVBoxLayout):
        """Create and connect the controls for single fluence integral group."""
        self.diff_fluence_integral_group = QGroupBox("Fluence Integral Plot Settings")
        figg = QGridLayout()
        self.diff_fluence_integral_group.setLayout(figg)
        layout.addWidget(self.diff_fluence_integral_group)

        figg.addWidget(QLabel("fluence_unit:"), 0, 0)
        self.diff_fluence_unit = QLineEdit("mJ/cm$^2$")
        figg.addWidget(self.diff_fluence_unit, 0, 1)

        figg.addWidget(QLabel("Fluence offset:"), 1, 0)
        self.diff_fluence_offset = QLineEdit("0")
        self.diff_fluence_offset.setValidator(QDoubleValidator())
        figg.addWidget(self.diff_fluence_offset, 1, 1)

        self.diff_fluence_plot_abs_and_diffs = QCheckBox("plot_abs_and_diffs")
        self.diff_fluence_plot_abs_and_diffs.setChecked(True)
        figg.addWidget(self.diff_fluence_plot_abs_and_diffs, 2, 0, 1, 2)

        self.diff_fluence_show_errorbars = QCheckBox("show_errorbars")
        self.diff_fluence_show_errorbars.setChecked(True)
        figg.addWidget(self.diff_fluence_show_errorbars, 3, 0, 1, 2)

        figg.addWidget(QLabel("errorbar_scale:"), 4, 0)
        self.diff_fluence_errorbar_scale = QLineEdit("1.0")
        self.diff_fluence_errorbar_scale.setValidator(QDoubleValidator())
        figg.addWidget(self.diff_fluence_errorbar_scale, 4, 1)

    def _init_single_fft_group(self, layout: QVBoxLayout):
        """Create and connect the controls for single FFT group."""
        self.diff_delay_fft_group = QGroupBox("FFT Settings")
        fg2 = QGridLayout()
        self.diff_delay_fft_group.setLayout(fg2)
        layout.addWidget(self.diff_delay_fft_group)

        fg2.addWidget(QLabel("region:"), 0, 0)
        self.diff_region = QComboBox()
        self.diff_region.addItems(["peak", "background"])
        fg2.addWidget(self.diff_region, 0, 1)

        fg2.addWidget(QLabel("kind:"), 1, 0)
        self.diff_kind = QComboBox()
        self.diff_kind.addItems(["diff", "absdiff"])
        fg2.addWidget(self.diff_kind, 1, 1)

        fg2.addWidget(QLabel("time_window_select_ps:"), 2, 0)
        self.diff_time_window_select = QLineEdit("(-1, 200)")
        fg2.addWidget(self.diff_time_window_select, 2, 1)

        fg2.addWidget(QLabel("poly_order:"), 3, 0)
        self.diff_poly_order = QLineEdit("2")
        self.diff_poly_order.setValidator(QDoubleValidator())
        fg2.addWidget(self.diff_poly_order, 3, 1)

        fg2.addWidget(QLabel("freq_unit:"), 4, 0)
        self.diff_freq_unit = QComboBox()
        self.diff_freq_unit.addItems(["cm^-1", "1/ps"])
        fg2.addWidget(self.diff_freq_unit, 4, 1)

        fg2.addWidget(QLabel("xlim_freq:"), 5, 0)
        self.diff_xlim_freq = QLineEdit("(-50, 850)")
        fg2.addWidget(self.diff_xlim_freq, 5, 1)

    def _init_single_runtime_group(self, layout: QVBoxLayout):
        """Create and connect the controls for single runtime group."""
        runtime_group = QGroupBox("Runtime and Save Options")
        rg = QGridLayout()
        runtime_group.setLayout(rg)
        layout.addWidget(runtime_group)

        rg.addWidget(QLabel("Number of q points:"), 0, 0)
        self.diff_npt = QLineEdit("1000")
        self.diff_npt.setValidator(QDoubleValidator())
        rg.addWidget(self.diff_npt, 0, 1)

        self.diff_normalize_xy = QCheckBox("normalize_xy")
        self.diff_normalize_xy.setChecked(True)
        rg.addWidget(self.diff_normalize_xy, 1, 0, 1, 2)

        rg.addWidget(QLabel("Q normalization range:"), 2, 0)
        self.diff_q_norm_range = QLineEdit("(2.65, 2.75)")
        rg.addWidget(self.diff_q_norm_range, 2, 1)

        self.diff_compute_if_missing = QCheckBox("compute_if_missing")
        self.diff_compute_if_missing.setChecked(True)
        rg.addWidget(self.diff_compute_if_missing, 3, 0, 1, 2)

        self.diff_overwrite_xy = QCheckBox("overwrite_xy")
        rg.addWidget(self.diff_overwrite_xy, 4, 0, 1, 2)

        self.diff_save = QCheckBox("save")
        self.diff_save.setChecked(True)
        rg.addWidget(self.diff_save, 5, 0, 1, 2)

        rg.addWidget(QLabel("save_format:"), 6, 0)
        self.diff_save_format = QComboBox()
        self.diff_save_format.addItems(["png", "pdf", "svg"])
        rg.addWidget(self.diff_save_format, 6, 1)

        rg.addWidget(QLabel("Save DPI:"), 7, 0)
        self.diff_save_dpi = QLineEdit("400")
        self.diff_save_dpi.setValidator(QDoubleValidator())
        rg.addWidget(self.diff_save_dpi, 7, 1)

        self.diff_save_overwrite = QCheckBox("save_overwrite")
        self.diff_save_overwrite.setChecked(True)
        rg.addWidget(self.diff_save_overwrite, 8, 0, 1, 2)

    def _init_single_actions(self, layout: QVBoxLayout):
        """Create single actions."""
        btn_row = QHBoxLayout()

        self.diff_integrals_btn = QPushButton("Plot Differential Integrals")
        self.diff_integrals_btn.clicked.connect(self._run_diff_integrals)
        btn_row.addWidget(self.diff_integrals_btn)

        self.diff_fft_btn = QPushButton("Plot Differential FFT")
        self.diff_fft_btn.clicked.connect(self._run_diff_fft)
        btn_row.addWidget(self.diff_fft_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _init_multi_experiment_widget(self, layout: QVBoxLayout):
        """Create and connect the controls for multi experiment widget."""
        self.diff_multi_widget = QWidget()
        dml = QVBoxLayout()
        self.diff_multi_widget.setLayout(dml)
        layout.addWidget(self.diff_multi_widget)

        series_group = QGroupBox("Experiment Type")
        sm = QHBoxLayout()
        series_group.setLayout(sm)
        dml.addWidget(series_group)

        sm.addWidget(QLabel("Experiment type:"))

        self.diff_multi_series_combo = QComboBox()
        self.diff_multi_series_combo.addItems(["Delay scan", "Fluence scan"])
        self.diff_multi_series_combo.currentIndexChanged.connect(
            self._refresh_series_widgets
        )
        sm.addWidget(self.diff_multi_series_combo)

        sm.addStretch()

        self.diff_multi_editor = MultiExperimentEditor(
            "Multiple-experiment Definitions",
            allow_merge=False,
            defaults=DEFAULT_MULTI_EXPERIMENTS_DIFF,
            series_kind="delay",
        )
        dml.addWidget(self.diff_multi_editor)

        self.diff_multi_editor_fluence = MultiExperimentEditor(
            "Multiple-experiment Fluence Definitions",
            allow_merge=False,
            defaults=[],
            series_kind="fluence",
        )
        dml.addWidget(self.diff_multi_editor_fluence)

        self._init_multi_delay_settings_group(dml)
        self._init_multi_delay_integral_group(dml)
        self._init_multi_delay_fft_group(dml)
        self._init_multi_fluence_settings_group(dml)
        self._init_multi_fluence_integral_group(dml)
        self._init_multi_actions(dml)

        dml.addStretch()

    def _init_multi_delay_settings_group(self, layout: QVBoxLayout):
        """Create and connect the controls for multi delay settings group."""
        self.diff_multi_delay_settings_group = QGroupBox(
            "Multiple-experiment Delay Plot Settings"
        )
        grid = QGridLayout()
        self.diff_multi_delay_settings_group.setLayout(grid)
        layout.addWidget(self.diff_multi_delay_settings_group)

        row = 0

        grid.addWidget(QLabel("delays_fs:"), row, 0)
        self.diff_multi_delays = QLineEdit("all")
        grid.addWidget(self.diff_multi_delays, row, 1)
        row += 1

        grid.addWidget(QLabel("Azimuthal window [deg]:"), row, 0)
        self.diff_multi_azim_window = QLineEdit("(-90, 90)")
        grid.addWidget(self.diff_multi_azim_window, row, 1)
        row += 1

        grid.addWidget(QLabel("peak:"), row, 0)
        self.diff_multi_peak = QLineEdit("110")
        grid.addWidget(self.diff_multi_peak, row, 1)
        row += 1

        grid.addWidget(QLabel("Peak definitions:"), row, 0)
        self.diff_multi_peak_specs = QPlainTextEdit(
            pretty_literal(DEFAULT_DIFF_PEAK_SPECS)
        )
        grid.addWidget(self.diff_multi_peak_specs, row, 1)

        grid.setRowMinimumHeight(row, 100)
        row += 1

        grid.addWidget(QLabel("Number of q points:"), row, 0)
        self.diff_multi_npt = QLineEdit("1000")
        self.diff_multi_npt.setValidator(QDoubleValidator())
        grid.addWidget(self.diff_multi_npt, row, 1)
        row += 1

        self.diff_multi_normalize_xy = QCheckBox("normalize_xy")
        self.diff_multi_normalize_xy.setChecked(True)
        grid.addWidget(self.diff_multi_normalize_xy, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("Q normalization range:"), row, 0)
        self.diff_multi_q_norm_range = QLineEdit("(2.65, 2.75)")
        grid.addWidget(self.diff_multi_q_norm_range, row, 1)
        row += 1

        self.diff_multi_compute_if_missing = QCheckBox("compute_if_missing")
        self.diff_multi_compute_if_missing.setChecked(True)
        grid.addWidget(self.diff_multi_compute_if_missing, row, 0, 1, 2)
        row += 1

        self.diff_multi_overwrite_xy = QCheckBox("overwrite_xy")
        grid.addWidget(self.diff_multi_overwrite_xy, row, 0, 1, 2)
        row += 1

        self.diff_multi_save = QCheckBox("save")
        self.diff_multi_save.setChecked(True)
        grid.addWidget(self.diff_multi_save, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("save_format:"), row, 0)
        self.diff_multi_save_format = QComboBox()
        self.diff_multi_save_format.addItems(["png", "pdf", "svg"])
        grid.addWidget(self.diff_multi_save_format, row, 1)
        row += 1

        grid.addWidget(QLabel("Save DPI:"), row, 0)
        self.diff_multi_save_dpi = QLineEdit("400")
        self.diff_multi_save_dpi.setValidator(QDoubleValidator())
        grid.addWidget(self.diff_multi_save_dpi, row, 1)
        row += 1

        self.diff_multi_save_overwrite = QCheckBox("save_overwrite")
        self.diff_multi_save_overwrite.setChecked(True)
        grid.addWidget(self.diff_multi_save_overwrite, row, 0, 1, 2)

    def _init_multi_delay_integral_group(self, layout: QVBoxLayout):
        """Create and connect the controls for multi delay integral group."""
        self.diff_multi_delay_integral_group = QGroupBox(
            "Multiple-experiment Delay Integrals"
        )
        grid = QGridLayout()
        self.diff_multi_delay_integral_group.setLayout(grid)
        layout.addWidget(self.diff_multi_delay_integral_group)

        grid.addWidget(QLabel("unit:"), 0, 0)
        self.diff_multi_unit = QComboBox()
        self.diff_multi_unit.addItems(["ps", "fs", "ns", "µs", "ms", "s"])
        grid.addWidget(self.diff_multi_unit, 0, 1)

        self.diff_multi_show_errorbars = QCheckBox("show_errorbars")
        self.diff_multi_show_errorbars.setChecked(True)
        grid.addWidget(self.diff_multi_show_errorbars, 1, 0, 1, 2)

        grid.addWidget(QLabel("errorbar_scale:"), 2, 0)
        self.diff_multi_errorbar_scale = QLineEdit("1.0")
        self.diff_multi_errorbar_scale.setValidator(QDoubleValidator())
        grid.addWidget(self.diff_multi_errorbar_scale, 2, 1)

        self.diff_multi_as_lines = QCheckBox("as_lines")
        grid.addWidget(self.diff_multi_as_lines, 3, 0, 1, 2)

    def _init_multi_delay_fft_group(self, layout: QVBoxLayout):
        """Create and connect the controls for multi delay FFT group."""
        self.diff_multi_delay_fft_group = QGroupBox("Multiple-experiment Delay FFT")
        grid = QGridLayout()
        self.diff_multi_delay_fft_group.setLayout(grid)
        layout.addWidget(self.diff_multi_delay_fft_group)

        grid.addWidget(QLabel("kind:"), 0, 0)
        self.diff_multi_kind = QComboBox()
        self.diff_multi_kind.addItems(["diff", "absdiff"])
        grid.addWidget(self.diff_multi_kind, 0, 1)

        grid.addWidget(QLabel("time_window_select_ps:"), 1, 0)
        self.diff_multi_time_window_select = QLineEdit("(-1, 200)")
        grid.addWidget(self.diff_multi_time_window_select, 1, 1)

        grid.addWidget(QLabel("poly_order:"), 2, 0)
        self.diff_multi_poly_order = QLineEdit("2")
        self.diff_multi_poly_order.setValidator(QDoubleValidator())
        grid.addWidget(self.diff_multi_poly_order, 2, 1)

        grid.addWidget(QLabel("freq_unit:"), 3, 0)
        self.diff_multi_freq_unit = QComboBox()
        self.diff_multi_freq_unit.addItems(["cm^-1", "1/ps"])
        grid.addWidget(self.diff_multi_freq_unit, 3, 1)

        grid.addWidget(QLabel("xlim_freq:"), 4, 0)
        self.diff_multi_xlim_freq = QLineEdit("(-50, 850)")
        grid.addWidget(self.diff_multi_xlim_freq, 4, 1)

    def _init_multi_fluence_settings_group(self, layout: QVBoxLayout):
        """Create and connect the controls for multi fluence settings group."""
        self.diff_multi_fluence_settings_group = QGroupBox(
            "Multiple-experiment Fluence Plot Settings"
        )
        grid = QGridLayout()
        self.diff_multi_fluence_settings_group.setLayout(grid)
        layout.addWidget(self.diff_multi_fluence_settings_group)

        row = 0

        grid.addWidget(QLabel("Fluences [mJ/cm²]:"), row, 0)
        self.diff_multi_fluences = QLineEdit("all")
        grid.addWidget(self.diff_multi_fluences, row, 1)
        row += 1

        grid.addWidget(QLabel("Azimuthal window [deg]:"), row, 0)
        self.diff_multi_fluence_azim_window = QLineEdit("(-90, 90)")
        grid.addWidget(self.diff_multi_fluence_azim_window, row, 1)
        row += 1

        grid.addWidget(QLabel("peak:"), row, 0)
        self.diff_multi_fluence_peak = QLineEdit("110")
        grid.addWidget(self.diff_multi_fluence_peak, row, 1)
        row += 1

        grid.addWidget(QLabel("Peak definitions:"), row, 0)
        self.diff_multi_fluence_peak_specs = QPlainTextEdit(
            pretty_literal(DEFAULT_DIFF_PEAK_SPECS)
        )
        grid.addWidget(self.diff_multi_fluence_peak_specs, row, 1)

        grid.setRowMinimumHeight(row, 100)
        row += 1

        grid.addWidget(QLabel("Number of q points:"), row, 0)
        self.diff_multi_fluence_npt = QLineEdit("1000")
        self.diff_multi_fluence_npt.setValidator(QDoubleValidator())
        grid.addWidget(self.diff_multi_fluence_npt, row, 1)
        row += 1

        self.diff_multi_fluence_normalize_xy = QCheckBox("normalize_xy")
        self.diff_multi_fluence_normalize_xy.setChecked(True)
        grid.addWidget(self.diff_multi_fluence_normalize_xy, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("Q normalization range:"), row, 0)
        self.diff_multi_fluence_q_norm_range = QLineEdit("(2.65, 2.75)")
        grid.addWidget(self.diff_multi_fluence_q_norm_range, row, 1)
        row += 1

        self.diff_multi_fluence_compute_if_missing = QCheckBox("compute_if_missing")
        self.diff_multi_fluence_compute_if_missing.setChecked(True)
        grid.addWidget(self.diff_multi_fluence_compute_if_missing, row, 0, 1, 2)
        row += 1

        self.diff_multi_fluence_overwrite_xy = QCheckBox("overwrite_xy")
        grid.addWidget(self.diff_multi_fluence_overwrite_xy, row, 0, 1, 2)
        row += 1

        self.diff_multi_fluence_save = QCheckBox("save")
        self.diff_multi_fluence_save.setChecked(True)
        grid.addWidget(self.diff_multi_fluence_save, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("save_format:"), row, 0)
        self.diff_multi_fluence_save_format = QComboBox()
        self.diff_multi_fluence_save_format.addItems(["png", "pdf", "svg"])
        grid.addWidget(self.diff_multi_fluence_save_format, row, 1)
        row += 1

        grid.addWidget(QLabel("Save DPI:"), row, 0)
        self.diff_multi_fluence_save_dpi = QLineEdit("400")
        self.diff_multi_fluence_save_dpi.setValidator(QDoubleValidator())
        grid.addWidget(self.diff_multi_fluence_save_dpi, row, 1)
        row += 1

        self.diff_multi_fluence_save_overwrite = QCheckBox("save_overwrite")
        self.diff_multi_fluence_save_overwrite.setChecked(True)
        grid.addWidget(self.diff_multi_fluence_save_overwrite, row, 0, 1, 2)

    def _init_multi_fluence_integral_group(self, layout: QVBoxLayout):
        """Create and connect the controls for multi fluence integral group."""
        self.diff_multi_fluence_integral_group = QGroupBox(
            "Multiple-experiment Fluence Integrals"
        )
        grid = QGridLayout()
        self.diff_multi_fluence_integral_group.setLayout(grid)
        layout.addWidget(self.diff_multi_fluence_integral_group)

        grid.addWidget(QLabel("fluence_unit:"), 0, 0)
        self.diff_multi_fluence_unit = QLineEdit("mJ/cm$^2$")
        grid.addWidget(self.diff_multi_fluence_unit, 0, 1)

        self.diff_multi_fluence_show_errorbars = QCheckBox("show_errorbars")
        self.diff_multi_fluence_show_errorbars.setChecked(True)
        grid.addWidget(self.diff_multi_fluence_show_errorbars, 1, 0, 1, 2)

        grid.addWidget(QLabel("errorbar_scale:"), 2, 0)
        self.diff_multi_fluence_errorbar_scale = QLineEdit("1.0")
        self.diff_multi_fluence_errorbar_scale.setValidator(QDoubleValidator())
        grid.addWidget(self.diff_multi_fluence_errorbar_scale, 2, 1)

        self.diff_multi_fluence_as_lines = QCheckBox("as_lines")
        grid.addWidget(self.diff_multi_fluence_as_lines, 3, 0, 1, 2)

    def _init_multi_actions(self, layout: QVBoxLayout):
        """Create multi actions."""
        btn_row = QHBoxLayout()

        self.diff_multi_integrals_btn = QPushButton("Plot Multi Differential Integrals")
        self.diff_multi_integrals_btn.clicked.connect(self._run_diff_integrals_multi)
        btn_row.addWidget(self.diff_multi_integrals_btn)

        self.diff_multi_fft_btn = QPushButton("Plot Multi Differential FFT")
        self.diff_multi_fft_btn.clicked.connect(self._run_diff_fft_multi)
        btn_row.addWidget(self.diff_multi_fft_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def set_facility(self, facility: str):
        """Store the active facility and refresh facility-dependent controls."""

        self.state.facility = facility
        self._refresh_series_widgets()

    def _refresh_mode_widgets(self):
        """Refresh mode widgets."""
        single_mode = self.diff_mode_combo.currentText() == "Single experiment"

        self.diff_single_widget.setVisible(single_mode)
        self.diff_multi_widget.setVisible(not single_mode)

    def _refresh_series_widgets(self):
        """Update series widgets visibility and defaults from the active mode."""
        delay_mode_single = self.diff_series_combo.currentText() == "Delay scan"
        is_id09 = self.state.facility == "ID09"

        self.diff_delay_primary_group.setVisible(delay_mode_single)
        self.diff_fluence_primary_group.setVisible(not delay_mode_single)
        self.diff_delay_integral_group.setVisible(delay_mode_single)
        self.diff_fluence_integral_group.setVisible(not delay_mode_single)
        self.diff_delay_fft_group.setVisible(delay_mode_single)
        self.diff_fft_btn.setVisible(delay_mode_single)

        self.diff_single_metadata.set_field_visible(
            "fluence_mJ_cm2",
            delay_mode_single,
        )
        self.diff_single_metadata.set_id09_visible(is_id09 and delay_mode_single)

        delay_mode_multi = self.diff_multi_series_combo.currentText() == "Delay scan"

        self.diff_multi_editor.setVisible(delay_mode_multi)
        self.diff_multi_editor_fluence.setVisible(not delay_mode_multi)
        self.diff_multi_delay_settings_group.setVisible(delay_mode_multi)
        self.diff_multi_delay_integral_group.setVisible(delay_mode_multi)
        self.diff_multi_delay_fft_group.setVisible(delay_mode_multi)
        self.diff_multi_fluence_settings_group.setVisible(not delay_mode_multi)
        self.diff_multi_fluence_integral_group.setVisible(not delay_mode_multi)
        self.diff_multi_fft_btn.setVisible(delay_mode_multi)
    
    def _build_analysis_paths(self):
        """Build analysis paths."""
        return self.path_service.build_analysis_paths(
            path_root=self.state.path_root,
            analysis_subdir=self.state.analysis_subdir,
            raw_subdir=self.state.raw_subdir,
        )


    def _poni_path(self):
        """Return PONI path."""
        return getattr(self.state, "poni_path", None)


    def _mask_path(self):
        """Return mask path."""
        return getattr(self.state, "mask_edf_path", None) or getattr(
            self.state,
            "mask_path",
            None,
        )


    def _azim_offset_deg(self):
        """Return azimuthal offset deg."""
        return self.integration_service.parse_azim_offset_deg(
            getattr(self.state, "azim_offset_deg", "-90.0")
        )

    def _polarization_factor(self):
        """Return polarization factor."""
        if not getattr(self.state, "polarization_enabled", True):
            return None
        return self.integration_service.parse_polarization_factor(
            getattr(self.state, "polarization_factor", 0.99)
        )


    def _diff_peak_specs(self):
        """Return diff peak specs."""
        peak_specs = parse_python_literal(self.diff_peak_specs.toPlainText())

        if not isinstance(peak_specs, dict) or not peak_specs:
            raise ValueError("Differential peak_specs must be a non-empty dict.")

        return peak_specs


    def _base_diff_kwargs(self):
        """Validate shared single-experiment fields for differential analysis."""
        kwargs = self.integration_service.build_experiment_kwargs(
            self.diff_single_metadata.values()
        )
        kwargs.update(
            self.integration_service.build_poni_mask_kwargs(
                poni_path=self._poni_path(),
                mask_edf_path=self._mask_path(),
            )
        )
        kwargs.update(
            delays_fs=self.integration_service.parse_delays_value(
                self.diff_delays.text()
            ),
            ref_type=self.diff_ref_type.currentText(),
            ref_value=self.integration_service.parse_ref_value(
                self.diff_ref_value.text()
            ),
            azim_window=self.integration_service.parse_range_tuple(
                self.diff_azim_window.text(),
                name="azim_window",
            ),
            azim_offset_deg=self._azim_offset_deg(),
            polarization_factor=self._polarization_factor(),
            peak=self.diff_peak.text().strip(),
            peak_specs=self._diff_peak_specs(),
            npt=parse_int_like(self.diff_npt.text(), name="npt"),
            normalize_xy=self.diff_normalize_xy.isChecked(),
            q_norm_range=self.integration_service.parse_range_tuple(
                self.diff_q_norm_range.text(),
                name="q_norm_range",
            ),
            compute_if_missing=self.diff_compute_if_missing.isChecked(),
            overwrite_xy=self.diff_overwrite_xy.isChecked(),
            save=self.diff_save.isChecked(),
            save_format=self.diff_save_format.currentText(),
            save_dpi=parse_int_like(self.diff_save_dpi.text(), name="save_dpi"),
            save_overwrite=self.diff_save_overwrite.isChecked(),
            paths=self._build_analysis_paths(),
        )
        return kwargs


    def _validated_diff_multi_experiments(self):
        """Return validated diff multi experiments."""
        required = [
            "sample_name",
            "temperature_K",
            "excitation_wl_nm",
            "fluence_mJ_cm2",
            "time_window_fs",
            "ref_type",
            "ref_value",
        ]

        return self.differential_service.validate_multi_experiments(
            self.diff_multi_editor.get_experiments(),
            required_fields=required,
        )


    def _base_diff_multi_kwargs(self):
        """Validate shared multi-experiment fields for differential analysis."""
        peak_specs = parse_python_literal(self.diff_multi_peak_specs.toPlainText())

        if not isinstance(peak_specs, dict) or not peak_specs:
            raise ValueError(
                "Multiple-experiment differential peak_specs must be a non-empty dict."
            )

        return dict(
            experiments=self._validated_diff_multi_experiments(),
            delays_fs=self.integration_service.parse_delays_value(
                self.diff_multi_delays.text()
            ),
            azim_window=self.integration_service.parse_range_tuple(
                self.diff_multi_azim_window.text(),
                name="azim_window",
            ),
            peak=self.diff_multi_peak.text().strip(),
            peak_specs=peak_specs,
            npt=parse_int_like(self.diff_multi_npt.text(), name="npt"),
            normalize_xy=self.diff_multi_normalize_xy.isChecked(),
            q_norm_range=self.integration_service.parse_range_tuple(
                self.diff_multi_q_norm_range.text(),
                name="q_norm_range",
            ),
            azim_offset_deg=self._azim_offset_deg(),
            polarization_factor=self._polarization_factor(),
            compute_if_missing=self.diff_multi_compute_if_missing.isChecked(),
            overwrite_xy=self.diff_multi_overwrite_xy.isChecked(),
            save=self.diff_multi_save.isChecked(),
            save_format=self.diff_multi_save_format.currentText(),
            save_dpi=parse_int_like(self.diff_multi_save_dpi.text(), name="save_dpi"),
            save_overwrite=self.diff_multi_save_overwrite.isChecked(),
            poni_path=self._poni_path(),
            mask_edf_path=self._mask_path(),
            paths=self._build_analysis_paths(),
        )


    def _run_diff_integrals(self):
        """Parse the differential integrals controls, invoke the service workflow, and log completion or errors."""
        try:
            if self.diff_series_combo.currentText() == "Delay scan":
                kwargs = self._base_diff_kwargs()
                kwargs.update(
                    unit=self.diff_unit.currentText(),
                    delay_offset=parse_float_like(
                        self.diff_delay_offset.text(),
                        name="delay_offset",
                    ),
                    plot_abs_and_diffs=self.diff_plot_abs_and_diffs.isChecked(),
                    show_errorbars=self.diff_show_errorbars.isChecked(),
                    errorbar_scale=parse_float_like(
                        self.diff_errorbar_scale.text(),
                        name="errorbar_scale",
                    ),
                )

                self.differential_service.plot_differential_integrals(**kwargs)

            else:
                metadata_values = self.diff_single_metadata.values()
                kwargs = self.integration_service.build_experiment_kwargs(metadata_values)
                kwargs.pop("fluence_mJ_cm2", None)
                kwargs.update(
                    self.integration_service.build_poni_mask_kwargs(
                        poni_path=self._poni_path(),
                        mask_edf_path=self._mask_path(),
                    )
                )

                azim_window = self.integration_service.parse_range_tuple(
                    self.diff_fluence_azim_window.text(),
                    name="azim_window",
                )
                fluences = self.integration_service.parse_fluences_value(
                    self.diff_fluences.text()
                )
                delay_fs = parse_int_like(
                    self.diff_fluence_delay_fs.text(),
                    name="delay_fs",
                )

                if self.state.facility == "ID09" and self.diff_compute_if_missing.isChecked():
                    self.integration_service.ensure_id09_fluence_cache(
                        sample_name=kwargs["sample_name"],
                        temperature_K=kwargs["temperature_K"],
                        excitation_wl_nm=kwargs["excitation_wl_nm"],
                        delay_fs=delay_fs,
                        time_window_fs=kwargs["time_window_fs"],
                        fluences_mJ_cm2=fluences,
                        azim_windows=[azim_window],
                        copy_2d_image=False,
                        overwrite=self.diff_overwrite_xy.isChecked(),
                        paths=self._build_analysis_paths(),
                    )

                kwargs.update(
                    delay_fs=delay_fs,
                    fluences_mJ_cm2=fluences,
                    ref_type=self.diff_fluence_ref_type.currentText(),
                    ref_value=self.integration_service.parse_ref_value(
                        self.diff_fluence_ref_value.text()
                    ),
                    azim_window=azim_window,
                    azim_offset_deg=self._azim_offset_deg(),
                    polarization_factor=self._polarization_factor(),
                    peak=self.diff_fluence_peak.text().strip(),
                    peak_specs=parse_python_literal(
                        self.diff_fluence_peak_specs.toPlainText()
                    ),
                    npt=parse_int_like(self.diff_npt.text(), name="npt"),
                    normalize_xy=self.diff_normalize_xy.isChecked(),
                    q_norm_range=self.integration_service.parse_range_tuple(
                        self.diff_q_norm_range.text(),
                        name="q_norm_range",
                    ),
                    compute_if_missing=self.diff_compute_if_missing.isChecked(),
                    overwrite_xy=self.diff_overwrite_xy.isChecked(),
                    fluence_unit=self.diff_fluence_unit.text().strip() or "mJ/cm$^2$",
                    fluence_offset=parse_float_like(
                        self.diff_fluence_offset.text(),
                        name="fluence_offset",
                    ),
                    show_errorbars=self.diff_fluence_show_errorbars.isChecked(),
                    errorbar_scale=parse_float_like(
                        self.diff_fluence_errorbar_scale.text(),
                        name="errorbar_scale",
                    ),
                    plot_abs_and_diffs=self.diff_fluence_plot_abs_and_diffs.isChecked(),
                    save=self.diff_save.isChecked(),
                    save_format=self.diff_save_format.currentText(),
                    save_dpi=parse_int_like(self.diff_save_dpi.text(), name="save_dpi"),
                    save_overwrite=self.diff_save_overwrite.isChecked(),
                    paths=self._build_analysis_paths(),
                )

                self.differential_service.plot_differential_integrals_fluence(**kwargs)

            self.log("Differential integral plot finished.")

        except Exception as exc:
            self.log(f"Differential Integrals Error: {exc}")


    def _run_diff_fft(self):
        """Parse the differential FFT controls, invoke the service workflow, and log completion or errors."""
        try:
            if self.diff_series_combo.currentText() != "Delay scan":
                raise NotImplementedError(
                    "Fluence differential FFT is not exposed in this GUI/backend."
                )

            kwargs = self._base_diff_kwargs()
            kwargs.update(
                delay_offset=parse_float_like(
                    self.diff_delay_offset.text(),
                    name="delay_offset",
                ),
                time_unit=self.diff_unit.currentText(),
                region=self.diff_region.currentText(),
                kind=self.diff_kind.currentText(),
                time_window_select_ps=self.integration_service.parse_range_tuple(
                    self.diff_time_window_select.text(),
                    name="time_window_select_ps",
                ),
                poly_order=parse_int_like(self.diff_poly_order.text(), name="poly_order"),
                freq_unit=self.diff_freq_unit.currentText(),
                xlim_freq=self.integration_service.parse_range_tuple(
                    self.diff_xlim_freq.text(),
                    name="xlim_freq",
                ),
            )

            self.differential_service.plot_differential_fft(**kwargs)
            self.log("Differential FFT plot finished.")

        except Exception as exc:
            self.log(f"Differential FFT Error: {exc}")


    def _run_diff_integrals_multi(self):
        """Compare differential integral traces across configured experiments."""
        try:
            if self.diff_multi_series_combo.currentText() == "Delay scan":
                kwargs = self._base_diff_multi_kwargs()
                kwargs.update(
                    unit=self.diff_multi_unit.currentText(),
                    show_errorbars=self.diff_multi_show_errorbars.isChecked(),
                    errorbar_scale=parse_float_like(
                        self.diff_multi_errorbar_scale.text(),
                        name="errorbar_scale",
                    ),
                    as_lines=self.diff_multi_as_lines.isChecked(),
                )

                self.differential_service.plot_differential_integrals_multi(**kwargs)

            else:
                experiments = self.diff_multi_editor_fluence.get_experiments()
                azim_window = self.integration_service.parse_range_tuple(
                    self.diff_multi_fluence_azim_window.text(),
                    name="azim_window",
                )
                fluences = self.integration_service.parse_fluences_value(
                    self.diff_multi_fluences.text()
                )

                if (
                    self.state.facility == "ID09"
                    and self.diff_multi_fluence_compute_if_missing.isChecked()
                ):
                    for experiment in experiments:
                        if "merge" in experiment:
                            raise ValueError(
                                "Differential fluence multi does not support merged experiments."
                            )

                        self.integration_service.ensure_id09_fluence_cache(
                            sample_name=str(experiment.get("sample_name")),
                            temperature_K=int(experiment.get("temperature_K")),
                            excitation_wl_nm=float(experiment.get("excitation_wl_nm")),
                            delay_fs=int(experiment.get("delay_fs")),
                            time_window_fs=int(experiment.get("time_window_fs")),
                            fluences_mJ_cm2=fluences,
                            azim_windows=[azim_window],
                            copy_2d_image=False,
                            overwrite=self.diff_multi_fluence_overwrite_xy.isChecked(),
                            paths=self._build_analysis_paths(),
                        )

                kwargs = dict(
                    experiments=experiments,
                    fluences_mJ_cm2=fluences,
                    azim_window=azim_window,
                    peak=self.diff_multi_fluence_peak.text().strip(),
                    peak_specs=parse_python_literal(
                        self.diff_multi_fluence_peak_specs.toPlainText()
                    ),
                    npt=parse_int_like(
                        self.diff_multi_fluence_npt.text(),
                        name="npt",
                    ),
                    normalize_xy=self.diff_multi_fluence_normalize_xy.isChecked(),
                    q_norm_range=self.integration_service.parse_range_tuple(
                        self.diff_multi_fluence_q_norm_range.text(),
                        name="q_norm_range",
                    ),
                    azim_offset_deg=self._azim_offset_deg(),
                    polarization_factor=self._polarization_factor(),
                    compute_if_missing=self.diff_multi_fluence_compute_if_missing.isChecked(),
                    overwrite_xy=self.diff_multi_fluence_overwrite_xy.isChecked(),
                    fluence_unit=self.diff_multi_fluence_unit.text().strip()
                    or "mJ/cm$^2$",
                    show_errorbars=self.diff_multi_fluence_show_errorbars.isChecked(),
                    errorbar_scale=parse_float_like(
                        self.diff_multi_fluence_errorbar_scale.text(),
                        name="errorbar_scale",
                    ),
                    as_lines=self.diff_multi_fluence_as_lines.isChecked(),
                    save=self.diff_multi_fluence_save.isChecked(),
                    save_format=self.diff_multi_fluence_save_format.currentText(),
                    save_dpi=parse_int_like(
                        self.diff_multi_fluence_save_dpi.text(),
                        name="save_dpi",
                    ),
                    save_overwrite=self.diff_multi_fluence_save_overwrite.isChecked(),
                    poni_path=self._poni_path(),
                    mask_edf_path=self._mask_path(),
                    paths=self._build_analysis_paths(),
                )

                self.differential_service.plot_differential_integrals_fluence_multi(
                    **kwargs
                )

            self.log("Multiple-experiment differential integrals finished.")

        except Exception as exc:
            self.log(f"Multi Differential Integrals Error: {exc}")


    def _run_diff_fft_multi(self):
        """Compare differential time traces and FFT spectra across experiments."""
        try:
            if self.diff_multi_series_combo.currentText() != "Delay scan":
                raise NotImplementedError(
                    "Fluence differential FFT is not exposed in this GUI/backend."
                )

            kwargs = self._base_diff_multi_kwargs()
            kwargs.update(
                kind=self.diff_multi_kind.currentText(),
                time_unit=self.diff_multi_unit.currentText(),
                time_window_select_ps=self.integration_service.parse_range_tuple(
                    self.diff_multi_time_window_select.text(),
                    name="time_window_select_ps",
                ),
                poly_order=parse_int_like(
                    self.diff_multi_poly_order.text(),
                    name="poly_order",
                ),
                freq_unit=self.diff_multi_freq_unit.currentText(),
                xlim_freq=self.integration_service.parse_range_tuple(
                    self.diff_multi_xlim_freq.text(),
                    name="xlim_freq",
                ),
            )

            self.differential_service.plot_differential_fft_multi(**kwargs)
            self.log("Multiple-experiment differential FFT finished.")

        except Exception as exc:
            self.log(f"Multi Differential FFT Error: {exc}")
