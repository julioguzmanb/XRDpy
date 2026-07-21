"""Facility data-reduction tab for the analysis GUI."""
from __future__ import annotations

import threading
from typing import Callable, Optional

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
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from trxrdpy.analysis.gui.services import (
    IntegrationService,
    PathService,
    PreparationService,
)
from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.widgets import (
    DropPathLineEdit,
    ExperimentMetadataWidget,
    PolarizationControlWidget,
)
from trxrdpy.analysis.gui.widgets.task_output_dialog import run_task_with_output_dialog


class PreparationTab(QWidget):
    """Create reduction metadata, representative 2D images, and shot caches.

    Attributes
    ----------
    state : AnalysisGuiState
        Shared facility and filesystem configuration.
    path_service : PathService
        Builds raw and analysis roots for reduction backends.
    preparation_service : PreparationService
        Parses inputs and dispatches ID09 or FemtoMAX reduction.
    experiment_metadata : ExperimentMetadataWidget
        Sample, acquisition, and facility-specific metadata editor.
    datared_id09_group, datared_femto_group : QGroupBox
        Facility-specific reduction-control containers.
    datared_ref_delay, datared_delays : QLineEdit
        ID09 dark-reference and delay-series selectors.
    datared_show_progress, datared_show_frame_progress : QCheckBox
        ID09 task-level and batched frame-level progress controls.
    datared_femto_scans, datared_femto_scan_type : QWidget
        FemtoMAX scan selection and reduction-mode controls.
    datared_femto_use_parallel, datared_femto_max_workers : QWidget
        FemtoMAX multiprocessing configuration.
    log : callable
        Callback receiving task status and errors.
    """

    def __init__(
        self,
        state: AnalysisGuiState,
        path_service: PathService,
        preparation_service: PreparationService,
        log: Optional[Callable[[str], None]] = None,
        parent=None,
        integration_service: Optional[IntegrationService] = None,
        polarization_changed_callback: Optional[
            Callable[[bool, float], None]
        ] = None,
    ):
        """Initialize ``PreparationTab``, bind shared state and services, and create its controls."""
        super().__init__(parent)

        self.state = state
        self.path_service = path_service
        self.preparation_service = preparation_service
        self.integration_service = integration_service or IntegrationService()
        self.log = log or (lambda message: None)
        self.polarization_changed_callback = polarization_changed_callback

        layout = self._make_scroll_layout()

        self.experiment_metadata = ExperimentMetadataWidget(
            title="Experiment Metadata",
            include_id09=True,
        )
        layout.addWidget(self.experiment_metadata)

        self._init_overview_group(layout)
        self._init_id09_groups(layout)
        self._init_femtomax_groups(layout)
        self._init_single_shot_group(layout)
        self._init_other_facilities_group(layout)

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

    def _init_overview_group(self, layout: QVBoxLayout):
        """Create explanatory text describing the data-reduction stage."""
        note_group = QGroupBox("Data Reduction Overview")
        note_layout = QVBoxLayout()
        note_group.setLayout(note_layout)
        layout.addWidget(note_group)

        self.datared_note = QLabel()
        self.datared_note.setWordWrap(True)
        note_layout.addWidget(self.datared_note)

        msg = QLabel(
            "This tab is the entry point for facility metadata and raw-data reduction. "
            "Where supported, raw frames can produce either representative 2D images "
            "or a progressively available single-shot 1D cache."
        )
        msg.setWordWrap(True)
        note_layout.addWidget(msg)

    def _init_id09_groups(self, layout: QVBoxLayout):
        """Create and connect the controls for ID09 groups."""
        self.datared_id09_group = QGroupBox("ID09 2D Image Production")
        id09_grid = QGridLayout()
        self.datared_id09_group.setLayout(id09_grid)
        layout.addWidget(self.datared_id09_group)

        id09_grid.addWidget(QLabel("dark ref_delay:"), 0, 0)
        self.datared_ref_delay = QLineEdit("-5ns")
        self.datared_ref_delay.setPlaceholderText("Example: -5ns")
        id09_grid.addWidget(self.datared_ref_delay, 0, 1)

        id09_grid.addWidget(QLabel("delay selection:"), 1, 0)
        self.datared_delays = QLineEdit("all")
        self.datared_delays.setPlaceholderText(
            "Examples: all, [0, 1, 5], ['-5ns', '0ps']"
        )
        id09_grid.addWidget(self.datared_delays, 1, 1)

        helper = QLabel(
            "Create dark from a reference delay, or create the delay-resolved 2D images "
            "that populate the standard analysis structure."
        )
        helper.setWordWrap(True)
        id09_grid.addWidget(helper, 2, 0, 1, 2)

        self.datared_runtime_group = QGroupBox("ID09 Runtime Options")
        rg = QGridLayout()
        self.datared_runtime_group.setLayout(rg)
        layout.addWidget(self.datared_runtime_group)

        self.datared_overwrite = QCheckBox("overwrite")
        self.datared_overwrite.setChecked(True)
        rg.addWidget(self.datared_overwrite, 0, 0, 1, 2)

        self.datared_show_progress = QCheckBox("show_progress")
        self.datared_show_progress.setChecked(True)
        rg.addWidget(self.datared_show_progress, 1, 0, 1, 2)

        self.datared_show_frame_progress = QCheckBox("show_frame_progress")
        self.datared_show_frame_progress.setChecked(False)
        rg.addWidget(self.datared_show_frame_progress, 2, 0, 1, 2)

        self.datared_actions_group = QGroupBox("Actions")
        al = QHBoxLayout()
        self.datared_actions_group.setLayout(al)
        layout.addWidget(self.datared_actions_group)

        self.datared_create_dark_btn = QPushButton("Create ID09 Dark 2D")
        self.datared_create_dark_btn.clicked.connect(self._create_id09_dark_2d)
        al.addWidget(self.datared_create_dark_btn)

        self.datared_create_delay_btn = QPushButton("Create ID09 Delay 2D Images")
        self.datared_create_delay_btn.clicked.connect(self._create_id09_delay_2d_images)
        al.addWidget(self.datared_create_delay_btn)

        al.addStretch()
    
    def _create_id09_dark_2d(self):
        """Validate the ID09 dark 2D fields and delegate artifact creation to the active facility service."""
        try:
            def error_summary(traceback_text):
                """Extract the final informative message from a captured worker traceback."""
                lines = [
                    line.strip()
                    for line in str(traceback_text).splitlines()
                    if line.strip()
                ]
                return lines[-1] if lines else "unknown error"

            if self.state.facility != "ID09":
                raise ValueError(
                    "This 2D-preparation backend is currently implemented only for ID09."
                )

            delay_ref_text = self.datared_ref_delay.text().strip()

            if not delay_ref_text:
                raise ValueError("dark ref_delay cannot be empty.")

            paths = self._build_analysis_paths()

            kwargs = self.preparation_service.build_id09_dark_kwargs(
                metadata_values=self.experiment_metadata.values(),
                paths=paths,
            )
            kwargs.update(
                delay_ref=self.preparation_service.parse_delays_value(delay_ref_text),
                overwrite=self.datared_overwrite.isChecked(),
                show_progress=self.datared_show_progress.isChecked(),
            )

            def task():
                """Execute the validated backend operation inside the background worker thread."""
                return self.preparation_service.create_id09_dark_from_ref_delay(**kwargs)

            run_task_with_output_dialog(
                self,
                "Create ID09 Dark 2D",
                task,
                on_success=lambda out_path: self.log(
                    f"ID09 dark 2D image created: {out_path}"
                ),
                on_error=lambda tb: self.log(
                    f"ID09 Create Dark 2D Error: {error_summary(tb)}"
                ),
            )

        except Exception as exc:
            self.log(f"ID09 Create Dark 2D Error: {exc}")


    def _create_id09_delay_2d_images(self):
        """Validate the ID09 delay 2D images fields and delegate artifact creation to the active facility service."""
        try:
            def error_summary(traceback_text):
                """Extract the final informative message from a captured worker traceback."""
                lines = [
                    line.strip()
                    for line in str(traceback_text).splitlines()
                    if line.strip()
                ]
                return lines[-1] if lines else "unknown error"

            if self.state.facility != "ID09":
                raise ValueError(
                    "This 2D-preparation backend is currently implemented only for ID09."
                )

            paths = self._build_analysis_paths()
            metadata_values = self.experiment_metadata.values()

            kwargs = self.preparation_service.build_experiment_kwargs(metadata_values)
            kwargs.update(self.preparation_service.build_id09_kwargs(metadata_values))
            kwargs.update(
                delays=self.preparation_service.parse_delays_value(
                    self.datared_delays.text()
                ),
                overwrite=self.datared_overwrite.isChecked(),
                show_progress=self.datared_show_progress.isChecked(),
                show_frame_progress=self.datared_show_frame_progress.isChecked(),
                paths=paths,
            )

            def task():
                """Execute the validated backend operation inside the background worker thread."""
                return self.preparation_service.create_id09_final_2d_images(**kwargs)

            def success(out_paths):
                """Summarize the completed background operation and update the GUI log."""
                n_saved = len(out_paths) if hasattr(out_paths, "__len__") else "?"
                self.log(f"ID09 delay 2D image creation finished. Saved {n_saved} files.")

            run_task_with_output_dialog(
                self,
                "Create ID09 Delay 2D Images",
                task,
                on_success=success,
                on_error=lambda tb: self.log(
                    f"ID09 Create Delay 2D Images Error: {error_summary(tb)}"
                ),
            )

        except Exception as exc:
            self.log(f"ID09 Create Delay 2D Images Error: {exc}")


    def _init_femtomax_groups(self, layout: QVBoxLayout):
        """Create and connect the controls for FemtoMAX groups."""
        self.datared_femto_group = QGroupBox("FemtoMAX Data Reduction")
        fg = QGridLayout()
        self.datared_femto_group.setLayout(fg)
        layout.addWidget(self.datared_femto_group)

        row = 0

        fg.addWidget(QLabel("scans:"), row, 0)
        self.datared_femto_scans = QLineEdit("")
        self.datared_femto_scans.setPlaceholderText(
            "Examples: [181661], [181661, 181662], 181661"
        )
        fg.addWidget(self.datared_femto_scans, row, 1)
        row += 1

        fg.addWidget(QLabel("Session ping references:"), row, 0)
        self.datared_femto_ping_reference_status = QLabel("")
        self.datared_femto_ping_reference_status.setWordWrap(True)
        fg.addWidget(self.datared_femto_ping_reference_status, row, 1)
        row += 1

        self.datared_femto_scans.editingFinished.connect(
            self._load_femtomax_ping_references
        )

        fg.addWidget(QLabel("scan_type:"), row, 0)
        self.datared_femto_scan_type = QComboBox()
        self.datared_femto_scan_type.addItems(["delay", "fluence", "dark"])
        self.datared_femto_scan_type.currentIndexChanged.connect(
            self._refresh_femtomax_scan_type_widgets
        )
        fg.addWidget(self.datared_femto_scan_type, row, 1)
        row += 1

        self.datared_femto_fluences_label = QLabel(
            "fluences_mJ_cm2 (one per scan):"
        )
        fg.addWidget(self.datared_femto_fluences_label, row, 0)
        self.datared_femto_fluences = QLineEdit("")
        self.datared_femto_fluences.setPlaceholderText(
            "Example: [15, 1, 7.7, 2.2, 11.7]"
        )
        fg.addWidget(self.datared_femto_fluences, row, 1)
        row += 1

        fg.addWidget(QLabel("selected_delays:"), row, 0)
        self.datared_femto_selected_delays = QLineEdit("auto")
        self.datared_femto_selected_delays.setPlaceholderText(
            "Examples: auto, [-1000, 0, 5000], -1000"
        )
        fg.addWidget(self.datared_femto_selected_delays, row, 1)
        row += 1

        fg.addWidget(QLabel("delay_source:"), row, 0)
        self.datared_femto_delay_source = QComboBox()
        self.datared_femto_delay_source.addItems(["avg", "p2", "p4"])
        fg.addWidget(self.datared_femto_delay_source, row, 1)
        row += 1

        self.datared_femto_require_both = QCheckBox("require_both")
        self.datared_femto_require_both.setChecked(True)
        fg.addWidget(self.datared_femto_require_both, row, 0, 1, 2)
        row += 1

        fg.addWidget(QLabel("nb_shot_threshold:"), row, 0)
        self.datared_femto_nb_shot_threshold = QLineEdit("")
        self.datared_femto_nb_shot_threshold.setPlaceholderText("Optional")
        self.datared_femto_nb_shot_threshold.setValidator(QDoubleValidator())
        fg.addWidget(self.datared_femto_nb_shot_threshold, row, 1)

        self.datared_femto_dist_group = QGroupBox("FemtoMAX Ping / Delay Distribution")
        dg = QGridLayout()
        self.datared_femto_dist_group.setLayout(dg)
        layout.addWidget(self.datared_femto_dist_group)

        dg.addWidget(QLabel("mode:"), 0, 0)
        self.datared_femto_dist_mode = QComboBox()
        self.datared_femto_dist_mode.addItems(["overlay", "stacked", "per_scan"])
        dg.addWidget(self.datared_femto_dist_mode, 0, 1)

        dg.addWidget(QLabel("unit:"), 1, 0)
        self.datared_femto_dist_unit = QComboBox()
        self.datared_femto_dist_unit.addItems(["fs", "ps", "ns", "µs", "ms", "s"])
        dg.addWidget(self.datared_femto_dist_unit, 1, 1)

        dg.addWidget(QLabel("view:"), 2, 0)
        self.datared_femto_dist_view = QComboBox()
        self.datared_femto_dist_view.addItems(["scatter", "hist"])
        self.datared_femto_dist_view.currentIndexChanged.connect(
            self._refresh_femtomax_distribution_widgets
        )
        dg.addWidget(self.datared_femto_dist_view, 2, 1)

        self.datared_femto_dist_bins_label = QLabel("bins:")
        dg.addWidget(self.datared_femto_dist_bins_label, 3, 0)
        self.datared_femto_dist_bins = QLineEdit("250")
        self.datared_femto_dist_bins.setValidator(QIntValidator(1, 100000, self))
        dg.addWidget(self.datared_femto_dist_bins, 3, 1)

        self.datared_femto_dist_range_label = QLabel("histogram range:")
        dg.addWidget(self.datared_femto_dist_range_label, 4, 0)
        self.datared_femto_dist_range = QLineEdit("")
        self.datared_femto_dist_range.setPlaceholderText(
            "Optional, in selected unit. Example: (-5000, 5000)"
        )
        dg.addWidget(self.datared_femto_dist_range, 4, 1)

        self.datared_femto_dist_density = QCheckBox("normalize histogram density")
        dg.addWidget(self.datared_femto_dist_density, 5, 0, 1, 2)

        self.datared_femto_dist_show_median = QCheckBox("show median")
        self.datared_femto_dist_show_median.setChecked(True)
        dg.addWidget(self.datared_femto_dist_show_median, 6, 0, 1, 2)

        self._refresh_femtomax_distribution_widgets()

        self.datared_femto_runtime_group = QGroupBox(
            "FemtoMAX 2D Export Runtime Options"
        )
        frg = QGridLayout()
        self.datared_femto_runtime_group.setLayout(frg)
        layout.addWidget(self.datared_femto_runtime_group)

        self.datared_femto_overwrite = QCheckBox("overwrite")
        self.datared_femto_overwrite.setChecked(True)
        frg.addWidget(self.datared_femto_overwrite, 0, 0, 1, 2)

        frg.addWidget(QLabel("2D image batch_size:"), 1, 0)
        self.datared_femto_batch_size = QLineEdit("1000")
        self.datared_femto_batch_size.setValidator(QDoubleValidator())
        frg.addWidget(self.datared_femto_batch_size, 1, 1)

        self.datared_femto_use_parallel = QCheckBox("use_parallel")
        self.datared_femto_use_parallel.setChecked(True)
        frg.addWidget(self.datared_femto_use_parallel, 2, 0, 1, 2)

        frg.addWidget(QLabel("max_workers:"), 3, 0)
        self.datared_femto_max_workers = QLineEdit("4")
        self.datared_femto_max_workers.setValidator(QDoubleValidator())
        frg.addWidget(self.datared_femto_max_workers, 3, 1)

        frg.addWidget(QLabel("2D group chunk_size:"), 4, 0)
        self.datared_femto_chunk_size = QLineEdit("1")
        self.datared_femto_chunk_size.setValidator(QDoubleValidator())
        frg.addWidget(self.datared_femto_chunk_size, 4, 1)

        frg.addWidget(QLabel("start_method:"), 5, 0)
        self.datared_femto_start_method = QComboBox()
        self.datared_femto_start_method.addItems(["spawn", "forkserver", "fork"])
        frg.addWidget(self.datared_femto_start_method, 5, 1)

        self.datared_femto_actions_group = QGroupBox("Actions")
        fal = QHBoxLayout()
        self.datared_femto_actions_group.setLayout(fal)
        layout.addWidget(self.datared_femto_actions_group)

        self.datared_femto_plot_dist_btn = QPushButton("Plot FemtoMAX Ping Distribution")
        self.datared_femto_plot_dist_btn.clicked.connect(
            self._plot_femtomax_ping_distribution
        )
        fal.addWidget(self.datared_femto_plot_dist_btn)

        self.datared_femto_create_h5_btn = QPushButton("Create FemtoMAX Metadata H5")
        self.datared_femto_create_h5_btn.clicked.connect(
            self._create_femtomax_metadata_h5
        )
        fal.addWidget(self.datared_femto_create_h5_btn)

        self.datared_femto_create_2d_btn = QPushButton("Create FemtoMAX 2D Images")
        self.datared_femto_create_2d_btn.clicked.connect(
            self._create_femtomax_2d_images
        )
        fal.addWidget(self.datared_femto_create_2d_btn)
        fal.addStretch()

        self._refresh_femtomax_scan_type_widgets()
        self._load_femtomax_ping_references(log_success=False)

    def _femtomax_ping_reference_path(self) -> str:
        """Return the selected FemtoMAX ping-reference CSV, or the packaged default."""
        path = getattr(self.state, "femtomax_ping_reference_path", None)
        if path is None:
            return self.preparation_service.default_femtomax_ping_reference_path()
        return str(path)

    def sync_femtomax_ping_reference_from_state(self):
        """Refresh the preparation summary after a Session-tab selection."""
        self._load_femtomax_ping_references(log_success=False)

    def _load_femtomax_ping_references(
        self,
        *_args,
        log_success: bool = True,
    ):
        """Load and validate FemtoMAX ping references, then update the relevant controls."""
        try:
            table = self.preparation_service.validate_femtomax_ping_reference_file(
                self._femtomax_ping_reference_path(),
                scans_text=self.datared_femto_scans.text(),
            )
            self.state.femtomax_ping_reference_path = self.path_service.normalize(
                table.path
            )
            self.datared_femto_ping_reference_status.setText(
                f"{table.path} — {len(table.ranges)} ranges; coverage "
                f"{table.scan_min}-{table.scan_max}; SHA-256 "
                f"{table.sha256[:12]}…"
            )
            self.datared_femto_ping_reference_status.setStyleSheet(
                "color: #287a3d;"
            )
            if log_success:
                self.log(
                    f"FemtoMAX ping references loaded: {table.path} "
                    f"({len(table.ranges)} ranges)."
                )
            return table
        except Exception as exc:
            self.datared_femto_ping_reference_status.setText(
                f"Ping references not loaded: {exc}"
            )
            self.datared_femto_ping_reference_status.setStyleSheet(
                "color: #b33a3a;"
            )
            if log_success:
                self.log(f"FemtoMAX Ping Reference Error: {exc}")
            return None

    def _refresh_femtomax_scan_type_widgets(self, *_args):
        """Update FemtoMAX controls whose meaning depends on scan type."""
        scan_type = self.datared_femto_scan_type.currentText().strip().lower()
        is_delay = scan_type == "delay"
        is_fluence = scan_type == "fluence"
        is_dark = scan_type == "dark"

        self.datared_femto_fluences_label.setVisible(is_fluence)
        self.datared_femto_fluences.setVisible(is_fluence)

        self.experiment_metadata.set_field_visible("excitation_wl_nm", not is_dark)
        self.experiment_metadata.set_field_visible("fluence_mJ_cm2", is_delay)
        self.experiment_metadata.set_field_visible("time_window_fs", not is_dark)

        if is_fluence and self.datared_femto_selected_delays.text().strip().lower() in {
            "",
            "auto",
        }:
            self.datared_femto_selected_delays.setText("[-1000]")

    def _refresh_femtomax_distribution_widgets(self, *_args):
        """Update delay-distribution controls for the selected plot view."""
        is_histogram = self.datared_femto_dist_view.currentText() == "hist"
        for widget in (
            self.datared_femto_dist_bins_label,
            self.datared_femto_dist_bins,
            self.datared_femto_dist_range_label,
            self.datared_femto_dist_range,
            self.datared_femto_dist_density,
        ):
            widget.setEnabled(is_histogram)

    def _build_analysis_paths(self):
        """Build normalized raw and analysis paths from the shared GUI state."""
        return self.path_service.build_analysis_paths(
            path_root=self.state.path_root,
            analysis_subdir=self.state.analysis_subdir,
            raw_subdir=self.state.raw_subdir,
        )
    
    def _plot_femtomax_ping_distribution(self):
        """Parse the FemtoMAX ping distribution settings, invoke the plot backend, and report the saved path."""
        try:
            if self.state.facility != "FemtoMAX":
                raise ValueError(
                    "This 2D-preparation backend is currently implemented only for FemtoMAX."
                )

            paths = self._build_analysis_paths()

            self.preparation_service.plot_femtomax_ping_distribution(
                scans_text=self.datared_femto_scans.text(),
                mode=self.datared_femto_dist_mode.currentText(),
                delay_source=self.datared_femto_delay_source.currentText(),
                unit=self.datared_femto_dist_unit.currentText(),
                view=self.datared_femto_dist_view.currentText(),
                bins_text=self.datared_femto_dist_bins.text(),
                hist_range_text=self.datared_femto_dist_range.text(),
                density=self.datared_femto_dist_density.isChecked(),
                show_median=self.datared_femto_dist_show_median.isChecked(),
                require_both=self.datared_femto_require_both.isChecked(),
                reference_path_text=self._femtomax_ping_reference_path(),
                paths=paths,
            )

            self.log("FemtoMAX ping / delay distribution plotted.")

        except Exception as exc:
            self.log(f"FemtoMAX Ping Distribution Error: {exc}")

    def _init_single_shot_group(self, layout: QVBoxLayout):
        """Create raw-frame to single-shot 1D production controls."""
        self.datared_single_shot_group = QGroupBox("Single-Shot 1D Production")
        grid = QGridLayout()
        self.datared_single_shot_group.setLayout(grid)
        layout.addWidget(self.datared_single_shot_group)

        grid.addWidget(QLabel("Metadata HDF5 (optional):"), 0, 0)
        self.datared_single_shot_metadata_h5 = DropPathLineEdit("", mode="file")
        self.datared_single_shot_metadata_h5.setPlaceholderText(
            "FemtoMAX: inferred from Experiment Metadata; SACLA: select or drop"
        )
        grid.addWidget(self.datared_single_shot_metadata_h5, 0, 1, 1, 2)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_single_shot_metadata)
        grid.addWidget(browse, 0, 3)

        grid.addWidget(QLabel("Azimuthal edges [deg]:"), 1, 0)
        self.datared_single_shot_azimuthal_edges = QLineEdit(
            "-90, -60, -30, 0, 30, 60, 90"
        )
        grid.addWidget(self.datared_single_shot_azimuthal_edges, 1, 1, 1, 3)

        self.datared_single_shot_include_full = QCheckBox("include_full")
        self.datared_single_shot_include_full.setChecked(True)
        grid.addWidget(self.datared_single_shot_include_full, 2, 0)
        grid.addWidget(QLabel("Full azimuthal range [deg]:"), 2, 1)
        self.datared_single_shot_full_range = QLineEdit("(-90, 90)")
        grid.addWidget(self.datared_single_shot_full_range, 2, 2, 1, 2)

        grid.addWidget(QLabel("Number of q points:"), 3, 0)
        self.datared_single_shot_npt = QLineEdit("1000")
        self.datared_single_shot_npt.setValidator(QIntValidator(1, 1000000))
        grid.addWidget(self.datared_single_shot_npt, 3, 1)

        grid.addWidget(QLabel("Azimuth offset [deg]:"), 3, 2)
        self.datared_single_shot_azim_offset_deg = QLineEdit(
            str(getattr(self.state, "azim_offset_deg", -90.0))
        )
        self.datared_single_shot_azim_offset_deg.setValidator(QDoubleValidator())
        self.datared_single_shot_azim_offset_deg.editingFinished.connect(
            self._sync_single_shot_geometry_to_state
        )
        grid.addWidget(self.datared_single_shot_azim_offset_deg, 3, 3)

        self.datared_single_shot_normalize_final = QCheckBox(
            "normalize final 1D pattern"
        )
        self.datared_single_shot_normalize_final.setChecked(True)
        grid.addWidget(self.datared_single_shot_normalize_final, 4, 0)
        grid.addWidget(QLabel("Final Q normalization range:"), 4, 1)
        self.datared_single_shot_q_norm_range = QLineEdit(
            str(getattr(self.state, "q_norm_range", "(2.65, 2.75)"))
        )
        grid.addWidget(self.datared_single_shot_q_norm_range, 4, 2, 1, 2)

        self.datared_single_shot_polarization_control = PolarizationControlWidget(
            enabled=getattr(self.state, "polarization_enabled", True),
            factor=(
                0.99
                if getattr(self.state, "polarization_factor", 0.99) is None
                else getattr(self.state, "polarization_factor", 0.99)
            ),
        )
        self.datared_single_shot_polarization_control.valueChanged.connect(
            self._on_polarization_changed
        )
        grid.addWidget(
            self.datared_single_shot_polarization_control,
            5,
            0,
            1,
            4,
        )
        self.datared_single_shot_group.setToolTip(
            "Single-shot cache patterns use the PONI and mask selected in Session, "
            "plus these radial-grid, azimuthal, and polarization settings. Final "
            "q-range normalization is applied later in Azimuthal Integration after "
            "shot averaging."
        )

        self.datared_sacla_labels = []
        self.datared_sacla_fields = []
        beamline_label = QLabel("SACLA beamline:")
        self.datared_sacla_beamline = QLineEdit("")
        self.datared_sacla_beamline.setPlaceholderText("From metadata (default 3)")
        detector_label = QLabel("Detector ID:")
        self.datared_sacla_detector_id = QLineEdit("MPCCD-8N0-3-002")
        grid.addWidget(beamline_label, 6, 0)
        grid.addWidget(self.datared_sacla_beamline, 6, 1)
        grid.addWidget(detector_label, 6, 2)
        grid.addWidget(self.datared_sacla_detector_id, 6, 3)

        background_label = QLabel("Background run:")
        self.datared_sacla_background = QLineEdit("")
        self.datared_sacla_background.setPlaceholderText("Optional")
        threshold_label = QLabel("Threshold [counts]:")
        self.datared_sacla_threshold_counts = QLineEdit("40")
        grid.addWidget(background_label, 7, 0)
        grid.addWidget(self.datared_sacla_background, 7, 1)
        grid.addWidget(threshold_label, 7, 2)
        grid.addWidget(self.datared_sacla_threshold_counts, 7, 3)

        intensity_label = QLabel("Pulse intensity column:")
        self.datared_sacla_intensity_col = QLineEdit("")
        self.datared_sacla_intensity_col.setPlaceholderText(
            "From metadata (xfel_bl_3_st_2_pd_user_9_fitting_peak/voltage)"
        )
        grid.addWidget(intensity_label, 8, 0)
        grid.addWidget(self.datared_sacla_intensity_col, 8, 1, 1, 3)

        chunks_label = QLabel("PBS array chunks:")
        self.datared_sacla_n_chunks = QLineEdit("20")
        self.datared_sacla_n_chunks.setValidator(QIntValidator(1, 1000000))
        self.datared_sacla_n_chunks.setToolTip(
            "Number of disjoint scheduler array tasks used for SACLA detector access."
        )
        grid.addWidget(chunks_label, 9, 0)
        grid.addWidget(self.datared_sacla_n_chunks, 9, 1)
        self.datared_sacla_labels.extend(
            [
                beamline_label,
                detector_label,
                background_label,
                threshold_label,
                intensity_label,
                chunks_label,
            ]
        )
        self.datared_sacla_fields.extend(
            [
                self.datared_sacla_beamline,
                self.datared_sacla_detector_id,
                self.datared_sacla_background,
                self.datared_sacla_threshold_counts,
                self.datared_sacla_intensity_col,
                self.datared_sacla_n_chunks,
            ]
        )

        self.datared_femtomax_single_shot_labels = []
        self.datared_femtomax_single_shot_fields = []
        batch_label = QLabel("HDF5 frame batch size:")
        self.datared_femtomax_read_batch_size = QLineEdit("16")
        self.datared_femtomax_read_batch_size.setValidator(
            QIntValidator(1, 4096)
        )
        self.datared_femtomax_read_batch_size.setToolTip(
            "Selected FemtoMAX frames read per HDF5 operation. Larger values "
            "reduce I/O overhead but use more memory."
        )
        grid.addWidget(batch_label, 9, 0)
        grid.addWidget(self.datared_femtomax_read_batch_size, 9, 1)
        self.datared_femtomax_single_shot_labels.append(batch_label)
        self.datared_femtomax_single_shot_fields.append(
            self.datared_femtomax_read_batch_size
        )

        work_chunk_label = QLabel("Shots per worker task:")
        self.datared_femtomax_work_chunk_size = QLineEdit("64")
        self.datared_femtomax_work_chunk_size.setValidator(
            QIntValidator(1, 1000000)
        )
        self.datared_femtomax_work_chunk_size.setToolTip(
            "Coarse scheduling unit. Tasks are interleaved across delays or "
            "fluences, while progress is reported after each HDF5 frame batch."
        )
        grid.addWidget(work_chunk_label, 9, 2)
        grid.addWidget(self.datared_femtomax_work_chunk_size, 9, 3)

        self.datared_femtomax_single_shot_use_parallel = QCheckBox(
            "parallel processing"
        )
        self.datared_femtomax_single_shot_use_parallel.setChecked(True)
        grid.addWidget(self.datared_femtomax_single_shot_use_parallel, 10, 0)

        workers_label = QLabel("Parallel workers:")
        self.datared_femtomax_single_shot_max_workers = QLineEdit("4")
        self.datared_femtomax_single_shot_max_workers.setValidator(
            QIntValidator(1, 1024)
        )
        grid.addWidget(workers_label, 10, 1)
        grid.addWidget(self.datared_femtomax_single_shot_max_workers, 10, 2)

        start_method_label = QLabel("Start method:")
        self.datared_femtomax_single_shot_start_method = QComboBox()
        self.datared_femtomax_single_shot_start_method.addItems(
            ["spawn", "forkserver", "fork"]
        )
        grid.addWidget(start_method_label, 11, 0)
        grid.addWidget(self.datared_femtomax_single_shot_start_method, 11, 1)

        self.datared_femtomax_single_shot_labels.extend(
            [work_chunk_label, workers_label, start_method_label]
        )
        self.datared_femtomax_single_shot_fields.extend(
            [
                self.datared_femtomax_work_chunk_size,
                self.datared_femtomax_single_shot_use_parallel,
                self.datared_femtomax_single_shot_max_workers,
                self.datared_femtomax_single_shot_start_method,
            ]
        )

        self.datared_overwrite_single_shot = QCheckBox(
            "overwrite_single_shot_1d"
        )
        self.datared_overwrite_single_shot.setChecked(False)
        grid.addWidget(self.datared_overwrite_single_shot, 11, 2, 1, 2)

        self.datared_integrate_single_shot_btn = QPushButton(
            "Produce Single-Shot 1D Patterns"
        )
        self.datared_integrate_single_shot_btn.clicked.connect(
            self._integrate_single_shot_1d
        )
        grid.addWidget(self.datared_integrate_single_shot_btn, 12, 0, 1, 4)

    def _browse_single_shot_metadata(self):
        """Select the facility metadata HDF5 used for shot production."""
        start = self.datared_single_shot_metadata_h5.text().strip()
        if not start:
            root = getattr(self.state, "path_root", None)
            start = "" if root is None else str(root)
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select single-shot metadata HDF5",
            start,
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if selected:
            self.datared_single_shot_metadata_h5.setText(selected)

    def _sync_single_shot_geometry_to_state(self):
        """Persist the visible single-shot azimuthal offset in shared state."""
        offset_text = self.datared_single_shot_azim_offset_deg.text().strip()
        if offset_text:
            try:
                self.state.azim_offset_deg = float(offset_text)
            except ValueError:
                self.log("Azimuth offset must be a number in degrees.")

    def _on_polarization_changed(self, enabled: bool, factor: float):
        """Persist and propagate the single-shot polarization correction."""
        self.state.polarization_enabled = bool(enabled)
        self.state.polarization_factor = float(factor)
        if self.polarization_changed_callback is not None:
            self.polarization_changed_callback(bool(enabled), float(factor))

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

    def _polarization_factor(self):
        """Return the shared polarization factor when correction is enabled."""
        control = getattr(
            self,
            "datared_single_shot_polarization_control",
            None,
        )
        if control is not None:
            return control.effective_factor()
        if not bool(getattr(self.state, "polarization_enabled", True)):
            return None
        factor = getattr(self.state, "polarization_factor", 0.99)
        return 0.99 if factor is None else float(factor)

    def _integrate_single_shot_1d(self):
        """Start facility raw-frame integration into the separate shot cache."""
        try:
            facility = self.state.facility
            if facility not in {"FemtoMAX", "SACLA"}:
                raise ValueError(
                    "Single-shot 1D integration is available for FemtoMAX and SACLA."
                )
            paths = self._build_analysis_paths()
            metadata_path = self.datared_single_shot_metadata_h5.text().strip()
            if not metadata_path:
                if facility != "FemtoMAX":
                    raise ValueError(
                        "Select the metadata HDF5 file for SACLA single-shot processing."
                    )
                scan_type = self.datared_femto_scan_type.currentText().strip().lower()
                scans = self.preparation_service.parse_femtomax_scans(
                    self.datared_femto_scans.text()
                )
                if isinstance(scans, int):
                    scans = [scans]
                delay_fs = None
                if scan_type == "fluence":
                    selected_delays = (
                        self.preparation_service.parse_femtomax_selected_delays(
                            self.datared_femto_selected_delays.text()
                        )
                    )
                    if isinstance(selected_delays, (int, float)):
                        selected_delays = [selected_delays]
                    if isinstance(selected_delays, str) or len(selected_delays) != 1:
                        raise ValueError(
                            "Automatic FemtoMAX fluence metadata selection requires "
                            "exactly one selected delay."
                        )
                    delay_fs = int(selected_delays[0])
                metadata_path = self.integration_service.resolve_single_shot_metadata_h5(
                    facility=facility,
                    explicit_path="",
                    scan_type=scan_type,
                    metadata_values=self.experiment_metadata.values(),
                    paths=paths,
                    delay_fs=delay_fs,
                    scans=scans,
                )
                self.datared_single_shot_metadata_h5.setText(metadata_path)

            kwargs = self.integration_service.build_single_shot_integration_kwargs(
                metadata_h5_path=metadata_path,
                poni_path=self._poni_path(),
                mask_edf_path=self._mask_path(),
                azimuthal_edges_text=(
                    self.datared_single_shot_azimuthal_edges.text()
                ),
                include_full=self.datared_single_shot_include_full.isChecked(),
                full_range_text=self.datared_single_shot_full_range.text(),
                npt_text=self.datared_single_shot_npt.text(),
                overwrite=self.datared_overwrite_single_shot.isChecked(),
                paths=paths,
                polarization_factor=self._polarization_factor(),
                azim_offset_deg=self.integration_service.parse_azim_offset_deg(
                    self.datared_single_shot_azim_offset_deg.text()
                ),
                facility=facility,
                sacla_beamline_text=self.datared_sacla_beamline.text(),
                sacla_detector_id_text=self.datared_sacla_detector_id.text(),
                sacla_background_text=self.datared_sacla_background.text(),
                sacla_threshold_counts_text=(
                    self.datared_sacla_threshold_counts.text()
                ),
                sacla_intensity_col_text=self.datared_sacla_intensity_col.text(),
                femtomax_read_batch_size_text=(
                    self.datared_femtomax_read_batch_size.text()
                ),
                femtomax_work_chunk_size_text=(
                    self.datared_femtomax_work_chunk_size.text()
                ),
                femtomax_use_parallel=(
                    self.datared_femtomax_single_shot_use_parallel.isChecked()
                ),
                femtomax_max_workers_text=(
                    self.datared_femtomax_single_shot_max_workers.text()
                ),
                femtomax_start_method=(
                    self.datared_femtomax_single_shot_start_method.currentText()
                ),
            )

            sacla_n_chunks = None
            if facility == "SACLA":
                sacla_n_chunks = int(self.datared_sacla_n_chunks.text())
                if sacla_n_chunks < 1:
                    raise ValueError("SACLA PBS array chunks must be at least 1.")

            cancel_event = threading.Event() if facility == "FemtoMAX" else None
            if cancel_event is not None:
                kwargs["cancel_event"] = cancel_event

            def task():
                if facility == "SACLA":
                    return self.integration_service.submit_sacla_single_shot_1d(
                        integration_kwargs=kwargs,
                        n_chunks=sacla_n_chunks,
                    )
                return self.integration_service.integrate_single_shot_1d(
                    facility=facility,
                    **kwargs,
                )

            def success(report):
                if report.get("submitted"):
                    self.log(
                        "SACLA single-shot 1D PBS array submitted. "
                        "Job: {}; chunks: {}. Production continues on the scheduler.".format(
                            report.get("job_id", "unknown"),
                            report.get("n_chunks", "?"),
                        )
                    )
                    return
                if report.get("cancelled"):
                    self.log(
                        f"{facility} single-shot 1D production stopped safely. "
                        f"Written before stopping: {report['written_patterns']}; "
                        "relaunch with overwrite disabled to resume."
                    )
                    return
                execution = ""
                if facility == "FemtoMAX":
                    if report.get("use_parallel"):
                        execution = " Parallel workers: {}.".format(
                            report.get("max_workers", "?")
                        )
                    else:
                        execution = " Serial execution."
                elif int(report.get("n_chunks", 1)) > 1:
                    execution = " Array chunk {}/{}.".format(
                        report.get("chunk_id", "?"),
                        report.get("n_chunks", "?"),
                    )
                self.log(
                    f"{facility} single-shot 1D production finished. "
                    f"Written: {report['written_patterns']}; "
                    f"already present: {report['existing_patterns']}; "
                    f"invalid shots skipped: {report['invalid_shots']}."
                    + execution
                )

            run_task_with_output_dialog(
                self,
                f"Produce {facility} Single-Shot 1D Patterns",
                task,
                on_success=success,
                on_error=lambda tb: self.log(
                    "Single-Shot 1D Production Error: "
                    + next(
                        (
                            line.strip()
                            for line in reversed(str(tb).splitlines())
                            if line.strip()
                        ),
                        "unknown error",
                    )
                ),
                auto_close_on_success=False,
                cancel_callback=(
                    cancel_event.set if cancel_event is not None else None
                ),
            )
        except Exception as exc:
            self.log(f"Single-Shot 1D Production Error: {exc}")

    def _init_other_facilities_group(self, layout: QVBoxLayout):
        """Create the placeholder shown when no preparation backend is available."""
        self.datared_placeholder_group = QGroupBox("Other Facilities")
        placeholder_layout = QVBoxLayout()
        self.datared_placeholder_group.setLayout(placeholder_layout)
        layout.addWidget(self.datared_placeholder_group)

        placeholder_text = QLabel(
            "No facility-specific data-reduction controls are available for "
            "the selected facility."
        )
        placeholder_text.setWordWrap(True)
        placeholder_layout.addWidget(placeholder_text)
    
    def set_facility(self, facility: str):
        """Activate the preparation controls supported by a facility.

        The facility is stored in shared GUI state. FemtoMAX, ID09, and SACLA
        control groups are shown or hidden accordingly, and the overview text is
        refreshed to describe the active workflow.
        """

        is_id09 = facility == "ID09"
        is_femto = facility == "FemtoMAX"
        is_sacla = facility == "SACLA"
        supports_single_shot = is_femto or is_sacla

        self.experiment_metadata.id09_group.setVisible(is_id09)

        self.datared_id09_group.setVisible(is_id09)
        self.datared_runtime_group.setVisible(is_id09)
        self.datared_actions_group.setVisible(is_id09)

        self.datared_femto_group.setVisible(is_femto)
        self.datared_femto_dist_group.setVisible(is_femto)
        self.datared_femto_runtime_group.setVisible(is_femto)
        self.datared_femto_actions_group.setVisible(is_femto)

        self.datared_single_shot_group.setVisible(supports_single_shot)
        for widget in self.datared_sacla_labels + self.datared_sacla_fields:
            widget.setVisible(is_sacla)
        for widget in (
            self.datared_femtomax_single_shot_labels
            + self.datared_femtomax_single_shot_fields
        ):
            widget.setVisible(is_femto)

        self.datared_placeholder_group.setVisible(
            not is_id09 and not supports_single_shot
        )

        if is_femto:
            self._refresh_femtomax_scan_type_widgets()
            self._load_femtomax_ping_references(log_success=False)
        else:
            self.experiment_metadata.set_field_visible("excitation_wl_nm", True)
            self.experiment_metadata.set_field_visible("fluence_mJ_cm2", True)
            self.experiment_metadata.set_field_visible("time_window_fs", True)

        if is_id09:
            self.datared_note.setText(
                "This section is active for ESRF-ID09.\n"
                "Use it to create the homogeneous dark 2D image and the delay-resolved 2D images "
                "inside the standard analysis structure."
            )
        elif is_femto:
            self.datared_note.setText(
                "This section is active for MAX IV FemtoMAX.\n"
                "Use it to inspect ping / delay distributions, create metadata HDF5 files, "
                "export averaged 2D detector images, or progressively produce single-shot "
                "1D patterns."
            )
        elif is_sacla:
            self.datared_note.setText(
                "This section is active for SACLA.\n"
                "Use the facility metadata HDF5 to progressively produce the single-shot "
                "1D cache. Existing representative-2D preparation remains available through "
                "the SACLA reduction workflow."
            )
        else:
            self.datared_note.setText(
                "This section is the general home for facility metadata and raw-data reduction."
            )
    
    def _build_femtomax_common_kwargs(self):
        """Validate shared FemtoMAX fields and assemble backend keyword arguments."""
        paths = self.path_service.build_analysis_paths(
            path_root=self.state.path_root,
            analysis_subdir=self.state.analysis_subdir,
            raw_subdir=self.state.raw_subdir,
        )

        return self.preparation_service.build_femtomax_common_kwargs(
            metadata_values=self.experiment_metadata.values(),
            scans_text=self.datared_femto_scans.text(),
            scan_type=self.datared_femto_scan_type.currentText(),
            selected_delays_text=self.datared_femto_selected_delays.text(),
            delay_source=self.datared_femto_delay_source.currentText(),
            require_both=self.datared_femto_require_both.isChecked(),
            nb_shot_threshold_text=self.datared_femto_nb_shot_threshold.text(),
            overwrite=self.datared_femto_overwrite.isChecked(),
            paths=paths,
            fluences_text=self.datared_femto_fluences.text(),
            reference_path_text=self._femtomax_ping_reference_path(),
        )


    def _create_femtomax_metadata_h5(self):
        """Validate FemtoMAX controls and generate standardized metadata HDF5."""
        try:
            def error_summary(traceback_text):
                lines = [
                    line.strip()
                    for line in str(traceback_text).splitlines()
                    if line.strip()
                ]
                return lines[-1] if lines else "unknown error"

            if self.state.facility != "FemtoMAX":
                raise ValueError(
                    "This 2D-preparation backend is currently implemented only for FemtoMAX."
                )

            kwargs = self._build_femtomax_common_kwargs()

            def task():
                return self.preparation_service.create_femtomax_metadata_h5(
                    **kwargs
                )

            run_task_with_output_dialog(
                self,
                "Create FemtoMAX Metadata H5",
                task,
                on_success=lambda _result: self.log(
                    "FemtoMAX metadata H5 creation finished."
                ),
                on_error=lambda tb: self.log(
                    f"FemtoMAX Create H5 Error: {error_summary(tb)}"
                ),
                auto_close_on_success=False,
            )

        except Exception as exc:
            self.log(f"FemtoMAX Create H5 Error: {exc}")
    
    def _create_femtomax_2d_images(self):
        """Validate the FemtoMAX 2D images fields and delegate artifact creation to the active facility service."""
        try:
            def error_summary(traceback_text):
                lines = [
                    line.strip()
                    for line in str(traceback_text).splitlines()
                    if line.strip()
                ]
                return lines[-1] if lines else "unknown error"

            if self.state.facility != "FemtoMAX":
                raise ValueError(
                    "This 2D-preparation backend is currently implemented only for FemtoMAX."
                )

            kwargs = self._build_femtomax_common_kwargs()
            kwargs.update(
                batch_size=int(float(self.datared_femto_batch_size.text())),
                use_parallel=self.datared_femto_use_parallel.isChecked(),
                max_workers=int(float(self.datared_femto_max_workers.text())),
                chunk_size=int(float(self.datared_femto_chunk_size.text())),
                start_method=self.datared_femto_start_method.currentText(),
            )

            def task():
                return self.preparation_service.generate_femtomax_2d_images(
                    **kwargs
                )

            def success(task_result):
                _exp, result = task_result
                if isinstance(result, dict):
                    summary = f"{len(result)} result entries"
                else:
                    summary = str(result)
                self.log(
                    f"FemtoMAX 2D image creation finished. Result: {summary}"
                )

            run_task_with_output_dialog(
                self,
                "Create FemtoMAX 2D Images",
                task,
                on_success=success,
                on_error=lambda tb: self.log(
                    f"FemtoMAX Create 2D Images Error: {error_summary(tb)}"
                ),
                auto_close_on_success=False,
            )

        except Exception as exc:
            self.log(f"FemtoMAX Create 2D Images Error: {exc}")
