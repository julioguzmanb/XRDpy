"""
2D Preparation tab for the analysis GUI.

This reproduces the legacy 2D Preparation tab layout while keeping backend
actions separated from the main window.
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
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from trxrdpy.analysis.gui.services import PathService, PreparationService
from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.widgets import ExperimentMetadataWidget
from trxrdpy.analysis.gui.widgets.task_output_dialog import run_task_with_output_dialog


class PreparationTab(QWidget):
    """
    Legacy-compatible 2D Preparation tab.
    """

    def __init__(
        self,
        state: AnalysisGuiState,
        path_service: PathService,
        preparation_service: PreparationService,
        log: Optional[Callable[[str], None]] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.state = state
        self.path_service = path_service
        self.preparation_service = preparation_service
        self.log = log or (lambda message: None)

        layout = self._make_scroll_layout()

        self.experiment_metadata = ExperimentMetadataWidget(
            title="Experiment Metadata",
            include_id09=True,
        )
        layout.addWidget(self.experiment_metadata)

        self._init_overview_group(layout)
        self._init_id09_groups(layout)
        self._init_femtomax_groups(layout)
        self._init_other_facilities_group(layout)

        self.set_facility(self.state.facility or "SACLA")

        layout.addStretch()

    def _make_scroll_layout(self) -> QVBoxLayout:
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
        note_group = QGroupBox("2D Preparation Overview")
        note_layout = QVBoxLayout()
        note_group.setLayout(note_layout)
        layout.addWidget(note_group)

        self.datared_note = QLabel()
        self.datared_note.setWordWrap(True)
        note_layout.addWidget(self.datared_note)

        msg = QLabel(
            "This tab is the general entry point for facility-specific 2D image production:\n"
            "raw data → homogeneous dark / delay 2D images → shared downstream analysis."
        )
        msg.setWordWrap(True)
        note_layout.addWidget(msg)

    def _init_id09_groups(self, layout: QVBoxLayout):
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
        try:
            def error_summary(traceback_text):
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
        try:
            def error_summary(traceback_text):
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
                return self.preparation_service.create_id09_final_2d_images(**kwargs)

            def success(out_paths):
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

        fg.addWidget(QLabel("scan_type:"), row, 0)
        self.datared_femto_scan_type = QComboBox()
        self.datared_femto_scan_type.addItems(["delay", "fluence", "dark"])
        fg.addWidget(self.datared_femto_scan_type, row, 1)
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
        self.datared_femto_delay_source.addItems(["avg", "p1", "p2", "p3", "p4"])
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
        self.datared_femto_dist_mode.addItems(["overlay", "stacked"])
        dg.addWidget(self.datared_femto_dist_mode, 0, 1)

        dg.addWidget(QLabel("unit:"), 1, 0)
        self.datared_femto_dist_unit = QComboBox()
        self.datared_femto_dist_unit.addItems(["fs", "ps"])
        dg.addWidget(self.datared_femto_dist_unit, 1, 1)

        dg.addWidget(QLabel("view:"), 2, 0)
        self.datared_femto_dist_view = QComboBox()
        self.datared_femto_dist_view.addItems(["scatter", "hist"])
        dg.addWidget(self.datared_femto_dist_view, 2, 1)

        dg.addWidget(QLabel("bins:"), 3, 0)
        self.datared_femto_dist_bins = QLineEdit("250")
        self.datared_femto_dist_bins.setValidator(QDoubleValidator())
        dg.addWidget(self.datared_femto_dist_bins, 3, 1)

        self.datared_femto_runtime_group = QGroupBox("FemtoMAX Export Runtime Options")
        frg = QGridLayout()
        self.datared_femto_runtime_group.setLayout(frg)
        layout.addWidget(self.datared_femto_runtime_group)

        self.datared_femto_overwrite = QCheckBox("overwrite")
        self.datared_femto_overwrite.setChecked(True)
        frg.addWidget(self.datared_femto_overwrite, 0, 0, 1, 2)

        frg.addWidget(QLabel("batch_size:"), 1, 0)
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

        frg.addWidget(QLabel("chunk_size:"), 4, 0)
        self.datared_femto_chunk_size = QLineEdit("1")
        self.datared_femto_chunk_size.setValidator(QDoubleValidator())
        frg.addWidget(self.datared_femto_chunk_size, 4, 1)

        frg.addWidget(QLabel("start_method:"), 5, 0)
        self.datared_femto_start_method = QComboBox()
        self.datared_femto_start_method.addItems(["fork", "spawn", "forkserver"])
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

    def _build_analysis_paths(self):
        return self.path_service.build_analysis_paths(
            path_root=self.state.path_root,
            analysis_subdir=self.state.analysis_subdir,
            raw_subdir=self.state.raw_subdir,
        )
    
    def _plot_femtomax_ping_distribution(self):
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
                paths=paths,
            )

            self.log("FemtoMAX ping / delay distribution plotted.")

        except Exception as exc:
            self.log(f"FemtoMAX Ping Distribution Error: {exc}")

    def _init_other_facilities_group(self, layout: QVBoxLayout):
        self.datared_placeholder_group = QGroupBox("Other Facilities")
        placeholder_layout = QVBoxLayout()
        self.datared_placeholder_group.setLayout(placeholder_layout)
        layout.addWidget(self.datared_placeholder_group)

        placeholder_text = QLabel(
            "SACLA can be added here later without changing the GUI structure.\n"
            "For now, this tab implements the ID09 and FemtoMAX backends."
        )
        placeholder_text.setWordWrap(True)
        placeholder_layout.addWidget(placeholder_text)
    
    def set_facility(self, facility: str):
        """
        Show/hide facility-specific widgets following the legacy GUI behavior.
        """

        is_id09 = facility == "ID09"
        is_femto = facility == "FemtoMAX"

        self.experiment_metadata.id09_group.setVisible(is_id09)

        self.datared_id09_group.setVisible(is_id09)
        self.datared_runtime_group.setVisible(is_id09)
        self.datared_actions_group.setVisible(is_id09)

        self.datared_femto_group.setVisible(is_femto)
        self.datared_femto_dist_group.setVisible(is_femto)
        self.datared_femto_runtime_group.setVisible(is_femto)
        self.datared_femto_actions_group.setVisible(is_femto)

        self.datared_placeholder_group.setVisible(not is_id09 and not is_femto)

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
                "and export averaged 2D detector images into the standard analysis structure."
            )
        else:
            self.datared_note.setText(
                "This section is designed as the general home for facility-specific 2D image production.\n"
                "In this version, ESRF-ID09 and MAX IV FemtoMAX backends are implemented here."
            )
    
    def _build_femtomax_common_kwargs(self):
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
        )


    def _create_femtomax_metadata_h5(self):
        try:
            if self.state.facility != "FemtoMAX":
                raise ValueError(
                    "This 2D-preparation backend is currently implemented only for FemtoMAX."
                )

            kwargs = self._build_femtomax_common_kwargs()
            self.preparation_service.create_femtomax_metadata_h5(**kwargs)

            self.log("FemtoMAX metadata H5 creation finished.")

        except Exception as exc:
            self.log(f"FemtoMAX Create H5 Error: {exc}")
    
    def _create_femtomax_2d_images(self):
        try:
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

            _exp, result = self.preparation_service.generate_femtomax_2d_images(**kwargs)

            if isinstance(result, dict):
                summary = f"{len(result)} result entries"
            else:
                summary = str(result)

            self.log(f"FemtoMAX 2D image creation finished. Result: {summary}")

        except Exception as exc:
            self.log(f"FemtoMAX Create 2D Images Error: {exc}")
