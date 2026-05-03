from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt, QTimer, QByteArray
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .services import SimulationService
from .state import AUTOSAVE_FILENAME, GuiState
from .tabs import PolycrystallineTab, SingleCrystalTab
from .widgets import MatrixRotationWindow


class MainWindow(QMainWindow):
    """
    Main simulation GUI window.

    This is the new top-level window for the refactored GUI package. It owns
    session persistence, autosave, summary/logging, and composes the extracted
    polycrystalline tab, single-crystal tab, and matrix-rotation helper.
    """

    def __init__(self) -> None:
        super().__init__()

        self._loading_gui_state = False
        self._run_counter = 0

        self.state = GuiState()
        self.service = SimulationService()

        self.matrix_window = MatrixRotationWindow(self)

        self.setWindowTitle("XRDpy Simulation GUI")
        self.resize(750, 850)

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._autosave_now)

        self._summary_timer = QTimer(self)
        self._summary_timer.setSingleShot(True)
        self._summary_timer.timeout.connect(self._refresh_summary)

        self._build_ui()
        self._connect_stateful_widgets()

        self.single_tab.set_matrix_rotation_window(self.matrix_window)

        self._default_state = self._gather_gui_state(include_log=False)
        self._refresh_summary()
        self._maybe_restore_autosave()
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        container = QWidget()
        self.setCentralWidget(container)

        root_layout = QVBoxLayout()
        container.setLayout(root_layout)

        self._build_session_controls(root_layout)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self.poly_tab = PolycrystallineTab(service=self.service, state=self.state, parent=self)
        self.single_tab = SingleCrystalTab(service=self.service, state=self.state, parent=self)

        self.tabs.addTab(self.poly_tab, "Polycrystalline")
        self.tabs.addTab(self.single_tab, "Single Crystal")

        self._build_session_tab()
        self.tabs.addTab(self.session_tab, "Session / Log")

        bottom_buttons_layout = QHBoxLayout()

        self.open_rotation_button = QPushButton("Open Matrix Rotation Tool")
        self.open_rotation_button.clicked.connect(self._open_matrix_rotation_window)
        bottom_buttons_layout.addWidget(self.open_rotation_button)

        self.close_all_plots_button = QPushButton("Close All Plots")
        self.close_all_plots_button.clicked.connect(self._close_all_plots)
        bottom_buttons_layout.addWidget(self.close_all_plots_button)

        bottom_buttons_layout.addStretch()
        root_layout.addLayout(bottom_buttons_layout)

    def _build_session_controls(self, root_layout: QVBoxLayout) -> None:
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

        root_layout.addWidget(self.session_group)

    def _build_session_tab(self) -> None:
        self.session_tab = QWidget()
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
    # External actions
    # ------------------------------------------------------------------
    def _open_matrix_rotation_window(self) -> None:
        self.matrix_window.show()
        self.matrix_window.raise_()
        self.matrix_window.activateWindow()

    def _close_all_plots(self) -> None:
        try:
            plt.close("all")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Plot Close Error",
                f"Could not close matplotlib windows:\n{str(e)}",
            )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _autosave_path(self) -> Path:
        return Path.home() / AUTOSAVE_FILENAME

    def _save_state_dict_to_path(self, state: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_state_dict_from_path(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

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

    def _set_plain_text(self, widget: QPlainTextEdit, value: str | None) -> None:
        with self._blocked(widget):
            widget.setPlainText("" if value is None else str(value))

    def _matrix_texts(self, matrix_edits) -> list[list[str]]:
        return [[cell.text() for cell in row] for row in matrix_edits]

    def _apply_matrix_texts(self, matrix_edits, values) -> None:
        if not values:
            return
        for i, row in enumerate(values[:len(matrix_edits)]):
            for j, value in enumerate(row[:len(matrix_edits[i])]):
                matrix_edits[i][j].setText(str(value))

    def _encode_geometry(self) -> str:
        return bytes(self.saveGeometry().toBase64()).decode("ascii")

    def _restore_geometry(self, value: str) -> None:
        if not value:
            return
        try:
            self.restoreGeometry(QByteArray.fromBase64(value.encode("ascii")))
        except Exception:
            pass

    def _sync_state_from_widgets(self) -> None:
        self.poly_tab.save_to_state()
        self.single_tab.save_to_state()

        self.state.ui.current_tab_index = self.tabs.currentIndex()
        self.state.ui.session_name = self.session_name_line.text()
        self.state.ui.session_notes = self.session_notes.toPlainText()
        self.state.geometry = self._encode_geometry()
        self.state.log = self.run_log.toPlainText()

        self.state.matrix_tool.space_group = self.matrix_window.line_space_group.text()
        self.state.matrix_tool.a = self.matrix_window.line_a.text()
        self.state.matrix_tool.b = self.matrix_window.line_b.text()
        self.state.matrix_tool.c = self.matrix_window.line_c.text()
        self.state.matrix_tool.alpha = self.matrix_window.line_alpha.text()
        self.state.matrix_tool.beta = self.matrix_window.line_beta.text()
        self.state.matrix_tool.gamma = self.matrix_window.line_gamma.text()
        self.state.matrix_tool.orientation_matrix = self._matrix_texts(self.matrix_window.matrix_edits)
        self.state.matrix_tool.rotx = self.matrix_window.line_rotx.text()
        self.state.matrix_tool.roty = self.matrix_window.line_roty.text()
        self.state.matrix_tool.rotz = self.matrix_window.line_rotz.text()

    def _gather_gui_state(self, *, include_log: bool = True) -> dict:
        self._sync_state_from_widgets()
        return self.state.to_dict(include_log=include_log)

    def _apply_gui_state(self, state_dict: dict) -> None:
        self._loading_gui_state = True
        try:
            new_state = GuiState.from_dict(state_dict)
            self.state = new_state

            self.poly_tab.state = self.state
            self.single_tab.state = self.state

            self._set_line_text(self.session_name_line, self.state.ui.session_name)
            self._set_plain_text(self.session_notes, self.state.ui.session_notes)

            self._set_line_text(self.matrix_window.line_space_group, self.state.matrix_tool.space_group)
            self._set_line_text(self.matrix_window.line_a, self.state.matrix_tool.a)
            self._set_line_text(self.matrix_window.line_b, self.state.matrix_tool.b)
            self._set_line_text(self.matrix_window.line_c, self.state.matrix_tool.c)
            self._set_line_text(self.matrix_window.line_alpha, self.state.matrix_tool.alpha)
            self._set_line_text(self.matrix_window.line_beta, self.state.matrix_tool.beta)
            self._set_line_text(self.matrix_window.line_gamma, self.state.matrix_tool.gamma)
            self._apply_matrix_texts(
                self.matrix_window.matrix_edits,
                self.state.matrix_tool.orientation_matrix,
            )
            self._set_line_text(self.matrix_window.line_rotx, self.state.matrix_tool.rotx)
            self._set_line_text(self.matrix_window.line_roty, self.state.matrix_tool.roty)
            self._set_line_text(self.matrix_window.line_rotz, self.state.matrix_tool.rotz)
            self.matrix_window._invalidate_result_matrix()

            self.poly_tab.load_from_state()
            self.single_tab.load_from_state()

            self.run_log.setPlainText(self.state.log)
            self._restore_geometry(self.state.geometry)
            self.tabs.setCurrentIndex(int(self.state.ui.current_tab_index))
        finally:
            self._loading_gui_state = False
            self._refresh_summary()
            self._schedule_autosave()

    def _save_gui_state_to_file(self) -> None:
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

    def _load_gui_state_from_file(self) -> None:
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

    def _load_autosave_from_disk(self) -> None:
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

    def _reset_to_defaults(self) -> None:
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

    def _maybe_restore_autosave(self) -> None:
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
    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.run_log.appendPlainText(f"[{timestamp}] {message}")

    def _clear_log(self) -> None:
        self.run_log.clear()
        self._log("Cleared run log.")

    def _schedule_autosave(self) -> None:
        if self._loading_gui_state:
            return
        self._autosave_timer.start(900)
        self._schedule_summary_refresh()

    def _schedule_summary_refresh(self) -> None:
        if self._loading_gui_state:
            return
        self._summary_timer.start(250)

    def _autosave_now(self) -> None:
        try:
            path = self._autosave_path()
            self._save_state_dict_to_path(self._gather_gui_state(include_log=True), path)
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.autosave_status_label.setText(f"Autosave: saved at {stamp}")
        except Exception as exc:
            self.autosave_status_label.setText(f"Autosave failed: {exc}")

    def _refresh_summary(self) -> None:
        current_tab = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else ""
        geometry_mode = self.single_tab.single_geometry_mode_combo.currentText()
        geometry_active = self.single_tab._single_geometry_enabled()
        legacy_suffix = " (ignored in geometry mode)" if geometry_active else ""

        lines = [
            f"Session name: {self.session_name_line.text().strip() or '(unnamed)'}",
            f"Active tab: {current_tab}",
            "",
            "Polycrystalline",
            f"  Function: {self.poly_tab.poly_func_combo.currentText()}",
            f"  Reflection source: {self.poly_tab.poly_combo_refsrc.currentText()}",
            f"  CIF path: {self.poly_tab.poly_line_cif_path.text().strip() or '(none)'}",
            (
                "  Lattice: SG={sg}, a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}".format(
                    sg=self.poly_tab.poly_line_space_group.text(),
                    a=self.poly_tab.poly_line_sam_a.text(),
                    b=self.poly_tab.poly_line_sam_b.text(),
                    c=self.poly_tab.poly_line_sam_c.text(),
                    alpha=self.poly_tab.poly_line_sam_alpha.text(),
                    beta=self.poly_tab.poly_line_sam_beta.text(),
                    gamma=self.poly_tab.poly_line_sam_gamma.text(),
                )
            ),
            f"  Beam: E={self.poly_tab.poly_line_energy.text()} eV, ΔE/E={self.poly_tab.poly_line_ebw.text()} %",
            f"  Detector: {self.poly_tab.poly_combo_det_type.currentText()}, dist={self.poly_tab.poly_line_dist.text()} m, binning=({self.poly_tab.poly_line_bin_h.text()}, {self.poly_tab.poly_line_bin_v.text()})",
            f"  qmax={self.poly_tab.poly_line_qmax.text()} Å^-1, cones={self.poly_tab.poly_line_cones.text()}, x_axis={self.poly_tab.poly_combo_xaxis.currentText()}, lorentz/polarization={self.poly_tab.poly_chk_lorpol.isChecked()}, FWHM={self.poly_tab.poly_line_fwhm.text()}",
            "",
            "Single Crystal",
            f"  Function: {self.single_tab.single_func_combo.currentText()}",
            f"  Detector: {self.single_tab.single_combo_det_type.currentText()}, dist={self.single_tab.single_line_dist.text()} m, binning=({self.single_tab.single_line_bin_h.text()}, {self.single_tab.single_line_bin_v.text()})",
            f"  Beam: E={self.single_tab.single_line_energy.text()} eV, ΔE/E={self.single_tab.single_line_ebw.text()} %",
            (
                "  Lattice: SG={sg}, a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}".format(
                    sg=self.single_tab.single_line_space_group.text(),
                    a=self.single_tab.single_line_sam_a.text(),
                    b=self.single_tab.single_line_sam_b.text(),
                    c=self.single_tab.single_line_sam_c.text(),
                    alpha=self.single_tab.single_line_sam_alpha.text(),
                    beta=self.single_tab.single_line_sam_beta.text(),
                    gamma=self.single_tab.single_line_sam_gamma.text(),
                )
            ),
            f"  qmax={self.single_tab.single_line_qmax.text()} Å^-1, custom orientation={self.single_tab.single_orientation_checkbox.isChecked()}",
            f"  Geometry mode: {geometry_mode}",
            f"  Legacy sample rotations: ({self.single_tab.single_line_sam_rotx.text()}, {self.single_tab.single_line_sam_roty.text()}, {self.single_tab.single_line_sam_rotz.text()}) deg{legacy_suffix}",
            f"  Legacy detector Euler rotations: ({self.single_tab.single_line_rotx.text()}, {self.single_tab.single_line_roty.text()}, {self.single_tab.single_line_rotz.text()}) deg{legacy_suffix}",
            f"  Geometry kind: {self.single_tab.geometry_panel.current_geometry_name() if self.single_tab._single_predefined_geometry_enabled() else '(custom/legacy)'}",
            f"  Geometry kwargs: {self.single_tab.single_geometry_kwargs.toPlainText().strip() or '(none)'}",
            f"  Sample motor angles: {(json.dumps(self.single_tab.geometry_panel.current_sample_angles(), indent=2) if self.single_tab._single_predefined_geometry_enabled() else (self.single_tab.single_geometry_sample_angles.toPlainText().strip() or '(none)'))}",
            f"  Detector motor angles: {(json.dumps(self.single_tab.geometry_panel.current_detector_angles(), indent=2) if self.single_tab._single_predefined_geometry_enabled() else (self.single_tab.single_geometry_detector_angles.toPlainText().strip() or '(none)'))}",
            f"  Custom sample chain: {self.single_tab.single_custom_sample_chain.toPlainText().strip() or '(none)'}",
            f"  Custom detector chain: {self.single_tab.single_custom_detector_chain.toPlainText().strip() or '(none)'}",
            f"  Bragg hkls: {self.single_tab.single_line_names.text().strip() or '(none)'}",
            f"  Forced extra HKLs: {self.single_tab.single_line_extra_hkls.text().strip() or '(none)'}",
            f"  Legacy detector angle range: {self.single_tab.single_line_angle.text().strip() or '(default)'}{' (ignored in geometry mode)' if geometry_active else ''}",
            f"  Legacy parameter scan: {self.single_tab.single_param1_combo.currentText()} in {self.single_tab.single_line_param1_range.text().strip() or '(none)'}, {self.single_tab.single_param2_combo.currentText()} in {self.single_tab.single_line_param2_range.text().strip() or '(none)'}{' (ignored in geometry mode)' if geometry_active else ''}",
            f"  Geometry sample scan: {self.single_tab.single_geometry_motor1_name.text().strip() or '(motor1)'} in {self.single_tab.single_geometry_motor1_range.text().strip() or '(none)'}, {self.single_tab.single_geometry_motor2_name.text().strip() or '(motor2)'} in {self.single_tab.single_geometry_motor2_range.text().strip() or '(none)'}",
            f"  Geometry detector scan ranges: {self.single_tab.single_geometry_detector_scan_ranges.toPlainText().strip() or '(none)'}",
            f"  Fixed-energy target: hkl={self.single_tab.single_line_target_hkl.text().strip() or '(none)'}, pixel=({self.single_tab.single_line_target_pixel_h.text().strip() or '?'}, {self.single_tab.single_line_target_pixel_v.text().strip() or '?'}) ± {self.single_tab.single_line_target_pixel_tol.text().strip() or '?'} px",
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
    # Signal wiring
    # ------------------------------------------------------------------
    def _connect_stateful_widgets(self) -> None:
        self.session_name_line.textChanged.connect(self._schedule_autosave)
        self.session_notes.textChanged.connect(self._schedule_autosave)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.poly_tab.state_changed.connect(self._schedule_autosave)
        self.single_tab.state_changed.connect(self._schedule_autosave)

        self.poly_tab.run_completed.connect(self._handle_poly_run_completed)
        self.single_tab.run_completed.connect(self._handle_single_run_completed)

        self.poly_tab.poly_run_btn.pressed.connect(
            lambda: self._note_run_requested("polycrystalline", self.poly_tab.poly_func_combo.currentText())
        )
        self.single_tab.single_run_btn.pressed.connect(
            lambda: self._note_run_requested("single-crystal", self.single_tab.single_func_combo.currentText())
        )

        for line in self.matrix_window.findChildren(QLineEdit):
            line.textChanged.connect(self._schedule_autosave)
        for plain in self.matrix_window.findChildren(QPlainTextEdit):
            plain.textChanged.connect(self._schedule_autosave)

    def _on_tab_changed(self, index: int) -> None:
        self.state.ui.current_tab_index = index
        self._schedule_autosave()

    def _note_run_requested(self, family: str, func_name: str) -> None:
        self._run_counter += 1
        self._log(f"Run #{self._run_counter}: requested {family} function '{func_name}'.")

    def _handle_poly_run_completed(self, ok: bool, func_name: str) -> None:
        self._refresh_summary()
        self._schedule_autosave()
        if ok:
            self.statusBar().showMessage(f"Ran polycrystalline function: {func_name}", 4000)
        else:
            self._log(f"Polycrystalline function '{func_name}' failed.")
            self.statusBar().showMessage(f"Polycrystalline function failed: {func_name}", 4000)

    def _handle_single_run_completed(self, ok: bool, func_name: str) -> None:
        self._refresh_summary()
        self._schedule_autosave()
        if ok:
            self.statusBar().showMessage(f"Ran single-crystal function: {func_name}", 4000)
        else:
            self._log(f"Single-crystal function '{func_name}' failed.")
            self.statusBar().showMessage(f"Single-crystal function failed: {func_name}", 4000)

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        try:
            self._autosave_now()
        finally:
            super().closeEvent(event)


__all__ = [
    "MainWindow",
    "launch_gui",
    "main",
]


def launch_gui() -> MainWindow:
    app = QApplication.instance()
    if app is None:
        raise RuntimeError(
            "No QApplication instance exists. Use main() to start the GUI as a script."
        )

    window = MainWindow()
    window.show()
    return window


def main() -> int:
    app = QApplication.instance()
    owns_app = app is None

    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    if owns_app:
        return app.exec_()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())