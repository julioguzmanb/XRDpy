"""
Main window for the new analysis GUI.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PyQt5.QtCore import QRectF, QTimer, QUrl
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainterPath, QPalette, QRegion

from PyQt5.QtWidgets import (
    QApplication,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QLabel,
    QHBoxLayout,
    QDialog,
    QAction,
    QDockWidget,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QComboBox,
    QListView,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from trxrdpy.analysis.gui.services import (
    CalibrationService,
    DifferentialService,
    FacilityService,
    FittingService,
    IntegrationService,
    PathService,
    PreparationService,
)
from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.style import LEGACY_MAIN_WINDOW_STYLESHEET, STYLE
from trxrdpy.analysis.gui.runtime_guard import GuiRuntimeGuard
from trxrdpy.analysis.gui.tabs import (
    CalibrationTab,
    DifferentialTab,
    FittingTab,
    PatternCreationTab,
    PreparationTab,
    SessionTab,
    ViewerTab,
)
from trxrdpy.analysis.gui.widgets import LogWidget


_MAIN_WINDOW = None
_RUNTIME_GUARD = None


class AnalysisMainWindow(QMainWindow):
    """Own the analysis GUI tabs, shared state, services, and session lifecycle.

    The window synchronizes experiment metadata across tabs, propagates facility
    and polarization changes, manages the dockable log, and serializes widget
    state for explicit saves and crash-safe autosaves. Scientific work is
    delegated to service objects so callbacks remain orchestration-only.

    Attributes
    ----------
    state : AnalysisGuiState
        Mutable session configuration shared by every tab.
    facility_service, path_service : object
        Facility registry and path-normalization services.
    calibration_service, preparation_service, integration_service : object
        Backend adapters for preparation and integration workflows.
    fitting_service, differential_service : object
        Backend adapters for fitting and differential workflows.
    tabs : QTabWidget
        Primary tab container.
    session_tab, preparation_tab, calibration_tab : QWidget
        Session, 2D-preparation, and calibration workflow tabs.
    pattern_creation_tab, viewer_tab, fitting_tab, differential_tab : QWidget
        Pattern creation, visualization, fitting, and differential-analysis tabs.
    log_widget : LogWidget
        Dockable session message log.
    _metadata_sync_in_progress : bool
        Reentrancy guard used while synchronizing experiment metadata widgets.
    """

    def __init__(self, parent=None, *, launch_directory=None):
        """Initialize ``AnalysisMainWindow``, bind shared state and services, and create its controls."""
        super().__init__(parent)

        self.state = AnalysisGuiState()
        self._allow_close_without_confirmation = False

        self.facility_service = FacilityService()
        self.path_service = PathService(launch_directory=launch_directory)
        # GUI-state files have their own dialog history.  Loading a state from
        # a directory should make a subsequent Save start in that directory,
        # regardless of paths selected elsewhere in the application.
        self._gui_state_directory = self.path_service.launch_directory
        self.calibration_service = CalibrationService()
        self.preparation_service = PreparationService()
        self.integration_service = IntegrationService()
        self.fitting_service = FittingService()
        self.differential_service = DifferentialService()

        self.setWindowTitle(STYLE.main_window_title)
        self.resize(max(STYLE.main_window_width, 1200), STYLE.main_window_height)
        self.setMinimumWidth(900)
        self.setStyleSheet(LEGACY_MAIN_WINDOW_STYLESHEET)

        container = QWidget()
        self.main_container = container
        self.main_splitter = None
        self.setCentralWidget(container)
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.AnimatedDocks
            | QMainWindow.AllowNestedDocks
            | QMainWindow.AllowTabbedDocks
            | QMainWindow.GroupedDragging
        )
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

        layout = QVBoxLayout()
        self.main_layout = layout
        container.setLayout(layout)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.log_widget = LogWidget()
        self.log_widget.text_edit.setObjectName("MainLogText")
        self.log_widget.text_edit.setMaximumBlockCount(STYLE.log_max_block_count)
        self.log_widget.text_edit.setMinimumHeight(STYLE.log_box_fixed_height)

        self._init_log_dock()
        self._init_menu_bar()
        self._init_toolbar()

        self.session_tab = SessionTab(
            state=self.state,
            facility_service=self.facility_service,
            path_service=self.path_service,
            preparation_service=self.preparation_service,
            log=self.log_widget.log,
            save_state_callback=self._save_gui_state_to_file,
            load_state_callback=self._load_gui_state_from_file,
            load_state_path_callback=self._load_gui_state_from_path,
            load_autosave_callback=self._load_autosave_from_disk,
            facility_changed_callback=self._on_facility_changed,
            ping_reference_changed_callback=(
                self._on_femtomax_ping_reference_changed
            ),
        )

        self.preparation_tab = PreparationTab(
            state=self.state,
            path_service=self.path_service,
            preparation_service=self.preparation_service,
            log=self.log_widget.log,
        )
        self.calibration_tab = CalibrationTab(
            state=self.state,
            path_service=self.path_service,
            calibration_service=self.calibration_service,
            log=self.log_widget.log,
            polarization_changed_callback=self._on_polarization_changed,
        )
        self.pattern_creation_tab = PatternCreationTab(
            state=self.state,
            path_service=self.path_service,
            integration_service=self.integration_service,
            log=self.log_widget.log,
            polarization_changed_callback=self._on_polarization_changed,
        )
        self.viewer_tab = ViewerTab(
            state=self.state,
            path_service=self.path_service,
            integration_service=self.integration_service,
            log=self.log_widget.log,
        )
        self.differential_tab = DifferentialTab(
            state=self.state,
            path_service=self.path_service,
            integration_service=self.integration_service,
            differential_service=self.differential_service,
            log=self.log_widget.log,
        )
        self.fitting_tab = FittingTab(
            state=self.state,
            path_service=self.path_service,
            integration_service=self.integration_service,
            fitting_service=self.fitting_service,
            log=self.log_widget.log,
        )

        self.tabs.addTab(self.session_tab, "Session")
        self.tabs.addTab(self.preparation_tab, "2D Image\nCreation")
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self.tabs.addTab(self.pattern_creation_tab, "1D Pattern\nCreation")
        self.tabs.addTab(self.viewer_tab, "1D Viewer")
        self.tabs.addTab(self.differential_tab, "Differential\nAnalysis")
        self.tabs.addTab(self.fitting_tab, "Fitting\nAnalysis")

        self.tabs.currentChanged.connect(
            lambda _index: self._sync_polarization_widgets()
        )
        self._sync_polarization_widgets()


        self._polish_controls()
        self._set_log_bottom_dock()
        self._connect_single_experiment_metadata_sync()
        self._connect_fluence_plot_control_sync()
        self._connect_delay_plot_control_sync()
        self._connect_q_norm_range_sync()
        self._enable_collapsible_group_boxes()
        self._last_single_metadata_widget = self._metadata_widget_for_tab_index(self.tabs.currentIndex())
        self.tabs.currentChanged.connect(self._sync_metadata_on_tab_change)
        self.log_widget.log(
            f"GUI ready. Launch directory: {self.path_service.launch_directory}"
        )
        QTimer.singleShot(0, self._maybe_prompt_restore_autosave)

    def log(self, message: str):
        """Forward a user-facing message to the shared log widget."""
        self.log_widget.log(message)

        if self.statusBar() is not None:
            self.statusBar().showMessage(message, 8000)

    def _enable_collapsible_group_boxes(self):
        """Make each subsection group box collapsible without changing saved state."""
        for group in self.tabs.findChildren(QGroupBox):
            if bool(group.property("_xrdpy_collapsible")):
                continue

            group.setProperty("_xrdpy_collapsible", True)
            group.setCheckable(True)
            group.setChecked(True)

            collapsed_height = max(34, group.fontMetrics().height() + 22)
            expanded_max_height = group.maximumHeight()
            if expanded_max_height <= collapsed_height:
                expanded_max_height = 16777215

            def apply_collapsed_state(
                checked,
                *,
                target=group,
                collapsed=collapsed_height,
                expanded=expanded_max_height,
            ):
                target.setMaximumHeight(expanded if checked else collapsed)

            group.toggled.connect(apply_collapsed_state)

    def _single_experiment_metadata_widgets(self):
        """Return all experiment metadata editors participating in cross-tab synchronization."""
        widgets = []

        candidates = [
            ("preparation_tab", "experiment_metadata"),
            ("pattern_creation_tab", "experiment_metadata"),
            ("viewer_tab", "experiment_metadata"),
            ("differential_tab", "diff_single_metadata"),
            ("fitting_tab", "fit_single_metadata"),
        ]

        for tab_name, widget_name in candidates:
            tab = getattr(self, tab_name, None)
            widget = getattr(tab, widget_name, None)

            if widget is not None and hasattr(widget, "values") and hasattr(widget, "set_values"):
                widgets.append(widget)

        return widgets


    def _calibration_context_widget(self):
        """Return the calibration metadata editor when the tab is available."""
        tab = getattr(self, "calibration_tab", None)

        if tab is None:
            return None

        widget = getattr(tab, "calibration_context", None)

        if widget is not None and hasattr(widget, "values") and hasattr(widget, "set_values"):
            return widget

        return None

    def _sync_single_experiment_metadata_from(self, source_widget):
        """Synchronize single experiment metadata from without recursively emitting changes."""
        if getattr(self, "_metadata_sync_in_progress", False):
            return

        if source_widget is None or not hasattr(source_widget, "values"):
            return

        try:
            values = dict(source_widget.values())
        except Exception:
            return

        calibration_widget = self._calibration_context_widget()
        general_widgets = self._single_experiment_metadata_widgets()

        sample_temperature_values = {}
        for key in ("sample_name", "temperature_K"):
            if key in values:
                sample_temperature_values[key] = values[key]

        self._metadata_sync_in_progress = True

        try:
            # If the source is Calibration Context, only push sample/temperature
            # to the full experiment metadata widgets. Do not overwrite excitation,
            # fluence, time window, ID09 dataset, etc.
            if source_widget is calibration_widget:
                for widget in general_widgets:
                    try:
                        current = dict(widget.values())
                        current.update(sample_temperature_values)
                        widget.blockSignals(True)
                        widget.set_values(current)
                    except Exception:
                        pass
                    finally:
                        try:
                            widget.blockSignals(False)
                        except Exception:
                            pass

                return

            # If the source is a full experiment metadata widget, sync full
            # metadata to other full widgets.
            for widget in general_widgets:
                if widget is source_widget:
                    continue

                try:
                    widget.blockSignals(True)
                    widget.set_values(values)
                except Exception:
                    pass
                finally:
                    try:
                        widget.blockSignals(False)
                    except Exception:
                        pass

            # Also update Calibration Context, but only sample/temperature.
            if calibration_widget is not None and calibration_widget is not source_widget:
                try:
                    current = dict(calibration_widget.values())
                    current.update(sample_temperature_values)
                    calibration_widget.blockSignals(True)
                    calibration_widget.set_values(current)
                except Exception:
                    pass
                finally:
                    try:
                        calibration_widget.blockSignals(False)
                    except Exception:
                        pass

        finally:
            self._metadata_sync_in_progress = False


    def _connect_single_experiment_metadata_sync(self):
        """Connect metadata-change signals for every synchronized experiment editor."""
        self._metadata_sync_in_progress = False

        widgets = self._single_experiment_metadata_widgets()
        calibration_widget = self._calibration_context_widget()

        if calibration_widget is not None:
            widgets.append(calibration_widget)

        for widget in widgets:
            fields = getattr(widget, "fields", {})

            if not isinstance(fields, dict):
                continue

            for field in fields.values():
                if hasattr(field, "editingFinished"):
                    try:
                        field.editingFinished.connect(
                            lambda w=widget: self._sync_single_experiment_metadata_from(w)
                        )
                    except Exception:
                        pass

                if hasattr(field, "textEdited"):
                    try:
                        field.textEdited.connect(
                            lambda _text, w=widget: self._sync_single_experiment_metadata_from(w)
                        )
                    except Exception:
                        pass

    def _q_norm_range_specs(self):
        """Return q-normalization range controls that should stay synchronized."""
        return [
            ("calibration_tab", "calib_q_norm_range"),
            ("pattern_creation_tab", "pattern_q_norm_range"),
            ("viewer_tab", "viewer_q_norm_range"),
            ("differential_tab", "diff_q_norm_range"),
            ("differential_tab", "diff_multi_q_norm_range"),
            ("differential_tab", "diff_multi_fluence_q_norm_range"),
            ("fitting_tab", "fit_q_norm_range"),
        ]

    def _apply_q_norm_range_state_to_controls(self, *, exclude=None):
        """Push the shared q-normalization range into every matching widget."""
        exclude = exclude or set()

        for tab_name, widget_name in self._q_norm_range_specs():
            if (tab_name, widget_name) in exclude:
                continue
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "setText"):
                continue
            try:
                widget.blockSignals(True)
                widget.setText(str(self.state.q_norm_range))
            except Exception:
                pass
            finally:
                try:
                    widget.blockSignals(False)
                except Exception:
                    pass

    def _sync_q_norm_range_from(self, tab_name, widget_name):
        """Persist one edited q-normalization range and propagate it."""
        widget = self._resolve_tab_widget_attr(tab_name, widget_name)
        if widget is None or not hasattr(widget, "text"):
            return
        try:
            self.state.q_norm_range = str(widget.text())
        except Exception:
            return
        self._apply_q_norm_range_state_to_controls(exclude={(tab_name, widget_name)})

    def _capture_q_norm_range_state_from_controls(self):
        """Read the current q-normalization range controls into shared state."""
        for tab_name, widget_name in self._q_norm_range_specs():
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "text"):
                continue
            try:
                self.state.q_norm_range = str(widget.text())
            except Exception:
                pass

    def _connect_q_norm_range_sync(self):
        """Synchronize q-normalization range controls across workflow tabs."""
        self._apply_q_norm_range_state_to_controls()

        for tab_name, widget_name in self._q_norm_range_specs():
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None:
                continue
            if hasattr(widget, "editingFinished"):
                try:
                    widget.editingFinished.connect(
                        lambda tn=tab_name, wn=widget_name:
                        self._sync_q_norm_range_from(tn, wn)
                    )
                except Exception:
                    pass
            if hasattr(widget, "textEdited"):
                try:
                    widget.textEdited.connect(
                        lambda _text, tn=tab_name, wn=widget_name:
                        self._sync_q_norm_range_from(tn, wn)
                    )
                except Exception:
                    pass

    def _fluence_line_edit_specs(self):
        """Return line edits that should stay synchronized across fluence workflows."""
        return [
            ("viewer_tab", "viewer_fluence_delay_fs", "fluence_delay_fs"),
            ("differential_tab", "diff_fluence_delay_fs", "fluence_delay_fs"),
            ("fitting_tab", "fit_fluence_delay_fs", "fluence_delay_fs"),
            ("viewer_tab", "viewer_fluences", "fluence_values"),
            ("differential_tab", "diff_fluences", "fluence_values"),
            ("fitting_tab", "fit_fluences", "fluence_values"),
            ("viewer_tab", "viewer_fluence_ref_value", "fluence_ref_value"),
            ("differential_tab", "diff_fluence_ref_value", "fluence_ref_value"),
            ("fitting_tab", "fit_fluence_ref_value", "fluence_ref_value"),
            ("viewer_tab", "viewer_fluence_scale", "fluence_scale"),
            ("differential_tab", "diff_fluence_scale", "fluence_scale"),
            ("differential_tab", "diff_multi_fluence_scale", "fluence_scale"),
            ("fitting_tab", "fit_fluence_scale", "fluence_scale"),
            ("fitting_tab", "fit_multi_fluence_scale", "fluence_scale"),
            ("viewer_tab", "viewer_fluence_offset", "fluence_offset"),
            ("differential_tab", "diff_fluence_offset", "fluence_offset"),
            ("fitting_tab", "fit_fluence_offset", "fluence_offset"),
            ("viewer_tab", "viewer_fluence_delay_offset_fs", "fluence_delay_offset_fs"),
            ("differential_tab", "diff_fluence_delay_offset_fs", "fluence_delay_offset_fs"),
            ("fitting_tab", "fit_fluence_delay_offset_fs", "fluence_delay_offset_fs"),
            ("viewer_tab", "viewer_digits", "fluence_delay_digits"),
            ("differential_tab", "diff_fluence_delay_digits", "fluence_delay_digits"),
            ("differential_tab", "diff_multi_fluence_delay_digits", "fluence_delay_digits"),
            ("fitting_tab", "fit_fluence_delay_digits", "fluence_delay_digits"),
            ("fitting_tab", "fit_multi_fluence_delay_digits", "fluence_delay_digits"),
        ]

    def _fluence_combo_specs(self):
        """Return combo boxes that should stay synchronized across fluence workflows."""
        return [
            ("viewer_tab", "viewer_fluence_ref_type", "fluence_ref_type"),
            ("differential_tab", "diff_fluence_ref_type", "fluence_ref_type"),
            ("fitting_tab", "fit_fluence_ref_type", "fluence_ref_type"),
            ("viewer_tab", "viewer_fs_or_ps", "fluence_delay_display_unit"),
            ("differential_tab", "diff_fluence_delay_unit", "fluence_delay_display_unit"),
            ("differential_tab", "diff_multi_fluence_delay_unit", "fluence_delay_display_unit"),
            ("fitting_tab", "fit_fluence_delay_unit", "fluence_delay_display_unit"),
            ("fitting_tab", "fit_multi_fluence_delay_unit", "fluence_delay_display_unit"),
        ]

    def _delay_line_edit_specs(self):
        """Return line edits that should stay synchronized across delay workflows."""
        return [
            ("viewer_tab", "viewer_delay_offset_fs", "delay_offset_fs"),
            ("viewer_tab", "viewer_delay_fluence_scale", "delay_fluence_scale"),
            ("viewer_tab", "viewer_delay_fluence_offset", "delay_fluence_offset"),
            ("differential_tab", "diff_delay_offset", "delay_offset_fs"),
            ("fitting_tab", "fit_delay_offset", "delay_offset_fs"),
            ("differential_tab", "diff_delay_fluence_scale", "delay_fluence_scale"),
            ("fitting_tab", "fit_delay_fluence_scale", "delay_fluence_scale"),
            ("differential_tab", "diff_delay_fluence_offset", "delay_fluence_offset"),
            ("fitting_tab", "fit_delay_fluence_offset", "delay_fluence_offset"),
        ]

    def _delay_combo_specs(self):
        """Return combo boxes that should stay synchronized across delay workflows."""
        return [
            ("viewer_tab", "viewer_fs_or_ps", "delay_display_unit"),
            ("differential_tab", "diff_unit", "delay_display_unit"),
            ("fitting_tab", "fit_time_unit", "delay_display_unit"),
        ]

    def _resolve_tab_widget_attr(self, tab_name, widget_name):
        """Return a tab child widget if it exists."""
        tab = getattr(self, tab_name, None)
        if tab is None:
            return None
        return getattr(tab, widget_name, None)

    def _apply_fluence_state_to_controls(self, *, exclude=None):
        """Push shared fluence state into all synchronized controls."""
        exclude = exclude or set()

        for tab_name, widget_name, state_name in self._fluence_line_edit_specs():
            if (tab_name, widget_name) in exclude:
                continue
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "setText"):
                continue
            try:
                widget.blockSignals(True)
                widget.setText(str(getattr(self.state, state_name)))
            except Exception:
                pass
            finally:
                try:
                    widget.blockSignals(False)
                except Exception:
                    pass

        for tab_name, widget_name, state_name in self._fluence_combo_specs():
            if (tab_name, widget_name) in exclude:
                continue
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "findText"):
                continue
            try:
                widget.blockSignals(True)
                text = str(getattr(self.state, state_name))
                idx = widget.findText(text)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            except Exception:
                pass
            finally:
                try:
                    widget.blockSignals(False)
                except Exception:
                    pass

    def _sync_fluence_line_edit_from(self, tab_name, widget_name, state_name):
        """Persist one edited fluence line edit and propagate it to sibling tabs."""
        widget = self._resolve_tab_widget_attr(tab_name, widget_name)
        if widget is None or not hasattr(widget, "text"):
            return
        try:
            setattr(self.state, state_name, str(widget.text()))
        except Exception:
            return
        self._apply_fluence_state_to_controls(exclude={(tab_name, widget_name)})

    def _sync_fluence_combo_from(self, tab_name, widget_name, state_name, text):
        """Persist one edited fluence combo box and propagate it to sibling tabs."""
        try:
            setattr(self.state, state_name, str(text))
        except Exception:
            return
        self._apply_fluence_state_to_controls(exclude={(tab_name, widget_name)})

    def _capture_fluence_state_from_controls(self):
        """Read the current synchronized fluence controls into shared state."""
        for tab_name, widget_name, state_name in self._fluence_line_edit_specs():
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "text"):
                continue
            try:
                setattr(self.state, state_name, str(widget.text()))
            except Exception:
                pass

        for tab_name, widget_name, state_name in self._fluence_combo_specs():
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "currentText"):
                continue
            try:
                setattr(self.state, state_name, str(widget.currentText()))
            except Exception:
                pass

    def _connect_fluence_plot_control_sync(self):
        """Synchronize common fluence plotting controls across analysis tabs."""
        self._apply_fluence_state_to_controls()

        for tab_name, widget_name, state_name in self._fluence_line_edit_specs():
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "editingFinished"):
                continue
            try:
                widget.editingFinished.connect(
                    lambda tn=tab_name, wn=widget_name, sn=state_name:
                    self._sync_fluence_line_edit_from(tn, wn, sn)
                )
            except Exception:
                pass
            if hasattr(widget, "textEdited"):
                try:
                    widget.textEdited.connect(
                        lambda _text, tn=tab_name, wn=widget_name, sn=state_name:
                        self._sync_fluence_line_edit_from(tn, wn, sn)
                    )
                except Exception:
                    pass

        for tab_name, widget_name, state_name in self._fluence_combo_specs():
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "currentTextChanged"):
                continue
            try:
                widget.currentTextChanged.connect(
                    lambda text, tn=tab_name, wn=widget_name, sn=state_name:
                    self._sync_fluence_combo_from(tn, wn, sn, text)
                )
            except Exception:
                pass


    def _metadata_widget_for_tab_index(self, index):
        """Resolve the experiment metadata editor associated with a tab index."""
        tab = self.tabs.widget(index)

        mapping = {
            getattr(self, "preparation_tab", None): getattr(getattr(self, "preparation_tab", None), "experiment_metadata", None),
            getattr(self, "calibration_tab", None): getattr(getattr(self, "calibration_tab", None), "calibration_context", None),
            getattr(self, "pattern_creation_tab", None): getattr(getattr(self, "pattern_creation_tab", None), "experiment_metadata", None),
            getattr(self, "viewer_tab", None): getattr(getattr(self, "viewer_tab", None), "experiment_metadata", None),
            getattr(self, "differential_tab", None): getattr(getattr(self, "differential_tab", None), "diff_single_metadata", None),
            getattr(self, "fitting_tab", None): getattr(getattr(self, "fitting_tab", None), "fit_single_metadata", None),
        }

        return mapping.get(tab)


    def _sync_metadata_on_tab_change(self, index):
        """Copy shared experiment values into the newly selected workflow tab."""
        previous_widget = getattr(self, "_last_single_metadata_widget", None)

        if previous_widget is not None:
            self._sync_single_experiment_metadata_from(previous_widget)

        self._last_single_metadata_widget = self._metadata_widget_for_tab_index(index)

    def _init_log_dock(self):
        """Create the dockable log panel and its visibility actions."""
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setObjectName("LogDock")
        self.log_dock.setWidget(self.log_widget)
        self.log_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.log_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)

    def _init_menu_bar(self):
        """Create session, view, and layout actions in the application menu bar."""
        view_menu = self.menuBar().addMenu("View")

        self.show_log_action = QAction("Show Log", self)
        self.show_log_action.setCheckable(True)
        self.show_log_action.setChecked(True)
        self.show_log_action.triggered.connect(self._set_log_visible)
        view_menu.addAction(self.show_log_action)

        self.clear_log_action = QAction("Clear Log", self)
        self.clear_log_action.triggered.connect(lambda checked=False: self.log_widget.text_edit.clear())
        view_menu.addAction(self.clear_log_action)

        view_menu.addSeparator()

        log_position_menu = view_menu.addMenu("Log Position")

        bottom_action = QAction("Bottom Dock", self)
        bottom_action.triggered.connect(lambda checked=False: self._set_log_bottom_dock())
        log_position_menu.addAction(bottom_action)

        top_action = QAction("Top Dock", self)
        top_action.triggered.connect(lambda checked=False: self._set_log_top_dock())
        log_position_menu.addAction(top_action)

        left_action = QAction("Left Split", self)
        left_action.triggered.connect(lambda checked=False: self._set_log_split_layout("left"))
        log_position_menu.addAction(left_action)

        right_action = QAction("Right Split", self)
        right_action.triggered.connect(lambda checked=False: self._set_log_split_layout("right"))
        log_position_menu.addAction(right_action)

        floating_action = QAction("Floating", self)
        floating_action.triggered.connect(lambda checked=False: self._set_log_floating())
        log_position_menu.addAction(floating_action)


        plots_menu = self.menuBar().addMenu("Plots")

        close_all_plots_action = QAction("Close All Plots", self)
        close_all_plots_action.triggered.connect(
            lambda checked=False: self._close_all_plots()
        )
        plots_menu.addAction(close_all_plots_action)


    def _reset_central_content(self):
        """Detach reusable widgets and clear the current central-window layout."""
        self.tabs.setParent(None)
        self.log_widget.setParent(None)

        if self.main_splitter is not None:
            self.main_splitter.setParent(None)
            self.main_splitter.deleteLater()
            self.main_splitter = None

        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget not in (self.tabs, self.log_widget):
                widget.setParent(None)
                widget.deleteLater()

    def _detach_log_from_dock(self):
        """Detach the log widget from its dock without deleting either object."""
        if self.log_dock.widget() is self.log_widget:
            try:
                self.log_dock.setWidget(None)
            except Exception:
                pass

        self.log_dock.hide()
        self.removeDockWidget(self.log_dock)

    def _set_tabs_only_central(self):
        """Display only the workflow tabs in the central window area."""
        self._reset_central_content()
        self.main_layout.addWidget(self.tabs)

    def _set_log_split_layout(self, side):

        """Place tabs and log in a central splitter with the requested orientation."""
        self._detach_log_from_dock()
        self._reset_central_content()

        self.main_splitter = QSplitter(Qt.Horizontal, self.main_container)
        self.main_splitter.setChildrenCollapsible(False)

        if side == "left":
            self.main_splitter.addWidget(self.log_widget)
            self.main_splitter.addWidget(self.tabs)
            self.main_splitter.setStretchFactor(0, 0)
            self.main_splitter.setStretchFactor(1, 1)
            self.main_splitter.setSizes([360, max(900, self.width() - 360)])
        else:
            self.main_splitter.addWidget(self.tabs)
            self.main_splitter.addWidget(self.log_widget)
            self.main_splitter.setStretchFactor(0, 1)
            self.main_splitter.setStretchFactor(1, 0)
            self.main_splitter.setSizes([max(900, self.width() - 360), 360])

        self.main_layout.addWidget(self.main_splitter)
        self.log_widget.show()

    def _set_log_bottom_dock(self):

        """Dock the log below the tab area using the main-window dock system."""
        self._set_tabs_only_central()
        self.log_dock.setWidget(self.log_widget)
        self._dock_log_at(Qt.BottomDockWidgetArea)

    def _set_log_top_dock(self):

        """Dock the log above the tab area using the main-window dock system."""
        self._set_tabs_only_central()
        self.log_dock.setWidget(self.log_widget)
        self._dock_log_at(Qt.TopDockWidgetArea)

    def _set_log_floating(self):

        """Detach the log dock into an independent floating window."""
        self._set_tabs_only_central()
        self.log_dock.setWidget(self.log_widget)
        self._float_log_dock()

    def _dock_log_at(self, area):
        """Move the log dock to a supported Qt dock area."""
        self.log_dock.hide()
        self.removeDockWidget(self.log_dock)

        if self.log_dock.widget() is not self.log_widget:
            self.log_widget.setParent(None)
            self.log_dock.setWidget(self.log_widget)

        self.log_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.log_dock.setFloating(False)
        self.addDockWidget(area, self.log_dock)
        self.log_dock.show()
        self.log_dock.raise_()

        try:
            if area in (Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea):
                self.resizeDocks([self.log_dock], [360], Qt.Horizontal)
            else:
                self.resizeDocks([self.log_dock], [220], Qt.Vertical)
        except Exception:
            pass


    def _float_log_dock(self):
        """Show the log dock as a floating window and raise it."""
        if self.log_dock.widget() is not self.log_widget:
            self.log_widget.setParent(None)
            self.log_dock.setWidget(self.log_widget)

        self.log_dock.setFloating(True)
        self.log_dock.show()
        self.log_dock.raise_()


    def _init_toolbar(self):
        """Create file, layout, and session actions in the main toolbar."""
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("MainToolBar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(Qt.BottomToolBarArea, toolbar)

        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)



        self.close_all_plots_action = QAction("Close All Plots", self)
        self.close_all_plots_action.setToolTip("Close all currently open matplotlib plot windows.")
        self.close_all_plots_action.triggered.connect(
            lambda checked=False: self._close_all_plots()
        )
        toolbar.addAction(self.close_all_plots_action)


    def _set_log_visible(self, visible):
        """Show or hide the log while keeping its menu action synchronized."""
        visible = bool(visible)

        if self.log_dock.widget() is self.log_widget:
            self.log_dock.setVisible(visible)
        else:
            self.log_widget.setVisible(visible)


    def _polish_combo_box(self, combo):
        """Apply consistent GUI sizing and styling to combo box."""
        try:
            # Force a controllable Qt popup instead of the native macOS popup,
            # whose width can ignore stylesheet min-width rules.
            combo.setView(QListView(combo))
        except Exception:
            pass

        try:
            fm = combo.fontMetrics()
            texts = [combo.itemText(i) for i in range(combo.count())]
            current = combo.currentText()
            if current:
                texts.append(current)

            longest_px = max([fm.horizontalAdvance(str(t)) for t in texts] or [0])

            # Extra room for arrow, frame, padding, and popup margins.
            closed_width = max(180, longest_px + 76)
            popup_width = max(230, longest_px + 96)

            combo.setMinimumWidth(closed_width)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)

            if hasattr(combo, "setMinimumContentsLength"):
                longest_chars = max([len(str(t)) for t in texts] or [0])
                combo.setMinimumContentsLength(max(12, longest_chars + 4))

            view = combo.view()
            if view is not None:
                view.setMinimumWidth(popup_width)
                view.setTextElideMode(Qt.ElideNone)
                view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        except Exception:
            pass

    def _polish_tabs(self):
        """Apply consistent GUI sizing and styling to tabs."""
        try:
            bar = self.tabs.tabBar()
            bar.setElideMode(Qt.ElideNone)
            bar.setExpanding(True)
            bar.setUsesScrollButtons(False)

            fm = bar.fontMetrics()
            min_height = max(60, fm.height() + 40)
            bar.setMinimumHeight(min_height)

            # Direct tabbar stylesheet overrides global styles that may under-size
            # selected tabs after font changes.
            bar.setStyleSheet(
                """
                QTabBar::tab {
                    font-size: 13px;
                    min-height: 42px;
                    min-width: 80px;
                    padding: 6px 8px;
                    margin-right: 2px;
                }

                QTabBar::tab:selected {
                    font-size: 13px;
                    min-height: 42px;
                    min-width: 84px;
                    padding: 6px 10px;
                    font-weight: 700;
                }
                """
            )
        except Exception:
            pass

    def _polish_peak_definition_editors(self):
        """Apply consistent GUI sizing and styling to peak definition editors."""
        for tab_name in ("differential_tab", "fitting_tab"):
            tab = getattr(self, tab_name, None)

            if tab is None:
                continue

            for widget_name, editor in vars(tab).items():
                if "peak_specs" not in widget_name:
                    continue

                if not hasattr(editor, "setMinimumHeight"):
                    continue

                try:
                    editor.setMinimumHeight(100)
                    editor.setMaximumHeight(180)
                    editor.setLineWrapMode(editor.WidgetWidth)
                except Exception:
                    pass


    def _apply_delay_state_to_controls(self, *, exclude=None):
        """Push shared delay-display state into synchronized controls."""
        exclude = exclude or set()

        for tab_name, widget_name, state_name in self._delay_line_edit_specs():
            if (tab_name, widget_name) in exclude:
                continue
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "setText"):
                continue
            try:
                widget.blockSignals(True)
                widget.setText(str(getattr(self.state, state_name)))
            except Exception:
                pass
            finally:
                try:
                    widget.blockSignals(False)
                except Exception:
                    pass

        for tab_name, widget_name, state_name in self._delay_combo_specs():
            if (tab_name, widget_name) in exclude:
                continue
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "findText"):
                continue
            try:
                widget.blockSignals(True)
                text = str(getattr(self.state, state_name))
                idx = widget.findText(text)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            except Exception:
                pass
            finally:
                try:
                    widget.blockSignals(False)
                except Exception:
                    pass

    def _sync_delay_line_edit_from(self, tab_name, widget_name, state_name):
        """Persist one edited delay-display line edit and propagate it."""
        widget = self._resolve_tab_widget_attr(tab_name, widget_name)
        if widget is None or not hasattr(widget, "text"):
            return
        try:
            setattr(self.state, state_name, str(widget.text()))
        except Exception:
            return
        self._apply_delay_state_to_controls(exclude={(tab_name, widget_name)})

    def _sync_delay_combo_from(self, tab_name, widget_name, state_name, text):
        """Persist one edited delay combo box and propagate it."""
        try:
            setattr(self.state, state_name, str(text))
        except Exception:
            return
        self._apply_delay_state_to_controls(exclude={(tab_name, widget_name)})

    def _capture_delay_state_from_controls(self):
        """Read synchronized delay-display controls into shared state."""
        for tab_name, widget_name, state_name in self._delay_line_edit_specs():
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "text"):
                continue
            try:
                setattr(self.state, state_name, str(widget.text()))
            except Exception:
                pass

        for tab_name, widget_name, state_name in self._delay_combo_specs():
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "currentText"):
                continue
            try:
                setattr(self.state, state_name, str(widget.currentText()))
            except Exception:
                pass

    def _connect_delay_plot_control_sync(self):
        """Synchronize common delay-scan display controls across analysis tabs."""
        self._apply_delay_state_to_controls()

        for tab_name, widget_name, state_name in self._delay_line_edit_specs():
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None:
                continue
            if hasattr(widget, "editingFinished"):
                try:
                    widget.editingFinished.connect(
                        lambda tn=tab_name, wn=widget_name, sn=state_name:
                        self._sync_delay_line_edit_from(tn, wn, sn)
                    )
                except Exception:
                    pass
            if hasattr(widget, "textEdited"):
                try:
                    widget.textEdited.connect(
                        lambda _text, tn=tab_name, wn=widget_name, sn=state_name:
                        self._sync_delay_line_edit_from(tn, wn, sn)
                    )
                except Exception:
                    pass

        for tab_name, widget_name, state_name in self._delay_combo_specs():
            widget = self._resolve_tab_widget_attr(tab_name, widget_name)
            if widget is None or not hasattr(widget, "currentTextChanged"):
                continue
            try:
                widget.currentTextChanged.connect(
                    lambda text, tn=tab_name, wn=widget_name, sn=state_name:
                    self._sync_delay_combo_from(tn, wn, sn, text)
                )
            except Exception:
                pass


    def _polish_controls(self):
        """Apply shared sizing and styling rules to interactive controls."""
        self._polish_tabs()
        self._polish_form_labels()
        self._polish_peak_definition_editors()

        for combo in self.findChildren(QComboBox):
            self._polish_combo_box(combo)

    def _polish_form_labels(self):
        """Keep form labels visually attached to the fields they describe."""
        for label in self.tabs.findChildren(QLabel):
            try:
                text = label.text().strip()
            except Exception:
                continue

            if not text.endswith(":"):
                continue

            try:
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                label.setSizePolicy(
                    QSizePolicy.Maximum,
                    label.sizePolicy().verticalPolicy(),
                )
            except Exception:
                pass

    def _close_all_plots(self):
        """Close every open Matplotlib figure created by analysis workflows."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib._pylab_helpers import Gcf

            n_figures = len(Gcf.get_all_fig_managers())
            plt.close("all")

            if n_figures == 1:
                self.log_widget.log("Closed 1 plot window.")
            else:
                self.log_widget.log(f"Closed {n_figures} plot windows.")

        except Exception as exc:
            self.log_widget.log(f"Close All Plots Error: {exc}")

    def _on_facility_changed(self, facility_key: str):
        """Persist a facility change and refresh every facility-dependent tab."""
        if hasattr(self, "preparation_tab"):
            self.preparation_tab.set_facility(facility_key)

        if hasattr(self, "pattern_creation_tab"):
            self.pattern_creation_tab.set_facility(facility_key)

        if hasattr(self, "viewer_tab"):
            self.viewer_tab.set_facility(facility_key)

        if hasattr(self, "differential_tab"):
            self.differential_tab.set_facility(facility_key)

        if hasattr(self, "fitting_tab"):
            self.fitting_tab.set_facility(facility_key)

    def _on_polarization_changed(self, enabled: bool, factor: float):
        """Persist a changed polarization setting and notify the synchronization callback."""
        self.state.polarization_enabled = bool(enabled)
        self.state.polarization_factor = float(factor)
        self._sync_polarization_widgets()

    def _on_femtomax_ping_reference_changed(self, path: str):
        """Persist a changed FemtoMAX ping-reference table and autosave state."""
        self.state.femtomax_ping_reference_path = self.path_service.normalize(path)
        if hasattr(self, "preparation_tab"):
            self.preparation_tab.sync_femtomax_ping_reference_from_state()

    def _sync_polarization_widgets(self):
        """Propagate one polarization setting across calibration and integration tabs."""
        enabled = bool(getattr(self.state, "polarization_enabled", True))
        factor = getattr(self.state, "polarization_factor", 0.99)
        if factor is None:
            factor = 0.99
        for tab_name, control_name in (
            ("calibration_tab", "calib_polarization_control"),
            ("pattern_creation_tab", "pattern_polarization_control"),
        ):
            tab = getattr(self, tab_name, None)
            control = getattr(tab, control_name, None) if tab is not None else None
            if control is not None:
                control.set_configuration(
                    enabled=enabled,
                    factor=float(factor),
                    emit=False,
                )


    def _autosave_path(self):
        """Return the per-user JSON path used for crash-safe GUI autosaves."""
        return Path.home() / "xrdpy_gui_autosave.json"

    def _state_widget_roots(self):
        """Return the top-level tab widgets traversed during GUI-state serialization."""
        roots = {}

        for key, attr_name in (
            ("session", "session_tab"),
            ("preparation", "preparation_tab"),
            ("calibration", "calibration_tab"),
            ("pattern", "pattern_tab"),
            ("pattern", "pattern_creation_tab"),
            ("viewer", "viewer_tab"),
            ("differential", "differential_tab"),
            ("fitting", "fitting_tab"),
        ):
            if key not in roots and hasattr(self, attr_name):
                roots[key] = getattr(self, attr_name)

        return roots

    def _collect_widget_state(self, root):
        """Collect widget state into a serializable state mapping."""
        state = {}

        for name, widget in vars(root).items():
            if name.startswith("_"):
                continue

            if hasattr(widget, "values") and hasattr(widget, "set_values"):
                try:
                    state[name] = {
                        "type": "ValueWidget",
                        "value": widget.values(),
                    }
                    continue
                except Exception:
                    pass

            if isinstance(widget, QLineEdit):
                state[name] = {"type": "QLineEdit", "value": widget.text()}

            elif isinstance(widget, QCheckBox):
                state[name] = {"type": "QCheckBox", "value": widget.isChecked()}

            elif isinstance(widget, QComboBox):
                state[name] = {"type": "QComboBox", "value": widget.currentText()}

            elif isinstance(widget, QPlainTextEdit):
                state[name] = {"type": "QPlainTextEdit", "value": widget.toPlainText()}

            elif hasattr(widget, "get_experiments") and hasattr(widget, "set_experiments"):
                try:
                    value = widget.get_experiments()
                except Exception:
                    value = None

                state[name] = {"type": "MultiExperimentEditor", "value": value}

        return state

    def _apply_widget_state(self, root, state):
        """Apply widget state to the existing widgets and shared state."""
        if not isinstance(state, dict):
            return

        for name, item in state.items():
            if not isinstance(item, dict) or not hasattr(root, name):
                continue

            widget = getattr(root, name)
            value = item.get("value")

            if hasattr(widget, "set_values") and item.get("type") == "ValueWidget":
                if isinstance(value, dict):
                    try:
                        widget.set_values(value)
                        continue
                    except Exception:
                        pass

            if isinstance(widget, QLineEdit):
                widget.setText("" if value is None else str(value))

            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))

            elif isinstance(widget, QComboBox):
                text = "" if value is None else str(value)
                index = widget.findText(text)

                # GUI labels use the typographic micro symbol, while API/state
                # files may use the ASCII spelling or the Greek mu character.
                if index < 0 and text.strip().lower() in {
                    "us",
                    "µs",
                    "μs",
                    "microsecond",
                    "microseconds",
                }:
                    index = widget.findText("µs")
                if index < 0 and text.strip().lower() == "abs":
                    index = widget.findText("absdiff")

                if index >= 0:
                    widget.setCurrentIndex(index)
                elif widget.isEditable():
                    widget.setEditText(text)

            elif isinstance(widget, QPlainTextEdit):
                widget.setPlainText("" if value is None else str(value))

            elif hasattr(widget, "set_experiments") and item.get("type") == "MultiExperimentEditor":
                if value is not None:
                    try:
                        widget.set_experiments(value)
                    except Exception:
                        # Some legacy/default editors contain merged experiments
                        # even when their UI does not expose "Add merged experiment".
                        # Do not abort loading the whole GUI state because of one editor.
                        pass

    def _collect_gui_state(self):
        """Serialize shared state and supported widget values to a JSON-compatible mapping."""
        return {
            "state_version": 1,
            "window": {
                "width": self.width(),
                "height": self.height(),
                "current_tab_index": self.tabs.currentIndex(),
            },
            "tabs": {
                name: self._collect_widget_state(root)
                for name, root in self._state_widget_roots().items()
            },
        }


    def _line_state(self, value):
        """Return the text stored by a line-edit widget."""
        return {"type": "QLineEdit", "value": "" if value is None else str(value)}

    def _check_state(self, value):
        """Validate whether stored GUI state is structurally compatible with this version."""
        return {"type": "QCheckBox", "value": bool(value)}

    def _combo_state(self, value):
        """Return the current text stored by a combo-box widget."""
        return {"type": "QComboBox", "value": "" if value is None else str(value)}

    def _plain_state(self, value):
        """Return the full text stored by a plain-text editor."""
        return {"type": "QPlainTextEdit", "value": "" if value is None else str(value)}

    def _editor_state(self, value):
        """Serialize a multi-experiment editor to its validated mapping list."""
        return {"type": "MultiExperimentEditor", "value": value}

    def _value_widget_state(self, value):
        """Return the value exposed by a parameter widget's uniform interface."""
        return {"type": "ValueWidget", "value": value if isinstance(value, dict) else {}}

    def _legacy_facility_label(self, value):
        """Translate a legacy saved facility label to the current stable key."""
        if value in (None, ""):
            return ""

        try:
            return self.facility_service.label_from_key(str(value))
        except Exception:
            return str(value)


    def _legacy_experiment_with_id09_fallback(self, experiment, fallback):
        """Upgrade legacy experiment metadata and supply missing ID09 fields."""
        out = dict(experiment) if isinstance(experiment, dict) else {}

        if isinstance(fallback, dict):
            for key in ("raw_sample_name", "dataset", "scan_nb"):
                if out.get(key) in (None, "") and fallback.get(key) not in (None, ""):
                    out[key] = fallback.get(key)

        return out

    def _convert_legacy_gui_state(self, state):
        """Convert the old monolithic gui_legacy.py JSON schema into the new
        generic per-tab widget-state schema.
        """
        tabs = {}
        datared = state.get("datared", {})
        session = state.get("session", {})
        legacy_polarization = self._value_widget_state(
            {
                "enabled": bool(
                    session.get(
                        "polarization_enabled",
                        (
                            session.get("polarization_factor") not in (None, "")
                            if "polarization_factor" in session
                            else True
                        ),
                    )
                ),
                "factor": (
                    0.99
                    if session.get("polarization_factor") in (None, "")
                    else session.get("polarization_factor")
                ),
            }
        )

        # -------------------------
        # Session
        # -------------------------
        tabs["session"] = {
            "session_facility_combo": self._combo_state(
                self._legacy_facility_label(session.get("facility"))
            ),
            "session_path_root": self._line_state(session.get("path_root", "")),
            "session_analysis_subdir": self._line_state(session.get("analysis_subdir", "analysis")),
            "session_raw_subdir": self._line_state(session.get("raw_subdir", "")),
            "session_poni_path": self._line_state(session.get("poni_path", "")),
            "session_mask_path": self._line_state(session.get("mask_edf_path", "")),
            "session_azim_offset_deg": self._line_state(session.get("azim_offset_deg", "-90.0")),
            "session_femtomax_ping_reference_path": self._line_state(
                datared.get("femto_ping_reference_path", "")
            ),
        }

        # -------------------------
        # Preparation / datared
        # -------------------------
        datared_experiment = datared.get("experiment", {})

        tabs["preparation"] = {
            "experiment_metadata": self._value_widget_state(datared_experiment),
            "datared_metadata": self._value_widget_state(datared_experiment),
            "preparation_metadata": self._value_widget_state(datared_experiment),
            "datared_ref_delay": self._line_state(datared.get("ref_delay", "-5ns")),
            "datared_delays": self._line_state(datared.get("delays", "all")),
            "datared_overwrite": self._check_state(datared.get("overwrite", True)),
            "datared_show_progress": self._check_state(datared.get("show_progress", True)),
            "datared_show_frame_progress": self._check_state(datared.get("show_frame_progress", False)),
            "datared_femto_scans": self._line_state(datared.get("femto_scans", "")),
            "datared_femto_scan_type": self._combo_state(datared.get("femto_scan_type", "delay")),
            "datared_femto_fluences": self._line_state(
                datared.get(
                    "femto_fluences_mJ_cm2",
                    datared.get("femto_fluences", ""),
                )
            ),
            "datared_femto_selected_delays": self._line_state(datared.get("femto_selected_delays", "auto")),
            "datared_femto_delay_source": self._combo_state(datared.get("femto_delay_source", "avg")),
            "datared_femto_require_both": self._check_state(datared.get("femto_require_both", True)),
            "datared_femto_nb_shot_threshold": self._line_state(datared.get("femto_nb_shot_threshold", "")),
            "datared_femto_dist_mode": self._combo_state(datared.get("femto_dist_mode", "overlay")),
            "datared_femto_dist_unit": self._combo_state(datared.get("femto_dist_unit", "fs")),
            "datared_femto_dist_view": self._combo_state(datared.get("femto_dist_view", "scatter")),
            "datared_femto_dist_bins": self._line_state(datared.get("femto_dist_bins", "250")),
            "datared_femto_dist_range": self._line_state(
                datared.get("femto_dist_range", "")
            ),
            "datared_femto_dist_density": self._check_state(
                datared.get("femto_dist_density", False)
            ),
            "datared_femto_dist_show_median": self._check_state(
                datared.get("femto_dist_show_median", True)
            ),
            "datared_femto_overwrite": self._check_state(datared.get("femto_overwrite", True)),
            "datared_femto_batch_size": self._line_state(datared.get("femto_batch_size", "1000")),
            "datared_femto_use_parallel": self._check_state(datared.get("femto_use_parallel", True)),
            "datared_femto_max_workers": self._line_state(datared.get("femto_max_workers", "4")),
            "datared_femto_chunk_size": self._line_state(datared.get("femto_chunk_size", "1")),
            "datared_femto_start_method": self._combo_state(datared.get("femto_start_method", "fork")),
        }

        # -------------------------
        # Calibration
        # -------------------------
        calibration = state.get("calibration", {})
        tabs["calibration"] = {
            "calibration_context": self._value_widget_state(calibration.get("experiment", {})),
            "calib_polarization_control": legacy_polarization,
            "calib_pyfai_image_path": self._line_state(calibration.get("pyfai_image_path", "")),
            "calib_azimuthal_edges": self._line_state(calibration.get("azimuthal_edges", "")),
            "calib_include_full": self._check_state(calibration.get("include_full", True)),
            "calib_full_range": self._line_state(calibration.get("full_range", "(-90, 90)")),
            "calib_npt": self._line_state(calibration.get("npt", "1000")),
            "calib_normalize": self._check_state(calibration.get("normalize", True)),
            "calib_q_norm_range": self._line_state(calibration.get("q_norm_range", "(2.65, 2.75)")),
            "calib_overwrite_xy": self._check_state(calibration.get("overwrite_xy", False)),
            "calib_q_fit_range": self._line_state(calibration.get("q_fit_range", "(2.4, 2.65)")),
            "calib_eta": self._line_state(calibration.get("eta", "0.3")),
            "calib_eta_vary": self._check_state(calibration.get("eta_vary", False)),
            "calib_fit_method": self._combo_state(calibration.get("fit_method", "leastsq")),
            "calib_force_refit": self._check_state(calibration.get("force_refit", True)),
            "calib_out_csv_name": self._line_state(calibration.get("out_csv_name", "peak_fits.csv")),
            "calib_caked_xlim": self._line_state(calibration.get("caked_xlim", "(2.45, 2.60)")),
            "calib_caked_ylim": self._line_state(calibration.get("caked_ylim", "")),
            "calib_caked_figure_title": self._line_state(calibration.get("caked_figure_title", "")),
            "calib_caked_save": self._check_state(calibration.get("caked_save", True)),
            "calib_property_name": self._line_state(calibration.get("property_name", "pv_center")),
            "calib_property_only_success": self._check_state(calibration.get("property_only_success", True)),
            "calib_property_ylim": self._line_state(calibration.get("property_ylim", "")),
            "calib_peak_name": self._line_state(calibration.get("peak_name", "")),
            "calib_property_figure_title": self._line_state(calibration.get("property_figure_title", "")),
            "calib_property_save": self._check_state(calibration.get("property_save", True)),
            "calib_figures_subdir": self._line_state(calibration.get("figures_subdir", "figures/calibration/")),
            "calib_save_format": self._combo_state(calibration.get("save_format", "png")),
            "calib_save_dpi": self._line_state(calibration.get("save_dpi", "400")),
        }

        # -------------------------
        # Pattern
        # -------------------------
        pattern = state.get("pattern", {})
        pattern_experiment = pattern.get("experiment", {})
        tabs["pattern"] = {
            "experiment_metadata": self._value_widget_state(pattern_experiment),
            "pattern_polarization_control": legacy_polarization,
            "pattern_metadata": self._value_widget_state(pattern_experiment),
            "pattern_experiment_metadata": self._value_widget_state(pattern_experiment),
            "pattern_series_combo": self._combo_state(pattern.get("series_type", "Delay scan")),
            "pattern_delays": self._line_state(pattern.get("delays_fs", "all")),
            "pattern_fluence_delay_fs": self._line_state(pattern.get("fluence_delay_fs", "0")),
            "pattern_fluences": self._line_state(pattern.get("fluences_mJ_cm2", "all")),
            "pattern_copy_2d_image": self._check_state(pattern.get("copy_2d_image", False)),
            "pattern_dark_tag": self._line_state(pattern.get("dark_tag", "")),
            "pattern_ref_delay": self._line_state(pattern.get("ref_delay", "-5ns")),
            "pattern_force_checkbox": self._check_state(pattern.get("force", True)),
            "pattern_azimuthal_edges": self._line_state(pattern.get("azimuthal_edges", "-90, -60, -30, 0, 30, 60, 90")),
            "pattern_include_full": self._check_state(pattern.get("include_full", True)),
            "pattern_full_range": self._line_state(pattern.get("full_range", "(-90, 90)")),
            "pattern_npt": self._line_state(pattern.get("npt", "1000")),
            "pattern_normalize_checkbox": self._check_state(pattern.get("normalize", True)),
            "pattern_q_norm_range": self._line_state(pattern.get("q_norm_range", "(2.65, 2.75)")),
            "pattern_overwrite_xy": self._check_state(pattern.get("overwrite_xy", True)),
        }

        # -------------------------
        # Viewer
        # -------------------------
        viewer = state.get("viewer", {})
        viewer_experiment = viewer.get("experiment", {})
        tabs["viewer"] = {
            "experiment_metadata": self._value_widget_state(viewer_experiment),
            "viewer_metadata": self._value_widget_state(viewer_experiment),
            "viewer_experiment_metadata": self._value_widget_state(viewer_experiment),
            "viewer_series_combo": self._combo_state(viewer.get("series_type", "Delay scan")),
            "viewer_delays": self._line_state(viewer.get("delays_fs", "all")),
            "viewer_max_curves": self._line_state(viewer.get("max_curves", "")),
            "viewer_ref_type": self._combo_state(viewer.get("ref_type", "dark")),
            "viewer_ref_value": self._line_state(viewer.get("ref_value", "[1466556]")),
            "viewer_delay_offset_fs": self._line_state(viewer.get("delay_offset_fs", viewer.get("delay_offset", "0"))),
            "viewer_delay_fluence_scale": self._line_state(viewer.get("delay_fluence_scale", "1.0")),
            "viewer_delay_fluence_offset": self._line_state(viewer.get("delay_fluence_offset", "0")),
            "viewer_fluence_delay_fs": self._line_state(viewer.get("fluence_delay_fs", "0")),
            "viewer_fluences": self._line_state(viewer.get("fluences_mJ_cm2", "all")),
            "viewer_fluence_max_curves": self._line_state(viewer.get("fluence_max_curves", "")),
            "viewer_fluence_ref_type": self._combo_state(viewer.get("fluence_ref_type", "dark")),
            "viewer_fluence_ref_value": self._line_state(viewer.get("fluence_ref_value", "[1466556]")),
            "viewer_fluence_scale": self._line_state(viewer.get("fluence_scale", "1.0")),
            "viewer_fluence_offset": self._line_state(viewer.get("fluence_offset", "0")),
            "viewer_fluence_delay_offset_fs": self._line_state(viewer.get("fluence_delay_offset_fs", "0")),
            "viewer_fluence_compute_if_missing": self._check_state(viewer.get("fluence_compute_if_missing", True)),
            "viewer_fluence_copy_2d": self._check_state(viewer.get("fluence_copy_2d", False)),
            "viewer_azim_window": self._line_state(viewer.get("azim_window", "(-90, 90)")),
            "viewer_xlim": self._line_state(viewer.get("xlim", "(1.5, 4.5)")),
            "viewer_ylim_top": self._line_state(viewer.get("ylim_top", "")),
            "viewer_ylim_diff": self._line_state(viewer.get("ylim_diff", "")),
            "viewer_digits": self._line_state(viewer.get("digits", "2")),
            "viewer_fluence_digits": self._line_state(viewer.get("fluence_digits", "3")),
            "viewer_ref_delay": self._line_state(viewer.get("ref_delay", "-5ns")),
            "viewer_compute_if_missing": self._check_state(viewer.get("compute_if_missing", True)),
            "viewer_fs_or_ps": self._combo_state(viewer.get("fs_or_ps", "ps")),
            "viewer_overwrite_xy": self._check_state(viewer.get("overwrite_xy", False)),
            "viewer_from2d_checkbox": self._check_state(viewer.get("from_2D_imgs", False)),
            "viewer_normalize": self._check_state(viewer.get("normalize", True)),
            "viewer_q_norm_range": self._line_state(viewer.get("q_norm_range", "(2.65, 2.75)")),
            "viewer_save_plots": self._check_state(viewer.get("save_plots", True)),
            "viewer_save_format": self._combo_state(viewer.get("save_format", "png")),
            "viewer_save_dpi": self._line_state(viewer.get("save_dpi", "400")),
            "viewer_save_overwrite": self._check_state(viewer.get("save_overwrite", True)),
        }

        # -------------------------
        # Differential
        # -------------------------
        differential = state.get("differential", {})
        diff_single = differential.get("single", {})
        diff_multi = differential.get("multi", {})
        id09_experiment_fallback = viewer_experiment or pattern_experiment
        diff_single_experiment = self._legacy_experiment_with_id09_fallback(
            diff_single.get("experiment", {}),
            id09_experiment_fallback,
        )

        tabs["differential"] = {
            "diff_mode_combo": self._combo_state(differential.get("mode", "Single experiment")),
            "diff_series_combo": self._combo_state(differential.get("single_series_type", "Delay scan")),
            "diff_multi_series_combo": self._combo_state(differential.get("multi_series_type", "Delay scan")),
            "diff_single_metadata": self._value_widget_state(diff_single_experiment),
            "diff_delays": self._line_state(diff_single.get("delays_fs", "all")),
            "diff_ref_type": self._combo_state(diff_single.get("ref_type", "dark")),
            "diff_ref_value": self._line_state(diff_single.get("ref_value", "[1466556]")),
            "diff_azim_window": self._line_state(diff_single.get("azim_window", "(-90, 90)")),
            "diff_peak": self._line_state(diff_single.get("peak", "110")),
            "diff_peak_specs": self._plain_state(diff_single.get("peak_specs", "")),
            "diff_fluence_delay_fs": self._line_state(diff_single.get("delay_fs", "0")),
            "diff_fluences": self._line_state(diff_single.get("fluences_mJ_cm2", "all")),
            "diff_fluence_ref_type": self._combo_state(diff_single.get("fluence_ref_type", "dark")),
            "diff_fluence_ref_value": self._line_state(diff_single.get("fluence_ref_value", "[1466556]")),
            "diff_fluence_azim_window": self._line_state(diff_single.get("fluence_azim_window", "(-90, 90)")),
            "diff_fluence_peak": self._line_state(diff_single.get("fluence_peak", "110")),
            "diff_fluence_peak_specs": self._plain_state(diff_single.get("fluence_peak_specs", "")),
            "diff_unit": self._combo_state(diff_single.get("unit", "ps")),
            "diff_delay_offset": self._line_state(diff_single.get("delay_offset_fs", diff_single.get("delay_offset", "0"))),
            "diff_delay_fluence_scale": self._line_state(diff_single.get("delay_fluence_scale", "1.0")),
            "diff_delay_fluence_offset": self._line_state(diff_single.get("delay_fluence_offset", "0")),
            "diff_plot_abs_and_diffs": self._check_state(diff_single.get("plot_abs_and_diffs", True)),
            "diff_show_errorbars": self._check_state(diff_single.get("show_errorbars", True)),
            "diff_errorbar_scale": self._line_state(diff_single.get("errorbar_scale", "1.0")),
            "diff_xlim": self._line_state(diff_single.get("xlim", "")),
            "diff_ylim_signed": self._line_state(diff_single.get("ylim_signed", "")),
            "diff_ylim_abs": self._line_state(diff_single.get("ylim_abs", "")),
            "diff_fluence_unit": self._line_state(diff_single.get("fluence_unit", "mJ/cm$^2$")),
            "diff_fluence_scale": self._line_state(diff_single.get("fluence_scale", "1.0")),
            "diff_fluence_offset": self._line_state(diff_single.get("fluence_offset", "0")),
            "diff_fluence_delay_offset_fs": self._line_state(diff_single.get("fluence_delay_offset_fs", "0")),
            "diff_fluence_delay_unit": self._combo_state(diff_single.get("fluence_delay_unit", "ps")),
            "diff_fluence_delay_digits": self._line_state(diff_single.get("fluence_delay_digits", "2")),
            "diff_fluence_plot_abs_and_diffs": self._check_state(diff_single.get("fluence_plot_abs_and_diffs", True)),
            "diff_fluence_show_errorbars": self._check_state(diff_single.get("fluence_show_errorbars", True)),
            "diff_fluence_errorbar_scale": self._line_state(diff_single.get("fluence_errorbar_scale", "1.0")),
            "diff_fluence_xlim": self._line_state(diff_single.get("fluence_xlim", "")),
            "diff_fluence_ylim_signed": self._line_state(diff_single.get("fluence_ylim_signed", "")),
            "diff_fluence_ylim_abs": self._line_state(diff_single.get("fluence_ylim_abs", "")),
            "diff_region": self._combo_state(diff_single.get("region", "peak")),
            "diff_kind": self._combo_state(diff_single.get("kind", "diff")),
            "diff_time_window_select": self._line_state(diff_single.get("time_window_select_ps", "(-1, 200)")),
            "diff_poly_order": self._line_state(diff_single.get("poly_order", "2")),
            "diff_freq_unit": self._combo_state(diff_single.get("freq_unit", "cm^-1")),
            "diff_xlim_freq": self._line_state(diff_single.get("xlim_freq", "(-50, 850)")),
            "diff_npt": self._line_state(diff_single.get("npt", "1000")),
            "diff_normalize_xy": self._check_state(diff_single.get("normalize_xy", True)),
            "diff_q_norm_range": self._line_state(diff_single.get("q_norm_range", "(2.65, 2.75)")),
            "diff_compute_if_missing": self._check_state(diff_single.get("compute_if_missing", True)),
            "diff_overwrite_xy": self._check_state(diff_single.get("overwrite_xy", False)),
            "diff_save": self._check_state(diff_single.get("save", True)),
            "diff_save_format": self._combo_state(diff_single.get("save_format", "png")),
            "diff_save_dpi": self._line_state(diff_single.get("save_dpi", "400")),
            "diff_save_overwrite": self._check_state(diff_single.get("save_overwrite", True)),
            "diff_multi_editor": self._editor_state(diff_multi.get("delay_experiments", diff_multi.get("experiments"))),
            "diff_multi_editor_fluence": self._editor_state(diff_multi.get("fluence_experiments")),
            "diff_multi_delays": self._line_state(diff_multi.get("delays_fs", "all")),
            "diff_multi_azim_window": self._line_state(diff_multi.get("azim_window", "(-90, 90)")),
            "diff_multi_peak": self._line_state(diff_multi.get("peak", "110")),
            "diff_multi_peak_specs": self._plain_state(diff_multi.get("peak_specs", "")),
            "diff_multi_npt": self._line_state(diff_multi.get("npt", "1000")),
            "diff_multi_normalize_xy": self._check_state(diff_multi.get("normalize_xy", True)),
            "diff_multi_q_norm_range": self._line_state(diff_multi.get("q_norm_range", "(2.65, 2.75)")),
            "diff_multi_compute_if_missing": self._check_state(diff_multi.get("compute_if_missing", True)),
            "diff_multi_overwrite_xy": self._check_state(diff_multi.get("overwrite_xy", False)),
            "diff_multi_save": self._check_state(diff_multi.get("save", True)),
            "diff_multi_save_format": self._combo_state(diff_multi.get("save_format", "png")),
            "diff_multi_save_dpi": self._line_state(diff_multi.get("save_dpi", "400")),
            "diff_multi_save_overwrite": self._check_state(diff_multi.get("save_overwrite", True)),
            "diff_multi_unit": self._combo_state(diff_multi.get("unit", "ps")),
            "diff_multi_show_errorbars": self._check_state(diff_multi.get("show_errorbars", True)),
            "diff_multi_errorbar_scale": self._line_state(diff_multi.get("errorbar_scale", "1.0")),
            "diff_multi_as_lines": self._check_state(diff_multi.get("as_lines", False)),
            "diff_multi_kind": self._combo_state(diff_multi.get("kind", "diff")),
            "diff_multi_time_window_select": self._line_state(diff_multi.get("time_window_select_ps", "(-1, 200)")),
            "diff_multi_poly_order": self._line_state(diff_multi.get("poly_order", "2")),
            "diff_multi_freq_unit": self._combo_state(diff_multi.get("freq_unit", "cm^-1")),
            "diff_multi_xlim_freq": self._line_state(diff_multi.get("xlim_freq", "(-50, 850)")),
            "diff_multi_fluences": self._line_state(diff_multi.get("fluences_mJ_cm2", "all")),
            "diff_multi_fluence_azim_window": self._line_state(diff_multi.get("fluence_azim_window", "(-90, 90)")),
            "diff_multi_fluence_peak": self._line_state(diff_multi.get("fluence_peak", "110")),
            "diff_multi_fluence_peak_specs": self._plain_state(diff_multi.get("fluence_peak_specs", "")),
            "diff_multi_fluence_npt": self._line_state(diff_multi.get("fluence_npt", "1000")),
            "diff_multi_fluence_normalize_xy": self._check_state(diff_multi.get("fluence_normalize_xy", True)),
            "diff_multi_fluence_q_norm_range": self._line_state(diff_multi.get("fluence_q_norm_range", "(2.65, 2.75)")),
            "diff_multi_fluence_compute_if_missing": self._check_state(diff_multi.get("fluence_compute_if_missing", True)),
            "diff_multi_fluence_overwrite_xy": self._check_state(diff_multi.get("fluence_overwrite_xy", False)),
            "diff_multi_fluence_save": self._check_state(diff_multi.get("fluence_save", True)),
            "diff_multi_fluence_save_format": self._combo_state(diff_multi.get("fluence_save_format", "png")),
            "diff_multi_fluence_save_dpi": self._line_state(diff_multi.get("fluence_save_dpi", "400")),
            "diff_multi_fluence_save_overwrite": self._check_state(diff_multi.get("fluence_save_overwrite", True)),
            "diff_multi_fluence_unit": self._line_state(diff_multi.get("fluence_unit", "mJ/cm$^2$")),
            "diff_multi_fluence_scale": self._line_state(diff_multi.get("fluence_scale", "1.0")),
            "diff_multi_fluence_delay_unit": self._combo_state(diff_multi.get("fluence_delay_unit", "ps")),
            "diff_multi_fluence_delay_digits": self._line_state(diff_multi.get("fluence_delay_digits", "2")),
            "diff_multi_fluence_show_errorbars": self._check_state(diff_multi.get("fluence_show_errorbars", True)),
            "diff_multi_fluence_errorbar_scale": self._line_state(diff_multi.get("fluence_errorbar_scale", "1.0")),
            "diff_multi_fluence_as_lines": self._check_state(diff_multi.get("fluence_as_lines", False)),
        }

        # -------------------------
        # Fitting
        # -------------------------
        fitting = state.get("fitting", {})
        fit_single = fitting.get("single", {})
        fit_multi = fitting.get("multi", {})
        fit_single_experiment = self._legacy_experiment_with_id09_fallback(
            fit_single.get("experiment", {}),
            id09_experiment_fallback,
        )

        tabs["fitting"] = {
            "fit_mode_combo": self._combo_state(fitting.get("mode", "Single experiment")),
            "fit_series_combo": self._combo_state(fitting.get("single_series_type", "Delay scan")),
            "fit_multi_series_combo": self._combo_state(fitting.get("multi_series_type", "Delay scan")),
            "fit_single_metadata": self._value_widget_state(fit_single_experiment),
            "fit_delays": self._line_state(fit_single.get("delays_fs", "all")),
            "fit_ref_type": self._combo_state(fit_single.get("ref_type", "dark")),
            "fit_ref_value": self._line_state(fit_single.get("ref_value", "[1466556]")),
            "fit_fluence_delay_fs": self._line_state(fit_single.get("delay_fs", "0")),
            "fit_fluences": self._line_state(fit_single.get("fluences_mJ_cm2", "all")),
            "fit_fluence_ref_type": self._combo_state(fit_single.get("fluence_ref_type", "dark")),
            "fit_fluence_ref_value": self._line_state(fit_single.get("fluence_ref_value", "[1466556]")),
            "fit_ref_values_mode": self._combo_state(fit_single.get("ref_values_mode", "combine")),
            "fit_fluence_ref_values_mode": self._combo_state(fit_single.get("fluence_ref_values_mode", "combine")),
            "fit_peak_specs": self._plain_state(fit_single.get("peak_specs", "")),
            "fit_azim_windows": self._plain_state(fit_single.get("azim_windows", "")),
            "fit_phi_mode": self._combo_state(fit_single.get("phi_mode", "phi_avg")),
            "fit_phi_reduce": self._combo_state(fit_single.get("phi_reduce", "sum")),
            "fit_default_eta": self._line_state(fit_single.get("default_eta", "0.3")),
            "fit_eta_mode": self._combo_state(fit_single.get("eta_mode", "fixed")),
            "fit_npt": self._line_state(fit_single.get("npt", "1000")),
            "fit_q_norm_range": self._line_state(fit_single.get("q_norm_range", "(2.65, 2.75)")),
            "fit_out_csv_name": self._line_state(fit_single.get("out_csv_name", "peak_fits_delay.csv")),
            "fit_normalize_xy": self._check_state(fit_single.get("normalize_xy", True)),
            "fit_compute_if_missing": self._check_state(fit_single.get("compute_if_missing", True)),
            "fit_overwrite_xy": self._check_state(fit_single.get("overwrite_xy", False)),
            "fit_include_reference": self._check_state(fit_single.get("include_reference_in_output", True)),
            "fit_show_fit_figures": self._check_state(fit_single.get("show_fit_figures", False)),
            "fit_save_fit_figures": self._check_state(fit_single.get("save_fit_figures", False)),
            "fit_fig_format": self._combo_state(fit_single.get("fit_figures_format", "png")),
            "fit_fig_dpi": self._line_state(fit_single.get("fit_figures_dpi", "300")),
            "fit_plot_only_success": self._check_state(fit_single.get("plot_only_success", True)),
            "fit_oversample": self._line_state(fit_single.get("fit_oversample", "10")),
            "fit_overlay_peak": self._line_state(fit_single.get("overlay_peak", "110")),
            "fit_overlay_delay": self._line_state(fit_single.get("overlay_delay_fs", "0")),
            "fit_overlay_group": self._line_state(fit_single.get("overlay_group", "Full")),
            "fit_overlay_is_reference": self._check_state(fit_single.get("overlay_is_reference", False)),
            "fit_overlay_ensure_csv": self._check_state(fit_single.get("overlay_ensure_csv", True)),
            "fit_overlay_show": self._check_state(fit_single.get("overlay_show", True)),
            "fit_overlay_save": self._check_state(fit_single.get("overlay_save", True)),
            "fit_fluence_overlay_peak": self._line_state(fit_single.get("fluence_overlay_peak", "110")),
            "fit_fluence_overlay_fluence": self._line_state(fit_single.get("fluence_overlay_fluence", "1.5")),
            "fit_fluence_overlay_group_name": self._line_state(fit_single.get("fluence_overlay_group", "Full")),
            "fit_fluence_overlay_is_reference": self._check_state(fit_single.get("fluence_overlay_is_reference", False)),
            "fit_fluence_overlay_ensure_csv": self._check_state(fit_single.get("fluence_overlay_ensure_csv", True)),
            "fit_fluence_overlay_show": self._check_state(fit_single.get("fluence_overlay_show", True)),
            "fit_fluence_overlay_save": self._check_state(fit_single.get("fluence_overlay_save", True)),
            "fit_time_peak": self._line_state(fit_single.get("time_peak", "110")),
            "fit_property": self._combo_state(fit_single.get("_property", "hkl_pos")),
            "fit_time_unit": self._combo_state(fit_single.get("time_unit", "ps")),
            "fit_groups": self._line_state(fit_single.get("groups", "['Full', 60, 30, 0]")),
            "fit_time_title": self._line_state(fit_single.get("time_title", "")),
            "fit_delay_offset": self._line_state(fit_single.get("delay_offset_fs", fit_single.get("delay_offset", "0"))),
            "fit_delay_fluence_scale": self._line_state(fit_single.get("delay_fluence_scale", "1.0")),
            "fit_delay_fluence_offset": self._line_state(fit_single.get("delay_fluence_offset", "0")),
            "fit_time_xlim": self._line_state(fit_single.get("time_xlim", "")),
            "fit_time_ylim": self._line_state(fit_single.get("time_ylim", "")),
            "fit_as_lines": self._check_state(fit_single.get("as_lines", False)),
            "fit_show_baseline_sigma": self._check_state(fit_single.get("show_baseline_sigma", True)),
            "fit_baseline_sigma": self._line_state(fit_single.get("baseline_sigma", "1")),
            "fit_baseline_alpha": self._line_state(fit_single.get("baseline_alpha", "1")),
            "fit_baseline_mode": self._combo_state(fit_single.get("baseline_mode", "errorbar")),
            "fit_time_save": self._check_state(fit_single.get("time_save", True)),
            "fit_time_save_fmt": self._combo_state(fit_single.get("time_save_fmt", "png")),
            "fit_time_save_dpi": self._line_state(fit_single.get("time_save_dpi", "300")),
            "fit_fluence_time_peak": self._line_state(fit_single.get("fluence_time_peak", "110")),
            "fit_fluence_property": self._combo_state(fit_single.get("fluence_property", "hkl_pos")),
            "fit_fluence_unit": self._line_state(fit_single.get("fluence_unit", "mJ/cm$^2$")),
            "fit_fluence_groups": self._line_state(fit_single.get("fluence_groups", "['Full', 60, 30, 0]")),
            "fit_fluence_time_title": self._line_state(fit_single.get("fluence_time_title", "")),
            "fit_fluence_scale": self._line_state(fit_single.get("fluence_scale", "1.0")),
            "fit_fluence_offset": self._line_state(fit_single.get("fluence_offset", "0")),
            "fit_fluence_delay_offset_fs": self._line_state(fit_single.get("fluence_delay_offset_fs", "0")),
            "fit_fluence_delay_unit": self._combo_state(fit_single.get("fluence_delay_unit", "ps")),
            "fit_fluence_delay_digits": self._line_state(fit_single.get("fluence_delay_digits", "2")),
            "fit_fluence_as_lines": self._check_state(fit_single.get("fluence_as_lines", False)),
            "fit_fluence_show_baseline_sigma": self._check_state(fit_single.get("fluence_show_baseline_sigma", True)),
            "fit_fluence_baseline_sigma": self._line_state(fit_single.get("fluence_baseline_sigma", "1")),
            "fit_fluence_baseline_alpha": self._line_state(fit_single.get("fluence_baseline_alpha", "0.18")),
            "fit_fluence_baseline_mode": self._combo_state(fit_single.get("fluence_baseline_mode", "errorbar")),
            "fit_fluence_time_save": self._check_state(fit_single.get("fluence_time_save", True)),
            "fit_fluence_time_save_fmt": self._combo_state(fit_single.get("fluence_time_save_fmt", "png")),
            "fit_fluence_time_save_dpi": self._line_state(fit_single.get("fluence_time_save_dpi", "300")),
            "fit_multi_editor": self._editor_state(fit_multi.get("delay_experiments", fit_multi.get("experiments"))),
            "fit_multi_editor_fluence": self._editor_state(fit_multi.get("fluence_experiments")),
            "fit_multi_peak": self._line_state(fit_multi.get("peak", "110")),
            "fit_multi_property": self._combo_state(fit_multi.get("_property", "hkl_pos")),
            "fit_multi_out_csv_name": self._line_state(fit_multi.get("out_csv_name", "peak_fits_delay.csv")),
            "fit_multi_unit": self._combo_state(fit_multi.get("unit", "ps")),
            "fit_multi_phi_mode": self._combo_state(fit_multi.get("phi_mode", "auto")),
            "fit_multi_phi_reduce": self._combo_state(fit_multi.get("phi_reduce", "sum")),
            "fit_multi_phi_window": self._line_state(fit_multi.get("phi_window", "Full")),
            "fit_multi_title": self._line_state(fit_multi.get("title", "")),
            "fit_multi_only_success": self._check_state(fit_multi.get("only_success", True)),
            "fit_multi_include_reference": self._check_state(fit_multi.get("include_reference", True)),
            "fit_multi_as_lines": self._check_state(fit_multi.get("as_lines", False)),
            "fit_multi_delay_offset": self._line_state(fit_multi.get("delay_offset", "")),
            "fit_multi_show_baseline_sigma": self._check_state(fit_multi.get("show_baseline_sigma", True)),
            "fit_multi_baseline_sigma": self._line_state(fit_multi.get("baseline_sigma", "1")),
            "fit_multi_baseline_alpha": self._line_state(fit_multi.get("baseline_alpha", "0.18")),
            "fit_multi_baseline_mode": self._combo_state(fit_multi.get("baseline_mode", "errorbar")),
            "fit_multi_norm_min_max": self._check_state(fit_multi.get("norm_min_max", False)),
            "fit_multi_delay_for_norm_max": self._line_state(fit_multi.get("delay_for_norm_max", "")),
            "fit_multi_cmap": self._line_state(fit_multi.get("cmap", "jet")),
            "fit_multi_save": self._check_state(fit_multi.get("save", True)),
            "fit_multi_save_fmt": self._combo_state(fit_multi.get("save_fmt", "png")),
            "fit_multi_save_dpi": self._line_state(fit_multi.get("save_dpi", "300")),
            "fit_multi_save_overwrite": self._check_state(fit_multi.get("save_overwrite", True)),
            "fit_multi_fluence_peak": self._line_state(fit_multi.get("fluence_peak", "110")),
            "fit_multi_fluence_property": self._combo_state(fit_multi.get("fluence_property", "hkl_pos")),
            "fit_multi_fluence_group_by": self._combo_state(fit_multi.get("fluence_group_by", "azim_range_str")),
            "fit_multi_fluence_group_name": self._line_state(fit_multi.get("fluence_group", "Full")),
            "fit_multi_fluence_unit": self._line_state(fit_multi.get("fluence_unit", "mJ/cm$^2$")),
            "fit_multi_fluence_scale": self._line_state(fit_multi.get("fluence_scale", "1.0")),
            "fit_multi_fluence_delay_unit": self._combo_state(fit_multi.get("fluence_delay_unit", "ps")),
            "fit_multi_fluence_delay_digits": self._line_state(fit_multi.get("fluence_delay_digits", "2")),
            "fit_multi_fluence_title": self._line_state(fit_multi.get("fluence_title", "")),
            "fit_multi_fluence_only_success": self._check_state(fit_multi.get("fluence_only_success", True)),
            "fit_multi_fluence_include_reference": self._check_state(fit_multi.get("fluence_include_reference", True)),
            "fit_multi_fluence_as_lines": self._check_state(fit_multi.get("fluence_as_lines", False)),
            "fit_multi_fluence_show_baseline_sigma": self._check_state(fit_multi.get("fluence_show_baseline_sigma", True)),
            "fit_multi_fluence_baseline_sigma": self._line_state(fit_multi.get("fluence_baseline_sigma", "1")),
            "fit_multi_fluence_baseline_mode": self._combo_state(fit_multi.get("fluence_baseline_mode", "errorbar")),
            "fit_multi_fluence_save": self._check_state(fit_multi.get("fluence_save", True)),
            "fit_multi_fluence_save_fmt": self._combo_state(fit_multi.get("fluence_save_fmt", "png")),
            "fit_multi_fluence_save_dpi": self._line_state(fit_multi.get("fluence_save_dpi", "300")),
            "fit_multi_fluence_save_overwrite": self._check_state(fit_multi.get("fluence_save_overwrite", True)),
        }

        return {
            "state_version": 1,
            "window": state.get("window", {}),
            "tabs": tabs,
        }


    def _apply_gui_state(self, state):
        """Apply gui state to the existing widgets and shared state."""
        if not isinstance(state, dict):
            raise ValueError("GUI state must be a dictionary.")

        if "tabs" not in state:
            state = self._convert_legacy_gui_state(state)

        window_state = state.get("window", {})
        if isinstance(window_state, dict):
            width = window_state.get("width")
            height = window_state.get("height")

            if width and height:
                self.resize(int(width), int(height))

        tabs_state = state.get("tabs", {})
        if isinstance(tabs_state, dict):
            tabs_state = {
                name: dict(tab_state) if isinstance(tab_state, dict) else tab_state
                for name, tab_state in tabs_state.items()
            }

            # Migrate state files written by the immediately preceding GUI:
            # ping references lived in Preparation and polarization was also
            # duplicated in Session.
            session_state = tabs_state.setdefault("session", {})
            preparation_state = tabs_state.get("preparation", {})
            if isinstance(session_state, dict) and isinstance(
                preparation_state,
                dict,
            ):
                if "session_femtomax_ping_reference_path" not in session_state:
                    old_ping = preparation_state.get(
                        "datared_femto_ping_reference_path"
                    )
                    if old_ping is not None:
                        session_state["session_femtomax_ping_reference_path"] = old_ping

            old_polarization = (
                session_state.get("session_polarization_control")
                if isinstance(session_state, dict)
                else None
            )
            if old_polarization is not None:
                for tab_name, control_name in (
                    ("calibration", "calib_polarization_control"),
                    ("pattern", "pattern_polarization_control"),
                ):
                    tab_state = tabs_state.setdefault(tab_name, {})
                    if isinstance(tab_state, dict):
                        tab_state.setdefault(control_name, old_polarization)

            for name, root in self._state_widget_roots().items():
                self._apply_widget_state(root, tabs_state.get(name, {}))

        if hasattr(self.session_tab, "_sync_state_from_widgets"):
            self.session_tab._sync_state_from_widgets()
        if self.state.facility:
            self.session_tab.set_facility(self.state.facility)
        self._sync_polarization_widgets()

        if self.state.facility:
            self._on_facility_changed(self.state.facility)

        if isinstance(window_state, dict):
            current_tab_index = window_state.get("current_tab_index")
            if current_tab_index is not None:
                try:
                    self.tabs.setCurrentIndex(int(current_tab_index))
                except Exception:
                    pass

        for root in self._state_widget_roots().values():
            for method_name in (
                "_refresh_mode_widgets",
                "_refresh_series_widgets",
                "_refresh_pattern_series_widgets",
                "_refresh_viewer_series_widgets",
                "_refresh_diff_mode_widgets",
                "_refresh_diff_series_widgets",
                "_refresh_fit_mode_widgets",
                "_refresh_fit_series_widgets",
            ):
                method = getattr(root, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass

            for widget in vars(root).values():
                if hasattr(widget, "update_preview"):
                    try:
                        widget.update_preview()
                    except Exception:
                        pass

        self._capture_fluence_state_from_controls()
        self._apply_fluence_state_to_controls()
        self._capture_delay_state_from_controls()
        self._apply_delay_state_to_controls()
        self._capture_q_norm_range_state_from_controls()
        self._apply_q_norm_range_state_to_controls()

    def _save_state_dict_to_path(self, state, path):
        """Atomically write a JSON-compatible GUI-state mapping to disk."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    def _load_state_dict_from_path(self, path):
        """Read and validate a GUI-state JSON mapping from disk."""
        path = self._normalize_gui_state_file_path(path)
        if path is None:
            raise ValueError("GUI state path is empty.")
        if not path.is_file():
            raise FileNotFoundError(f"GUI state file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _normalize_gui_state_file_path(self, file_name):
        """Normalize an explicit GUI-state path before loading or saving."""
        if file_name is None:
            return None

        text = str(file_name).strip().strip('"').strip("'")
        if not text:
            return None

        path = Path(text).expanduser()
        if not path.is_absolute():
            path = self._gui_state_directory / path

        try:
            return path.resolve()
        except FileNotFoundError:
            return path.absolute()

    def _set_gui_state_path_field(self, file_name) -> None:
        """Display the loaded GUI-state file after widget state has been applied."""
        normalized = self._normalize_gui_state_file_path(file_name)
        if normalized is None:
            return

        widget = getattr(self.session_tab, "session_gui_state_path", None)
        if widget is None:
            return

        text = str(normalized)
        blocked = widget.blockSignals(True)
        try:
            widget.setText(text)
            widget.setCursorPosition(len(text))
        finally:
            widget.blockSignals(blocked)

    def _build_gui_state_dialog(self, *, save: bool) -> QFileDialog:
        """Build a native state-file dialog with predictable directory history."""

        start_directory = self._gui_state_directory
        dialog = QFileDialog(self)
        # Keep the platform-native dialog.  DontUseNativeDialog was previously
        # enabled to work around directory handling, but it also caused the
        # visible change in file-picker appearance on macOS.
        dialog.setOption(QFileDialog.DontUseNativeDialog, False)
        dialog.setWindowTitle("Save GUI state" if save else "Load GUI state")
        dialog.setDirectory(str(start_directory))
        dialog.setDirectoryUrl(QUrl.fromLocalFile(str(start_directory)))
        dialog.setNameFilters(["JSON Files (*.json)", "All Files (*)"])
        dialog.setViewMode(QFileDialog.Detail)

        if save:
            dialog.setAcceptMode(QFileDialog.AcceptSave)
            dialog.setFileMode(QFileDialog.AnyFile)
            dialog.setDefaultSuffix("json")
            dialog.selectFile("xrdpy_gui_state.json")
        else:
            dialog.setAcceptMode(QFileDialog.AcceptOpen)
            dialog.setFileMode(QFileDialog.ExistingFile)

        return dialog

    def _select_gui_state_path(self, *, save: bool):
        """Open a save/load dialog and return the selected GUI-state JSON path."""
        dialog = self._build_gui_state_dialog(save=save)
        if dialog.exec_() != QDialog.Accepted:
            return None
        selected = dialog.selectedFiles()
        return selected[0] if selected else None

    def _remember_gui_state_selection(self, file_name) -> None:
        """Use a loaded/saved state's directory for the next state dialog."""

        normalized = self._normalize_gui_state_file_path(file_name)
        if normalized is None:
            return
        self._gui_state_directory = (
            normalized if normalized.is_dir() else normalized.parent
        )
        self.path_service.remember_dialog_selection(normalized)

    def _save_gui_state_to_file(self):
        """Collect current controls and save them to a user-selected JSON file."""
        try:
            state = self._collect_gui_state()
            file_name = self._select_gui_state_path(save=True)

            if not file_name:
                return

            self._save_state_dict_to_path(state, file_name)
            self._remember_gui_state_selection(file_name)
            self.log_widget.log(f"GUI state saved to: {file_name}")

        except Exception as exc:
            self.log_widget.log(f"Save GUI State Error: {exc}")

    def _load_gui_state_from_file(self):
        """Load a user-selected JSON state file and apply it to the GUI."""
        try:
            file_name = self._select_gui_state_path(save=False)

            if not file_name:
                return

            normalized = self._normalize_gui_state_file_path(file_name)
            self._remember_gui_state_selection(normalized)
            state = self._load_state_dict_from_path(normalized)
            self._apply_gui_state(state)
            self._set_gui_state_path_field(normalized)
            self.log_widget.log(f"GUI state loaded from: {normalized}")

        except Exception as exc:
            self.log_widget.log(f"Load GUI State Error: {exc}")

    def _load_gui_state_from_path(self, file_name):
        """Load a GUI-state JSON file from an explicit pasted path."""
        try:
            if not file_name:
                return
            normalized = self._normalize_gui_state_file_path(file_name)
            self._remember_gui_state_selection(normalized)
            state = self._load_state_dict_from_path(normalized)
            self._apply_gui_state(state)
            self._set_gui_state_path_field(normalized)
            self.log_widget.log(f"GUI state loaded from: {normalized}")

        except Exception as exc:
            self.log_widget.log(f"Load GUI State Error: {exc}")

    def _load_autosave_from_disk(self):
        """Restore the per-user autosave file after structural validation."""
        try:
            path = self._autosave_path()

            if not path.exists():
                QMessageBox.information(
                    self,
                    "No Autosave",
                    f"No autosave file found at:\n{path}",
                )
                return

            state = self._load_state_dict_from_path(path)
            self._apply_gui_state(state)
            self.log_widget.log(f"Autosaved GUI state restored from: {path}")

        except Exception as exc:
            self.log_widget.log(f"Load Autosave Error: {exc}")


    def _maybe_prompt_restore_autosave(self):
        """Prompt the user about restore autosave when recovery data exist."""
        path = self._autosave_path()

        if not path.exists():
            return

        dialog = QDialog(self)
        dialog.setObjectName("RestoreAutosaveDialog")
        dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        dialog.setModal(True)
        dialog.resize(580, 250)

        layout = QVBoxLayout()
        dialog.setLayout(layout)

        header = QWidget(dialog)
        header.setObjectName("RestoreAutosaveHeader")
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(18, 10, 14, 10)
        header_layout.setSpacing(10)
        header.setLayout(header_layout)

        header_label = QLabel("Restore Autosave")
        header_label.setObjectName("RestoreAutosaveHeaderLabel")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        close_button = QPushButton("×")
        close_button.setObjectName("RestoreAutosaveCloseButton")
        close_button.setFixedSize(28, 28)
        close_button.clicked.connect(dialog.reject)
        header_layout.addWidget(close_button)

        layout.addWidget(header)

        body_widget = QWidget(dialog)
        body_widget.setObjectName("RestoreAutosaveBodyWidget")
        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(26, 22, 26, 22)
        body_layout.setSpacing(15)
        body_widget.setLayout(body_layout)
        layout.addWidget(body_widget)

        title = QLabel("Restore autosaved GUI state?")
        title.setObjectName("RestoreAutosaveTitle")
        title.setWordWrap(True)
        body_layout.addWidget(title)

        body = QLabel(
            "A previous autosaved GUI state was found:\n"
            f"{path}\n\n"
            "Restore it now?"
        )
        body.setObjectName("RestoreAutosaveBody")
        body.setWordWrap(True)
        body_layout.addWidget(body)

        button_row = QHBoxLayout()
        button_row.addStretch()

        no_button = QPushButton("No")
        yes_button = QPushButton("Yes")

        no_button.clicked.connect(dialog.reject)
        yes_button.clicked.connect(dialog.accept)

        button_row.addWidget(no_button)
        button_row.addWidget(yes_button)
        body_layout.addLayout(button_row)

        no_button.setDefault(True)
        no_button.setFocus()

        if dialog.exec_() == QDialog.Accepted:
            self._load_autosave_from_disk()


    def _autosave_gui_state(self):
        """Persist current GUI state to the crash-safe autosave location."""
        try:
            self._save_state_dict_to_path(
                self._collect_gui_state(),
                self._autosave_path(),
            )
        except Exception as exc:
            self.log_widget.log(f"GUI autosave error: {exc}")
            guard = getattr(self, "runtime_guard", None)
            if guard is not None:
                guard.record("GUI autosave error", repr(exc))

    def closeEvent(self, event):
        """Close event.

        Parameters
        ----------
        event : object
            Qt or Matplotlib event supplied by the framework callback.
        """
        if self.isVisible() and not self._allow_close_without_confirmation:
            answer = QMessageBox.question(
                self,
                "Quit XRDpy Analysis?",
                "Closing the main window stops the analysis GUI and any active "
                "work. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                guard = getattr(self, "runtime_guard", None)
                if guard is not None:
                    guard.record("Main-window close request cancelled")
                return
            self._allow_close_without_confirmation = True

        guard = getattr(self, "runtime_guard", None)
        if guard is not None:
            guard.record("Main-window close accepted")
        self._autosave_gui_state()
        super().closeEvent(event)
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(0, app.quit)



def _apply_light_palette(app):
    """Apply light palette to the existing widgets and shared state."""
    palette = QPalette()

    window = QColor("#eef2f5")
    panel = QColor("#f6f7f9")
    base = QColor("#f3f4f6")
    text = QColor("#17212b")
    disabled = QColor("#7d8c98")
    highlight = QColor("#c7dff4")

    palette.setColor(QPalette.Window, window)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, base)
    palette.setColor(QPalette.AlternateBase, panel)
    palette.setColor(QPalette.ToolTipBase, panel)
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, QColor("#eceff3"))
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, text)

    palette.setColor(QPalette.Disabled, QPalette.Text, disabled)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled)

    app.setPalette(palette)



def main(*, launch_directory=None):
    """Launch the analysis GUI application.

    Parameters
    ----------
    launch_directory : object
        Shell working directory captured before GUI imports and used to initialize dialogs.
    """
    global _MAIN_WINDOW, _RUNTIME_GUARD

    launch_directory = Path(
        Path.cwd() if launch_directory is None else launch_directory
    ).expanduser().resolve()
    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    app.setQuitOnLastWindowClosed(False)

    _RUNTIME_GUARD = GuiRuntimeGuard(launch_directory=launch_directory)
    _RUNTIME_GUARD.install(app)

    _MAIN_WINDOW = AnalysisMainWindow(launch_directory=launch_directory)
    _RUNTIME_GUARD.attach_window(_MAIN_WINDOW)
    app._xrdpy_analysis_main_window = _MAIN_WINDOW
    _MAIN_WINDOW.show()

    exit_code = app.exec_()
    _RUNTIME_GUARD.record(f"Qt event loop returned exit code {exit_code}")
    return exit_code


if __name__ == "__main__":
    main()
