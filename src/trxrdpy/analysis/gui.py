import ast
import json
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

plt.ion()

PACKAGE_IMPORT_ERROR = None
PACKAGE_NAME = None
GUI_STATE_VERSION = 1
AUTOSAVE_FILENAME = ".xrdpy_analysis_gui_last_session.json"

try:
    from .common.paths import AnalysisPaths
    from .Spring8_SACLA import azimint as sacla_azimint
    from .MaxIV_FemtoMAX import azimint as femto_azimint
    from .ESRF_ID09 import azimint as id09_azimint
    from .ESRF_ID09 import datared as id09_datared
    from . import calibration, differential_analysis, fitting

    PACKAGE_NAME = "trxrdpy"
except Exception as exc_1:
    try:
        from trxrdpy.analysis.common.paths import AnalysisPaths
        from trxrdpy.analysis.Spring8_SACLA import azimint as sacla_azimint
        from trxrdpy.analysis.MaxIV_FemtoMAX import azimint as femto_azimint
        from trxrdpy.analysis.ESRF_ID09 import azimint as id09_azimint
        from trxrdpy.analysis.ESRF_ID09 import datared as id09_datared
        from trxrdpy.analysis import calibration, differential_analysis, fitting


        PACKAGE_NAME = "XRDpy"
    except Exception as exc_2:
        PACKAGE_IMPORT_ERROR = (exc_1, exc_2)
        AnalysisPaths = None
        sacla_azimint = None
        femto_azimint = None
        id09_azimint = None
        id09_datared = None
        calibration = None
        differential_analysis = None
        fitting = None


DEFAULT_DIFF_PEAK_SPECS = {
    "012": {"q_range": (1.6438, 1.8), "bg_side": "right"},
    "104": {"q_range": (2.21, 2.40), "bg_side": "left"},
    "110": {"q_range": (2.45, 2.6), "bg_side": "right"},
    "116": {"q_range": (3.58, 3.82), "bg_side": "right"},
    "300": {"q_range": (4.30, 4.46), "bg_side": "left"},
}

DEFAULT_FIT_PEAK_SPECS = {
    "104": {"q_fit_range": (2.20, 2.40), "eta": 0.3},
    "110": {"q_fit_range": (2.40, 2.65), "eta": 0.3},
    "116": {"q_fit_range": (3.577, 3.823), "eta": 0.3},
    "300": {"q_fit_range": (4.3, 4.46), "eta": 0.3},
}

DEFAULT_AZIM_WINDOWS = [
    (-90, 90),
    (-75, -45),
    (-45, -15),
    (-15, 15),
    (15, 45),
    (45, 75),
]

DEFAULT_CALIBRATION_AZIMUTHAL_EDGES = [-75, -45, -15, 15, 45, 75]
DEFAULT_CALIBRATION_FIGURES_SUBDIR = "figures/calibration/"

DEFAULT_MULTI_EXPERIMENTS_DIFF = [
    dict(
        sample_name="DET70",
        temperature_K=110,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=1.5,
        time_window_fs=250,
        phi_mode="phi_avg",
        delay_offset_ps=9.6,
        ref_type="dark",
        ref_value=[1466560],
        label=r"V$_2$O$_3$, 110, 1500, 1.5",
        delay_for_norm_max=40,
    ),
    dict(
        label=r"V$_2$O$_3$, 110, 1500, 2.4",
        merge=[
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=2.4,
                time_window_fs=250,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466559],
                delay_for_norm_max=40,
            ),
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=2.4,
                time_window_fs=40,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466586],
                delay_for_norm_max=40,
            ),
        ],
        delay_for_norm_max=40,
    ),
]

DEFAULT_MULTI_EXPERIMENTS_FIT = [
    dict(
        sample_name="DET70",
        temperature_K=110,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=5,
        time_window_fs=250,
        phi_mode="phi_avg",
        delay_offset_ps=9.6,
        ref_type="dark",
        ref_value=[1466558],
        label=r"V$_2$O$_3$, 110, 1500, 5",
        delay_for_norm_max=40,
    ),
    dict(
        sample_name="DET70",
        temperature_K=110,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=9,
        time_window_fs=250,
        phi_mode="phi_avg",
        delay_offset_ps=9.6,
        ref_type="dark",
        ref_value=[1466588],
        label=r"V$_2$O$_3$, 110, 1500, 9",
        delay_for_norm_max=40,
    ),
    dict(
        label=r"V$_2$O$_3$, 110, 1500, 12",
        merge=[
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=12,
                time_window_fs=250,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466557],
                delay_for_norm_max=40,
            ),
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=12,
                time_window_fs=40,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466584],
                delay_for_norm_max=40,
            ),
        ],
        delay_for_norm_max=40,
    ),
    dict(
        label=r"V$_2$O$_3$, 110, 1500, 25",
        merge=[
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=25,
                time_window_fs=250,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466556],
                delay_for_norm_max=40,
            ),
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=25,
                time_window_fs=40,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466583],
                delay_for_norm_max=40,
            ),
        ],
        delay_for_norm_max=40,
    ),
]


def pretty_literal(obj) -> str:
    return repr(obj)


def parse_python_literal(text: str, *, empty=None):
    text = (text or "").strip()
    if text == "":
        return empty
    low = text.lower()
    if low == "none":
        return None
    if low == "all":
        return "all"
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def parse_int_like(text: str, *, name: str) -> int:
    text = (text or "").strip()
    if text == "":
        raise ValueError(f"{name} cannot be empty.")
    return int(float(text))


def parse_float_like(text: str, *, name: str) -> float:
    text = (text or "").strip()
    if text == "":
        raise ValueError(f"{name} cannot be empty.")
    return float(text)


def parse_optional_int_like(text: str):
    text = (text or "").strip()
    if text == "":
        return None
    return int(float(text))


def parse_optional_float_like(text: str):
    text = (text or "").strip()
    if text == "":
        return None
    return float(text)


def parse_tuple2(text: str, *, name: str, cast=float):
    value = parse_python_literal(text)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a tuple/list with two values, e.g. (-90, 90).")
    return cast(value[0]), cast(value[1])


def parse_optional_tuple2(text: str, *, name: str, cast=float):
    text = (text or "").strip()
    if text == "" or text.lower() == "none":
        return None
    return parse_tuple2(text, name=name, cast=cast)


def parse_edges(text: str):
    text = (text or "").strip()
    if text == "":
        raise ValueError("Azimuthal edges cannot be empty.")
    try:
        value = ast.literal_eval(text)
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float)
            if arr.ndim != 1 or arr.size < 2:
                raise ValueError
            return arr
    except Exception:
        pass
    parts = [p.strip() for p in text.split(",") if p.strip()]
    arr = np.asarray([float(p) for p in parts], dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("Azimuthal edges must contain at least two numeric values.")
    return arr


def parse_windows(text: str):
    value = parse_python_literal(text, empty=None)
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        return [tuple(float(x) for x in value)]
    if not isinstance(value, (list, tuple)):
        raise ValueError("Azimuth windows must be a list of tuples, e.g. [(-90,90), (-75,-45)].")
    out = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each azimuth window must contain two values.")
        out.append((float(item[0]), float(item[1])))
    return out


def parse_groups(text: str):
    value = parse_python_literal(text, empty=None)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def parse_scan_spec(text: str):
    text = (text or "").strip()
    if text == "":
        raise ValueError("scan / scan_spec cannot be empty.")
    return parse_python_literal(text)


class ExperimentLeafWidget(QFrame):
    def __init__(self, *, title="Experiment", show_label=True, show_remove=True, remove_callback=None, data=None):
        super().__init__()
        self.remove_callback = remove_callback
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #d9d9d9; border-radius: 4px; }")

        outer = QVBoxLayout()
        self.setLayout(outer)

        header = QHBoxLayout()
        outer.addLayout(header)
        header.addWidget(QLabel(f"<b>{title}</b>"))
        header.addStretch()
        if show_remove:
            btn = QPushButton("Remove")
            btn.clicked.connect(self._remove_me)
            header.addWidget(btn)

        grid = QGridLayout()
        outer.addLayout(grid)

        self.show_label = show_label
        row = 0
        if show_label:
            grid.addWidget(QLabel("label:"), row, 0)
            self.label_edit = QLineEdit("")
            grid.addWidget(self.label_edit, row, 1)
            row += 1
        else:
            self.label_edit = None

        grid.addWidget(QLabel("sample_name:"), row, 0)
        self.sample_name = QLineEdit("DET70")
        grid.addWidget(self.sample_name, row, 1)
        row += 1

        grid.addWidget(QLabel("temperature_K:"), row, 0)
        self.temperature = QLineEdit("110")
        self.temperature.setValidator(QDoubleValidator())
        grid.addWidget(self.temperature, row, 1)
        row += 1

        grid.addWidget(QLabel("excitation_wl_nm:"), row, 0)
        self.excitation = QLineEdit("1500")
        self.excitation.setValidator(QDoubleValidator())
        grid.addWidget(self.excitation, row, 1)
        row += 1

        grid.addWidget(QLabel("fluence_mJ_cm2:"), row, 0)
        self.fluence = QLineEdit("25")
        self.fluence.setValidator(QDoubleValidator())
        grid.addWidget(self.fluence, row, 1)
        row += 1

        grid.addWidget(QLabel("time_window_fs:"), row, 0)
        self.time_window = QLineEdit("250")
        self.time_window.setValidator(QDoubleValidator())
        grid.addWidget(self.time_window, row, 1)
        row += 1

        grid.addWidget(QLabel("phi_mode:"), row, 0)
        self.phi_mode = QComboBox()
        self.phi_mode.addItems(["phi_avg", "separate_phi"])
        grid.addWidget(self.phi_mode, row, 1)
        row += 1

        grid.addWidget(QLabel("delay_offset_ps:"), row, 0)
        self.delay_offset_ps = QLineEdit("")
        self.delay_offset_ps.setValidator(QDoubleValidator())
        grid.addWidget(self.delay_offset_ps, row, 1)
        row += 1

        grid.addWidget(QLabel("ref_type:"), row, 0)
        self.ref_type = QComboBox()
        self.ref_type.addItems(["dark", "delay"])
        grid.addWidget(self.ref_type, row, 1)
        row += 1

        grid.addWidget(QLabel("ref_value:"), row, 0)
        self.ref_value = QLineEdit("")
        self.ref_value.setPlaceholderText("Example: [1466556], -95000, '-5ns'")
        grid.addWidget(self.ref_value, row, 1)
        row += 1

        grid.addWidget(QLabel("delay_for_norm_max:"), row, 0)
        self.delay_for_norm_max = QLineEdit("")
        self.delay_for_norm_max.setValidator(QDoubleValidator())
        grid.addWidget(self.delay_for_norm_max, row, 1)
        row += 1

        grid.addWidget(QLabel("csv_path (optional):"), row, 0)
        csv_box = QHBoxLayout()
        self.csv_path = QLineEdit("")
        csv_box.addWidget(self.csv_path)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_csv)
        csv_box.addWidget(browse)
        grid.addLayout(csv_box, row, 1)

        if data:
            self.set_data(data)

    def _browse_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV Files (*.csv);;All Files (*)")
        if file_name:
            self.csv_path.setText(file_name)

    def _remove_me(self):
        if callable(self.remove_callback):
            self.remove_callback(self)

    def set_data(self, data):
        if self.label_edit is not None:
            self.label_edit.setText(str(data.get("label", "")))
        self.sample_name.setText(str(data.get("sample_name", "")))
        if "temperature_K" in data:
            self.temperature.setText(str(data.get("temperature_K", "")))
        if "excitation_wl_nm" in data:
            self.excitation.setText(str(data.get("excitation_wl_nm", "")))
        if "fluence_mJ_cm2" in data:
            self.fluence.setText(str(data.get("fluence_mJ_cm2", "")))
        if "time_window_fs" in data:
            self.time_window.setText(str(data.get("time_window_fs", "")))
        phi_mode = data.get("phi_mode", "phi_avg")
        idx = self.phi_mode.findText(str(phi_mode))
        if idx >= 0:
            self.phi_mode.setCurrentIndex(idx)
        if "delay_offset_ps" in data:
            self.delay_offset_ps.setText(str(data.get("delay_offset_ps", "")))
        ref_type = data.get("ref_type", "dark")
        idx = self.ref_type.findText(str(ref_type))
        if idx >= 0:
            self.ref_type.setCurrentIndex(idx)
        if "ref_value" in data:
            self.ref_value.setText(pretty_literal(data.get("ref_value")))
        if "delay_for_norm_max" in data:
            self.delay_for_norm_max.setText(str(data.get("delay_for_norm_max", "")))
        if "csv_path" in data:
            self.csv_path.setText(str(data.get("csv_path", "")))

    def to_dict(self):
        out = {}
        if self.label_edit is not None:
            label = self.label_edit.text().strip()
            if label:
                out["label"] = label

        sample_name = self.sample_name.text().strip()
        if not sample_name:
            raise ValueError("sample_name cannot be empty.")
        out["sample_name"] = sample_name
        out["temperature_K"] = parse_int_like(self.temperature.text(), name="temperature_K")
        out["excitation_wl_nm"] = parse_float_like(self.excitation.text(), name="excitation_wl_nm")
        out["fluence_mJ_cm2"] = parse_float_like(self.fluence.text(), name="fluence_mJ_cm2")
        out["time_window_fs"] = parse_int_like(self.time_window.text(), name="time_window_fs")
        out["phi_mode"] = self.phi_mode.currentText()

        delay_offset = parse_optional_float_like(self.delay_offset_ps.text())
        if delay_offset is not None:
            out["delay_offset_ps"] = delay_offset

        out["ref_type"] = self.ref_type.currentText()
        ref_value_text = self.ref_value.text().strip()
        if ref_value_text:
            out["ref_value"] = parse_python_literal(ref_value_text)

        delay_for_norm_max = parse_optional_float_like(self.delay_for_norm_max.text())
        if delay_for_norm_max is not None:
            out["delay_for_norm_max"] = delay_for_norm_max

        csv_path = self.csv_path.text().strip()
        if csv_path:
            out["csv_path"] = csv_path

        return out


class MergeExperimentWidget(QFrame):
    def __init__(self, *, title="Merged experiment", remove_callback=None, data=None):
        super().__init__()
        self.remove_callback = remove_callback
        self.sub_entries = []
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #cfcfcf; border-radius: 4px; }")

        outer = QVBoxLayout()
        self.setLayout(outer)

        header = QHBoxLayout()
        outer.addLayout(header)
        header.addWidget(QLabel(f"<b>{title}</b>"))
        header.addStretch()
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self._remove_me)
        header.addWidget(btn_remove)

        top_grid = QGridLayout()
        outer.addLayout(top_grid)
        top_grid.addWidget(QLabel("label:"), 0, 0)
        self.label_edit = QLineEdit("")
        top_grid.addWidget(self.label_edit, 0, 1)
        top_grid.addWidget(QLabel("delay_for_norm_max:"), 1, 0)
        self.delay_for_norm_max = QLineEdit("")
        self.delay_for_norm_max.setValidator(QDoubleValidator())
        top_grid.addWidget(self.delay_for_norm_max, 1, 1)

        sub_header = QHBoxLayout()
        outer.addLayout(sub_header)
        sub_header.addWidget(QLabel("<b>Sub-experiments</b>"))
        sub_header.addStretch()
        add_btn = QPushButton("Add sub-experiment")
        add_btn.clicked.connect(self.add_sub_experiment)
        sub_header.addWidget(add_btn)

        self.sub_container = QWidget()
        self.sub_layout = QVBoxLayout()
        self.sub_layout.setContentsMargins(0, 0, 0, 0)
        self.sub_container.setLayout(self.sub_layout)
        outer.addWidget(self.sub_container)

        if data:
            self.set_data(data)
        else:
            self.add_sub_experiment()
            self.add_sub_experiment()

    def _remove_me(self):
        if callable(self.remove_callback):
            self.remove_callback(self)

    def add_sub_experiment(self, data=None):
        widget = ExperimentLeafWidget(
            title="Merged source",
            show_label=False,
            show_remove=True,
            remove_callback=self.remove_sub_experiment,
            data=data,
        )
        self.sub_entries.append(widget)
        self.sub_layout.addWidget(widget)

    def remove_sub_experiment(self, widget):
        if widget in self.sub_entries:
            self.sub_entries.remove(widget)
            widget.setParent(None)
            widget.deleteLater()

    def set_data(self, data):
        self.label_edit.setText(str(data.get("label", "")))
        if "delay_for_norm_max" in data:
            self.delay_for_norm_max.setText(str(data.get("delay_for_norm_max", "")))
        for widget in list(self.sub_entries):
            self.remove_sub_experiment(widget)
        for sub in data.get("merge", []):
            self.add_sub_experiment(sub)
        if not self.sub_entries:
            self.add_sub_experiment()

    def to_dict(self):
        label = self.label_edit.text().strip()
        if not label:
            raise ValueError("Merged experiment label cannot be empty.")
        if not self.sub_entries:
            raise ValueError("Merged experiment must contain at least one sub-experiment.")
        out = {"label": label, "merge": [w.to_dict() for w in self.sub_entries]}
        delay_for_norm_max = parse_optional_float_like(self.delay_for_norm_max.text())
        if delay_for_norm_max is not None:
            out["delay_for_norm_max"] = delay_for_norm_max
        return out


class MultiExperimentEditor(QGroupBox):
    def __init__(self, title="Experiments", *, allow_merge=True, defaults=None):
        super().__init__(title)
        self.allow_merge = allow_merge
        self.entries = []
        self.defaults = defaults or []

        outer = QVBoxLayout()
        self.setLayout(outer)

        btn_row = QHBoxLayout()
        outer.addLayout(btn_row)

        add_btn = QPushButton("Add experiment")
        add_btn.clicked.connect(self.add_experiment)
        btn_row.addWidget(add_btn)

        if self.allow_merge:
            add_merge_btn = QPushButton("Add merged experiment")
            add_merge_btn.clicked.connect(self.add_merged_experiment)
            btn_row.addWidget(add_merge_btn)

        load_btn = QPushButton("Load example")
        load_btn.clicked.connect(self.load_defaults)
        btn_row.addWidget(load_btn)

        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self.clear_entries)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(360)
        outer.addWidget(self.scroll)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll.setWidget(self.scroll_content)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFixedHeight(75)
        outer.addWidget(self.preview)

        outer.setStretchFactor(self.scroll, 1)
        outer.setStretchFactor(self.preview, 0)

        self.load_defaults()

    def update_preview(self):
        try:
            self.preview.setPlainText(pretty_literal(self.get_experiments()))
        except Exception as exc:
            self.preview.setPlainText(f"Preview unavailable: {exc}")

    def clear_entries(self):
        for widget in list(self.entries):
            self.remove_entry(widget)
        self.update_preview()

    def load_defaults(self):
        self.clear_entries()
        if not self.defaults:
            self.add_experiment()
            return
        for item in self.defaults:
            if isinstance(item, dict) and "merge" in item:
                self.add_merged_experiment(item)
            else:
                self.add_experiment(item)
        self.update_preview()

    def set_experiments(self, experiments):
        self.clear_entries()
        if not experiments:
            self.update_preview()
            return
        for item in experiments:
            if isinstance(item, dict) and "merge" in item:
                if not self.allow_merge:
                    raise ValueError("This editor does not support merged experiments.")
                self.add_merged_experiment(item)
            else:
                self.add_experiment(item)
        self.update_preview()

    def add_experiment(self, data=None):
        widget = ExperimentLeafWidget(remove_callback=self.remove_entry, data=data)
        self.entries.append(widget)
        self.scroll_layout.addWidget(widget)
        self.update_preview()

    def add_merged_experiment(self, data=None):
        widget = MergeExperimentWidget(remove_callback=self.remove_entry, data=data)
        self.entries.append(widget)
        self.scroll_layout.addWidget(widget)
        self.update_preview()

    def remove_entry(self, widget):
        if widget in self.entries:
            self.entries.remove(widget)
            widget.setParent(None)
            widget.deleteLater()
            self.update_preview()

    def get_experiments(self):
        if not self.entries:
            raise ValueError("At least one experiment must be defined.")
        return [w.to_dict() for w in self.entries]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRDpy Analysis GUI")
        self.resize(900, 800)
        self._experiment_widgets = {}
        self._syncing_experiment_fields = False
        self._loading_gui_state = False

        self.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #cfcfcf;
                top: -1px;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 8px;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                font-weight: bold;
            }
            QPushButton {
                padding: 6px 10px;
            }
            """
        )

        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout()
        container.setLayout(layout)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.session_tab = QWidget()
        self.tabs.addTab(self.session_tab, "Session")
        self._init_session_tab()

        self.datared_tab = QWidget()
        self.tabs.addTab(self.datared_tab, "2D Preparation")
        self._init_datared_tab()

        self.calibration_tab = QWidget()
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self._init_calibration_tab()

        self.pattern_tab = QWidget()
        self.tabs.addTab(self.pattern_tab, "1D Pattern Creation")
        self._init_pattern_tab()

        self.viewer_tab = QWidget()
        self.tabs.addTab(self.viewer_tab, "1D Viewer")
        self._init_viewer_tab()

        self.diff_tab = QWidget()
        self.tabs.addTab(self.diff_tab, "Differential")
        self._init_diff_tab()

        self.fit_tab = QWidget()
        self.tabs.addTab(self.fit_tab, "Fitting")
        self._init_fit_tab()

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        log_group.setMaximumHeight(120)
        layout.addWidget(log_group)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(1000)
        self.log_box.setFixedHeight(60)
        log_layout.addWidget(self.log_box)

        self._refresh_facility_dependent_widgets()
        self._refresh_diff_mode_widgets()
        self._refresh_fit_mode_widgets()
        self._log("GUI ready.")
        if PACKAGE_IMPORT_ERROR is not None:
            self._warn_package_import()
        else:
            self._log(f"Using package import namespace: {PACKAGE_NAME}")

        self._maybe_restore_autosave()

    def _validated_diff_multi_experiments(self):
        experiments = self.diff_multi_editor.get_experiments()
        required = [
            "sample_name",
            "temperature_K",
            "excitation_wl_nm",
            "fluence_mJ_cm2",
            "time_window_fs",
            "ref_type",
            "ref_value",
        ]

        for i, exp in enumerate(experiments, start=1):
            if "merge" in exp:
                raise ValueError(
                    f"Differential multi does not support merged experiments.\n"
                    f"Experiment #{i} is a merged entry."
                )

            missing = [key for key in required if exp.get(key, None) in (None, "")]
            if missing:
                raise ValueError(
                    f"Differential experiment #{i} is missing required fields: {', '.join(missing)}"
                )

        return experiments

    def _make_scroll_tab(self, tab_widget: QWidget):
        main_layout = QVBoxLayout()
        tab_widget.setLayout(main_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)
        content = QWidget()
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        scroll.setWidget(content)
        return content_layout

    def _warn_package_import(self):
        msg = (
            "Could not import the analysis package.\n\n"
            "Run this script in the Python environment where your package is installed."
        )
        self._log(msg)
        QMessageBox.warning(self, "Import Warning", msg)

    def _log(self, message: str):
        self.log_box.appendPlainText(message)
        if self.statusBar() is not None:
            self.statusBar().showMessage(message, 8000)

    def _show_exception(self, title: str, exc: Exception):
        tb = traceback.format_exc()
        self._log(f"{title}: {exc}")
        self._log(tb)
        QMessageBox.critical(self, title, f"{exc}\n\n{tb}")

    def _browse_directory_into(self, target: QLineEdit):
        directory = QFileDialog.getExistingDirectory(self, "Select directory", target.text().strip() or "")
        if directory:
            target.setText(directory)

    def _browse_file_into(self, target: QLineEdit, caption="Select file", file_filter="All Files (*)"):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption=caption,
            directory=str(Path(target.text().strip()).parent) if target.text().strip() else "",
            filter=file_filter,
        )
        if file_name:
            target.setText(file_name)

    def _facility(self):
        return self.session_facility_combo.currentText()

    def _current_azimint_module(self):
        facility = self._facility()
        if facility == "SACLA":
            return sacla_azimint
        if facility == "FemtoMAX":
            return femto_azimint
        return id09_azimint

    def _paths(self):
        if AnalysisPaths is None:
            raise ImportError("AnalysisPaths could not be imported from the package.")
        path_root = self.session_path_root.text().strip()
        if not path_root:
            raise ValueError("path_root cannot be empty.")
        return AnalysisPaths(
            path_root=Path(path_root),
            analysis_subdir=self.session_analysis_subdir.text().strip() or "analysis",
            raw_subdir=self.session_raw_subdir.text().strip(),
        )

    def _poni_mask_kwargs(self):
        return {
            "poni_path": self.session_poni_path.text().strip() or None,
            "mask_edf_path": self.session_mask_path.text().strip() or None,
        }

    def _azim_offset_deg(self):
        return parse_float_like(self.session_azim_offset_deg.text(), name="azim_offset_deg")

    def _ref_value_from_line(self, line_edit: QLineEdit):
        text = line_edit.text().strip()
        if not text:
            return None
        return parse_python_literal(text)

    def _delays_value_from_line(self, line_edit: QLineEdit):
        text = line_edit.text().strip()
        if not text:
            raise ValueError("Delays field cannot be empty.")
        return parse_python_literal(text)

    def _diff_peak_specs(self):
        value = parse_python_literal(self.diff_peak_specs.toPlainText())
        if not isinstance(value, dict) or not value:
            raise ValueError("Differential peak specs must be a non-empty dict.")
        return value

    def _fit_peak_specs(self):
        value = parse_python_literal(self.fit_peak_specs.toPlainText())
        if not isinstance(value, dict) or not value:
            raise ValueError("Fitting peak specs must be a non-empty dict.")
        return value

    def _build_experiment_group(self, parent_layout, *, prefix: str, title: str, defaults=None, include_id09=False):
        defaults = defaults or {}
        group = QGroupBox(title)
        grid = QGridLayout()
        group.setLayout(grid)
        parent_layout.addWidget(group)

        widgets = {}
        row = 0
        grid.addWidget(QLabel("sample_name:"), row, 0)
        widgets["sample_name"] = QLineEdit(str(defaults.get("sample_name", "DET70")))
        grid.addWidget(widgets["sample_name"], row, 1)
        row += 1

        grid.addWidget(QLabel("temperature_K:"), row, 0)
        widgets["temperature_K"] = QLineEdit(str(defaults.get("temperature_K", "110")))
        widgets["temperature_K"].setValidator(QDoubleValidator())
        grid.addWidget(widgets["temperature_K"], row, 1)
        row += 1

        grid.addWidget(QLabel("excitation_wl_nm:"), row, 0)
        widgets["excitation_wl_nm"] = QLineEdit(str(defaults.get("excitation_wl_nm", "1500")))
        widgets["excitation_wl_nm"].setValidator(QDoubleValidator())
        grid.addWidget(widgets["excitation_wl_nm"], row, 1)
        row += 1

        grid.addWidget(QLabel("fluence_mJ_cm2:"), row, 0)
        widgets["fluence_mJ_cm2"] = QLineEdit(str(defaults.get("fluence_mJ_cm2", "25")))
        widgets["fluence_mJ_cm2"].setValidator(QDoubleValidator())
        grid.addWidget(widgets["fluence_mJ_cm2"], row, 1)
        row += 1

        grid.addWidget(QLabel("time_window_fs:"), row, 0)
        widgets["time_window_fs"] = QLineEdit(str(defaults.get("time_window_fs", "250")))
        widgets["time_window_fs"].setValidator(QDoubleValidator())
        grid.addWidget(widgets["time_window_fs"], row, 1)

        id09_group = None
        if include_id09:
            id09_group = QGroupBox("ID09-specific Metadata")
            id09_grid = QGridLayout()
            id09_group.setLayout(id09_grid)
            parent_layout.addWidget(id09_group)

            id09_grid.addWidget(QLabel("dataset:"), 0, 0)
            widgets["dataset"] = QLineEdit(str(defaults.get("dataset", "3")))
            widgets["dataset"].setValidator(QIntValidator())
            id09_grid.addWidget(widgets["dataset"], 0, 1)

            id09_grid.addWidget(QLabel("scan_nb:"), 1, 0)
            widgets["scan_nb"] = QLineEdit(str(defaults.get("scan_nb", "7")))
            widgets["scan_nb"].setValidator(QIntValidator())
            id09_grid.addWidget(widgets["scan_nb"], 1, 1)

        widgets["_group"] = group
        widgets["_id09_group"] = id09_group
        self._experiment_widgets[prefix] = widgets

        for field_name in ("sample_name", "temperature_K", "excitation_wl_nm", "fluence_mJ_cm2", "time_window_fs"):
            widgets[field_name].textChanged.connect(
                lambda _text, p=prefix, f=field_name: self._sync_experiment_field(p, f)
            )
        if include_id09:
            widgets["dataset"].textChanged.connect(lambda _text, p=prefix: self._sync_experiment_field(p, "dataset"))
            widgets["scan_nb"].textChanged.connect(lambda _text, p=prefix: self._sync_experiment_field(p, "scan_nb"))
        return widgets

    def _build_calibration_experiment_group(self, parent_layout, *, prefix: str, title: str, defaults=None):
        defaults = defaults or {}
        group = QGroupBox(title)
        grid = QGridLayout()
        group.setLayout(grid)
        parent_layout.addWidget(group)

        widgets = {}
        row = 0
        grid.addWidget(QLabel("sample_name:"), row, 0)
        widgets["sample_name"] = QLineEdit(str(defaults.get("sample_name", "DET70")))
        grid.addWidget(widgets["sample_name"], row, 1)
        row += 1

        grid.addWidget(QLabel("temperature_K:"), row, 0)
        widgets["temperature_K"] = QLineEdit(str(defaults.get("temperature_K", "110")))
        widgets["temperature_K"].setValidator(QDoubleValidator())
        grid.addWidget(widgets["temperature_K"], row, 1)
        row += 1

        grid.addWidget(QLabel("scan / scan_spec:"), row, 0)
        widgets["scan_spec"] = QLineEdit(str(defaults.get("scan_spec", "[7]")))
        widgets["scan_spec"].setPlaceholderText("Examples: [7], [1466556], 181661, 'scan_181661'")
        grid.addWidget(widgets["scan_spec"], row, 1)

        widgets["_group"] = group
        widgets["_id09_group"] = None
        self._experiment_widgets[prefix] = widgets

        for field_name in ("sample_name", "temperature_K"):
            widgets[field_name].textChanged.connect(
                lambda _text, p=prefix, f=field_name: self._sync_experiment_field(p, f)
            )
        return widgets

    def _sync_experiment_field(self, source_prefix: str, field_name: str):
        if self._syncing_experiment_fields:
            return
        source = self._experiment_widgets.get(source_prefix, {}).get(field_name)
        if source is None:
            return
        self._syncing_experiment_fields = True
        try:
            value = source.text()
            for prefix, widgets in self._experiment_widgets.items():
                if prefix == source_prefix:
                    continue
                target = widgets.get(field_name)
                if target is None or target.text() == value:
                    continue
                blocked = target.blockSignals(True)
                target.setText(value)
                target.blockSignals(blocked)
        finally:
            self._syncing_experiment_fields = False

    def _experiment_kwargs(self, prefix: str):
        widgets = self._experiment_widgets[prefix]
        sample_name = widgets["sample_name"].text().strip()
        if not sample_name:
            raise ValueError("sample_name cannot be empty.")
        return {
            "sample_name": sample_name,
            "temperature_K": parse_int_like(widgets["temperature_K"].text(), name="temperature_K"),
            "excitation_wl_nm": parse_float_like(widgets["excitation_wl_nm"].text(), name="excitation_wl_nm"),
            "fluence_mJ_cm2": parse_float_like(widgets["fluence_mJ_cm2"].text(), name="fluence_mJ_cm2"),
            "time_window_fs": parse_int_like(widgets["time_window_fs"].text(), name="time_window_fs"),
        }

    def _id09_kwargs_from_prefix(self, prefix: str):
        widgets = self._experiment_widgets[prefix]
        return {
            "dataset": parse_int_like(widgets["dataset"].text(), name="dataset"),
            "scan_nb": parse_int_like(widgets["scan_nb"].text(), name="scan_nb"),
        }

    def _calibration_context_kwargs(self):
        widgets = self._experiment_widgets["calibration"]
        sample_name = widgets["sample_name"].text().strip()
        if not sample_name:
            raise ValueError("sample_name cannot be empty.")
        return {
            "sample_name": sample_name,
            "scan": parse_scan_spec(widgets["scan_spec"].text()),
            "temperature_K": parse_int_like(widgets["temperature_K"].text(), name="temperature_K"),
            "paths": self._paths(),
        }

    def _datared_id09_dark_kwargs(self):
        widgets = self._experiment_widgets["datared"]
        sample_name = widgets["sample_name"].text().strip()
        if not sample_name:
            raise ValueError("sample_name cannot be empty.")
        kwargs = {
            "sample_name": sample_name,
            "temperature_K": parse_int_like(widgets["temperature_K"].text(), name="temperature_K"),
            "paths": self._paths(),
        }
        kwargs.update(self._id09_kwargs_from_prefix("datared"))
        return kwargs

    def _refresh_facility_dependent_widgets(self):
        is_id09 = self._facility() == "ID09"
        for widgets in self._experiment_widgets.values():
            if widgets.get("_id09_group") is not None:
                widgets["_id09_group"].setVisible(is_id09)

        if hasattr(self, "datared_id09_group"):
            self.datared_id09_group.setVisible(is_id09)
        if hasattr(self, "datared_runtime_group"):
            self.datared_runtime_group.setVisible(is_id09)
        if hasattr(self, "datared_actions_group"):
            self.datared_actions_group.setVisible(is_id09)
        if hasattr(self, "datared_placeholder_group"):
            self.datared_placeholder_group.setVisible(not is_id09)

        self.pattern_dark_group.setVisible(not is_id09)
        self.pattern_id09_group.setVisible(is_id09)
        self.pattern_normalize_checkbox.setVisible(not is_id09)
        self.viewer_id09_group.setVisible(is_id09)
        self.viewer_from2d_checkbox.setVisible(not is_id09)

        if is_id09:
            self.datared_note.setText(
                "This section is active for ID09.\n"
                "Use it to create the homogeneous dark 2D image and the delay-resolved 2D images "
                "inside the standard analysis structure."
            )
        else:
            self.datared_note.setText(
                "This section is designed as the general home for facility-specific 2D image production.\n"
                "In this version, only the ID09 backend is implemented here."
            )

        self._log(f"Facility set to {self._facility()}.")

    def _refresh_diff_mode_widgets(self):
        single_mode = self.diff_mode_combo.currentText() == "Single experiment"
        self.diff_single_widget.setVisible(single_mode)
        self.diff_multi_widget.setVisible(not single_mode)

    def _refresh_fit_mode_widgets(self):
        single_mode = self.fit_mode_combo.currentText() == "Single experiment"
        self.fit_single_widget.setVisible(single_mode)
        self.fit_multi_widget.setVisible(not single_mode)

    def _init_session_tab(self):
        layout = self._make_scroll_tab(self.session_tab)

        group = QGroupBox("Facility and Paths")
        grid = QGridLayout()
        group.setLayout(grid)
        layout.addWidget(group)

        row = 0
        grid.addWidget(QLabel("Facility:"), row, 0)
        self.session_facility_combo = QComboBox()
        self.session_facility_combo.addItems(["SACLA", "FemtoMAX", "ID09"])
        self.session_facility_combo.currentIndexChanged.connect(self._refresh_facility_dependent_widgets)
        grid.addWidget(self.session_facility_combo, row, 1)
        row += 1

        grid.addWidget(QLabel("path_root:"), row, 0)
        h = QHBoxLayout()
        self.session_path_root = QLineEdit("")
        h.addWidget(self.session_path_root)
        b = QPushButton("Browse")
        b.clicked.connect(lambda: self._browse_directory_into(self.session_path_root))
        h.addWidget(b)
        grid.addLayout(h, row, 1)
        row += 1

        grid.addWidget(QLabel("analysis_subdir:"), row, 0)
        self.session_analysis_subdir = QLineEdit("analysis")
        grid.addWidget(self.session_analysis_subdir, row, 1)
        row += 1

        grid.addWidget(QLabel("raw_subdir:"), row, 0)
        self.session_raw_subdir = QLineEdit("")
        self.session_raw_subdir.setPlaceholderText("Optional. Example: RAW_DATA/")
        grid.addWidget(self.session_raw_subdir, row, 1)

        calib_group = QGroupBox("Calibration and Shared Geometry")
        calib_grid = QGridLayout()
        calib_group.setLayout(calib_grid)
        layout.addWidget(calib_group)

        row = 0
        calib_grid.addWidget(QLabel("poni_path:"), row, 0)
        h = QHBoxLayout()
        self.session_poni_path = QLineEdit("")
        h.addWidget(self.session_poni_path)
        b = QPushButton("Browse")
        b.clicked.connect(
            lambda: self._browse_file_into(
                self.session_poni_path, "Select PONI file", "PONI Files (*.poni);;All Files (*)"
            )
        )
        h.addWidget(b)
        calib_grid.addLayout(h, row, 1)
        row += 1

        calib_grid.addWidget(QLabel("mask_edf_path:"), row, 0)
        h = QHBoxLayout()
        self.session_mask_path = QLineEdit("")
        h.addWidget(self.session_mask_path)
        b = QPushButton("Browse")
        b.clicked.connect(
            lambda: self._browse_file_into(
                self.session_mask_path, "Select mask file", "EDF Files (*.edf);;All Files (*)"
            )
        )
        h.addWidget(b)
        calib_grid.addLayout(h, row, 1)
        row += 1

        calib_grid.addWidget(QLabel("azim_offset_deg:"), row, 0)
        self.session_azim_offset_deg = QLineEdit("-90.0")
        self.session_azim_offset_deg.setValidator(QDoubleValidator())
        calib_grid.addWidget(self.session_azim_offset_deg, row, 1)

        persist_group = QGroupBox("GUI State")
        persist_layout = QHBoxLayout()
        persist_group.setLayout(persist_layout)
        layout.addWidget(persist_group)

        self.btn_save_state = QPushButton("Save GUI State...")
        self.btn_save_state.clicked.connect(self._save_gui_state_to_file)
        persist_layout.addWidget(self.btn_save_state)

        self.btn_load_state = QPushButton("Load GUI State...")
        self.btn_load_state.clicked.connect(self._load_gui_state_from_file)
        persist_layout.addWidget(self.btn_load_state)

        self.btn_load_autosave = QPushButton("Restore Last Autosave")
        self.btn_load_autosave.clicked.connect(self._load_autosave_from_disk)
        persist_layout.addWidget(self.btn_load_autosave)

        persist_layout.addStretch()
        layout.addStretch()

    def _init_datared_tab(self):
        layout = self._make_scroll_tab(self.datared_tab)
        self._build_experiment_group(layout, prefix="datared", title="Experiment Metadata", include_id09=True)

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
        self.datared_delays.setPlaceholderText("Examples: all, [0, 1, 5], ['-5ns', '0ps']")
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
        self.datared_create_dark_btn.clicked.connect(self._run_id09_create_dark)
        al.addWidget(self.datared_create_dark_btn)

        self.datared_create_delay_btn = QPushButton("Create ID09 Delay 2D Images")
        self.datared_create_delay_btn.clicked.connect(self._run_id09_create_delay_images)
        al.addWidget(self.datared_create_delay_btn)

        al.addStretch()

        self.datared_placeholder_group = QGroupBox("Other Facilities")
        placeholder_layout = QVBoxLayout()
        self.datared_placeholder_group.setLayout(placeholder_layout)
        layout.addWidget(self.datared_placeholder_group)

        placeholder_text = QLabel(
            "SACLA and FemtoMAX can be added here later without changing the GUI structure.\n"
            "For now, this tab only implements the ID09 backend."
        )
        placeholder_text.setWordWrap(True)
        placeholder_layout.addWidget(placeholder_text)

        layout.addStretch()

    def _init_calibration_tab(self):
        layout = self._make_scroll_tab(self.calibration_tab)

        note = QLabel(
            "Calibration is facility agnostic at this stage.\n"
            "It assumes the homogeneous dark 2D image already exists in the analysis structure."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self._build_calibration_experiment_group(
            layout,
            prefix="calibration",
            title="Calibration Context",
            defaults=dict(sample_name="DET70", temperature_K=110, scan_spec="[1466556]"),
        )

        integration_group = QGroupBox("Azimuthal Integration Settings")
        ig = QGridLayout()
        integration_group.setLayout(ig)
        layout.addWidget(integration_group)

        ig.addWidget(QLabel("azimuthal_edges:"), 0, 0)
        self.calib_azimuthal_edges = QLineEdit(pretty_literal(DEFAULT_CALIBRATION_AZIMUTHAL_EDGES))
        ig.addWidget(self.calib_azimuthal_edges, 0, 1)

        self.calib_include_full = QCheckBox("include_full")
        self.calib_include_full.setChecked(True)
        ig.addWidget(self.calib_include_full, 1, 0, 1, 2)

        ig.addWidget(QLabel("full_range:"), 2, 0)
        self.calib_full_range = QLineEdit("(-90, 90)")
        ig.addWidget(self.calib_full_range, 2, 1)

        ig.addWidget(QLabel("npt:"), 3, 0)
        self.calib_npt = QLineEdit("1000")
        self.calib_npt.setValidator(QDoubleValidator())
        ig.addWidget(self.calib_npt, 3, 1)

        self.calib_normalize = QCheckBox("normalize")
        self.calib_normalize.setChecked(True)
        ig.addWidget(self.calib_normalize, 4, 0, 1, 2)

        ig.addWidget(QLabel("q_norm_range:"), 5, 0)
        self.calib_q_norm_range = QLineEdit("(2.65, 2.75)")
        ig.addWidget(self.calib_q_norm_range, 5, 1)

        self.calib_overwrite_xy = QCheckBox("overwrite_xy")
        self.calib_overwrite_xy.setChecked(False)
        ig.addWidget(self.calib_overwrite_xy, 6, 0, 1, 2)

        fit_group = QGroupBox("Peak Fitting Settings")
        fg = QGridLayout()
        fit_group.setLayout(fg)
        layout.addWidget(fit_group)

        fg.addWidget(QLabel("q_fit_range:"), 0, 0)
        self.calib_q_fit_range = QLineEdit("(2.4, 2.65)")
        fg.addWidget(self.calib_q_fit_range, 0, 1)

        fg.addWidget(QLabel("eta:"), 1, 0)
        self.calib_eta = QLineEdit("0.3")
        self.calib_eta.setValidator(QDoubleValidator())
        fg.addWidget(self.calib_eta, 1, 1)

        fg.addWidget(QLabel("fit_method:"), 2, 0)
        self.calib_fit_method = QLineEdit("leastsq")
        fg.addWidget(self.calib_fit_method, 2, 1)

        self.calib_force_refit = QCheckBox("force_refit")
        self.calib_force_refit.setChecked(True)
        fg.addWidget(self.calib_force_refit, 3, 0, 1, 2)

        fg.addWidget(QLabel("out_csv_name:"), 4, 0)
        self.calib_out_csv_name = QLineEdit("peak_fits.csv")
        fg.addWidget(self.calib_out_csv_name, 4, 1)

        caked_group = QGroupBox("Caked 1D Pattern Plot")
        cg = QGridLayout()
        caked_group.setLayout(cg)
        layout.addWidget(caked_group)

        cg.addWidget(QLabel("xlim:"), 0, 0)
        self.calib_caked_xlim = QLineEdit("(2.45, 2.60)")
        cg.addWidget(self.calib_caked_xlim, 0, 1)

        cg.addWidget(QLabel("ylim:"), 1, 0)
        self.calib_caked_ylim = QLineEdit("")
        self.calib_caked_ylim.setPlaceholderText("Optional")
        cg.addWidget(self.calib_caked_ylim, 1, 1)

        cg.addWidget(QLabel("figure_title:"), 2, 0)
        self.calib_caked_figure_title = QLineEdit("")
        self.calib_caked_figure_title.setPlaceholderText("Optional")
        cg.addWidget(self.calib_caked_figure_title, 2, 1)

        self.calib_caked_save = QCheckBox("save")
        self.calib_caked_save.setChecked(True)
        cg.addWidget(self.calib_caked_save, 3, 0, 1, 2)

        property_group = QGroupBox("Property vs Azimuth Plot")
        pg = QGridLayout()
        property_group.setLayout(pg)
        layout.addWidget(property_group)

        pg.addWidget(QLabel("property:"), 0, 0)
        self.calib_property_name = QLineEdit("pv_center")
        pg.addWidget(self.calib_property_name, 0, 1)

        self.calib_property_only_success = QCheckBox("only_success")
        self.calib_property_only_success.setChecked(True)
        pg.addWidget(self.calib_property_only_success, 1, 0, 1, 2)

        pg.addWidget(QLabel("ylim:"), 2, 0)
        self.calib_property_ylim = QLineEdit("")
        self.calib_property_ylim.setPlaceholderText("Optional")
        pg.addWidget(self.calib_property_ylim, 2, 1)

        pg.addWidget(QLabel("figure_title:"), 3, 0)
        self.calib_property_figure_title = QLineEdit("")
        self.calib_property_figure_title.setPlaceholderText("Optional")
        pg.addWidget(self.calib_property_figure_title, 3, 1)

        self.calib_property_save = QCheckBox("save")
        self.calib_property_save.setChecked(True)
        pg.addWidget(self.calib_property_save, 4, 0, 1, 2)

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

        sg.addWidget(QLabel("save_dpi:"), 2, 0)
        self.calib_save_dpi = QLineEdit("400")
        self.calib_save_dpi.setValidator(QDoubleValidator())
        sg.addWidget(self.calib_save_dpi, 2, 1)

        action_group = QGroupBox("Actions")
        al = QHBoxLayout()
        action_group.setLayout(al)
        layout.addWidget(action_group)

        btn = QPushButton("Compute XY Files")
        btn.clicked.connect(self._run_calibration_compute_xy)
        al.addWidget(btn)

        btn = QPushButton("Run Peak Fitting")
        btn.clicked.connect(self._run_calibration_peak_fitting)
        al.addWidget(btn)

        btn = QPushButton("Plot Caked 1D Patterns")
        btn.clicked.connect(self._run_calibration_plot_caked)
        al.addWidget(btn)

        btn = QPushButton("Plot Property vs Azimuth")
        btn.clicked.connect(self._run_calibration_plot_property)
        al.addWidget(btn)

        al.addStretch()
        layout.addStretch()

    def _init_pattern_tab(self):
        layout = self._make_scroll_tab(self.pattern_tab)
        self._build_experiment_group(layout, prefix="pattern", title="Experiment Metadata", include_id09=True)

        group = QGroupBox("Integration Target")
        grid = QGridLayout()
        group.setLayout(grid)
        layout.addWidget(group)
        grid.addWidget(QLabel("delays_fs:"), 0, 0)
        self.pattern_delays = QLineEdit("all")
        grid.addWidget(self.pattern_delays, 0, 1)

        self.pattern_dark_group = QGroupBox("Dark Integration (SACLA / FemtoMAX)")
        dark_grid = QGridLayout()
        self.pattern_dark_group.setLayout(dark_grid)
        layout.addWidget(self.pattern_dark_group)
        dark_grid.addWidget(QLabel("dark_tag:"), 0, 0)
        self.pattern_dark_tag = QLineEdit("")
        dark_grid.addWidget(self.pattern_dark_tag, 0, 1)

        self.pattern_id09_group = QGroupBox("ID09-specific Options")
        id09_grid = QGridLayout()
        self.pattern_id09_group.setLayout(id09_grid)
        layout.addWidget(self.pattern_id09_group)
        id09_grid.addWidget(QLabel("ref_delay:"), 0, 0)
        self.pattern_ref_delay = QLineEdit("-5ns")
        id09_grid.addWidget(self.pattern_ref_delay, 0, 1)
        self.pattern_force_checkbox = QCheckBox("force")
        self.pattern_force_checkbox.setChecked(True)
        id09_grid.addWidget(self.pattern_force_checkbox, 1, 0, 1, 2)

        az_group = QGroupBox("Azimuthal and Integration Settings")
        az_grid = QGridLayout()
        az_group.setLayout(az_grid)
        layout.addWidget(az_group)
        az_grid.addWidget(QLabel("azimuthal_edges:"), 0, 0)
        self.pattern_azimuthal_edges = QLineEdit("-90, -60, -30, 0, 30, 60, 90")
        az_grid.addWidget(self.pattern_azimuthal_edges, 0, 1)
        self.pattern_include_full = QCheckBox("include_full")
        self.pattern_include_full.setChecked(True)
        az_grid.addWidget(self.pattern_include_full, 1, 0, 1, 2)
        az_grid.addWidget(QLabel("full_range:"), 2, 0)
        self.pattern_full_range = QLineEdit("(-90, 90)")
        az_grid.addWidget(self.pattern_full_range, 2, 1)
        az_grid.addWidget(QLabel("npt:"), 3, 0)
        self.pattern_npt = QLineEdit("1000")
        self.pattern_npt.setValidator(QDoubleValidator())
        az_grid.addWidget(self.pattern_npt, 3, 1)
        self.pattern_normalize_checkbox = QCheckBox("normalize")
        self.pattern_normalize_checkbox.setChecked(True)
        az_grid.addWidget(self.pattern_normalize_checkbox, 4, 0, 1, 2)
        az_grid.addWidget(QLabel("q_norm_range:"), 5, 0)
        self.pattern_q_norm_range = QLineEdit("(2.65, 2.75)")
        az_grid.addWidget(self.pattern_q_norm_range, 5, 1)

        runtime_group = QGroupBox("Runtime Options")
        rg = QGridLayout()
        runtime_group.setLayout(rg)
        layout.addWidget(runtime_group)
        self.pattern_overwrite_xy = QCheckBox("overwrite_xy")
        self.pattern_overwrite_xy.setChecked(True)
        rg.addWidget(self.pattern_overwrite_xy, 0, 0, 1, 2)

        action_group = QGroupBox("Actions")
        al = QHBoxLayout()
        action_group.setLayout(al)
        layout.addWidget(action_group)
        self.pattern_integrate_dark_btn = QPushButton("Integrate Dark 1D")
        self.pattern_integrate_dark_btn.clicked.connect(self._run_integrate_dark)
        al.addWidget(self.pattern_integrate_dark_btn)
        self.pattern_integrate_delay_btn = QPushButton("Integrate Delay 1D")
        self.pattern_integrate_delay_btn.clicked.connect(self._run_integrate_delay)
        al.addWidget(self.pattern_integrate_delay_btn)
        al.addStretch()
        layout.addStretch()

    def _init_viewer_tab(self):
        layout = self._make_scroll_tab(self.viewer_tab)
        self._build_experiment_group(layout, prefix="viewer", title="Experiment Metadata", include_id09=True)

        group = QGroupBox("Reference and Delay Selection")
        grid = QGridLayout()
        group.setLayout(grid)
        layout.addWidget(group)
        grid.addWidget(QLabel("delays_fs:"), 0, 0)
        self.viewer_delays = QLineEdit("all")
        grid.addWidget(self.viewer_delays, 0, 1)
        grid.addWidget(QLabel("ref_type:"), 1, 0)
        self.viewer_ref_type = QComboBox()
        self.viewer_ref_type.addItems(["dark", "delay"])
        grid.addWidget(self.viewer_ref_type, 1, 1)
        grid.addWidget(QLabel("ref_value:"), 2, 0)
        self.viewer_ref_value = QLineEdit("[1466556]")
        grid.addWidget(self.viewer_ref_value, 2, 1)
        grid.addWidget(QLabel("azim_window:"), 3, 0)
        self.viewer_azim_window = QLineEdit("(-90, 90)")
        grid.addWidget(self.viewer_azim_window, 3, 1)
        grid.addWidget(QLabel("xlim:"), 4, 0)
        self.viewer_xlim = QLineEdit("(1.5, 4.5)")
        grid.addWidget(self.viewer_xlim, 4, 1)
        grid.addWidget(QLabel("digits:"), 5, 0)
        self.viewer_digits = QLineEdit("2")
        self.viewer_digits.setValidator(QDoubleValidator())
        grid.addWidget(self.viewer_digits, 5, 1)

        self.viewer_id09_group = QGroupBox("ID09-specific Viewer Options")
        vg = QGridLayout()
        self.viewer_id09_group.setLayout(vg)
        layout.addWidget(self.viewer_id09_group)
        vg.addWidget(QLabel("ref_delay:"), 0, 0)
        self.viewer_ref_delay = QLineEdit("-5ns")
        vg.addWidget(self.viewer_ref_delay, 0, 1)
        self.viewer_compute_if_missing = QCheckBox("compute_if_missing")
        self.viewer_compute_if_missing.setChecked(True)
        vg.addWidget(self.viewer_compute_if_missing, 1, 0, 1, 2)
        vg.addWidget(QLabel("fs_or_ps:"), 2, 0)
        self.viewer_fs_or_ps = QComboBox()
        self.viewer_fs_or_ps.addItems(["ps", "fs"])
        vg.addWidget(self.viewer_fs_or_ps, 2, 1)

        runtime_group = QGroupBox("Save / Runtime Options")
        rg = QGridLayout()
        runtime_group.setLayout(rg)
        layout.addWidget(runtime_group)
        self.viewer_overwrite_xy = QCheckBox("overwrite_xy")
        rg.addWidget(self.viewer_overwrite_xy, 0, 0, 1, 2)
        self.viewer_from2d_checkbox = QCheckBox("from_2D_imgs")
        rg.addWidget(self.viewer_from2d_checkbox, 1, 0, 1, 2)
        self.viewer_save_plots = QCheckBox("save_plots")
        self.viewer_save_plots.setChecked(True)
        rg.addWidget(self.viewer_save_plots, 2, 0, 1, 2)
        rg.addWidget(QLabel("save_format:"), 3, 0)
        self.viewer_save_format = QComboBox()
        self.viewer_save_format.addItems(["png", "pdf", "svg"])
        rg.addWidget(self.viewer_save_format, 3, 1)
        rg.addWidget(QLabel("save_dpi:"), 4, 0)
        self.viewer_save_dpi = QLineEdit("400")
        self.viewer_save_dpi.setValidator(QDoubleValidator())
        rg.addWidget(self.viewer_save_dpi, 4, 1)
        self.viewer_save_overwrite = QCheckBox("save_overwrite")
        self.viewer_save_overwrite.setChecked(True)
        rg.addWidget(self.viewer_save_overwrite, 5, 0, 1, 2)

        btn = QPushButton("Plot 1D Absolute + Differences")
        btn.clicked.connect(self._run_plot_1d)
        layout.addWidget(btn)
        layout.addStretch()

    def _init_diff_tab(self):
        layout = self._make_scroll_tab(self.diff_tab)

        mode_group = QGroupBox("Analysis Mode")
        ml = QHBoxLayout()
        mode_group.setLayout(ml)
        layout.addWidget(mode_group)
        ml.addWidget(QLabel("Differential mode:"))
        self.diff_mode_combo = QComboBox()
        self.diff_mode_combo.addItems(["Single experiment", "Multiple experiments"])
        self.diff_mode_combo.currentIndexChanged.connect(self._refresh_diff_mode_widgets)
        ml.addWidget(self.diff_mode_combo)
        ml.addStretch()

        self.diff_single_widget = QWidget()
        dsl = QVBoxLayout()
        self.diff_single_widget.setLayout(dsl)
        layout.addWidget(self.diff_single_widget)

        self._build_experiment_group(dsl, prefix="diff_single", title="Experiment Metadata")

        primary_group = QGroupBox("Single-experiment Differential Analysis")
        grid = QGridLayout()
        primary_group.setLayout(grid)
        dsl.addWidget(primary_group)
        row = 0
        grid.addWidget(QLabel("delays_fs:"), row, 0)
        self.diff_delays = QLineEdit("all")
        grid.addWidget(self.diff_delays, row, 1)
        row += 1
        grid.addWidget(QLabel("ref_type:"), row, 0)
        self.diff_ref_type = QComboBox()
        self.diff_ref_type.addItems(["dark", "delay"])
        grid.addWidget(self.diff_ref_type, row, 1)
        row += 1
        grid.addWidget(QLabel("ref_value:"), row, 0)
        self.diff_ref_value = QLineEdit("[1466556]")
        grid.addWidget(self.diff_ref_value, row, 1)
        row += 1
        grid.addWidget(QLabel("azim_window:"), row, 0)
        self.diff_azim_window = QLineEdit("(-90, 90)")
        grid.addWidget(self.diff_azim_window, row, 1)
        row += 1
        grid.addWidget(QLabel("peak:"), row, 0)
        self.diff_peak = QLineEdit("110")
        grid.addWidget(self.diff_peak, row, 1)
        row += 1
        grid.addWidget(QLabel("peak_specs:"), row, 0)
        self.diff_peak_specs = QPlainTextEdit(pretty_literal(DEFAULT_DIFF_PEAK_SPECS))
        self.diff_peak_specs.setMinimumHeight(120)
        grid.addWidget(self.diff_peak_specs, row, 1)

        int_group = QGroupBox("Integral Plot Settings")
        ig = QGridLayout()
        int_group.setLayout(ig)
        dsl.addWidget(int_group)
        ig.addWidget(QLabel("unit:"), 0, 0)
        self.diff_unit = QComboBox()
        self.diff_unit.addItems(["ps", "fs"])
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

        fft_group = QGroupBox("FFT Settings")
        fg = QGridLayout()
        fft_group.setLayout(fg)
        dsl.addWidget(fft_group)
        fg.addWidget(QLabel("region:"), 0, 0)
        self.diff_region = QComboBox()
        self.diff_region.addItems(["peak", "background"])
        fg.addWidget(self.diff_region, 0, 1)
        fg.addWidget(QLabel("kind:"), 1, 0)
        self.diff_kind = QComboBox()
        self.diff_kind.addItems(["diff", "abs"])
        fg.addWidget(self.diff_kind, 1, 1)
        fg.addWidget(QLabel("time_window_select_ps:"), 2, 0)
        self.diff_time_window_select = QLineEdit("(-1, 200)")
        fg.addWidget(self.diff_time_window_select, 2, 1)
        fg.addWidget(QLabel("poly_order:"), 3, 0)
        self.diff_poly_order = QLineEdit("2")
        self.diff_poly_order.setValidator(QDoubleValidator())
        fg.addWidget(self.diff_poly_order, 3, 1)
        fg.addWidget(QLabel("freq_unit:"), 4, 0)
        self.diff_freq_unit = QComboBox()
        self.diff_freq_unit.addItems(["cm^-1", "1/ps"])
        fg.addWidget(self.diff_freq_unit, 4, 1)
        fg.addWidget(QLabel("xlim_freq:"), 5, 0)
        self.diff_xlim_freq = QLineEdit("(-50, 850)")
        fg.addWidget(self.diff_xlim_freq, 5, 1)

        runtime_group = QGroupBox("Runtime and Save Options")
        rg = QGridLayout()
        runtime_group.setLayout(rg)
        dsl.addWidget(runtime_group)
        rg.addWidget(QLabel("npt:"), 0, 0)
        self.diff_npt = QLineEdit("1000")
        self.diff_npt.setValidator(QDoubleValidator())
        rg.addWidget(self.diff_npt, 0, 1)
        self.diff_normalize_xy = QCheckBox("normalize_xy")
        self.diff_normalize_xy.setChecked(True)
        rg.addWidget(self.diff_normalize_xy, 1, 0, 1, 2)
        rg.addWidget(QLabel("q_norm_range:"), 2, 0)
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
        rg.addWidget(QLabel("save_dpi:"), 7, 0)
        self.diff_save_dpi = QLineEdit("400")
        self.diff_save_dpi.setValidator(QDoubleValidator())
        rg.addWidget(self.diff_save_dpi, 7, 1)
        self.diff_save_overwrite = QCheckBox("save_overwrite")
        self.diff_save_overwrite.setChecked(True)
        rg.addWidget(self.diff_save_overwrite, 8, 0, 1, 2)

        btn_row = QHBoxLayout()
        b = QPushButton("Plot Differential Integrals")
        b.clicked.connect(self._run_diff_integrals)
        btn_row.addWidget(b)
        b = QPushButton("Plot Differential FFT")
        b.clicked.connect(self._run_diff_fft)
        btn_row.addWidget(b)
        btn_row.addStretch()
        dsl.addLayout(btn_row)
        dsl.addStretch()

        self.diff_multi_widget = QWidget()
        dml = QVBoxLayout()
        self.diff_multi_widget.setLayout(dml)
        layout.addWidget(self.diff_multi_widget)

        self.diff_multi_editor = MultiExperimentEditor(
            "Multiple-experiment Definitions",
            allow_merge=False,
            defaults=DEFAULT_MULTI_EXPERIMENTS_DIFF,
        )
        dml.addWidget(self.diff_multi_editor)

        group = QGroupBox("Multiple-experiment Plot Settings")
        grid = QGridLayout()
        group.setLayout(grid)
        dml.addWidget(group)
        row = 0
        grid.addWidget(QLabel("delays_fs:"), row, 0)
        self.diff_multi_delays = QLineEdit("all")
        grid.addWidget(self.diff_multi_delays, row, 1)
        row += 1
        grid.addWidget(QLabel("azim_window:"), row, 0)
        self.diff_multi_azim_window = QLineEdit("(-90, 90)")
        grid.addWidget(self.diff_multi_azim_window, row, 1)
        row += 1
        grid.addWidget(QLabel("peak:"), row, 0)
        self.diff_multi_peak = QLineEdit("110")
        grid.addWidget(self.diff_multi_peak, row, 1)
        row += 1
        grid.addWidget(QLabel("peak_specs:"), row, 0)
        self.diff_multi_peak_specs = QPlainTextEdit(pretty_literal(DEFAULT_DIFF_PEAK_SPECS))
        self.diff_multi_peak_specs.setMinimumHeight(120)
        grid.addWidget(self.diff_multi_peak_specs, row, 1)
        row += 1
        grid.addWidget(QLabel("npt:"), row, 0)
        self.diff_multi_npt = QLineEdit("1000")
        self.diff_multi_npt.setValidator(QDoubleValidator())
        grid.addWidget(self.diff_multi_npt, row, 1)
        row += 1
        self.diff_multi_normalize_xy = QCheckBox("normalize_xy")
        self.diff_multi_normalize_xy.setChecked(True)
        grid.addWidget(self.diff_multi_normalize_xy, row, 0, 1, 2)
        row += 1
        grid.addWidget(QLabel("q_norm_range:"), row, 0)
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
        grid.addWidget(QLabel("save_dpi:"), row, 0)
        self.diff_multi_save_dpi = QLineEdit("400")
        self.diff_multi_save_dpi.setValidator(QDoubleValidator())
        grid.addWidget(self.diff_multi_save_dpi, row, 1)
        row += 1
        self.diff_multi_save_overwrite = QCheckBox("save_overwrite")
        self.diff_multi_save_overwrite.setChecked(True)
        grid.addWidget(self.diff_multi_save_overwrite, row, 0, 1, 2)

        group = QGroupBox("Multiple-experiment Integrals")
        grid = QGridLayout()
        group.setLayout(grid)
        dml.addWidget(group)
        grid.addWidget(QLabel("unit:"), 0, 0)
        self.diff_multi_unit = QComboBox()
        self.diff_multi_unit.addItems(["ps", "fs"])
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

        group = QGroupBox("Multiple-experiment FFT")
        grid = QGridLayout()
        group.setLayout(grid)
        dml.addWidget(group)
        grid.addWidget(QLabel("kind:"), 0, 0)
        self.diff_multi_kind = QComboBox()
        self.diff_multi_kind.addItems(["diff", "abs"])
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

        btn_row = QHBoxLayout()
        b = QPushButton("Plot Multi Differential Integrals")
        b.clicked.connect(self._run_diff_integrals_multi)
        btn_row.addWidget(b)
        b = QPushButton("Plot Multi Differential FFT")
        b.clicked.connect(self._run_diff_fft_multi)
        btn_row.addWidget(b)
        btn_row.addStretch()
        dml.addLayout(btn_row)
        dml.addStretch()

    def _init_fit_tab(self):
        layout = self._make_scroll_tab(self.fit_tab)

        mode_group = QGroupBox("Analysis Mode")
        ml = QHBoxLayout()
        mode_group.setLayout(ml)
        layout.addWidget(mode_group)
        ml.addWidget(QLabel("Fitting mode:"))
        self.fit_mode_combo = QComboBox()
        self.fit_mode_combo.addItems(["Single experiment", "Multiple experiments"])
        self.fit_mode_combo.currentIndexChanged.connect(self._refresh_fit_mode_widgets)
        ml.addWidget(self.fit_mode_combo)
        ml.addStretch()

        self.fit_single_widget = QWidget()
        fsl = QVBoxLayout()
        self.fit_single_widget.setLayout(fsl)
        layout.addWidget(self.fit_single_widget)
        self._build_experiment_group(fsl, prefix="fit_single", title="Experiment Metadata")

        fit_group = QGroupBox("Delay-series Peak Fitting")
        grid = QGridLayout()
        fit_group.setLayout(grid)
        fsl.addWidget(fit_group)
        row = 0
        grid.addWidget(QLabel("delays_fs:"), row, 0)
        self.fit_delays = QLineEdit("all")
        grid.addWidget(self.fit_delays, row, 1)
        row += 1
        grid.addWidget(QLabel("ref_type:"), row, 0)
        self.fit_ref_type = QComboBox()
        self.fit_ref_type.addItems(["dark", "delay"])
        grid.addWidget(self.fit_ref_type, row, 1)
        row += 1
        grid.addWidget(QLabel("ref_value:"), row, 0)
        self.fit_ref_value = QLineEdit("[1466556]")
        grid.addWidget(self.fit_ref_value, row, 1)
        row += 1
        grid.addWidget(QLabel("peak_specs:"), row, 0)
        self.fit_peak_specs = QPlainTextEdit(pretty_literal(DEFAULT_FIT_PEAK_SPECS))
        self.fit_peak_specs.setMinimumHeight(120)
        grid.addWidget(self.fit_peak_specs, row, 1)
        row += 1
        grid.addWidget(QLabel("azim_windows:"), row, 0)
        self.fit_azim_windows = QPlainTextEdit(pretty_literal(DEFAULT_AZIM_WINDOWS))
        self.fit_azim_windows.setMinimumHeight(90)
        grid.addWidget(self.fit_azim_windows, row, 1)
        row += 1
        grid.addWidget(QLabel("phi_mode:"), row, 0)
        self.fit_phi_mode = QComboBox()
        self.fit_phi_mode.addItems(["phi_avg", "separate_phi"])
        grid.addWidget(self.fit_phi_mode, row, 1)
        row += 1
        grid.addWidget(QLabel("phi_reduce:"), row, 0)
        self.fit_phi_reduce = QComboBox()
        self.fit_phi_reduce.addItems(["sum", "mean"])
        grid.addWidget(self.fit_phi_reduce, row, 1)
        row += 1
        grid.addWidget(QLabel("default_eta:"), row, 0)
        self.fit_default_eta = QLineEdit("0.3")
        self.fit_default_eta.setValidator(QDoubleValidator())
        grid.addWidget(self.fit_default_eta, row, 1)
        row += 1
        grid.addWidget(QLabel("npt:"), row, 0)
        self.fit_npt = QLineEdit("1000")
        self.fit_npt.setValidator(QDoubleValidator())
        grid.addWidget(self.fit_npt, row, 1)
        row += 1
        grid.addWidget(QLabel("q_norm_range:"), row, 0)
        self.fit_q_norm_range = QLineEdit("(2.65, 2.75)")
        grid.addWidget(self.fit_q_norm_range, row, 1)
        row += 1
        grid.addWidget(QLabel("out_csv_name:"), row, 0)
        self.fit_out_csv_name = QLineEdit("peak_fits_delay.csv")
        grid.addWidget(self.fit_out_csv_name, row, 1)

        options_group = QGroupBox("Fit Runtime Options")
        og = QGridLayout()
        options_group.setLayout(og)
        fsl.addWidget(options_group)
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

        overlay_group = QGroupBox("Overlay Plot from CSV")
        ov = QGridLayout()
        overlay_group.setLayout(ov)
        fsl.addWidget(overlay_group)
        ov.addWidget(QLabel("peak:"), 0, 0)
        self.fit_overlay_peak = QLineEdit("110")
        ov.addWidget(self.fit_overlay_peak, 0, 1)
        ov.addWidget(QLabel("delay_fs:"), 1, 0)
        self.fit_overlay_delay = QLineEdit("0")
        ov.addWidget(self.fit_overlay_delay, 1, 1)
        ov.addWidget(QLabel("group:"), 2, 0)
        self.fit_overlay_group = QLineEdit("Full")
        ov.addWidget(self.fit_overlay_group, 2, 1)
        self.fit_overlay_is_reference = QCheckBox("is_reference")
        ov.addWidget(self.fit_overlay_is_reference, 3, 0, 1, 2)
        self.fit_overlay_ensure_csv = QCheckBox("ensure_csv")
        self.fit_overlay_ensure_csv.setChecked(True)
        ov.addWidget(self.fit_overlay_ensure_csv, 4, 0, 1, 2)
        self.fit_overlay_show = QCheckBox("show")
        self.fit_overlay_show.setChecked(True)
        ov.addWidget(self.fit_overlay_show, 5, 0, 1, 2)
        self.fit_overlay_save = QCheckBox("save")
        self.fit_overlay_save.setChecked(True)
        ov.addWidget(self.fit_overlay_save, 6, 0, 1, 2)

        time_group = QGroupBox("Time Evolution Plot")
        tg = QGridLayout()
        time_group.setLayout(tg)
        fsl.addWidget(time_group)
        tg.addWidget(QLabel("peak:"), 0, 0)
        self.fit_time_peak = QLineEdit("110")
        tg.addWidget(self.fit_time_peak, 0, 1)
        tg.addWidget(QLabel("_property:"), 1, 0)
        self.fit_property = QComboBox()
        self.fit_property.addItems(["hkl_pos", "hkl_fwhm", "amplitude", "eta"])
        tg.addWidget(self.fit_property, 1, 1)
        tg.addWidget(QLabel("unit:"), 2, 0)
        self.fit_time_unit = QComboBox()
        self.fit_time_unit.addItems(["ps", "fs"])
        tg.addWidget(self.fit_time_unit, 2, 1)
        tg.addWidget(QLabel("groups:"), 3, 0)
        self.fit_groups = QLineEdit("['Full', 60, 30, 0]")
        tg.addWidget(self.fit_groups, 3, 1)
        tg.addWidget(QLabel("delay_offset:"), 4, 0)
        self.fit_delay_offset = QLineEdit("0")
        self.fit_delay_offset.setValidator(QDoubleValidator())
        tg.addWidget(self.fit_delay_offset, 4, 1)
        self.fit_as_lines = QCheckBox("as_lines")
        tg.addWidget(self.fit_as_lines, 5, 0, 1, 2)
        self.fit_show_baseline_sigma = QCheckBox("show_baseline_sigma")
        self.fit_show_baseline_sigma.setChecked(True)
        tg.addWidget(self.fit_show_baseline_sigma, 6, 0, 1, 2)
        tg.addWidget(QLabel("baseline_sigma:"), 7, 0)
        self.fit_baseline_sigma = QLineEdit("1")
        self.fit_baseline_sigma.setValidator(QDoubleValidator())
        tg.addWidget(self.fit_baseline_sigma, 7, 1)
        tg.addWidget(QLabel("baseline_alpha:"), 8, 0)
        self.fit_baseline_alpha = QLineEdit("1")
        self.fit_baseline_alpha.setValidator(QDoubleValidator())
        tg.addWidget(self.fit_baseline_alpha, 8, 1)
        tg.addWidget(QLabel("baseline_mode:"), 9, 0)
        self.fit_baseline_mode = QComboBox()
        self.fit_baseline_mode.addItems(["errorbar", "band"])
        tg.addWidget(self.fit_baseline_mode, 9, 1)
        self.fit_time_save = QCheckBox("save")
        self.fit_time_save.setChecked(True)
        tg.addWidget(self.fit_time_save, 10, 0, 1, 2)
        tg.addWidget(QLabel("save_fmt:"), 11, 0)
        self.fit_time_save_fmt = QComboBox()
        self.fit_time_save_fmt.addItems(["png", "pdf", "svg"])
        tg.addWidget(self.fit_time_save_fmt, 11, 1)
        tg.addWidget(QLabel("save_dpi:"), 12, 0)
        self.fit_time_save_dpi = QLineEdit("300")
        self.fit_time_save_dpi.setValidator(QDoubleValidator())
        tg.addWidget(self.fit_time_save_dpi, 12, 1)

        btn_row = QHBoxLayout()
        b = QPushButton("Run Delay Peak Fitting")
        b.clicked.connect(self._run_delay_peak_fitting)
        btn_row.addWidget(b)
        b = QPushButton("Plot Fit Overlay")
        b.clicked.connect(self._run_fit_overlay)
        btn_row.addWidget(b)
        b = QPushButton("Plot Time Evolution")
        b.clicked.connect(self._run_time_evolution)
        btn_row.addWidget(b)
        btn_row.addStretch()
        fsl.addLayout(btn_row)
        fsl.addStretch()

        self.fit_multi_widget = QWidget()
        fml = QVBoxLayout()
        self.fit_multi_widget.setLayout(fml)
        layout.addWidget(self.fit_multi_widget)

        note = QLabel(
            "In multiple-experiment mode, only plotting/comparison workflows are exposed here.\n"
            "Peak fitting itself remains a single-experiment action."
        )
        note.setWordWrap(True)
        fml.addWidget(note)

        self.fit_multi_editor = MultiExperimentEditor(
            "Multiple-experiment Definitions", allow_merge=True, defaults=DEFAULT_MULTI_EXPERIMENTS_FIT
        )
        fml.addWidget(self.fit_multi_editor)

        group = QGroupBox("Multiple-experiment Time Evolution")
        grid = QGridLayout()
        group.setLayout(grid)
        fml.addWidget(group)
        row = 0
        grid.addWidget(QLabel("peak:"), row, 0)
        self.fit_multi_peak = QLineEdit("110")
        grid.addWidget(self.fit_multi_peak, row, 1)
        row += 1
        grid.addWidget(QLabel("_property:"), row, 0)
        self.fit_multi_property = QComboBox()
        self.fit_multi_property.addItems(["hkl_pos", "hkl_fwhm", "amplitude", "eta"])
        grid.addWidget(self.fit_multi_property, row, 1)
        row += 1
        grid.addWidget(QLabel("out_csv_name:"), row, 0)
        self.fit_multi_out_csv_name = QLineEdit("peak_fits_delay.csv")
        grid.addWidget(self.fit_multi_out_csv_name, row, 1)
        row += 1
        grid.addWidget(QLabel("unit:"), row, 0)
        self.fit_multi_unit = QComboBox()
        self.fit_multi_unit.addItems(["ps", "fs"])
        grid.addWidget(self.fit_multi_unit, row, 1)
        row += 1
        grid.addWidget(QLabel("phi_mode override:"), row, 0)
        self.fit_multi_phi_mode = QComboBox()
        self.fit_multi_phi_mode.addItems(["auto", "phi_avg", "separate_phi"])
        grid.addWidget(self.fit_multi_phi_mode, row, 1)
        row += 1
        grid.addWidget(QLabel("phi_reduce:"), row, 0)
        self.fit_multi_phi_reduce = QComboBox()
        self.fit_multi_phi_reduce.addItems(["sum", "mean"])
        grid.addWidget(self.fit_multi_phi_reduce, row, 1)
        row += 1
        grid.addWidget(QLabel("phi_window:"), row, 0)
        self.fit_multi_phi_window = QLineEdit("Full")
        grid.addWidget(self.fit_multi_phi_window, row, 1)
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
        grid.addWidget(QLabel("save_dpi:"), row, 0)
        self.fit_multi_save_dpi = QLineEdit("300")
        self.fit_multi_save_dpi.setValidator(QDoubleValidator())
        grid.addWidget(self.fit_multi_save_dpi, row, 1)
        row += 1
        self.fit_multi_save_overwrite = QCheckBox("save_overwrite")
        self.fit_multi_save_overwrite.setChecked(True)
        grid.addWidget(self.fit_multi_save_overwrite, row, 0, 1, 2)

        btn = QPushButton("Plot Multi Time Evolution")
        btn.clicked.connect(self._run_time_evolution_multi)
        fml.addWidget(btn)
        fml.addStretch()

    # -------------------------------------------------------------------------
    # GUI state persistence
    # -------------------------------------------------------------------------

    def _autosave_path(self) -> Path:
        return Path.home() / AUTOSAVE_FILENAME

    def _combo_set_text_if_present(self, combo: QComboBox, text: str):
        idx = combo.findText(str(text))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _set_line_text(self, widget: QLineEdit, value):
        widget.setText("" if value is None else str(value))

    def _set_plain_text(self, widget: QPlainTextEdit, value):
        widget.setPlainText("" if value is None else str(value))

    def _experiment_group_state(self, prefix: str):
        widgets = self._experiment_widgets[prefix]
        out = {}
        for key in (
            "sample_name",
            "temperature_K",
            "excitation_wl_nm",
            "fluence_mJ_cm2",
            "time_window_fs",
            "dataset",
            "scan_nb",
            "scan_spec",
        ):
            if key in widgets:
                out[key] = widgets[key].text()
        return out

    def _apply_experiment_group_state(self, prefix: str, state: dict):
        widgets = self._experiment_widgets[prefix]
        for key in (
            "sample_name",
            "temperature_K",
            "excitation_wl_nm",
            "fluence_mJ_cm2",
            "time_window_fs",
            "dataset",
            "scan_nb",
            "scan_spec",
        ):
            if key in state and key in widgets:
                widgets[key].setText("" if state[key] is None else str(state[key]))

    def _collect_gui_state(self):
        state = {
            "state_version": GUI_STATE_VERSION,
            "package_name": PACKAGE_NAME,
            "window": {
                "width": self.width(),
                "height": self.height(),
                "current_tab_index": self.tabs.currentIndex(),
            },
            "session": {
                "facility": self.session_facility_combo.currentText(),
                "path_root": self.session_path_root.text(),
                "analysis_subdir": self.session_analysis_subdir.text(),
                "raw_subdir": self.session_raw_subdir.text(),
                "poni_path": self.session_poni_path.text(),
                "mask_edf_path": self.session_mask_path.text(),
                "azim_offset_deg": self.session_azim_offset_deg.text(),
            },
            "datared": {
                "experiment": self._experiment_group_state("datared"),
                "ref_delay": self.datared_ref_delay.text(),
                "delays": self.datared_delays.text(),
                "overwrite": self.datared_overwrite.isChecked(),
                "show_progress": self.datared_show_progress.isChecked(),
                "show_frame_progress": self.datared_show_frame_progress.isChecked(),
            },
            "calibration": {
                "experiment": self._experiment_group_state("calibration"),
                "azimuthal_edges": self.calib_azimuthal_edges.text(),
                "include_full": self.calib_include_full.isChecked(),
                "full_range": self.calib_full_range.text(),
                "npt": self.calib_npt.text(),
                "normalize": self.calib_normalize.isChecked(),
                "q_norm_range": self.calib_q_norm_range.text(),
                "overwrite_xy": self.calib_overwrite_xy.isChecked(),
                "q_fit_range": self.calib_q_fit_range.text(),
                "eta": self.calib_eta.text(),
                "fit_method": self.calib_fit_method.text(),
                "force_refit": self.calib_force_refit.isChecked(),
                "out_csv_name": self.calib_out_csv_name.text(),
                "caked_xlim": self.calib_caked_xlim.text(),
                "caked_ylim": self.calib_caked_ylim.text(),
                "caked_figure_title": self.calib_caked_figure_title.text(),
                "caked_save": self.calib_caked_save.isChecked(),
                "property_name": self.calib_property_name.text(),
                "property_only_success": self.calib_property_only_success.isChecked(),
                "property_ylim": self.calib_property_ylim.text(),
                "property_figure_title": self.calib_property_figure_title.text(),
                "property_save": self.calib_property_save.isChecked(),
                "figures_subdir": self.calib_figures_subdir.text(),
                "save_format": self.calib_save_format.currentText(),
                "save_dpi": self.calib_save_dpi.text(),
            },
            "pattern": {
                "experiment": self._experiment_group_state("pattern"),
                "delays_fs": self.pattern_delays.text(),
                "dark_tag": self.pattern_dark_tag.text(),
                "ref_delay": self.pattern_ref_delay.text(),
                "force": self.pattern_force_checkbox.isChecked(),
                "azimuthal_edges": self.pattern_azimuthal_edges.text(),
                "include_full": self.pattern_include_full.isChecked(),
                "full_range": self.pattern_full_range.text(),
                "npt": self.pattern_npt.text(),
                "normalize": self.pattern_normalize_checkbox.isChecked(),
                "q_norm_range": self.pattern_q_norm_range.text(),
                "overwrite_xy": self.pattern_overwrite_xy.isChecked(),
            },
            "viewer": {
                "experiment": self._experiment_group_state("viewer"),
                "delays_fs": self.viewer_delays.text(),
                "ref_type": self.viewer_ref_type.currentText(),
                "ref_value": self.viewer_ref_value.text(),
                "azim_window": self.viewer_azim_window.text(),
                "xlim": self.viewer_xlim.text(),
                "digits": self.viewer_digits.text(),
                "ref_delay": self.viewer_ref_delay.text(),
                "compute_if_missing": self.viewer_compute_if_missing.isChecked(),
                "fs_or_ps": self.viewer_fs_or_ps.currentText(),
                "overwrite_xy": self.viewer_overwrite_xy.isChecked(),
                "from_2D_imgs": self.viewer_from2d_checkbox.isChecked(),
                "save_plots": self.viewer_save_plots.isChecked(),
                "save_format": self.viewer_save_format.currentText(),
                "save_dpi": self.viewer_save_dpi.text(),
                "save_overwrite": self.viewer_save_overwrite.isChecked(),
            },
            "differential": {
                "mode": self.diff_mode_combo.currentText(),
                "single": {
                    "experiment": self._experiment_group_state("diff_single"),
                    "delays_fs": self.diff_delays.text(),
                    "ref_type": self.diff_ref_type.currentText(),
                    "ref_value": self.diff_ref_value.text(),
                    "azim_window": self.diff_azim_window.text(),
                    "peak": self.diff_peak.text(),
                    "peak_specs": self.diff_peak_specs.toPlainText(),
                    "unit": self.diff_unit.currentText(),
                    "delay_offset": self.diff_delay_offset.text(),
                    "plot_abs_and_diffs": self.diff_plot_abs_and_diffs.isChecked(),
                    "show_errorbars": self.diff_show_errorbars.isChecked(),
                    "errorbar_scale": self.diff_errorbar_scale.text(),
                    "region": self.diff_region.currentText(),
                    "kind": self.diff_kind.currentText(),
                    "time_window_select_ps": self.diff_time_window_select.text(),
                    "poly_order": self.diff_poly_order.text(),
                    "freq_unit": self.diff_freq_unit.currentText(),
                    "xlim_freq": self.diff_xlim_freq.text(),
                    "npt": self.diff_npt.text(),
                    "normalize_xy": self.diff_normalize_xy.isChecked(),
                    "q_norm_range": self.diff_q_norm_range.text(),
                    "compute_if_missing": self.diff_compute_if_missing.isChecked(),
                    "overwrite_xy": self.diff_overwrite_xy.isChecked(),
                    "save": self.diff_save.isChecked(),
                    "save_format": self.diff_save_format.currentText(),
                    "save_dpi": self.diff_save_dpi.text(),
                    "save_overwrite": self.diff_save_overwrite.isChecked(),
                },
                "multi": {
                    "experiments": self.diff_multi_editor.get_experiments(),
                    "delays_fs": self.diff_multi_delays.text(),
                    "azim_window": self.diff_multi_azim_window.text(),
                    "peak": self.diff_multi_peak.text(),
                    "peak_specs": self.diff_multi_peak_specs.toPlainText(),
                    "npt": self.diff_multi_npt.text(),
                    "normalize_xy": self.diff_multi_normalize_xy.isChecked(),
                    "q_norm_range": self.diff_multi_q_norm_range.text(),
                    "compute_if_missing": self.diff_multi_compute_if_missing.isChecked(),
                    "overwrite_xy": self.diff_multi_overwrite_xy.isChecked(),
                    "save": self.diff_multi_save.isChecked(),
                    "save_format": self.diff_multi_save_format.currentText(),
                    "save_dpi": self.diff_multi_save_dpi.text(),
                    "save_overwrite": self.diff_multi_save_overwrite.isChecked(),
                    "unit": self.diff_multi_unit.currentText(),
                    "show_errorbars": self.diff_multi_show_errorbars.isChecked(),
                    "errorbar_scale": self.diff_multi_errorbar_scale.text(),
                    "as_lines": self.diff_multi_as_lines.isChecked(),
                    "kind": self.diff_multi_kind.currentText(),
                    "time_window_select_ps": self.diff_multi_time_window_select.text(),
                    "poly_order": self.diff_multi_poly_order.text(),
                    "freq_unit": self.diff_multi_freq_unit.currentText(),
                    "xlim_freq": self.diff_multi_xlim_freq.text(),
                },
            },
            "fitting": {
                "mode": self.fit_mode_combo.currentText(),
                "single": {
                    "experiment": self._experiment_group_state("fit_single"),
                    "delays_fs": self.fit_delays.text(),
                    "ref_type": self.fit_ref_type.currentText(),
                    "ref_value": self.fit_ref_value.text(),
                    "peak_specs": self.fit_peak_specs.toPlainText(),
                    "azim_windows": self.fit_azim_windows.toPlainText(),
                    "phi_mode": self.fit_phi_mode.currentText(),
                    "phi_reduce": self.fit_phi_reduce.currentText(),
                    "default_eta": self.fit_default_eta.text(),
                    "npt": self.fit_npt.text(),
                    "q_norm_range": self.fit_q_norm_range.text(),
                    "out_csv_name": self.fit_out_csv_name.text(),
                    "normalize_xy": self.fit_normalize_xy.isChecked(),
                    "compute_if_missing": self.fit_compute_if_missing.isChecked(),
                    "overwrite_xy": self.fit_overwrite_xy.isChecked(),
                    "include_reference_in_output": self.fit_include_reference.isChecked(),
                    "show_fit_figures": self.fit_show_fit_figures.isChecked(),
                    "save_fit_figures": self.fit_save_fit_figures.isChecked(),
                    "fit_figures_format": self.fit_fig_format.currentText(),
                    "fit_figures_dpi": self.fit_fig_dpi.text(),
                    "plot_only_success": self.fit_plot_only_success.isChecked(),
                    "fit_oversample": self.fit_oversample.text(),
                    "overlay_peak": self.fit_overlay_peak.text(),
                    "overlay_delay_fs": self.fit_overlay_delay.text(),
                    "overlay_group": self.fit_overlay_group.text(),
                    "overlay_is_reference": self.fit_overlay_is_reference.isChecked(),
                    "overlay_ensure_csv": self.fit_overlay_ensure_csv.isChecked(),
                    "overlay_show": self.fit_overlay_show.isChecked(),
                    "overlay_save": self.fit_overlay_save.isChecked(),
                    "time_peak": self.fit_time_peak.text(),
                    "_property": self.fit_property.currentText(),
                    "time_unit": self.fit_time_unit.currentText(),
                    "groups": self.fit_groups.text(),
                    "delay_offset": self.fit_delay_offset.text(),
                    "as_lines": self.fit_as_lines.isChecked(),
                    "show_baseline_sigma": self.fit_show_baseline_sigma.isChecked(),
                    "baseline_sigma": self.fit_baseline_sigma.text(),
                    "baseline_alpha": self.fit_baseline_alpha.text(),
                    "baseline_mode": self.fit_baseline_mode.currentText(),
                    "time_save": self.fit_time_save.isChecked(),
                    "time_save_fmt": self.fit_time_save_fmt.currentText(),
                    "time_save_dpi": self.fit_time_save_dpi.text(),
                },
                "multi": {
                    "experiments": self.fit_multi_editor.get_experiments(),
                    "peak": self.fit_multi_peak.text(),
                    "_property": self.fit_multi_property.currentText(),
                    "out_csv_name": self.fit_multi_out_csv_name.text(),
                    "unit": self.fit_multi_unit.currentText(),
                    "phi_mode": self.fit_multi_phi_mode.currentText(),
                    "phi_reduce": self.fit_multi_phi_reduce.currentText(),
                    "phi_window": self.fit_multi_phi_window.text(),
                    "only_success": self.fit_multi_only_success.isChecked(),
                    "include_reference": self.fit_multi_include_reference.isChecked(),
                    "as_lines": self.fit_multi_as_lines.isChecked(),
                    "delay_offset": self.fit_multi_delay_offset.text(),
                    "show_baseline_sigma": self.fit_multi_show_baseline_sigma.isChecked(),
                    "baseline_sigma": self.fit_multi_baseline_sigma.text(),
                    "baseline_alpha": self.fit_multi_baseline_alpha.text(),
                    "baseline_mode": self.fit_multi_baseline_mode.currentText(),
                    "norm_min_max": self.fit_multi_norm_min_max.isChecked(),
                    "delay_for_norm_max": self.fit_multi_delay_for_norm_max.text(),
                    "cmap": self.fit_multi_cmap.text(),
                    "save": self.fit_multi_save.isChecked(),
                    "save_fmt": self.fit_multi_save_fmt.currentText(),
                    "save_dpi": self.fit_multi_save_dpi.text(),
                    "save_overwrite": self.fit_multi_save_overwrite.isChecked(),
                },
            },
        }
        return state

    def _apply_gui_state(self, state: dict):
        if not isinstance(state, dict):
            raise ValueError("GUI state must be a dictionary.")

        self._loading_gui_state = True
        self._syncing_experiment_fields = True
        try:
            window_state = state.get("window", {})
            if isinstance(window_state, dict):
                width = window_state.get("width")
                height = window_state.get("height")
                if width and height:
                    self.resize(int(width), int(height))

            session = state.get("session", {})
            if isinstance(session, dict):
                self._combo_set_text_if_present(
                    self.session_facility_combo,
                    session.get("facility", self.session_facility_combo.currentText()),
                )
                self._set_line_text(self.session_path_root, session.get("path_root", ""))
                self._set_line_text(self.session_analysis_subdir, session.get("analysis_subdir", "analysis"))
                self._set_line_text(self.session_raw_subdir, session.get("raw_subdir", ""))
                self._set_line_text(self.session_poni_path, session.get("poni_path", ""))
                self._set_line_text(self.session_mask_path, session.get("mask_edf_path", ""))
                self._set_line_text(self.session_azim_offset_deg, session.get("azim_offset_deg", "-90.0"))

            datared = state.get("datared", {})
            if isinstance(datared, dict):
                if "experiment" in datared:
                    self._apply_experiment_group_state("datared", datared["experiment"])
                self._set_line_text(self.datared_ref_delay, datared.get("ref_delay", "-5ns"))
                self._set_line_text(self.datared_delays, datared.get("delays", "all"))
                self.datared_overwrite.setChecked(bool(datared.get("overwrite", True)))
                self.datared_show_progress.setChecked(bool(datared.get("show_progress", True)))
                self.datared_show_frame_progress.setChecked(bool(datared.get("show_frame_progress", False)))

            calibration_state = state.get("calibration", {})
            if isinstance(calibration_state, dict):
                if "experiment" in calibration_state:
                    self._apply_experiment_group_state("calibration", calibration_state["experiment"])
                self._set_line_text(
                    self.calib_azimuthal_edges,
                    calibration_state.get("azimuthal_edges", pretty_literal(DEFAULT_CALIBRATION_AZIMUTHAL_EDGES)),
                )
                self.calib_include_full.setChecked(bool(calibration_state.get("include_full", True)))
                self._set_line_text(self.calib_full_range, calibration_state.get("full_range", "(-90, 90)"))
                self._set_line_text(self.calib_npt, calibration_state.get("npt", "1000"))
                self.calib_normalize.setChecked(bool(calibration_state.get("normalize", True)))
                self._set_line_text(self.calib_q_norm_range, calibration_state.get("q_norm_range", "(2.65, 2.75)"))
                self.calib_overwrite_xy.setChecked(bool(calibration_state.get("overwrite_xy", False)))
                self._set_line_text(self.calib_q_fit_range, calibration_state.get("q_fit_range", "(2.4, 2.65)"))
                self._set_line_text(self.calib_eta, calibration_state.get("eta", "0.3"))
                self._set_line_text(self.calib_fit_method, calibration_state.get("fit_method", "leastsq"))
                self.calib_force_refit.setChecked(bool(calibration_state.get("force_refit", True)))
                self._set_line_text(self.calib_out_csv_name, calibration_state.get("out_csv_name", "peak_fits.csv"))
                self._set_line_text(self.calib_caked_xlim, calibration_state.get("caked_xlim", "(2.45, 2.60)"))
                self._set_line_text(self.calib_caked_ylim, calibration_state.get("caked_ylim", ""))
                self._set_line_text(self.calib_caked_figure_title, calibration_state.get("caked_figure_title", ""))
                self.calib_caked_save.setChecked(bool(calibration_state.get("caked_save", True)))
                self._set_line_text(self.calib_property_name, calibration_state.get("property_name", "pv_center"))
                self.calib_property_only_success.setChecked(bool(calibration_state.get("property_only_success", True)))
                self._set_line_text(self.calib_property_ylim, calibration_state.get("property_ylim", ""))
                self._set_line_text(self.calib_property_figure_title, calibration_state.get("property_figure_title", ""))
                self.calib_property_save.setChecked(bool(calibration_state.get("property_save", True)))
                self._set_line_text(
                    self.calib_figures_subdir,
                    calibration_state.get("figures_subdir", DEFAULT_CALIBRATION_FIGURES_SUBDIR),
                )
                self._combo_set_text_if_present(self.calib_save_format, calibration_state.get("save_format", "png"))
                self._set_line_text(self.calib_save_dpi, calibration_state.get("save_dpi", "400"))

            pattern = state.get("pattern", {})
            if isinstance(pattern, dict):
                if "experiment" in pattern:
                    self._apply_experiment_group_state("pattern", pattern["experiment"])
                self._set_line_text(self.pattern_delays, pattern.get("delays_fs", "all"))
                self._set_line_text(self.pattern_dark_tag, pattern.get("dark_tag", ""))
                self._set_line_text(self.pattern_ref_delay, pattern.get("ref_delay", "-5ns"))
                self.pattern_force_checkbox.setChecked(bool(pattern.get("force", True)))
                self._set_line_text(
                    self.pattern_azimuthal_edges, pattern.get("azimuthal_edges", "-90, -60, -30, 0, 30, 60, 90")
                )
                self.pattern_include_full.setChecked(bool(pattern.get("include_full", True)))
                self._set_line_text(self.pattern_full_range, pattern.get("full_range", "(-90, 90)"))
                self._set_line_text(self.pattern_npt, pattern.get("npt", "1000"))
                self.pattern_normalize_checkbox.setChecked(bool(pattern.get("normalize", True)))
                self._set_line_text(self.pattern_q_norm_range, pattern.get("q_norm_range", "(2.65, 2.75)"))
                self.pattern_overwrite_xy.setChecked(bool(pattern.get("overwrite_xy", True)))

            viewer = state.get("viewer", {})
            if isinstance(viewer, dict):
                if "experiment" in viewer:
                    self._apply_experiment_group_state("viewer", viewer["experiment"])
                self._set_line_text(self.viewer_delays, viewer.get("delays_fs", "all"))
                self._combo_set_text_if_present(self.viewer_ref_type, viewer.get("ref_type", "dark"))
                self._set_line_text(self.viewer_ref_value, viewer.get("ref_value", "[1466556]"))
                self._set_line_text(self.viewer_azim_window, viewer.get("azim_window", "(-90, 90)"))
                self._set_line_text(self.viewer_xlim, viewer.get("xlim", "(1.5, 4.5)"))
                self._set_line_text(self.viewer_digits, viewer.get("digits", "2"))
                self._set_line_text(self.viewer_ref_delay, viewer.get("ref_delay", "-5ns"))
                self.viewer_compute_if_missing.setChecked(bool(viewer.get("compute_if_missing", True)))
                self._combo_set_text_if_present(self.viewer_fs_or_ps, viewer.get("fs_or_ps", "ps"))
                self.viewer_overwrite_xy.setChecked(bool(viewer.get("overwrite_xy", False)))
                self.viewer_from2d_checkbox.setChecked(bool(viewer.get("from_2D_imgs", False)))
                self.viewer_save_plots.setChecked(bool(viewer.get("save_plots", True)))
                self._combo_set_text_if_present(self.viewer_save_format, viewer.get("save_format", "png"))
                self._set_line_text(self.viewer_save_dpi, viewer.get("save_dpi", "400"))
                self.viewer_save_overwrite.setChecked(bool(viewer.get("save_overwrite", True)))

            differential = state.get("differential", {})
            if isinstance(differential, dict):
                self._combo_set_text_if_present(self.diff_mode_combo, differential.get("mode", "Single experiment"))
                diff_single = differential.get("single", {})
                if isinstance(diff_single, dict):
                    if "experiment" in diff_single:
                        self._apply_experiment_group_state("diff_single", diff_single["experiment"])
                    self._set_line_text(self.diff_delays, diff_single.get("delays_fs", "all"))
                    self._combo_set_text_if_present(self.diff_ref_type, diff_single.get("ref_type", "dark"))
                    self._set_line_text(self.diff_ref_value, diff_single.get("ref_value", "[1466556]"))
                    self._set_line_text(self.diff_azim_window, diff_single.get("azim_window", "(-90, 90)"))
                    self._set_line_text(self.diff_peak, diff_single.get("peak", "110"))
                    self._set_plain_text(self.diff_peak_specs, diff_single.get("peak_specs", pretty_literal(DEFAULT_DIFF_PEAK_SPECS)))
                    self._combo_set_text_if_present(self.diff_unit, diff_single.get("unit", "ps"))
                    self._set_line_text(self.diff_delay_offset, diff_single.get("delay_offset", "0"))
                    self.diff_plot_abs_and_diffs.setChecked(bool(diff_single.get("plot_abs_and_diffs", True)))
                    self.diff_show_errorbars.setChecked(bool(diff_single.get("show_errorbars", True)))
                    self._set_line_text(self.diff_errorbar_scale, diff_single.get("errorbar_scale", "1.0"))
                    self._combo_set_text_if_present(self.diff_region, diff_single.get("region", "peak"))
                    self._combo_set_text_if_present(self.diff_kind, diff_single.get("kind", "diff"))
                    self._set_line_text(self.diff_time_window_select, diff_single.get("time_window_select_ps", "(-1, 200)"))
                    self._set_line_text(self.diff_poly_order, diff_single.get("poly_order", "2"))
                    self._combo_set_text_if_present(self.diff_freq_unit, diff_single.get("freq_unit", "cm^-1"))
                    self._set_line_text(self.diff_xlim_freq, diff_single.get("xlim_freq", "(-50, 850)"))
                    self._set_line_text(self.diff_npt, diff_single.get("npt", "1000"))
                    self.diff_normalize_xy.setChecked(bool(diff_single.get("normalize_xy", True)))
                    self._set_line_text(self.diff_q_norm_range, diff_single.get("q_norm_range", "(2.65, 2.75)"))
                    self.diff_compute_if_missing.setChecked(bool(diff_single.get("compute_if_missing", True)))
                    self.diff_overwrite_xy.setChecked(bool(diff_single.get("overwrite_xy", False)))
                    self.diff_save.setChecked(bool(diff_single.get("save", True)))
                    self._combo_set_text_if_present(self.diff_save_format, diff_single.get("save_format", "png"))
                    self._set_line_text(self.diff_save_dpi, diff_single.get("save_dpi", "400"))
                    self.diff_save_overwrite.setChecked(bool(diff_single.get("save_overwrite", True)))

                diff_multi = differential.get("multi", {})
                if isinstance(diff_multi, dict):
                    experiments = diff_multi.get("experiments")
                    if experiments is not None:
                        self.diff_multi_editor.set_experiments(experiments)
                    self._set_line_text(self.diff_multi_delays, diff_multi.get("delays_fs", "all"))
                    self._set_line_text(self.diff_multi_azim_window, diff_multi.get("azim_window", "(-90, 90)"))
                    self._set_line_text(self.diff_multi_peak, diff_multi.get("peak", "110"))
                    self._set_plain_text(self.diff_multi_peak_specs, diff_multi.get("peak_specs", pretty_literal(DEFAULT_DIFF_PEAK_SPECS)))
                    self._set_line_text(self.diff_multi_npt, diff_multi.get("npt", "1000"))
                    self.diff_multi_normalize_xy.setChecked(bool(diff_multi.get("normalize_xy", True)))
                    self._set_line_text(self.diff_multi_q_norm_range, diff_multi.get("q_norm_range", "(2.65, 2.75)"))
                    self.diff_multi_compute_if_missing.setChecked(bool(diff_multi.get("compute_if_missing", True)))
                    self.diff_multi_overwrite_xy.setChecked(bool(diff_multi.get("overwrite_xy", False)))
                    self.diff_multi_save.setChecked(bool(diff_multi.get("save", True)))
                    self._combo_set_text_if_present(self.diff_multi_save_format, diff_multi.get("save_format", "png"))
                    self._set_line_text(self.diff_multi_save_dpi, diff_multi.get("save_dpi", "400"))
                    self.diff_multi_save_overwrite.setChecked(bool(diff_multi.get("save_overwrite", True)))
                    self._combo_set_text_if_present(self.diff_multi_unit, diff_multi.get("unit", "ps"))
                    self.diff_multi_show_errorbars.setChecked(bool(diff_multi.get("show_errorbars", True)))
                    self._set_line_text(self.diff_multi_errorbar_scale, diff_multi.get("errorbar_scale", "1.0"))
                    self.diff_multi_as_lines.setChecked(bool(diff_multi.get("as_lines", False)))
                    self._combo_set_text_if_present(self.diff_multi_kind, diff_multi.get("kind", "diff"))
                    self._set_line_text(self.diff_multi_time_window_select, diff_multi.get("time_window_select_ps", "(-1, 200)"))
                    self._set_line_text(self.diff_multi_poly_order, diff_multi.get("poly_order", "2"))
                    self._combo_set_text_if_present(self.diff_multi_freq_unit, diff_multi.get("freq_unit", "cm^-1"))
                    self._set_line_text(self.diff_multi_xlim_freq, diff_multi.get("xlim_freq", "(-50, 850)"))

            fitting_state = state.get("fitting", {})
            if isinstance(fitting_state, dict):
                self._combo_set_text_if_present(self.fit_mode_combo, fitting_state.get("mode", "Single experiment"))
                fit_single = fitting_state.get("single", {})
                if isinstance(fit_single, dict):
                    if "experiment" in fit_single:
                        self._apply_experiment_group_state("fit_single", fit_single["experiment"])
                    self._set_line_text(self.fit_delays, fit_single.get("delays_fs", "all"))
                    self._combo_set_text_if_present(self.fit_ref_type, fit_single.get("ref_type", "dark"))
                    self._set_line_text(self.fit_ref_value, fit_single.get("ref_value", "[1466556]"))
                    self._set_plain_text(self.fit_peak_specs, fit_single.get("peak_specs", pretty_literal(DEFAULT_FIT_PEAK_SPECS)))
                    self._set_plain_text(self.fit_azim_windows, fit_single.get("azim_windows", pretty_literal(DEFAULT_AZIM_WINDOWS)))
                    self._combo_set_text_if_present(self.fit_phi_mode, fit_single.get("phi_mode", "phi_avg"))
                    self._combo_set_text_if_present(self.fit_phi_reduce, fit_single.get("phi_reduce", "sum"))
                    self._set_line_text(self.fit_default_eta, fit_single.get("default_eta", "0.3"))
                    self._set_line_text(self.fit_npt, fit_single.get("npt", "1000"))
                    self._set_line_text(self.fit_q_norm_range, fit_single.get("q_norm_range", "(2.65, 2.75)"))
                    self._set_line_text(self.fit_out_csv_name, fit_single.get("out_csv_name", "peak_fits_delay.csv"))
                    self.fit_normalize_xy.setChecked(bool(fit_single.get("normalize_xy", True)))
                    self.fit_compute_if_missing.setChecked(bool(fit_single.get("compute_if_missing", True)))
                    self.fit_overwrite_xy.setChecked(bool(fit_single.get("overwrite_xy", False)))
                    self.fit_include_reference.setChecked(bool(fit_single.get("include_reference_in_output", True)))
                    self.fit_show_fit_figures.setChecked(bool(fit_single.get("show_fit_figures", False)))
                    self.fit_save_fit_figures.setChecked(bool(fit_single.get("save_fit_figures", False)))
                    self._combo_set_text_if_present(self.fit_fig_format, fit_single.get("fit_figures_format", "png"))
                    self._set_line_text(self.fit_fig_dpi, fit_single.get("fit_figures_dpi", "300"))
                    self.fit_plot_only_success.setChecked(bool(fit_single.get("plot_only_success", True)))
                    self._set_line_text(self.fit_oversample, fit_single.get("fit_oversample", "10"))

                    self._set_line_text(self.fit_overlay_peak, fit_single.get("overlay_peak", "110"))
                    self._set_line_text(self.fit_overlay_delay, fit_single.get("overlay_delay_fs", "0"))
                    self._set_line_text(self.fit_overlay_group, fit_single.get("overlay_group", "Full"))
                    self.fit_overlay_is_reference.setChecked(bool(fit_single.get("overlay_is_reference", False)))
                    self.fit_overlay_ensure_csv.setChecked(bool(fit_single.get("overlay_ensure_csv", True)))
                    self.fit_overlay_show.setChecked(bool(fit_single.get("overlay_show", True)))
                    self.fit_overlay_save.setChecked(bool(fit_single.get("overlay_save", True)))

                    self._set_line_text(self.fit_time_peak, fit_single.get("time_peak", "110"))
                    self._combo_set_text_if_present(self.fit_property, fit_single.get("_property", "hkl_pos"))
                    self._combo_set_text_if_present(self.fit_time_unit, fit_single.get("time_unit", "ps"))
                    self._set_line_text(self.fit_groups, fit_single.get("groups", "['Full', 60, 30, 0]"))
                    self._set_line_text(self.fit_delay_offset, fit_single.get("delay_offset", "0"))
                    self.fit_as_lines.setChecked(bool(fit_single.get("as_lines", False)))
                    self.fit_show_baseline_sigma.setChecked(bool(fit_single.get("show_baseline_sigma", True)))
                    self._set_line_text(self.fit_baseline_sigma, fit_single.get("baseline_sigma", "1"))
                    self._set_line_text(self.fit_baseline_alpha, fit_single.get("baseline_alpha", "1"))
                    self._combo_set_text_if_present(self.fit_baseline_mode, fit_single.get("baseline_mode", "errorbar"))
                    self.fit_time_save.setChecked(bool(fit_single.get("time_save", True)))
                    self._combo_set_text_if_present(self.fit_time_save_fmt, fit_single.get("time_save_fmt", "png"))
                    self._set_line_text(self.fit_time_save_dpi, fit_single.get("time_save_dpi", "300"))

                fit_multi = fitting_state.get("multi", {})
                if isinstance(fit_multi, dict):
                    experiments = fit_multi.get("experiments")
                    if experiments is not None:
                        self.fit_multi_editor.set_experiments(experiments)
                    self._set_line_text(self.fit_multi_peak, fit_multi.get("peak", "110"))
                    self._combo_set_text_if_present(self.fit_multi_property, fit_multi.get("_property", "hkl_pos"))
                    self._set_line_text(self.fit_multi_out_csv_name, fit_multi.get("out_csv_name", "peak_fits_delay.csv"))
                    self._combo_set_text_if_present(self.fit_multi_unit, fit_multi.get("unit", "ps"))
                    self._combo_set_text_if_present(self.fit_multi_phi_mode, fit_multi.get("phi_mode", "auto"))
                    self._combo_set_text_if_present(self.fit_multi_phi_reduce, fit_multi.get("phi_reduce", "sum"))
                    self._set_line_text(self.fit_multi_phi_window, fit_multi.get("phi_window", "Full"))
                    self.fit_multi_only_success.setChecked(bool(fit_multi.get("only_success", True)))
                    self.fit_multi_include_reference.setChecked(bool(fit_multi.get("include_reference", True)))
                    self.fit_multi_as_lines.setChecked(bool(fit_multi.get("as_lines", False)))
                    self._set_line_text(self.fit_multi_delay_offset, fit_multi.get("delay_offset", ""))
                    self.fit_multi_show_baseline_sigma.setChecked(bool(fit_multi.get("show_baseline_sigma", True)))
                    self._set_line_text(self.fit_multi_baseline_sigma, fit_multi.get("baseline_sigma", "1"))
                    self._set_line_text(self.fit_multi_baseline_alpha, fit_multi.get("baseline_alpha", "0.18"))
                    self._combo_set_text_if_present(self.fit_multi_baseline_mode, fit_multi.get("baseline_mode", "errorbar"))
                    self.fit_multi_norm_min_max.setChecked(bool(fit_multi.get("norm_min_max", False)))
                    self._set_line_text(self.fit_multi_delay_for_norm_max, fit_multi.get("delay_for_norm_max", ""))
                    self._set_line_text(self.fit_multi_cmap, fit_multi.get("cmap", "jet"))
                    self.fit_multi_save.setChecked(bool(fit_multi.get("save", True)))
                    self._combo_set_text_if_present(self.fit_multi_save_fmt, fit_multi.get("save_fmt", "png"))
                    self._set_line_text(self.fit_multi_save_dpi, fit_multi.get("save_dpi", "300"))
                    self.fit_multi_save_overwrite.setChecked(bool(fit_multi.get("save_overwrite", True)))

            current_tab_index = state.get("window", {}).get("current_tab_index")
            if current_tab_index is not None:
                try:
                    self.tabs.setCurrentIndex(int(current_tab_index))
                except Exception:
                    pass

        finally:
            self._syncing_experiment_fields = False
            self._loading_gui_state = False

        self._refresh_facility_dependent_widgets()
        self._refresh_diff_mode_widgets()
        self._refresh_fit_mode_widgets()
        self.diff_multi_editor.update_preview()
        self.fit_multi_editor.update_preview()

    def _save_state_dict_to_path(self, state: dict, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _load_state_dict_from_path(self, path: Path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_gui_state_to_file(self):
        try:
            state = self._collect_gui_state()
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save GUI state",
                str(Path.home() / "xrdpy_gui_state.json"),
                "JSON Files (*.json);;All Files (*)",
            )
            if not file_name:
                return
            self._save_state_dict_to_path(state, Path(file_name))
            self._log(f"GUI state saved to: {file_name}")
        except Exception as exc:
            self._show_exception("Save GUI State Error", exc)

    def _load_gui_state_from_file(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load GUI state",
                str(Path.home()),
                "JSON Files (*.json);;All Files (*)",
            )
            if not file_name:
                return
            state = self._load_state_dict_from_path(Path(file_name))
            self._apply_gui_state(state)
            self._log(f"GUI state loaded from: {file_name}")
        except Exception as exc:
            self._show_exception("Load GUI State Error", exc)

    def _save_autosave_to_disk(self):
        state = self._collect_gui_state()
        path = self._autosave_path()
        self._save_state_dict_to_path(state, path)
        return path

    def _load_autosave_from_disk(self):
        try:
            path = self._autosave_path()
            if not path.exists():
                QMessageBox.information(self, "No Autosave", f"No autosave file found at:\n{path}")
                return
            state = self._load_state_dict_from_path(path)
            self._apply_gui_state(state)
            self._log(f"Autosaved GUI state restored from: {path}")
        except Exception as exc:
            self._show_exception("Load Autosave Error", exc)

    def _maybe_restore_autosave(self):
        try:
            path = self._autosave_path()
            if not path.exists():
                return
            answer = QMessageBox.question(
                self,
                "Restore previous GUI state?",
                f"A previous autosaved GUI state was found:\n{path}\n\nRestore it now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                state = self._load_state_dict_from_path(path)
                self._apply_gui_state(state)
                self._log(f"Restored autosaved GUI state from: {path}")
        except Exception as exc:
            self._log(f"Autosave restore failed: {exc}")

    def closeEvent(self, event):
        try:
            path = self._save_autosave_to_disk()
            self._log(f"Autosaved GUI state to: {path}")
        except Exception as exc:
            self._log(f"Autosave failed during close: {exc}")
        super().closeEvent(event)

    # -------------------------------------------------------------------------
    # 2D preparation actions
    # -------------------------------------------------------------------------

    def _run_id09_create_dark(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None or id09_datared is None:
                raise ImportError("Backend package is not available in this environment.")
            if self._facility() != "ID09":
                raise ValueError("This 2D-preparation backend is currently implemented only for ID09.")

            delay_ref_text = self.datared_ref_delay.text().strip()
            if not delay_ref_text:
                raise ValueError("dark ref_delay cannot be empty.")

            kwargs = self._datared_id09_dark_kwargs()
            kwargs.update(
                delay_ref=parse_python_literal(delay_ref_text),
                overwrite=self.datared_overwrite.isChecked(),
                show_progress=self.datared_show_progress.isChecked(),
            )
            out_path = id09_datared.create_dark_from_ref_delay(**kwargs)
            self._log(f"ID09 dark 2D image created: {out_path}")
        except Exception as exc:
            self._show_exception("ID09 Create Dark 2D Error", exc)

    def _run_id09_create_delay_images(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None or id09_datared is None:
                raise ImportError("Backend package is not available in this environment.")
            if self._facility() != "ID09":
                raise ValueError("This 2D-preparation backend is currently implemented only for ID09.")

            kwargs = self._experiment_kwargs("datared")
            kwargs.update(self._id09_kwargs_from_prefix("datared"))
            kwargs.update(
                delays=self._delays_value_from_line(self.datared_delays),
                overwrite=self.datared_overwrite.isChecked(),
                show_progress=self.datared_show_progress.isChecked(),
                show_frame_progress=self.datared_show_frame_progress.isChecked(),
                paths=self._paths(),
            )
            out_paths = id09_datared.create_final_2D_images(**kwargs)
            n_saved = len(out_paths) if hasattr(out_paths, "__len__") else "?"
            self._log(f"ID09 delay 2D image creation finished. Saved {n_saved} files.")
        except Exception as exc:
            self._show_exception("ID09 Create Delay 2D Images Error", exc)

    # -------------------------------------------------------------------------
    # Calibration actions
    # -------------------------------------------------------------------------

    def _run_calibration_compute_xy(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None or calibration is None:
                raise ImportError("Backend package is not available in this environment.")

            kwargs = self._calibration_context_kwargs()
            kwargs.update(self._poni_mask_kwargs())
            kwargs.update(
                azimuthal_ranges=parse_edges(self.calib_azimuthal_edges.text()),
                include_full=self.calib_include_full.isChecked(),
                full_range=parse_tuple2(self.calib_full_range.text(), name="full_range", cast=float),
                npt=parse_int_like(self.calib_npt.text(), name="npt"),
                normalize=self.calib_normalize.isChecked(),
                q_norm_range=parse_tuple2(self.calib_q_norm_range.text(), name="q_norm_range", cast=float),
                overwrite_xy=self.calib_overwrite_xy.isChecked(),
                azim_offset_deg=self._azim_offset_deg(),
            )
            calibration.compute_xy_files(**kwargs)
            self._log("Calibration XY computation finished.")
        except Exception as exc:
            self._show_exception("Calibration Compute XY Error", exc)

    def _run_calibration_peak_fitting(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None or calibration is None:
                raise ImportError("Backend package is not available in this environment.")

            kwargs = self._calibration_context_kwargs()
            kwargs.update(self._poni_mask_kwargs())
            kwargs.update(
                q_fit_range=parse_tuple2(self.calib_q_fit_range.text(), name="q_fit_range", cast=float),
                azimuthal_ranges=parse_edges(self.calib_azimuthal_edges.text()),
                include_full=self.calib_include_full.isChecked(),
                full_range=parse_tuple2(self.calib_full_range.text(), name="full_range", cast=float),
                npt=parse_int_like(self.calib_npt.text(), name="npt"),
                normalize=self.calib_normalize.isChecked(),
                q_norm_range=parse_tuple2(self.calib_q_norm_range.text(), name="q_norm_range", cast=float),
                eta=parse_float_like(self.calib_eta.text(), name="eta"),
                fit_method=self.calib_fit_method.text().strip() or "leastsq",
                force_refit=self.calib_force_refit.isChecked(),
                out_csv_name=self.calib_out_csv_name.text().strip() or "peak_fits.csv",
                overwrite_xy=self.calib_overwrite_xy.isChecked(),
                azim_offset_deg=self._azim_offset_deg(),
            )
            _df = calibration.do_peak_fitting(**kwargs)
            self._log("Calibration peak fitting finished.")
        except Exception as exc:
            self._show_exception("Calibration Peak Fitting Error", exc)

    def _run_calibration_plot_caked(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None or calibration is None:
                raise ImportError("Backend package is not available in this environment.")

            kwargs = self._calibration_context_kwargs()
            kwargs.update(self._poni_mask_kwargs())
            kwargs.update(
                azimuthal_ranges=parse_edges(self.calib_azimuthal_edges.text()),
                include_full=self.calib_include_full.isChecked(),
                full_range=parse_tuple2(self.calib_full_range.text(), name="full_range", cast=float),
                npt=parse_int_like(self.calib_npt.text(), name="npt"),
                normalize=self.calib_normalize.isChecked(),
                q_norm_range=parse_tuple2(self.calib_q_norm_range.text(), name="q_norm_range", cast=float),
                overwrite_xy=self.calib_overwrite_xy.isChecked(),
                azim_offset_deg=self._azim_offset_deg(),
                xlim=parse_optional_tuple2(self.calib_caked_xlim.text(), name="xlim", cast=float),
                ylim=parse_optional_tuple2(self.calib_caked_ylim.text(), name="ylim", cast=float),
                figure_title=self.calib_caked_figure_title.text().strip() or None,
                save=self.calib_caked_save.isChecked(),
                figures_subdir=self.calib_figures_subdir.text().strip() or DEFAULT_CALIBRATION_FIGURES_SUBDIR,
                save_format=self.calib_save_format.currentText(),
                save_dpi=parse_int_like(self.calib_save_dpi.text(), name="save_dpi"),
            )
            calibration.plot_caked_1D_patterns(**kwargs)
            self._log("Calibration caked 1D pattern plot finished.")
        except Exception as exc:
            self._show_exception("Calibration Caked Pattern Plot Error", exc)

    def _run_calibration_plot_property(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None or calibration is None:
                raise ImportError("Backend package is not available in this environment.")

            kwargs = self._calibration_context_kwargs()
            kwargs.update(
                _property=self.calib_property_name.text().strip() or "pv_center",
                figure_title=self.calib_property_figure_title.text().strip() or None,
                only_success=self.calib_property_only_success.isChecked(),
                out_csv_name=self.calib_out_csv_name.text().strip() or "peak_fits.csv",
                ylim=parse_optional_tuple2(self.calib_property_ylim.text(), name="ylim", cast=float),
                save=self.calib_property_save.isChecked(),
                figures_subdir=self.calib_figures_subdir.text().strip() or DEFAULT_CALIBRATION_FIGURES_SUBDIR,
                save_format=self.calib_save_format.currentText(),
                save_dpi=parse_int_like(self.calib_save_dpi.text(), name="save_dpi"),
            )
            calibration.plot_property_vs_azimuth(**kwargs)
            self._log("Calibration property-vs-azimuth plot finished.")
        except Exception as exc:
            self._show_exception("Calibration Property Plot Error", exc)

    # -------------------------------------------------------------------------
    # Existing run actions
    # -------------------------------------------------------------------------

    def _run_integrate_dark(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None:
                raise ImportError("Backend package is not available in this environment.")
            if self._facility() == "ID09":
                raise ValueError("Dark integration is not exposed here for ID09.")
            module = self._current_azimint_module()
            kwargs = self._experiment_kwargs("pattern")
            kwargs.update(self._poni_mask_kwargs())
            kwargs.update(
                dark_tag=parse_int_like(self.pattern_dark_tag.text(), name="dark_tag"),
                azimuthal_edges=parse_edges(self.pattern_azimuthal_edges.text()),
                include_full=self.pattern_include_full.isChecked(),
                overwrite_xy=self.pattern_overwrite_xy.isChecked(),
                paths=self._paths(),
            )
            module.integrate_dark_1d(**kwargs)
            self._log("Dark 1D integration finished.")
        except Exception as exc:
            self._show_exception("Integrate Dark 1D Error", exc)

    def _run_integrate_delay(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None:
                raise ImportError("Backend package is not available in this environment.")
            facility = self._facility()
            module = self._current_azimint_module()
            kwargs = self._experiment_kwargs("pattern")
            kwargs.update(self._poni_mask_kwargs())
            kwargs.update(
                delays_fs=self._delays_value_from_line(self.pattern_delays),
                azimuthal_edges=parse_edges(self.pattern_azimuthal_edges.text()),
                include_full=self.pattern_include_full.isChecked(),
                full_range=parse_tuple2(self.pattern_full_range.text(), name="full_range", cast=float),
                npt=parse_int_like(self.pattern_npt.text(), name="npt"),
                q_norm_range=parse_tuple2(self.pattern_q_norm_range.text(), name="q_norm_range", cast=float),
                overwrite_xy=self.pattern_overwrite_xy.isChecked(),
                paths=self._paths(),
            )
            if facility == "ID09":
                kwargs.update(self._id09_kwargs_from_prefix("pattern"))
                kwargs.update(
                    force=self.pattern_force_checkbox.isChecked(),
                    ref_delay=self.pattern_ref_delay.text().strip() or None,
                    azim_offset_deg=self._azim_offset_deg(),
                )
            else:
                kwargs.update(normalize=self.pattern_normalize_checkbox.isChecked())
            module.integrate_delay_1d(**kwargs)
            self._log("Delay 1D integration finished.")
        except Exception as exc:
            self._show_exception("Integrate Delay 1D Error", exc)

    def _run_plot_1d(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None:
                raise ImportError("Backend package is not available in this environment.")
            facility = self._facility()
            module = self._current_azimint_module()
            kwargs = self._experiment_kwargs("viewer")
            kwargs.update(self._poni_mask_kwargs())
            kwargs.update(
                delays_fs=self._delays_value_from_line(self.viewer_delays),
                ref_type=self.viewer_ref_type.currentText(),
                ref_value=self._ref_value_from_line(self.viewer_ref_value),
                azim_window=parse_tuple2(self.viewer_azim_window.text(), name="azim_window", cast=float),
                xlim=parse_tuple2(self.viewer_xlim.text(), name="xlim", cast=float),
                digits=parse_int_like(self.viewer_digits.text(), name="digits"),
                save_plots=self.viewer_save_plots.isChecked(),
                save_format=self.viewer_save_format.currentText(),
                save_dpi=parse_int_like(self.viewer_save_dpi.text(), name="save_dpi"),
                save_overwrite=self.viewer_save_overwrite.isChecked(),
                paths=self._paths(),
            )
            if facility == "ID09":
                kwargs.update(self._id09_kwargs_from_prefix("viewer"))
                kwargs.update(
                    npt=parse_int_like(self.pattern_npt.text(), name="npt"),
                    ref_delay=self.viewer_ref_delay.text().strip() or None,
                    q_norm_range=parse_tuple2(self.pattern_q_norm_range.text(), name="q_norm_range", cast=float),
                    compute_if_missing=self.viewer_compute_if_missing.isChecked(),
                    overwrite_xy=self.viewer_overwrite_xy.isChecked(),
                    ylim_top=None,
                    ylim_diff=None,
                    vlines_peak=None,
                    vlines_bckg=None,
                    fs_or_ps=self.viewer_fs_or_ps.currentText(),
                    title=None,
                    azim_offset_deg=self._azim_offset_deg(),
                )
            else:
                kwargs.update(from_2D_imgs=self.viewer_from2d_checkbox.isChecked())
            module.plot_1D_abs_and_diffs_delay(**kwargs)
            self._log("1D comparison plot finished.")
        except Exception as exc:
            self._show_exception("1D Viewer Error", exc)

    def _base_diff_kwargs(self):
        kwargs = self._experiment_kwargs("diff_single")
        kwargs.update(self._poni_mask_kwargs())
        kwargs.update(
            delays_fs=self._delays_value_from_line(self.diff_delays),
            ref_type=self.diff_ref_type.currentText(),
            ref_value=self._ref_value_from_line(self.diff_ref_value),
            azim_window=parse_tuple2(self.diff_azim_window.text(), name="azim_window", cast=float),
            azim_offset_deg=self._azim_offset_deg(),
            peak=self.diff_peak.text().strip(),
            peak_specs=self._diff_peak_specs(),
            npt=parse_int_like(self.diff_npt.text(), name="npt"),
            normalize_xy=self.diff_normalize_xy.isChecked(),
            q_norm_range=parse_tuple2(self.diff_q_norm_range.text(), name="q_norm_range", cast=float),
            compute_if_missing=self.diff_compute_if_missing.isChecked(),
            overwrite_xy=self.diff_overwrite_xy.isChecked(),
            save=self.diff_save.isChecked(),
            save_format=self.diff_save_format.currentText(),
            save_dpi=parse_int_like(self.diff_save_dpi.text(), name="save_dpi"),
            save_overwrite=self.diff_save_overwrite.isChecked(),
            paths=self._paths(),
        )
        return kwargs

    def _run_diff_integrals(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None:
                raise ImportError("Backend package is not available in this environment.")
            kwargs = self._base_diff_kwargs()
            kwargs.update(
                unit=self.diff_unit.currentText(),
                delay_offset=parse_float_like(self.diff_delay_offset.text(), name="delay_offset"),
                plot_abs_and_diffs=self.diff_plot_abs_and_diffs.isChecked(),
                show_errorbars=self.diff_show_errorbars.isChecked(),
                errorbar_scale=parse_float_like(self.diff_errorbar_scale.text(), name="errorbar_scale"),
            )
            differential_analysis.plot_differential_integrals(**kwargs)
            self._log("Differential integral plot finished.")
        except Exception as exc:
            self._show_exception("Differential Integrals Error", exc)

    def _run_diff_fft(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None:
                raise ImportError("Backend package is not available in this environment.")
            kwargs = self._base_diff_kwargs()
            kwargs.update(
                delay_offset=parse_float_like(self.diff_delay_offset.text(), name="delay_offset"),
                region=self.diff_region.currentText(),
                kind=self.diff_kind.currentText(),
                time_window_select_ps=parse_tuple2(
                    self.diff_time_window_select.text(), name="time_window_select_ps", cast=float
                ),
                poly_order=parse_int_like(self.diff_poly_order.text(), name="poly_order"),
                freq_unit=self.diff_freq_unit.currentText(),
                xlim_freq=parse_tuple2(self.diff_xlim_freq.text(), name="xlim_freq", cast=float),
            )
            differential_analysis.plot_differential_fft(**kwargs)
            self._log("Differential FFT plot finished.")
        except Exception as exc:
            self._show_exception("Differential FFT Error", exc)

    def _base_diff_multi_kwargs(self):
        peak_specs = parse_python_literal(self.diff_multi_peak_specs.toPlainText())
        if not isinstance(peak_specs, dict) or not peak_specs:
            raise ValueError("Multiple-experiment differential peak_specs must be a non-empty dict.")
        return dict(
            experiments=self._validated_diff_multi_experiments(),
            delays_fs=self._delays_value_from_line(self.diff_multi_delays),
            azim_window=parse_tuple2(self.diff_multi_azim_window.text(), name="azim_window", cast=float),
            peak=self.diff_multi_peak.text().strip(),
            peak_specs=peak_specs,
            npt=parse_int_like(self.diff_multi_npt.text(), name="npt"),
            normalize_xy=self.diff_multi_normalize_xy.isChecked(),
            q_norm_range=parse_tuple2(self.diff_multi_q_norm_range.text(), name="q_norm_range", cast=float),
            azim_offset_deg=self._azim_offset_deg(),
            compute_if_missing=self.diff_multi_compute_if_missing.isChecked(),
            overwrite_xy=self.diff_multi_overwrite_xy.isChecked(),
            save=self.diff_multi_save.isChecked(),
            save_format=self.diff_multi_save_format.currentText(),
            save_dpi=parse_int_like(self.diff_multi_save_dpi.text(), name="save_dpi"),
            save_overwrite=self.diff_multi_save_overwrite.isChecked(),
            poni_path=self.session_poni_path.text().strip() or None,
            mask_edf_path=self.session_mask_path.text().strip() or None,
            paths=self._paths(),
        )

    def _run_diff_integrals_multi(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None:
                raise ImportError("Backend package is not available in this environment.")
            kwargs = self._base_diff_multi_kwargs()
            kwargs.update(
                unit=self.diff_multi_unit.currentText(),
                show_errorbars=self.diff_multi_show_errorbars.isChecked(),
                errorbar_scale=parse_float_like(self.diff_multi_errorbar_scale.text(), name="errorbar_scale"),
                as_lines=self.diff_multi_as_lines.isChecked(),
            )
            differential_analysis.plot_differential_integrals_multi(**kwargs)
            self._log("Multiple-experiment differential integrals finished.")
        except Exception as exc:
            self._show_exception("Multi Differential Integrals Error", exc)

    def _run_diff_fft_multi(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None:
                raise ImportError("Backend package is not available in this environment.")
            kwargs = self._base_diff_multi_kwargs()
            kwargs.update(
                kind=self.diff_multi_kind.currentText(),
                time_window_select_ps=parse_tuple2(
                    self.diff_multi_time_window_select.text(), name="time_window_select_ps", cast=float
                ),
                poly_order=parse_int_like(self.diff_multi_poly_order.text(), name="poly_order"),
                freq_unit=self.diff_multi_freq_unit.currentText(),
                xlim_freq=parse_tuple2(self.diff_multi_xlim_freq.text(), name="xlim_freq", cast=float),
            )
            differential_analysis.plot_differential_fft_multi(**kwargs)
            self._log("Multiple-experiment differential FFT finished.")
        except Exception as exc:
            self._show_exception("Multi Differential FFT Error", exc)

    def _fit_base_kwargs(self):
        kwargs = self._experiment_kwargs("fit_single")
        kwargs.update(self._poni_mask_kwargs())
        kwargs.update(
            peak_specs=self._fit_peak_specs(),
            azim_windows=parse_windows(self.fit_azim_windows.toPlainText()),
            azim_offset_deg=self._azim_offset_deg(),
            npt=parse_int_like(self.fit_npt.text(), name="npt"),
            normalize_xy=self.fit_normalize_xy.isChecked(),
            q_norm_range=parse_tuple2(self.fit_q_norm_range.text(), name="q_norm_range", cast=float),
            compute_if_missing=self.fit_compute_if_missing.isChecked(),
            overwrite_xy=self.fit_overwrite_xy.isChecked(),
            default_eta=parse_float_like(self.fit_default_eta.text(), name="default_eta"),
            phi_mode=self.fit_phi_mode.currentText(),
            phi_reduce=self.fit_phi_reduce.currentText(),
            paths=self._paths(),
        )
        return kwargs

    def _run_delay_peak_fitting(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None:
                raise ImportError("Backend package is not available in this environment.")
            kwargs = self._fit_base_kwargs()
            kwargs.update(
                delays_fs=self._delays_value_from_line(self.fit_delays),
                ref_type=self.fit_ref_type.currentText(),
                ref_value=self._ref_value_from_line(self.fit_ref_value),
                include_reference_in_output=self.fit_include_reference.isChecked(),
                out_csv_name=self.fit_out_csv_name.text().strip() or "peak_fits_delay.csv",
                show_fit_figures=self.fit_show_fit_figures.isChecked(),
                save_fit_figures=self.fit_save_fit_figures.isChecked(),
                fit_figures_format=self.fit_fig_format.currentText(),
                fit_figures_dpi=parse_int_like(self.fit_fig_dpi.text(), name="fit_figures_dpi"),
                plot_only_success=self.fit_plot_only_success.isChecked(),
                fit_oversample=parse_int_like(self.fit_oversample.text(), name="fit_oversample"),
            )
            _df, csv_path = fitting.run_delay_peak_fitting(**kwargs)
            self._log(f"Delay peak fitting finished. CSV: {csv_path}")
        except Exception as exc:
            self._show_exception("Delay Peak Fitting Error", exc)

    def _run_fit_overlay(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None:
                raise ImportError("Backend package is not available in this environment.")
            kwargs = self._fit_base_kwargs()
            kwargs.update(
                peak=self.fit_overlay_peak.text().strip(),
                delay_fs=parse_int_like(self.fit_overlay_delay.text(), name="delay_fs"),
                is_reference=self.fit_overlay_is_reference.isChecked(),
                group=parse_python_literal(self.fit_overlay_group.text(), empty=None),
                out_csv_name=self.fit_out_csv_name.text().strip() or "peak_fits_delay.csv",
                ensure_csv=self.fit_overlay_ensure_csv.isChecked(),
                delays_fs=self._delays_value_from_line(self.fit_delays),
                ref_type=self.fit_ref_type.currentText(),
                ref_value=self._ref_value_from_line(self.fit_ref_value),
                show=self.fit_overlay_show.isChecked(),
                save=self.fit_overlay_save.isChecked(),
                save_format=self.fit_fig_format.currentText(),
                save_dpi=parse_int_like(self.fit_fig_dpi.text(), name="save_dpi"),
                fit_oversample=parse_int_like(self.fit_oversample.text(), name="fit_oversample"),
                only_success=self.fit_plot_only_success.isChecked(),
            )
            out = fitting.plot_fit_overlay_from_csv(**kwargs)
            self._log(
                f"Fit overlay finished. Saved path: {out.get('saved_path', None) if isinstance(out, dict) else None}"
            )
        except Exception as exc:
            self._show_exception("Fit Overlay Error", exc)

    def _run_time_evolution(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None:
                raise ImportError("Backend package is not available in this environment.")
            kwargs = self._experiment_kwargs("fit_single")
            kwargs.update(
                peak=self.fit_time_peak.text().strip(),
                _property=self.fit_property.currentText(),
                out_csv_name=self.fit_out_csv_name.text().strip() or "peak_fits_delay.csv",
                unit=self.fit_time_unit.currentText(),
                groups=parse_groups(self.fit_groups.text()),
                phi_mode=self.fit_phi_mode.currentText(),
                phi_reduce=self.fit_phi_reduce.currentText(),
                as_lines=self.fit_as_lines.isChecked(),
                delay_offset=parse_float_like(self.fit_delay_offset.text(), name="delay_offset"),
                show_baseline_sigma=self.fit_show_baseline_sigma.isChecked(),
                baseline_sigma=parse_float_like(self.fit_baseline_sigma.text(), name="baseline_sigma"),
                baseline_alpha=parse_float_like(self.fit_baseline_alpha.text(), name="baseline_alpha"),
                baseline_mode=self.fit_baseline_mode.currentText(),
                save=self.fit_time_save.isChecked(),
                save_fmt=self.fit_time_save_fmt.currentText(),
                save_dpi=parse_int_like(self.fit_time_save_dpi.text(), name="save_dpi"),
                paths=self._paths(),
            )
            fitting.plot_time_evolution(**kwargs)
            self._log("Time evolution plot finished.")
        except Exception as exc:
            self._show_exception("Time Evolution Error", exc)

    def _run_time_evolution_multi(self):
        try:
            if PACKAGE_IMPORT_ERROR is not None:
                raise ImportError("Backend package is not available in this environment.")
            phi_mode = self.fit_multi_phi_mode.currentText()
            if phi_mode == "auto":
                phi_mode = None
            phi_window_text = self.fit_multi_phi_window.text().strip()
            phi_window = None if phi_window_text == "" else parse_python_literal(phi_window_text)
            delay_offset = parse_optional_float_like(self.fit_multi_delay_offset.text())
            delay_for_norm_max = parse_optional_float_like(self.fit_multi_delay_for_norm_max.text())
            kwargs = dict(
                experiments=self.fit_multi_editor.get_experiments(),
                peak=self.fit_multi_peak.text().strip(),
                _property=self.fit_multi_property.currentText(),
                out_csv_name=self.fit_multi_out_csv_name.text().strip() or "peak_fits_delay.csv",
                unit=self.fit_multi_unit.currentText(),
                phi_mode=phi_mode,
                phi_reduce=self.fit_multi_phi_reduce.currentText(),
                phi_window=phi_window,
                only_success=self.fit_multi_only_success.isChecked(),
                include_reference=self.fit_multi_include_reference.isChecked(),
                as_lines=self.fit_multi_as_lines.isChecked(),
                delay_offset=delay_offset,
                show_baseline_sigma=self.fit_multi_show_baseline_sigma.isChecked(),
                baseline_sigma=parse_float_like(self.fit_multi_baseline_sigma.text(), name="baseline_sigma"),
                baseline_alpha=parse_float_like(self.fit_multi_baseline_alpha.text(), name="baseline_alpha"),
                baseline_mode=self.fit_multi_baseline_mode.currentText(),
                norm_min_max=self.fit_multi_norm_min_max.isChecked(),
                delay_for_norm_max=delay_for_norm_max,
                cmap=self.fit_multi_cmap.text().strip() or None,
                save=self.fit_multi_save.isChecked(),
                save_fmt=self.fit_multi_save_fmt.currentText(),
                save_dpi=parse_int_like(self.fit_multi_save_dpi.text(), name="save_dpi"),
                save_overwrite=self.fit_multi_save_overwrite.isChecked(),
                paths=self._paths(),
            )
            out = fitting.plot_time_evolution_multi(**kwargs)
            self._log(
                f"Multiple-experiment time evolution finished. Saved path: {out.get('saved_path', None) if isinstance(out, dict) else None}"
            )
        except Exception as exc:
            self._show_exception("Multi Time Evolution Error", exc)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()