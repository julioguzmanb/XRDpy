"""Central visual constants and stylesheet fragments for the simulation GUI."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationGuiStyle:
    """Immutable collection of window dimensions, spacing, colors, and fonts."""
    main_window_title: str = "XRDpy Simulation GUI"
    main_window_width: int = 1200
    main_window_height: int = 780
    log_box_min_height: int = 170
    log_max_block_count: int = 5000


STYLE = SimulationGuiStyle()


SIMULATION_MAIN_WINDOW_STYLESHEET = """
QMainWindow {
    background-color: #eef2f5;
}

QWidget {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #17212b;
    background-color: #eef2f5;
}

/* Main tabs */
QTabWidget::pane {
    border: 1px solid #c4ccd4;
    border-radius: 8px;
    background: #f6f7f9;
    top: -1px;
}

QTabBar::tab {
    background: #dde3ea;
    border: 1px solid #c4ccd4;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 7px 12px;
    margin-right: 2px;
    min-height: 32px;
    color: #17324a;
}

QTabBar::tab:selected {
    background: #f6f7f9;
    color: #0c3559;
    font-weight: 700;
}

QTabBar::tab:hover:!selected {
    background: #e8edf2;
}

/* Section panels */
QGroupBox {
    background-color: #f6f7f9;
    border: 1px solid #c8d0d8;
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    color: #17212b;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 2px;
    padding: 0 8px;
    background-color: #eef2f5;
    color: #0c3559;
    font-size: 14px;
    font-weight: 700;
}

/* Labels */
QLabel {
    background-color: transparent;
    color: #17212b;
}

QGroupBox QLabel {
    background-color: transparent;
}

/* Inputs */
QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QPlainTextEdit,
QTextEdit {
    background-color: #f3f4f6;
    border: 1px solid #b8c0c8;
    border-radius: 7px;
    padding: 5px 8px;
    min-height: 27px;
    color: #17212b;
    selection-background-color: #c7dff4;
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QPlainTextEdit:focus,
QTextEdit:focus {
    border: 1px solid #477da8;
    background-color: #ffffff;
}

/* Buttons */
QPushButton {
    background-color: #eceff3;
    border: 1px solid #9eb6ca;
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 28px;
    color: #17324a;
}

QPushButton:hover {
    background-color: #d9eaf8;
    border-color: #6f9fc5;
}

QPushButton:pressed {
    background-color: #c5dff3;
    border-color: #477da8;
}

QPushButton:disabled {
    background-color: #e3e9ef;
    border-color: #c5cfd8;
    color: #8393a0;
}

/* Check/radio */
QCheckBox,
QRadioButton {
    background-color: transparent;
    color: #17212b;
    spacing: 7px;
    padding: 2px 1px;
    min-height: 22px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
}

QCheckBox::indicator:unchecked {
    background-color: #ffffff;
    border: 2px solid #667b8d;
    border-radius: 3px;
}

QCheckBox::indicator:unchecked:hover {
    background-color: #edf5fb;
    border-color: #2f6f9f;
}

QCheckBox::indicator:unchecked:disabled {
    background-color: #e4e9ed;
    border-color: #9ba9b5;
}

/* Log/dock */
QDockWidget {
    background-color: #eef2f5;
    color: #17212b;
}

QDockWidget::title {
    background: #dde3ea;
    border: 1px solid #c4ccd4;
    padding: 5px 8px;
    text-align: left;
    color: #0c3559;
    font-weight: 700;
}

QTextEdit#SimulationLogText,
QPlainTextEdit#SimulationLogText {
    background-color: #f3f4f6;
    color: #17212b;
    font-size: 14px;
}

/* Menu */
QMenuBar {
    background-color: #eef2f5;
    border-bottom: 1px solid #c4ccd4;
    color: #17212b;
}

QMenuBar::item {
    padding: 4px 9px;
    background: transparent;
}

QMenuBar::item:selected {
    background: #dde3ea;
    border-radius: 4px;
}

QMenu {
    background-color: #f6f7f9;
    border: 1px solid #c4ccd4;
    padding: 4px;
    color: #17212b;
}

QMenu::item {
    padding: 5px 24px 5px 18px;
}

QMenu::item:selected {
    background-color: #dde3ea;
    color: #0c3559;
}

/* Toolbar */
QToolBar#SimulationMainToolBar {
    background-color: #dde3ea;
    border-top: 1px solid #c4ccd4;
    spacing: 6px;
    padding: 5px 8px;
}

QToolBar#SimulationMainToolBar QToolButton {
    background-color: #eceff3;
    border: 1px solid #9eb6ca;
    border-radius: 8px;
    padding: 5px 10px;
    color: #17324a;
}

QToolBar#SimulationMainToolBar QToolButton:hover {
    background-color: #d9eaf8;
    border-color: #6f9fc5;
}

/* Scrollbars */
QScrollBar:vertical {
    background: #e4e8ec;
    width: 12px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #b8c0c8;
    min-height: 24px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background: #9ea7b0;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #e4e8ec;
    height: 12px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background: #b8c0c8;
    min-width: 24px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal:hover {
    background: #9ea7b0;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
}
"""
