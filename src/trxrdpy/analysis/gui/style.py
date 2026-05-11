"""
Central GUI style and layout constants for the analysis GUI.

These values are intended to preserve the legacy analysis GUI appearance
while the internals are refactored.
"""

from dataclasses import dataclass


LEGACY_MAIN_WINDOW_STYLESHEET = """
/* -------------------------------------------------------------------------
   XRDpy Analysis GUI aesthetic stylesheet
   Muted blue scientific theme.
   ------------------------------------------------------------------------- */

QMainWindow {
    background-color: #eef2f5;
}

QWidget {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #17212b;
}

/* Modal dialogs / QMessageBox: keep text readable */
QDialog,
QMessageBox {
    background-color: #eef5fb;
    color: #17212b;
}

QDialog QLabel,
QMessageBox QLabel {
    color: #17212b;
    background: transparent;
}

QMessageBox QPushButton,
QDialog QPushButton {
    min-width: 82px;
}

/* Main tabs */
QTabWidget::pane {
    border: 1px solid #c4ccd4;
    border-radius: 6px;
    background: #f6f7f9;
    top: -1px;
}

QTabBar::tab {
    background: #dde3ea;
    border: 1px solid #c4ccd4;
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 7px 13px;
    margin-right: 2px;
    color: #213547;
}

QTabBar::tab:selected {
    background: #f6f7f9;
    color: #0c3559;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background: #e8edf2;
}

/* Section panels */
QGroupBox {
    background-color: #f6f7f9;
    border: 1px solid #c8d0d8;
    border-radius: 7px;
    margin-top: 13px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    color: #1c2d3e;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 2px;
    padding: 0 5px;
    background-color: #f6f7f9;
    color: #0c3559;
}

/* Labels */
QLabel {
    color: #1f2f3d;
}

QLabel:disabled {
    color: #8293a3;
}

/* Inputs */
QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox {
    background-color: #f3f4f6;
    border: 1px solid #b8c0c8;
    border-radius: 4px;
    padding: 4px 6px;
    min-height: 31px;
    color: #17212b;
    selection-background-color: #c7dff4;
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {
    border: 1px solid #477da8;
    background-color: #f3f4f6;
}

QLineEdit:disabled,
QComboBox:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled {
    background-color: #e4ebf2;
    color: #7d8c98;
}

/* Text editors */
QPlainTextEdit,
QTextEdit {
    background-color: #f3f4f6;
    border: 1px solid #b8c0c8;
    border-radius: 5px;
    padding: 5px;
    color: #17212b;
    selection-background-color: #c7dff4;
}

/* Buttons */
QPushButton {
    background-color: #eceff3;
    border: 1px solid #9eb6ca;
    border-radius: 5px;
    padding: 5px 11px;
    min-height: 31px;
    color: #17324a;
}

QPushButton:hover {
    background-color: #e2e7ed;
    border-color: #6f9fc5;
}

QPushButton:pressed {
    background-color: #d7dde5;
    border-color: #477da8;
}

QPushButton:disabled {
    background-color: #e3e9ef;
    border-color: #c5cfd8;
    color: #8393a0;
}

/* Check boxes */
QCheckBox {
    spacing: 6px;
    color: #17212b;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
}

/* Docked / floating log */
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
    font-weight: 600;
}

/* Menus */
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

/* Scrollbars */
QScrollBar:vertical {
    background: #e4e8ec;
    width: 12px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #b8c0c8;
    min-height: 31px;
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


/* -------------------------------------------------------------------------
   Force light backgrounds for containers that otherwise inherit macOS dark mode.
   ------------------------------------------------------------------------- */

QWidget {
    background-color: #eef2f5;
    color: #17212b;
}

QMainWindow,
QDialog,
QMessageBox {
    background-color: #eef2f5;
    color: #17212b;
}

QScrollArea,
QAbstractScrollArea,
QScrollArea > QWidget,
QScrollArea > QWidget > QWidget,
QFrame {
    background-color: #eef2f5;
    color: #17212b;
}

QTabWidget,
QTabWidget > QWidget,
QStackedWidget {
    background-color: #eef2f5;
    color: #17212b;
}

QGroupBox {
    background-color: #f6f7f9;
    color: #17212b;
}

QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QPlainTextEdit,
QTextEdit {
    background-color: #f3f4f6;
    color: #17212b;
}

QListView,
QTreeView,
QTableView {
    background-color: #f3f4f6;
    color: #17212b;
    alternate-background-color: #e8edf2;
}

QToolTip {
    background-color: #f6f7f9;
    color: #17212b;
    border: 1px solid #c4ccd4;
}



/* -------------------------------------------------------------------------
   Geometry fixes after larger font size.
   ------------------------------------------------------------------------- */

QTabBar::tab {
    min-height: 39px;
    min-width: 105px;
    padding: 9px 18px;
}

QTabBar::tab:selected {
    min-height: 39px;
    padding: 9px 18px;
}

QComboBox {
    min-width: 180px;
    min-height: 31px;
    padding: 4px 22px 4px 8px;
}


QComboBox QAbstractItemView {
    min-width: 230px;
    padding: 4px;
    outline: 0;
}

QLineEdit,
QPushButton,
QSpinBox,
QDoubleSpinBox {
    min-height: 31px;
}



/* -------------------------------------------------------------------------
   Custom autosave dialog and dock-control refinements.
   ------------------------------------------------------------------------- */

QDialog#RestoreAutosaveDialog {
    background-color: #eef2f5;
    color: #17212b;
}

QDialog#RestoreAutosaveDialog QLabel {
    background: transparent;
    color: #17212b;
}

QDialog#RestoreAutosaveDialog QLabel#RestoreAutosaveTitle {
    font-size: 15px;
    font-weight: 700;
    color: #0c3559;
}

QDialog#RestoreAutosaveDialog QLabel#RestoreAutosaveBody {
    font-size: 13px;
    color: #17212b;
}

QDialog#RestoreAutosaveDialog QPushButton {
    min-width: 96px;
    min-height: 31px;
}

/* Make default QDockWidget controls lighter where Qt allows styling. */
QDockWidget::close-button,
QDockWidget::float-button {
    background-color: #e6edf3;
    border: 1px solid #aebfce;
    border-radius: 4px;
    width: 15px;
    height: 15px;
    margin: 2px;
}

QDockWidget::close-button:hover,
QDockWidget::float-button:hover {
    background-color: #d9eaf8;
    border-color: #6f9fc5;
}

QDockWidget::close-button:pressed,
QDockWidget::float-button:pressed {
    background-color: #c5dff3;
    border-color: #477da8;
}



/* -------------------------------------------------------------------------
   Fully custom autosave dialog: no native gray title bar.
   ------------------------------------------------------------------------- */

QDialog#RestoreAutosaveDialog {
    background-color: #eef2f5;
    border: 1px solid #b8c7d6;
    border-radius: 8px;
}

QWidget#RestoreAutosaveHeader {
    background-color: #dde8f2;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border-bottom: 1px solid #b8c7d6;
}

QLabel#RestoreAutosaveHeaderLabel {
    background: transparent;
    color: #0c3559;
    font-weight: 700;
    font-size: 13px;
}

QPushButton#RestoreAutosaveCloseButton {
    background-color: #eaf1f8;
    border: 1px solid #aebfce;
    border-radius: 14px;
    color: #17212b;
    font-weight: 700;
    padding: 0;
    min-width: 28px;
    min-height: 28px;
}

QPushButton#RestoreAutosaveCloseButton:hover {
    background-color: #d9eaf8;
    border-color: #6f9fc5;
}

QWidget#RestoreAutosaveBodyWidget {
    background-color: #eef2f5;
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
}

QDialog#RestoreAutosaveDialog QLabel#RestoreAutosaveTitle {
    background: transparent;
    color: #0c3559;
    font-size: 15px;
    font-weight: 700;
}

QDialog#RestoreAutosaveDialog QLabel#RestoreAutosaveBody {
    background: transparent;
    color: #17212b;
    font-size: 13px;
}




/* ---------------- Rounded aesthetic patch ---------------- */

QTabWidget::pane {
    background: #eef3f7;
    border: 1px solid #c5cfda;
    border-radius: 12px;
    margin-top: 6px;
}

QTabBar::tab {
    background: #dfe8f2;
    border: 1px solid #bcc9d8;
    border-bottom: none;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    padding: 10px 16px;
    min-height: 34px;
}

QTabBar::tab:selected {
    background: #f4f8fc;
    color: #0f3a67;
    font-weight: 600;
}

QGroupBox {
    background: #f4f8fc;
    border: 1px solid #c5cfda;
    border-radius: 12px;
    margin-top: 12px;
    padding: 14px 12px 12px 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #0f3a67;
    background: #eef3f7;
    border-radius: 6px;
}

QLineEdit,
QComboBox,
QPlainTextEdit,
QTextEdit,
QSpinBox,
QDoubleSpinBox {
    background: #f9fbfd;
    border: 1px solid #b8c6d5;
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 28px;
}


QPushButton {
    background: #edf3f8;
    border: 1px solid #9fb6cf;
    border-radius: 10px;
    padding: 7px 14px;
    min-height: 28px;
}

QPushButton:hover {
    background: #e3edf7;
}

QPushButton:pressed {
    background: #d7e6f3;
}

QScrollArea {
    background: #eef3f7;
    border: 1px solid #c5cfda;
    border-radius: 12px;
}

QDockWidget {
    background: #eef3f7;
    border: 1px solid #c5cfda;
    border-radius: 12px;
}

QDockWidget::title {
    background: #dde6ef;
    padding-left: 10px;
    padding-top: 6px;
    padding-bottom: 6px;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    color: #1f2a36;
    font-weight: 600;
}

/* --------------------------------------------------------- */


/* -------------------------------------------------------------------------
   Final geometry repair: normal combo arrows and no oversized tabs.
   ------------------------------------------------------------------------- */

QComboBox {
    min-width: 150px;
    min-height: 31px;
    padding: 4px 8px;
}

QComboBox QAbstractItemView {
    min-width: 190px;
    padding: 4px;
    outline: 0;
}

QTabBar::tab {
    min-height: 36px;
    min-width: 0px;
    padding: 8px 11px;
}

QTabBar::tab:selected {
    min-height: 36px;
    padding: 8px 11px;
}






/* -------------------------------------------------------------------------
   Safe final layout override after reverting custom main-window chrome.
   ------------------------------------------------------------------------- */

QTabBar::tab {
    min-height: 36px;
    min-width: 0px;
    padding: 8px 11px;
}

QTabBar::tab:selected {
    min-height: 36px;
    padding: 8px 11px;
    font-weight: 700;
}

QTabWidget::pane {
    border-radius: 8px;
}

QGroupBox {
    border-radius: 9px;
}

QLineEdit,
QComboBox,
QPlainTextEdit,
QTextEdit,
QSpinBox,
QDoubleSpinBox {
    border-radius: 7px;
}

QPushButton {
    border-radius: 8px;
}

QComboBox {
    min-width: 150px;
    min-height: 31px;
    padding: 4px 8px;
}



/* -------------------------------------------------------------------------
   Remove label background tiles.
   Labels should blend into their parent panels.
   ------------------------------------------------------------------------- */

QLabel {
    background-color: transparent;
    color: #17212b;
}

QGroupBox QLabel {
    background-color: transparent;
}

QCheckBox,
QRadioButton {
    background-color: transparent;
    color: #17212b;
}

QGroupBox::title {
    background-color: #eef2f5;
    color: #0c3559;
}

/* Keep section-title labels transparent unless explicitly styled otherwise. */
QLabel#RestoreAutosaveTitle,
QLabel#RestoreAutosaveBody,
QLabel#RestoreAutosaveHeaderLabel {
    background-color: transparent;
}



/* -------------------------------------------------------------------------
   Final selected-tab anti-clipping fix.
   ------------------------------------------------------------------------- */

QTabBar::tab {
    min-height: 48px;
    min-width: 0px;
    padding: 12px 14px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    min-height: 50px;
    padding: 12px 14px;
    font-weight: 700;
}



/* -------------------------------------------------------------------------
   Final main-tab width fix.
   ------------------------------------------------------------------------- */

QTabBar::tab {
    min-height: 46px;
    min-width: 118px;
    padding: 10px 18px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    min-height: 46px;
    min-width: 124px;
    padding: 10px 20px;
    font-weight: 700;
}



/* -------------------------------------------------------------------------
   Final targeted typography.
   Larger main tabs, slightly larger log text.
   ------------------------------------------------------------------------- */

QTabBar::tab {
    font-size: 15px;
    min-height: 54px;
    min-width: 118px;
    padding: 10px 18px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    font-size: 15px;
    min-height: 54px;
    min-width: 124px;
    padding: 10px 20px;
    font-weight: 700;
}

QTextEdit#MainLogText {
    font-size: 13px;
}



/* -------------------------------------------------------------------------
   Final typography tuning.
   Regular text slightly larger, main tabs slightly smaller, section headers larger.
   ------------------------------------------------------------------------- */

/* Regular GUI text */
QWidget {
    font-size: 13px;
}

/* Main navigation tabs */
QTabBar::tab {
    font-size: 13px;
    min-height: 42px;
    min-width: 118px;
    padding: 9px 18px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    font-size: 13px;
    min-height: 42px;
    min-width: 124px;
    padding: 9px 20px;
    font-weight: 700;
}

/* Subsection titles, e.g. Experiment Metadata, Calibration Context */
QGroupBox {
    font-size: 13px;
}

QGroupBox::title {
    font-size: 15px;
    font-weight: 700;
    color: #0c3559;
    background-color: #eef2f5;
    padding: 0 8px;
}

/* Keep log readable but not larger than section headers */
QTextEdit#MainLogText {
    font-size: 13px;
}



/* -------------------------------------------------------------------------
   FINAL typography correction.
   Regular controls are 13px; main tabs and section headers are emphasized.
   ------------------------------------------------------------------------- */

QWidget {
    font-size: 13px;
}

QLabel,
QLineEdit,
QComboBox,
QPushButton,
QCheckBox,
QRadioButton,
QSpinBox,
QDoubleSpinBox,
QPlainTextEdit,
QTextEdit {
    font-size: 13px;
}

QTabBar::tab {
    font-size: 13px;
    min-height: 42px;
    min-width: 118px;
    padding: 9px 18px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    font-size: 13px;
    min-height: 42px;
    min-width: 124px;
    padding: 9px 20px;
    font-weight: 700;
}

QGroupBox {
    font-size: 13px;
}

QGroupBox::title {
    font-size: 13px;
    font-weight: 700;
    color: #0c3559;
}

QTextEdit#MainLogText {
    font-size: 13px;
}



/* -------------------------------------------------------------------------
   Toolbar styling.
   ------------------------------------------------------------------------- */

QToolBar#MainToolBar {
    background-color: #dde8f2;
    border-top: 1px solid #b8c7d6;
    spacing: 6px;
    padding: 5px 8px;
}

QToolBar#MainToolBar QToolButton {
    background-color: #e7f0f8;
    border: 1px solid #9eb6ca;
    border-radius: 8px;
    padding: 5px 10px;
    color: #17324a;
}

QToolBar#MainToolBar QToolButton:hover {
    background-color: #d9eaf8;
    border-color: #6f9fc5;
}

QToolBar#MainToolBar QToolButton:pressed {
    background-color: #c5dff3;
    border-color: #477da8;
}



/* -------------------------------------------------------------------------
   FINAL main-tab compact-width tuning.
   ------------------------------------------------------------------------- */

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


@dataclass(frozen=True)
class AnalysisGuiStyle:
    main_window_title: str = "XRDpy Analysis GUI"
    main_window_width: int = 900
    main_window_height: int = 800

    log_group_title: str = "Log"
    log_group_max_height: int = 120
    log_box_fixed_height: int = 60
    log_max_block_count: int = 1000


STYLE = AnalysisGuiStyle()
