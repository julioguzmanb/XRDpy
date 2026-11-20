"""Qt simulation GUI, launch helpers, session state, and reusable widgets."""
from __future__ import annotations
from .state import (
    AUTOSAVE_FILENAME,
    GUI_STATE_VERSION,
    GuiState,
    MatrixToolState,
    PathsState,
    PolyState,
    SingleState,
    UIState,
)
from .widgets import GeometryPanel, MatrixRotationWindow


def __getattr__(name):
    # Do not preload a module that runpy/spawn may execute as __main__.
    if name in {"MainWindow", "launch_gui", "main"}:
        from . import main_window
        return getattr(main_window, name)
    raise AttributeError(name)

__all__ = [
    "MainWindow",
    "launch_gui",
    "main",
    "GuiState",
    "UIState",
    "PathsState",
    "PolyState",
    "SingleState",
    "MatrixToolState",
    "GUI_STATE_VERSION",
    "AUTOSAVE_FILENAME",
    "GeometryPanel",
    "MatrixRotationWindow",
]
