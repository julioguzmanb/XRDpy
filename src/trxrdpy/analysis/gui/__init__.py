from __future__ import annotations


def main():
    # Capture the shell directory before importing the GUI and its scientific
    # backends.  This is the directory users expect file dialogs to open in.
    """Create the Qt application, install runtime diagnostics, and show the main window."""
    from pathlib import Path

    launch_directory = Path.cwd().resolve()
    from .main_window import main as _main

    return _main(launch_directory=launch_directory)


__all__ = ["main"]
