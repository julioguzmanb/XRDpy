"""Runtime diagnostics and exception containment for the analysis GUI."""
from __future__ import annotations

from datetime import datetime
import faulthandler
import os
from pathlib import Path
import sys
import tempfile
import threading
import traceback


class GuiRuntimeGuard:
    """Keep uncaught callback exceptions visible and persist crash diagnostics.

    Attributes
    ----------
    launch_directory : pathlib.Path
        Directory from which the GUI process was started.
    window : QMainWindow or None
        Attached analysis window used for user-visible error reporting.
    log_path : pathlib.Path
        Writable persistent or temporary diagnostic-log path.
    _handle : IO
        Line-buffered text stream backing ``log_path``.
    _previous_sys_hook, _previous_thread_hook : callable or None
        Exception hooks retained for diagnostics and compatibility.
    """

    def __init__(self, *, launch_directory: Path):
        """Initialize configuration, normalize inputs, and create the object runtime state."""
        self.launch_directory = Path(launch_directory).expanduser().resolve()
        self.window = None
        self.log_path, self._handle = self._open_log()
        self._previous_sys_hook = sys.excepthook
        self._previous_thread_hook = getattr(threading, "excepthook", None)

    def _open_log(self):
        """Open the crash log in append mode, creating its parent directory first."""
        candidates = (
            Path.home() / ".xrdpy" / "logs" / "analysis_gui.log",
            Path(tempfile.gettempdir()) / "xrdpy_analysis_gui.log",
        )
        last_error = None
        for path in candidates:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                return path, path.open("a", encoding="utf-8", buffering=1)
            except Exception as exc:
                last_error = exc
        raise OSError("Could not create the analysis GUI runtime log") from last_error

    def record(self, message: str, details: str | None = None) -> None:
        """Record .

        Parameters
        ----------
        message : str
            Concise user-facing error or diagnostic message.
        details : str | None
            Optional diagnostic traceback or contextual details stored with the message.
        """
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        header = f"[{timestamp}] pid={os.getpid()} {message}\n"
        try:
            self._handle.write(header)
            if details:
                self._handle.write(str(details).rstrip() + "\n")
            self._handle.flush()
        except Exception:
            pass

    def install(self, app) -> None:
        """Return install.

        Parameters
        ----------
        app : object
            Active Qt application whose exception hook is being installed.
        """
        try:
            faulthandler.enable(file=self._handle, all_threads=True)
        except Exception as exc:
            self.record("Could not enable faulthandler", repr(exc))

        sys.excepthook = self._handle_python_exception
        if self._previous_thread_hook is not None:
            threading.excepthook = self._handle_thread_exception

        app.aboutToQuit.connect(
            lambda: self.record("Qt application is about to quit cleanly")
        )
        self.record(
            "GUI runtime guard installed",
            f"launch_directory={self.launch_directory}\npython={sys.version}",
        )

    def attach_window(self, window) -> None:
        """Attach the main window and report the runtime-log location in its log."""
        self.window = window
        window.runtime_guard = self
        try:
            window.log_widget.log(f"Runtime log: {self.log_path}")
        except Exception:
            pass

    def _report_to_gui(self, summary: str) -> None:
        """Write a concise failure summary to the attached GUI log when available."""
        try:
            if self.window is not None:
                self.window.log_widget.log(
                    f"{summary}. Details were written to {self.log_path}"
                )
        except Exception:
            pass

    def _handle_python_exception(self, exc_type, exc_value, exc_traceback) -> None:
        """Persist an uncaught main-thread exception and surface it in the GUI."""
        details = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        self.record("UNHANDLED PYTHON EXCEPTION", details)
        self._report_to_gui(f"Unhandled error: {exc_value}")
        try:
            sys.__stderr__.write(details)
            sys.__stderr__.flush()
        except Exception:
            pass

    def _handle_thread_exception(self, args) -> None:
        """Persist an uncaught worker-thread exception and surface it in the GUI."""
        details = "".join(
            traceback.format_exception(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
            )
        )
        self.record(
            f"UNHANDLED THREAD EXCEPTION in {getattr(args.thread, 'name', 'unknown')}",
            details,
        )
        self._report_to_gui(f"Background-thread error: {args.exc_value}")


__all__ = ["GuiRuntimeGuard"]
