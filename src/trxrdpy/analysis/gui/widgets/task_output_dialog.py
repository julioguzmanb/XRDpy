"""
Live task-output dialog for long-running GUI actions.

It captures stdout/stderr from a worker thread so tqdm output can be shown in
a small Qt window while the GUI remains responsive.
"""
from __future__ import annotations

import os
import sys
import threading
import traceback

from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class _ThreadRoutingStream:
    """Route writes from registered worker threads to their own Qt signals.

    Unregistered threads, including the Qt GUI thread and multiprocessing
    children, continue writing to the stream that was active when the router
    was installed. Keys include the process ID so a forked child never tries
    to emit through a copied Qt signal owned by the parent process.
    """

    def __init__(self, fallback):
        self._fallback = fallback
        self._routes = {}
        self._lock = threading.RLock()

    @property
    def encoding(self):
        """Expose the wrapped stream encoding for terminal-aware libraries."""
        return getattr(self._fallback, "encoding", "utf-8")

    def register_current_thread(self, emit_func, *, tee=False):
        """Route the calling thread's writes to ``emit_func`` until unregistered."""
        key = (os.getpid(), threading.get_ident())
        with self._lock:
            self._routes[key] = (emit_func, bool(tee))
        return key

    def unregister(self, key):
        """Remove a previously registered process/thread route."""
        with self._lock:
            self._routes.pop(key, None)

    def _current_route(self):
        key = (os.getpid(), threading.get_ident())
        with self._lock:
            return self._routes.get(key)

    def write(self, text):
        """Forward worker output or fall back to the original process stream."""
        route = self._current_route()
        if route is not None:
            emit_func, tee = route
            if text:
                emit_func(str(text))
            if not tee:
                return len(text or "")
        if self._fallback is None:
            return len(text or "")
        written = self._fallback.write(text)
        if written is None:
            return len(text or "")
        return written

    def flush(self):
        """Flush the fallback stream when the current thread is not captured."""
        route = self._current_route()
        if self._fallback is not None and (route is None or route[1]):
            return self._fallback.flush()

    def isatty(self):
        """Let tqdm render progress for captured workers only."""
        if self._current_route() is not None:
            return True
        return bool(
            self._fallback is not None
            and getattr(self._fallback, "isatty", lambda: False)()
        )

    def fileno(self):
        """Delegate file-descriptor access when the fallback supports it."""
        if self._fallback is None or not hasattr(self._fallback, "fileno"):
            raise OSError("Stream has no file descriptor.")
        return self._fallback.fileno()

    def writable(self):
        """Report whether the wrapped stream accepts writes."""
        if self._fallback is None:
            return True
        return bool(getattr(self._fallback, "writable", lambda: True)())

    def __getattr__(self, name):
        """Delegate less common text-stream attributes to the wrapped stream."""
        if self._fallback is None:
            raise AttributeError(name)
        return getattr(self._fallback, name)


_STREAM_ROUTER_LOCK = threading.RLock()


def _ensure_thread_stream_router(stream_name):
    """Install or return the process-wide thread-aware stream router."""
    with _STREAM_ROUTER_LOCK:
        current = getattr(sys, stream_name)
        if isinstance(current, _ThreadRoutingStream):
            return current
        router = _ThreadRoutingStream(current)
        setattr(sys, stream_name, router)
        return router


class TaskWorker(QObject):
    """Execute a callable in a worker thread and report output through Qt signals.

    Attributes
    ----------
    func : callable
        Zero-argument task executed by :meth:`run`.
    output, result, error, finished : pyqtSignal
        Signals carrying captured text, return values, tracebacks, and completion.
    """
    output = pyqtSignal(str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, func):
        """Initialize configuration, normalize inputs, and create the object runtime state."""
        super().__init__()
        self.func = func

    def run(self):
        """Execute the task while forwarding stdout and stderr to the dialog.

        The callable's return value is emitted through ``result``. Unhandled
        exceptions are converted to traceback text and emitted through
        ``error``. The calling thread's stream routes are removed and
        ``finished`` is emitted in all cases.
        """
        stdout_router = _ensure_thread_stream_router("stdout")
        stderr_router = _ensure_thread_stream_router("stderr")
        stdout_key = stdout_router.register_current_thread(
            self.output.emit,
            tee=True,
        )
        stderr_key = stderr_router.register_current_thread(
            self.output.emit,
            tee=True,
        )

        try:
            result = self.func()
            self.result.emit(result)

        except Exception:
            self.error.emit(traceback.format_exc())

        finally:
            stdout_router.unregister(stdout_key)
            stderr_router.unregister(stderr_key)
            self.finished.emit()


class TaskOutputDialog(QDialog):
    """Display live output and completion status for a background task.

    Attributes
    ----------
    auto_close_on_success : bool
        Whether a successful task closes the dialog automatically.
    auto_close_delay_ms : int
        Delay before automatic closure, in milliseconds.
    status_label : QLabel
        Current running, success, or failure state.
    output_text : QPlainTextEdit
        Captured standard output, standard error, and progress updates.
    close_button : QPushButton
        Manual close control, disabled while a task is active.
    _running : bool
        Whether the worker has not yet emitted completion.
    _last_progress_line : str
        Most recent carriage-return progress message.
    """
    task_finished = pyqtSignal()

    def __init__(
        self,
        title="Running task",
        parent=None,
        *,
        auto_close_on_success=True,
        auto_close_delay_ms=1200,
        cancel_callback=None,
    ):
        """Initialize ``TaskOutputDialog``, bind shared state and services, and create its controls."""
        super().__init__(parent)
        self.auto_close_on_success = bool(auto_close_on_success)
        self.auto_close_delay_ms = int(auto_close_delay_ms)

        self.setWindowTitle(title)
        self.resize(760, 380)
        self._running = True
        self._cancel_requested = False
        self._cancel_callback = cancel_callback
        self._last_progress_line = ""

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.status_label = QLabel("Running...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        button_row = QHBoxLayout()
        button_row.addStretch()

        self.stop_button = QPushButton("Stop safely")
        self.stop_button.setVisible(cancel_callback is not None)
        self.stop_button.clicked.connect(self.request_cancel)
        button_row.addWidget(self.stop_button)

        self.close_button = QPushButton("Close")
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.close)
        button_row.addWidget(self.close_button)

        layout.addLayout(button_row)

    def append_output(self, text):
        """Append output.

        Parameters
        ----------
        text : object
            Text entered in the corresponding GUI field.
        """
        if not text:
            return

        text = str(text)

        # tqdm commonly updates one line using carriage returns.
        if "\r" in text:
            parts = text.split("\r")
            last = parts[-1].strip()
            if last:
                self._last_progress_line = last
                self.status_label.setText(last)
            if not text.endswith("\n"):
                return
            text = (last + "\n") if last else ""

        self.output_text.moveCursor(QTextCursor.End)
        self.output_text.insertPlainText(text)
        self.output_text.moveCursor(QTextCursor.End)

    def request_cancel(self):
        """Request a cooperative stop and leave the dialog open until it is safe."""
        if not self._running or self._cancel_requested:
            return
        self._cancel_requested = True
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping safely after the active HDF5 batch...")
        self.append_output("\nSafe stop requested. Finishing active file batches...\n")
        if self._cancel_callback is not None:
            self._cancel_callback()

    def mark_success(self):
        """Mark the task successful, update controls, and optionally schedule closure."""
        self._running = False
        self.status_label.setText("Finished.")
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.task_finished.emit()

        if self.auto_close_on_success:
            QTimer.singleShot(self.auto_close_delay_ms, self.close)

    def mark_error(self):
        """Mark the task failed, append its traceback, and enable manual closure."""
        self._running = False
        self.status_label.setText("Error.")
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.task_finished.emit()

    def mark_cancelled(self):
        """Mark a cooperative cancellation after workers have closed their batches."""
        self._running = False
        self.status_label.setText("Stopped safely. Existing files can be resumed.")
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.task_finished.emit()

    def closeEvent(self, event):
        """Prevent closing while a task is active unless cancellation is confirmed."""
        if self._running:
            self.hide()
            event.ignore()
            return

        super().closeEvent(event)


def run_task_with_output_dialog(
    parent,
    title,
    func,
    *,
    on_success=None,
    on_error=None,
    auto_close_on_success=True,
    auto_close_delay_ms=1200,
    cancel_callback=None,
):
    """Run func() in a QThread and show stdout/stderr in a TaskOutputDialog.

    on_success(result) and on_error(traceback_text) are called in the Qt thread.

    Parameters
    ----------
    parent : QWidget or None
        Owning widget used to retain the dialog for the task lifetime.
    title : str
        Dialog window title.
    func : callable
        Zero-argument operation executed in the worker thread.
    on_success : callable, optional
        Qt-thread callback receiving the operation's return value.
    on_error : callable, optional
        Qt-thread callback receiving formatted traceback text.
    auto_close_on_success : bool
        Close the dialog automatically after successful completion.
    auto_close_delay_ms : int
        Delay before automatic closure in milliseconds.

    Returns
    -------
    TaskOutputDialog
        Visible dialog retaining its worker and thread until completion.
    """
    dialog = TaskOutputDialog(
        title=title,
        parent=parent,
        auto_close_on_success=auto_close_on_success,
        auto_close_delay_ms=auto_close_delay_ms,
        cancel_callback=cancel_callback,
    )

    thread = QThread(dialog)
    worker = TaskWorker(func)
    worker.moveToThread(thread)

    dialog._thread = thread
    dialog._worker = worker

    # Keep dialog alive if parent exists.
    if parent is not None:
        dialogs = getattr(parent, "_active_task_output_dialogs", None)
        if dialogs is None:
            dialogs = []
            setattr(parent, "_active_task_output_dialogs", dialogs)
        dialogs.append(dialog)

        def cleanup_dialog_ref():
            """Release retained worker, thread, and dialog references after completion."""
            try:
                dialogs.remove(dialog)
            except ValueError:
                pass

        dialog.destroyed.connect(cleanup_dialog_ref)

    def handle_result(result):
        """Forward a successful background-task result to the caller callback."""
        if on_success is not None:
            on_success(result)
        if isinstance(result, dict) and result.get("cancelled"):
            dialog.mark_cancelled()
        else:
            dialog.mark_success()

    def handle_error(traceback_text):
        """Display a background-task traceback and invoke the error callback."""
        dialog.append_output("\n" + traceback_text + "\n")
        if on_error is not None:
            on_error(traceback_text)
        dialog.mark_error()

    worker.output.connect(dialog.append_output)
    worker.result.connect(handle_result)
    worker.error.connect(handle_error)

    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    dialog.show()
    thread.start()

    return dialog
