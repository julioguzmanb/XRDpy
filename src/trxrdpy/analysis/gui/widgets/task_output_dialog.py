"""
Live task-output dialog for long-running GUI actions.

It captures stdout/stderr from a worker thread so tqdm output can be shown in
a small Qt window while the GUI remains responsive.
"""
from __future__ import annotations

import sys
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


class _StreamProxy:
    """Forward redirected text streams to a Qt signal."""
    def __init__(self, emit_func):
        """Initialize the object and its runtime state."""
        self._emit_func = emit_func
        self.encoding = "utf-8"

    def write(self, text):
        """Forward nonempty stream text to the configured Qt signal."""
        if text:
            self._emit_func(str(text))
        return len(text or "")

    def flush(self):
        """Provide the no-op flush method required by text streams."""
        pass

    def isatty(self):
        # Helps tqdm behave as if it has a terminal-like sink.
        """Report terminal-like behavior for progress-bar compatibility."""
        return True


class TaskWorker(QObject):
    """Execute a callable in a worker thread and report output through Qt signals."""
    output = pyqtSignal(str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, func):
        """Initialize the object and its runtime state."""
        super().__init__()
        self.func = func

    def run(self):
        """Execute the task while forwarding stdout and stderr to the dialog.

        The callable's return value is emitted through ``result``. Unhandled
        exceptions are converted to traceback text and emitted through
        ``error``. Original streams are restored and ``finished`` is emitted in
        all cases.
        """
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        proxy = _StreamProxy(self.output.emit)

        try:
            sys.stdout = proxy
            sys.stderr = proxy
            result = self.func()
            self.result.emit(result)

        except Exception:
            self.error.emit(traceback.format_exc())

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.finished.emit()


class TaskOutputDialog(QDialog):
    """Display live output and completion status for a background task."""
    task_finished = pyqtSignal()

    def __init__(self, title="Running task", parent=None, *, auto_close_on_success=True, auto_close_delay_ms=1200):
        """Initialize ``TaskOutputDialog``, bind shared state and services, and create its controls."""
        super().__init__(parent)
        self.auto_close_on_success = bool(auto_close_on_success)
        self.auto_close_delay_ms = int(auto_close_delay_ms)

        self.setWindowTitle(title)
        self.resize(760, 380)
        self._running = True
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
            text = text.replace("\r", "\n")

        self.output_text.moveCursor(QTextCursor.End)
        self.output_text.insertPlainText(text)
        self.output_text.moveCursor(QTextCursor.End)

    def mark_success(self):
        """Return mark success."""
        self._running = False
        self.status_label.setText("Finished.")
        self.close_button.setEnabled(True)
        self.task_finished.emit()

        if self.auto_close_on_success:
            QTimer.singleShot(self.auto_close_delay_ms, self.close)

    def mark_error(self):
        """Return mark error."""
        self._running = False
        self.status_label.setText("Error.")
        self.close_button.setEnabled(True)
        self.task_finished.emit()

    def closeEvent(self, event):
        # No cancellation yet. While running, just hide instead of destroying.
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
):
    """Run func() in a QThread and show stdout/stderr in a TaskOutputDialog.

    on_success(result) and on_error(traceback_text) are called in the Qt thread.
    """
    dialog = TaskOutputDialog(
        title=title,
        parent=parent,
        auto_close_on_success=auto_close_on_success,
        auto_close_delay_ms=auto_close_delay_ms,
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
            """Return cleanup dialog ref."""
            try:
                dialogs.remove(dialog)
            except ValueError:
                pass

        dialog.destroyed.connect(cleanup_dialog_ref)

    def handle_result(result):
        """Forward a successful background-task result to the caller callback."""
        if on_success is not None:
            on_success(result)

    def handle_error(traceback_text):
        """Display a background-task traceback and invoke the error callback."""
        dialog.append_output("\n" + traceback_text + "\n")
        if on_error is not None:
            on_error(traceback_text)

    worker.output.connect(dialog.append_output)
    worker.result.connect(handle_result)
    worker.error.connect(handle_error)

    worker.result.connect(lambda _result: dialog.mark_success())
    worker.error.connect(lambda _traceback_text: dialog.mark_error())

    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    dialog.show()
    thread.start()

    return dialog
