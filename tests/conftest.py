import time

import pytest


@pytest.fixture
def wait_simulation():
    def wait(tab, timeout=15):
        from PyQt5.QtWidgets import QApplication
        deadline = time.monotonic() + timeout
        def busy():
            return getattr(tab, '_simulation_busy', False) or getattr(tab, '_mapping_plot_busy', False)
        while busy() and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(.005)
        assert not busy(), 'Simulation worker did not finish.'
        QApplication.processEvents()
    return wait
