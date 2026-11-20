"""Separate numerical work from GUI-thread-only figure creation."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_PLOTS = ContextVar('simulation_deferred_plots', default=None)


@contextmanager
def capture_plots():
    calls = []
    token = _PLOTS.set(calls)
    try:
        yield calls
    finally:
        _PLOTS.reset(token)


def gui_plot(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        calls = _PLOTS.get()
        if calls is not None:
            calls.append((function, args, kwargs))
            return None
        return function(*args, **kwargs)
    return wrapped
