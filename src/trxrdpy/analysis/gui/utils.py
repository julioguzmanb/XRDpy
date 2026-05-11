"""
Small utility functions used by the analysis GUI.

These functions are copied from the legacy GUI behavior.
"""

import ast

import numpy as np
import re


def pretty_literal(obj) -> str:
    return repr(obj)


def parse_python_literal(text, *, empty=None):
    if text is None:
        return empty

    if not isinstance(text, str):
        return text

    text = text.strip()

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

def parse_int_like(text, *, name: str) -> int:
    if text is None:
        text = ""

    text = str(text).strip()

    if text == "":
        raise ValueError(f"{name} cannot be empty.")

    return int(float(text))

def parse_float_like(text, *, name: str) -> float:
    if text is None:
        text = ""

    text = str(text).strip()

    if text == "":
        raise ValueError(f"{name} cannot be empty.")

    return float(text)

def parse_optional_float_like(text):
    if text is None:
        return None

    text = str(text).strip()

    if text == "":
        return None

    return float(text)

def parse_optional_int_like(text):
    if text is None:
        return None

    text = str(text).strip()

    if text == "":
        return None

    return int(float(text))

def parse_scan_spec(text):
    if text is None:
        raise ValueError("scan / scan_spec cannot be empty.")

    if isinstance(text, str) and text.strip() == "":
        raise ValueError("scan / scan_spec cannot be empty.")

    return parse_python_literal(text)

def parse_tuple2(text, *, name: str, cast=float):
    value = parse_python_literal(text)

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a tuple/list with two values, e.g. (-90, 90).")

    return cast(value[0]), cast(value[1])

def parse_optional_tuple2(text, *, name: str, cast=float):
    if text is None:
        return None

    if isinstance(text, str):
        stripped = text.strip()
        if stripped == "" or stripped.lower() == "none":
            return None

    return parse_tuple2(text, name=name, cast=cast)

def parse_edges(text):
    if text is None:
        raise ValueError("Azimuthal edges cannot be empty.")

    if not isinstance(text, str):
        arr = np.asarray(text, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("Azimuthal edges must contain at least two numeric values.")
        return arr

    text = text.strip()

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
        return [tuple(float(item) for item in value)]

    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "Azimuth windows must be a list of tuples, e.g. [(-90,90), (-75,-45)]."
        )

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


def parse_ref_value(text):
    """
    Parse reference values used by viewer/differential/fitting actions.

    Accepted examples:
        ""
        [1466556]
        -10e6
        -10ns
        -10 ns
        [-10ns]
        [-10ns, 0ns, 5ps]
        "-5ns"

    Unit-suffixed time values are converted to femtoseconds.
    """
    time_unit_to_fs = {
        "fs": 1.0,
        "fsec": 1.0,
        "femtosecond": 1.0,
        "femtoseconds": 1.0,
        "ps": 1.0e3,
        "psec": 1.0e3,
        "picosecond": 1.0e3,
        "picoseconds": 1.0e3,
        "ns": 1.0e6,
        "nsec": 1.0e6,
        "nanosecond": 1.0e6,
        "nanoseconds": 1.0e6,
        "us": 1.0e9,
        "µs": 1.0e9,
        "μs": 1.0e9,
        "usec": 1.0e9,
        "microsecond": 1.0e9,
        "microseconds": 1.0e9,
        "ms": 1.0e12,
        "msec": 1.0e12,
        "millisecond": 1.0e12,
        "milliseconds": 1.0e12,
        "s": 1.0e15,
        "sec": 1.0e15,
        "second": 1.0e15,
        "seconds": 1.0e15,
    }

    def normalize_number(value):
        value = float(value)
        if value.is_integer():
            return int(value)
        return value

    def convert_scalar(value):
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return normalize_number(value)

        if not isinstance(value, str):
            return value

        s = value.strip()

        if s == "" or s.lower() == "none":
            return None

        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1].strip()

        match = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-zµμ]+)",
            s,
        )

        if match:
            number = float(match.group(1))
            unit = match.group(2).lower()

            if unit not in time_unit_to_fs:
                raise ValueError(f"Unsupported ref_value time unit: {unit}")

            return normalize_number(number * time_unit_to_fs[unit])

        try:
            return normalize_number(s)
        except Exception:
            return s

    if text is None:
        return None

    if not isinstance(text, str):
        if isinstance(text, (list, tuple)):
            return [convert_scalar(item) for item in text]
        return convert_scalar(text)

    s = text.strip()

    if s == "" or s.lower() == "none":
        return None

    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [parse_ref_value(part.strip()) for part in inner.split(",") if part.strip()]

    value = parse_python_literal(s)

    if isinstance(value, (list, tuple)):
        return [convert_scalar(item) for item in value]

    return convert_scalar(value)
