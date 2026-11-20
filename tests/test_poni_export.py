from __future__ import annotations

import pyFAI
import pytest

from trxrdpy.simulation.poni import read_poni_file, write_poni_file
from trxrdpy.simulation.utils import energy_to_wavelength


def geometry_values():
    return dict(dist=0.1040658008471512, poni1=0.056283783395496195,
                poni2=0.11760363022346329, rot1=-0.0223, rot2=0.5306, rot3=0.12,
                pixel1=75e-6, pixel2=100e-6, max_shape=(831, 1475), energy_ev=15000)


@pytest.mark.parametrize("orientation", [1, 2, 3, 4])
def test_export_roundtrip_with_package_and_pyfai(tmp_path, orientation):
    values = geometry_values()
    path = write_poni_file(tmp_path / "manual.poni", **values, orientation=orientation)
    parsed = read_poni_file(path)
    native = pyFAI.load(str(path))
    assert parsed.version == "2.1"
    assert parsed.detector == "Detector"
    assert parsed.max_shape == (831, 1475)
    assert native.detector.max_shape == (831, 1475)
    assert int(native.detector.orientation) == orientation
    for name in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3", "pixel1", "pixel2"):
        assert getattr(parsed, name) == values[name]
        assert getattr(native, name) == values[name]
    assert parsed.wavelength == energy_to_wavelength(15000)
    assert native.wavelength == parsed.wavelength
    assert not list(tmp_path.glob(".poni-*.tmp"))


def test_export_with_wavelength_and_extension(tmp_path):
    values = geometry_values()
    values.pop("energy_ev")
    path = write_poni_file(tmp_path / "manual", **values, wavelength=1.25e-10)
    assert path.name == "manual.poni"
    assert read_poni_file(path).wavelength == 1.25e-10


def test_export_requires_explicit_overwrite_and_never_appends(tmp_path):
    path = tmp_path / "existing.poni"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        write_poni_file(path, **geometry_values())
    assert path.read_text() == "original"
    write_poni_file(path, **geometry_values(), overwrite=True)
    write_poni_file(path, **geometry_values(), overwrite=True)
    assert path.read_text().count("Distance:") == 1
    assert not list(tmp_path.glob(".poni-*.tmp"))


@pytest.mark.parametrize("changes", [
    {"dist": 0}, {"pixel1": -1}, {"pixel2": float("nan")}, {"rot2": float("inf")},
    {"energy_ev": 0}, {"energy_ev": None}, {"wavelength": 1e-10},
    {"max_shape": (0, 10)}, {"max_shape": (4.5, 10)}, {"max_shape": ("invalid", 10)},
    {"orientation": 5}, {"max_shape": (10,)},
])
def test_export_rejects_invalid_input_without_touching_existing_file(tmp_path, changes):
    path = tmp_path / "existing.poni"
    path.write_text("original")
    values = geometry_values()
    values.update(changes)
    with pytest.raises(ValueError):
        write_poni_file(path, **values, overwrite=True)
    assert path.read_text() == "original"
    assert not list(tmp_path.glob(".poni-*.tmp"))


def test_export_rejects_non_poni_extension(tmp_path):
    path = tmp_path / "settings.json"
    with pytest.raises(ValueError, match="extension"):
        write_poni_file(path, **geometry_values())
    assert not path.exists()
