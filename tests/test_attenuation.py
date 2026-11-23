"""Physics, data, and CXRO benchmark checks for offline attenuation."""
import numpy as np
import pytest

from trxrdpy.simulation.attenuation import (XRayMaterial, atomic_factors,
                                            nb_fraction_from_weight_fraction,
                                            parse_formula)
from trxrdpy.simulation.attenuation_plot import (AttenuationData, load_attenuation_csv,
                                                 load_attenuation_h5, plot_attenuation,
                                                 save_attenuation_csv, save_attenuation_h5,
                                                 stepped_range)


def test_composition_and_density_scaling():
    assert parse_formula("SrTi0.9901Nb0.0099O3") == {"Sr": 1, "Ti": .9901, "Nb": .0099, "O": 3}
    assert nb_fraction_from_weight_fraction(.005) == pytest.approx(.009898671, rel=1e-6)
    a = XRayMaterial("SrTiO3", 5.12)
    b = XRayMaterial("SrTiO3", 10.24)
    for original, doubled in zip(a.optical_constants([13, 15.5]), b.optical_constants([13, 15.5])):
        np.testing.assert_allclose(doubled, 2 * original)


def test_bulk_and_grazing_relations():
    material = XRayMaterial("SrTiO3", 5.12)
    energy = np.array([10., 13., 15.5])
    bulk = material.bulk_attenuation_length(energy, unit="m")
    mu = material.attenuation_coefficient(energy)
    np.testing.assert_allclose(bulk * mu, 1.)
    np.testing.assert_allclose(material.attenuation_coefficient(energy, unit="cm^-1"), mu / 100)
    np.testing.assert_allclose(material.grazing_penetration_depth(energy, 90, unit="m"), bulk, rtol=2e-5)
    np.testing.assert_allclose(material.grazing_penetration_depth(energy, 20, unit="m"),
                               bulk * np.sin(np.deg2rad(20)), rtol=.001)
    assert np.all(np.isfinite(material.grazing_penetration_depth(energy, 0)))
    assert np.all(material.grazing_penetration_depth(energy, 0) > 0)
    critical_deg = np.rad2deg(np.sqrt(2 * material.optical_constants(13)[0]))
    for angle in (critical_deg * .99, critical_deg, critical_deg * 1.01):
        assert np.isfinite(material.grazing_penetration_depth(13, angle))
    assert material.grazing_depth_from_rot_y(13, -1) == pytest.approx(material.grazing_depth_from_rot_y(13, 1))
    assert material.bulk_attenuation_length(13, unit="um") == pytest.approx(float(material.bulk_attenuation_length(np.array([13]), unit="um")[0]))


def test_cxro_reference_and_approximation():
    material = XRayMaterial("SrTi0.9901Nb0.0099O3", 5.175)
    delta, beta = material.optical_constants([13., 15.5])
    np.testing.assert_allclose(delta, [5.75478771e-6, 3.97494159e-6], rtol=2e-5)
    np.testing.assert_allclose(beta, [1.12482716e-7, 5.73427599e-8], rtol=.007)
    for energy, angle, reference_um in [(13, 1, 1.15498), (13, 11.25, 13.1613),
                                         (15.5, 1, 1.91174), (15.5, 11.25, 21.6540)]:
        approx = material.grazing_penetration_depth(energy, angle, unit="um")
        exact = material.grazing_penetration_depth(energy, angle, unit="um", exact=True)
        assert approx == pytest.approx(reference_um, rel=.01)
        assert exact == pytest.approx(approx, rel=2e-5)


def test_edges_and_validation():
    # The Nb table includes closely spaced edge points. Interpolation uses them.
    from trxrdpy.simulation.attenuation import _read_table
    from importlib.resources import files
    path = str(files("trxrdpy.simulation") / "data" / "henke" / "nb.nff")
    energies, f1, f2 = _read_table(path)
    valid_pairs = np.flatnonzero((f1[:-1] > -9990) & (f1[1:] > -9990))
    idx = valid_pairs[np.argmin(np.diff(energies)[valid_pairs])]
    mid = (energies[idx] + energies[idx+1]) / 2000
    got = atomic_factors("Nb", mid)
    assert got[0] == pytest.approx((f1[idx] + f1[idx+1]) / 2)
    assert got[1] == pytest.approx((f2[idx] + f2[idx+1]) / 2)
    for invalid in (-1, 0, 40, np.nan):
        with pytest.raises(ValueError):
            atomic_factors("Nb", invalid)
    with pytest.raises(ValueError):
        XRayMaterial("SrTiO3", -1)
    with pytest.raises(ValueError):
        XRayMaterial("SrTiO3", 5).grazing_penetration_depth(13, -1)
    with pytest.raises(ValueError):
        parse_formula("Sr(Ti)O3")


def test_all_cxro_elements_and_vanadium_oxide():
    from pathlib import Path
    root = Path(__file__).parents[1] / "src/trxrdpy/simulation/data/henke"
    tables = list(root.glob("*.nff"))
    assert len(tables) == 92
    for table in tables:
        f1, f2 = atomic_factors(table.stem.capitalize(), 13.)
        assert np.isfinite(f1) and f2 >= 0
    assert XRayMaterial("V2O3", 4.87).grazing_penetration_depth(13, 1) > 0


def test_eV_step_angle_scan_csv_and_slider(tmp_path):
    energy = stepped_range(10000, 10020, 10)
    angles = stepped_range(0, 1, .5)
    material = XRayMaterial("V2O3", 4.87)
    length = material.grazing_penetration_depth(energy[None, :] / 1000,
                                                angles[:, None], unit="m")
    data = AttenuationData(energy, length, angles, "grazing", "µm", "V2O3", 4.87)
    path = tmp_path / "attenuation.csv"
    save_attenuation_csv(path, data)
    loaded = load_attenuation_csv(path)
    np.testing.assert_array_equal(loaded.energy_eV, energy)
    np.testing.assert_array_equal(loaded.angles_deg, angles)
    np.testing.assert_allclose(loaded.length, length)
    assert loaded.unit == "µm"
    lines = path.read_text().splitlines()
    assert lines[0].startswith("# trxrdpy-attenuation-v2:")
    assert lines[1] == "photon_energy_eV,incidence_angle_deg,attenuation_length_m"
    assert "display_unit" not in path.read_text()
    selected = []
    ax = plot_attenuation(loaded, on_angle_change=lambda angle, curve: selected.append(angle), show=False)
    ax._attenuation_angle_box.set_val("0.75")
    assert selected[-1] == pytest.approx(.75)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), loaded.display_curve_at(.75))
    assert ax.get_xlabel() == "Photon energy [eV]"
    assert "Incidence angle: 0.75°" in ax.get_title()
    assert "—" not in ax.get_title()
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_fine_step_scan_is_lazy():
    energy = stepped_range(10000, 15000, 1)
    angles = stepped_range(0, 5, .001)
    material = XRayMaterial("V2O3", 4.87)
    from trxrdpy.simulation.attenuation import grazing_depth_from_constants
    delta, beta = material.optical_constants(energy / 1000)
    calculator = lambda angle: grazing_depth_from_constants(energy / 1000, delta, beta, angle, unit="m")
    data = AttenuationData(energy, None, angles, "grazing", "µm", "V2O3", 4.87, calculator)
    assert len(energy) * len(angles) > 1_000_000
    assert data.length is None
    np.testing.assert_allclose(data.curve_at(1.2345), calculator(1.2345))


def test_legacy_csv_units_are_converted_to_meters(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("photon_energy_eV,incidence_angle_deg,attenuation_length_µm,mode,formula,density_g_cm3\n"
                    "10000,1,2,grazing,V2O3,4.87\n"
                    "10010,1,3,grazing,V2O3,4.87\n", encoding="utf-8")
    data = load_attenuation_csv(path)
    np.testing.assert_allclose(data.length[0], [2e-6, 3e-6])
    np.testing.assert_allclose(data.display_curve_at(1), [2, 3])


def test_energy_motor_shows_depth_versus_angle():
    energy = np.array([10000., 10010., 10020.])
    angles = np.array([.5, 1., 1.5])
    material = XRayMaterial("V2O3", 4.87)
    length = material.grazing_penetration_depth(energy[None, :] / 1000,
                                                angles[:, None], unit="m")
    data = AttenuationData(energy, length, angles, "grazing", "nm", "V2O3", 4.87)
    ax = plot_attenuation(data, slider_motor="energy", show=False)
    assert ax.get_xlabel() == "Incidence angle [°]"
    ax._attenuation_angle_box.set_val("10015")
    np.testing.assert_allclose(ax.lines[0].get_ydata(), data.display_curve_at_energy(10015))
    assert "Photon energy: 10015 eV" in ax.get_title()
    previous, following = ax._attenuation_step_buttons
    assert ax._attenuation_slider.ax.get_position().width >= .42 - 1e-9
    assert ax._attenuation_angle_box.ax.get_position().width >= .15 - 1e-9
    assert ax._attenuation_slider.ax.get_position().x1 < previous.ax.get_position().x0
    assert following.ax.get_position().x1 < ax._attenuation_angle_box.ax.get_position().x0
    following._observers.process('clicked', None)
    assert ax._attenuation_angle_box.text == '10020'
    assert not following.get_active()
    previous._observers.process('clicked', None)
    assert ax._attenuation_angle_box.text == '10010'
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_hdf5_round_trip_and_display_unit_is_not_persisted(tmp_path):
    import h5py
    energy = np.array([10000., 10010., 10020.])
    angles = np.array([.5, 1., 1.5])
    material = XRayMaterial("V2O3", 4.87)
    delta, beta = material.optical_constants(energy / 1000)
    from trxrdpy.simulation.attenuation import grazing_depth_from_constants
    calculator = lambda angle: grazing_depth_from_constants(energy / 1000, delta, beta, angle, unit="m")
    data = AttenuationData(energy, None, angles, "grazing", "nm", "V2O3", 4.87, calculator)
    path = tmp_path / "attenuation.h5"
    save_attenuation_h5(path, data)
    with h5py.File(path, "r") as handle:
        assert set(handle) == {"photon_energy_eV", "incidence_angle_deg", "attenuation_length_m"}
        assert "display_unit" not in handle.attrs
        assert handle.attrs["formula"] == "V2O3"
        assert handle["attenuation_length_m"].compression == "gzip"
        assert handle["attenuation_length_m"].fletcher32
    loaded = load_attenuation_h5(path, display_unit="µm")
    assert loaded.length is None
    assert loaded.unit == "µm"
    np.testing.assert_allclose(loaded.curve_at(1), calculator(1))
    np.testing.assert_allclose(loaded.curve_at_energy(10010),
                               [calculator(angle)[1] for angle in angles])
    loaded._h5_finalizer()


@pytest.mark.parametrize("extension,save", [("csv", save_attenuation_csv),
                                            ("h5", save_attenuation_h5)])
def test_failed_export_preserves_existing_file(tmp_path, extension, save):
    path = tmp_path / f"scan.{extension}"
    path.write_bytes(b"previous complete export")
    calls = []

    def failing_curve(angle):
        calls.append(angle)
        if len(calls) == 2:
            raise RuntimeError("calculation interrupted")
        return np.array([1e-6, 2e-6])

    data = AttenuationData(np.array([10000., 10010.]), None, np.array([1., 2.]),
                           "grazing", "µm", "SrTiO3", 5.12, failing_curve)
    with pytest.raises(RuntimeError, match="interrupted"):
        save(path, data)
    assert path.read_bytes() == b"previous complete export"
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_lazy_curve_rejects_nonfinite_output():
    data = AttenuationData(np.array([10000.]), None, np.array([1.]),
                           "grazing", "µm", "SrTiO3", 5.12,
                           lambda angle: np.array([np.nan]))
    with pytest.raises(ValueError, match="nonpositive/nonfinite"):
        data.curve_at(1)
