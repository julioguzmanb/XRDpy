from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from trxrdpy.analysis import calibration
from trxrdpy.analysis.common import azimint_utils
from trxrdpy.analysis.common.azimint_utils import AzimIntegrator
from trxrdpy.analysis.common.plot_utils import DetectorCakePlotter


class _FakeIntegrate2dResult:
    def __init__(self):
        self.intensity = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
            ]
        )
        self.radial = np.array([1.0, 2.0, 3.0])
        self.azimuthal = np.array([-75.0, -25.0])


class _FakeAzimuthalIntegrator:
    def __init__(self):
        self.kwargs = None

    def integrate2d(self, image, **kwargs):
        self.image = np.asarray(image)
        self.kwargs = kwargs
        return _FakeIntegrate2dResult()


class _RangeAwareFakeAzimuthalIntegrator:
    def __init__(self):
        self.calls = []

    def integrate2d(self, image, **kwargs):
        self.calls.append(kwargs)
        start, stop = kwargs["azimuth_range"]
        bins = kwargs["npt_azim"]
        step = (stop - start) / bins

        result = _FakeIntegrate2dResult()
        result.radial = np.arange(kwargs["npt_rad"], dtype=float)
        result.azimuthal = start + step * (np.arange(bins) + 0.5)
        result.intensity = np.repeat(
            result.azimuthal[:, None],
            kwargs["npt_rad"],
            axis=1,
        )
        return result


def test_integrate2d_forwards_corrections_and_restores_display_azimuth():
    integrator = AzimIntegrator(
        npt=3,
        normalize=True,
        q_norm_range=(2.0, 3.0),
        azim_offset_deg=-90.0,
        polarization_factor=0.8,
    )
    fake_ai = _FakeAzimuthalIntegrator()
    integrator._ai = fake_ai
    integrator._mask = np.zeros((2, 2), dtype=np.uint8)

    cake, q, azimuth = integrator.integrate2d(
        np.ones((2, 2)),
        npt_rad=3,
        npt_azim=2,
        radial_range=(1.0, 3.0),
        azimuthal_range=(0.0, 90.0),
    )

    assert fake_ai.kwargs["npt_rad"] == 3
    assert fake_ai.kwargs["npt_azim"] == 2
    assert fake_ai.kwargs["unit"] == "q_A^-1"
    assert fake_ai.kwargs["azimuth_range"] == (-90.0, 0.0)
    assert fake_ai.kwargs["radial_range"] == (1.0, 3.0)
    assert fake_ai.kwargs["polarization_factor"] == 0.8
    assert fake_ai.kwargs["mask"] is integrator._mask

    np.testing.assert_allclose(q, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(azimuth, [15.0, 65.0])
    np.testing.assert_allclose(
        cake,
        [
            [0.4, 0.8, 1.2],
            [0.4, 0.8, 1.2],
        ],
    )


def test_integrate2d_wraps_full_circle_without_invalid_pyfai_range():
    integrator = AzimIntegrator(npt=2, normalize=False, azim_offset_deg=-90.0)
    fake_ai = _RangeAwareFakeAzimuthalIntegrator()
    integrator._ai = fake_ai
    integrator._mask = np.zeros((2, 2), dtype=np.uint8)

    cake, _q, azimuth = integrator.integrate2d(
        np.ones((2, 2)),
        npt_rad=2,
        npt_azim=4,
        azimuthal_range=(-180.0, 180.0),
    )

    assert len(fake_ai.calls) == 1
    assert fake_ai.calls[0]["azimuth_range"] == (-180.0, 180.0)
    np.testing.assert_allclose(azimuth, [-135.0, -45.0, 45.0, 135.0])
    np.testing.assert_allclose(cake[:, 0], [135.0, -135.0, -45.0, 45.0])


def test_integrate2d_accepts_no_mask_and_passes_none_to_pyfai():
    integrator = AzimIntegrator(npt=3, normalize=False, azim_offset_deg=-90.0)
    fake_ai = _FakeAzimuthalIntegrator()
    integrator._ai = fake_ai
    integrator._mask = None

    cake, _q, _azimuth = integrator.integrate2d(
        np.ones((2, 2)),
        npt_rad=3,
        npt_azim=2,
        azimuthal_range=(0.0, 90.0),
    )

    assert cake.shape == (2, 3)
    assert fake_ai.kwargs["mask"] is None


def test_mask_file_is_converted_to_pyfai_boolean_mask(monkeypatch):
    class FakeFabioImage:
        data = np.array([[0, 2], [-1, 0]], dtype=np.int16)

    monkeypatch.setattr(
        azimint_utils.fabio,
        "open",
        lambda _path: FakeFabioImage(),
    )
    integrator = AzimIntegrator(mask_edf_path="mask.edf")

    assert integrator._mask.dtype == np.bool_
    np.testing.assert_array_equal(
        integrator._mask,
        [[False, True], [True, False]],
    )


def test_integrate2d_stitches_partial_range_across_pyfai_boundary():
    integrator = AzimIntegrator(npt=2, normalize=False, azim_offset_deg=-90.0)
    fake_ai = _RangeAwareFakeAzimuthalIntegrator()
    integrator._ai = fake_ai
    integrator._mask = np.zeros((2, 2), dtype=np.uint8)

    cake, _q, azimuth = integrator.integrate2d(
        np.ones((2, 2)),
        npt_rad=2,
        npt_azim=8,
        azimuthal_range=(-120.0, 120.0),
    )

    assert [call["azimuth_range"] for call in fake_ai.calls] == [
        (150.0, 180.0),
        (-180.0, 30.0),
    ]
    assert cake.shape == (8, 2)
    assert np.all(np.diff(azimuth) > 0.0)
    assert azimuth[0] > -120.0
    assert azimuth[-1] < 120.0


def test_detector_cake_plotter_builds_two_physical_coordinate_panels(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(plt, "show", lambda: None)
    detector = np.arange(12, dtype=float).reshape(3, 4) + 1.0
    cake = np.arange(6, dtype=float).reshape(2, 3) + 1.0
    q = np.array([1.0, 2.0, 3.0])
    azimuth = np.array([-45.0, 45.0])

    fig, axes = DetectorCakePlotter().plot(
        detector,
        cake,
        q,
        azimuth,
        detector_log_scale=True,
        cake_log_scale=True,
        invert_detector_x=True,
        invert_detector_y=True,
        title="Calibration diagnostic",
        save=True,
        save_dir=tmp_path,
        save_name="detector-cake",
        save_overwrite=True,
    )

    assert len(axes) == 2
    assert axes[0].get_title() == "Detector image"
    assert axes[0].get_xlabel() == "x [px]"
    assert axes[0].get_xlim()[0] > axes[0].get_xlim()[1]
    assert axes[0].get_ylim()[0] > axes[0].get_ylim()[1]
    assert axes[1].get_title() == "2D cake"
    assert axes[1].get_xlabel() == "q [Å$^{-1}$]"
    assert axes[1].get_ylabel() == "Azimuth [deg]"
    assert (tmp_path / "detector-cake.png").is_file()
    plt.close(fig)


def test_detector_cake_plotter_rejects_transposed_cake(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    with np.testing.assert_raises_regex(ValueError, "cake_intensity shape"):
        DetectorCakePlotter().plot(
            np.ones((2, 2)),
            np.ones((3, 2)),
            np.array([1.0, 2.0, 3.0]),
            np.array([-30.0, 30.0]),
        )


def test_detector_default_y_direction_matches_pyfai_view(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    fig, axes = DetectorCakePlotter().plot(
        np.ones((2, 2)),
        np.ones((2, 2)),
        np.array([1.0, 2.0]),
        np.array([-30.0, 30.0]),
    )

    assert axes[0].get_ylim()[0] < axes[0].get_ylim()[1]
    plt.close(fig)


def test_public_calibration_api_connects_context_data_to_plotter(monkeypatch, tmp_path):
    detector = np.ones((2, 3))
    cake = np.ones((4, 5))
    q = np.linspace(1.0, 2.0, 5)
    azimuth = np.linspace(-75.0, 75.0, 4)

    class FakeContext:
        def compute_2d_cake(self, scan, **kwargs):
            self.scan = scan
            self.kwargs = kwargs
            return detector, cake, q, azimuth

        def analysis_dir(self, scan):
            return tmp_path

    context = FakeContext()
    monkeypatch.setattr(calibration, "_make_context", lambda **kwargs: context)

    class FakePlotter:
        def plot(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return "figure", "axes"

    plotter = FakePlotter()
    monkeypatch.setattr(
        calibration.plot_utils,
        "DetectorCakePlotter",
        lambda: plotter,
    )

    fig, axes, data = calibration.plot_detector_and_cake(
        "sample",
        7,
        110,
        npt_rad=5,
        npt_azim=4,
        radial_range=(1.0, 2.0),
        azimuthal_range=(-90.0, 90.0),
        use_mask=False,
        invert_detector_x=True,
        invert_detector_y=True,
        save=False,
    )

    assert (fig, axes) == ("figure", "axes")
    assert context.scan == 7
    assert context.kwargs["npt_rad"] == 5
    assert context.kwargs["npt_azim"] == 4
    assert context.kwargs["use_mask"] is False
    assert plotter.args[0] is detector
    assert plotter.args[1] is cake
    assert plotter.args[2] is q
    assert plotter.args[3] is azimuth
    assert plotter.kwargs["invert_detector_x"] is True
    assert plotter.kwargs["invert_detector_y"] is True
    assert data["detector_image"] is detector
    assert data["cake_intensity"] is cake
