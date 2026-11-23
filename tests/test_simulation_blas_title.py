"""The detector title must not start a parallel LAPACK solve in a GUI worker."""
import numpy as np

from trxrdpy.simulation import utils
from trxrdpy.simulation.detector import Detector
from trxrdpy.simulation.experiment import Experiment
from trxrdpy.simulation.sample import LatticeStructure


def test_relative_orientation_matches_solve_for_nonorthogonal_cell():
    initial = np.array([[2., 1., 0.], [0., 3., 1.], [1., 0., 4.]])
    current = initial @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    expected = np.linalg.solve(initial, current).T
    np.testing.assert_allclose(utils.relative_orientation(initial, current), expected,
                               atol=1e-14)


def test_detector_title_avoids_lapack_solve(monkeypatch):
    detector = Detector(detector_type="manual", pxsize_h=0.001, pxsize_v=0.001,
                        num_pixels_h=100, num_pixels_v=100, dist=0.1,
                        poni1=0.05, poni2=0.05)
    lattice = LatticeStructure(space_group=1, a=4, b=5, c=6,
                               alpha=90, beta=90, gamma=75)
    lattice.apply_rotation(roty=12)
    exp = Experiment(detector, lattice, energy=10000, e_bandwidth=1.5)

    def forbidden(*args, **kwargs):
        raise AssertionError("A tiny title matrix must not use LAPACK")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    assert "Detector:" in exp._detector_title()
