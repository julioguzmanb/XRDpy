import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import experiment, single_crystal, utils
from trxrdpy.simulation.detector import Detector
from trxrdpy.simulation.geometry import DiffractometerGeometry, MotorChain
from trxrdpy.simulation.gui.state import GuiState
from trxrdpy.simulation.gui.services import SimulationService
from trxrdpy.simulation.gui.services.background_simulation import scan_result
from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
from trxrdpy.simulation.results import save_result, load_result


def detector(**kwargs):
    values = dict(detector_type='manual', pxsize_h=.001, pxsize_v=.001,
                  num_pixels_h=60, num_pixels_v=60, dist=.1, poni1=.03, poni2=.03,
                  rotx=0, roty=0, rotz=0)
    return Detector(**{**values, **kwargs})


def q_for_rays(rays, wavelength=1e-10):
    rays = np.asarray(rays, dtype=float)
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    return (rays - [1, 0, 0]) * (2 * np.pi / (wavelength * 1e10))


def test_hit_filter_rejects_reverse_parallel_and_outside_rays():
    q = q_for_rays([[1, 0, 0], [1, 1, 0], [-1, 0, 0], [0, 1, 0]])
    np.testing.assert_equal(experiment.detector_hit_mask(q, 1e-10, detector()),
                            [True, False, False, False])


@pytest.mark.parametrize('binning', [(1, 1), (2, 2)])
def test_transformed_detector_bounds_and_forward_direction(binning):
    det = detector(rotx=23, roty=-31, rotz=17, binning=binning)
    transform = det.get_transform()
    transform[:3, 3] = [.02, -.01, .03]
    det.set_transform(transform)
    pixels = np.array([[2, 3], [-.1, 3], [det.num_pixels_h + .1, 3],
                       [2, det.num_pixels_v + .1], [2, -.1]])
    native = np.column_stack((np.full(5, det.dist),
                              det.poni2 - (pixels[:, 0] + .5) * det.pxsize_h,
                              -det.poni1 + (pixels[:, 1] + .5) * det.pxsize_v))
    rays = utils.apply_transform(native, transform)
    np.testing.assert_equal(experiment.detector_hit_mask(q_for_rays(rays), 1e-10, det),
                            [True, False, False, False, False])
    assert not experiment.detector_hit_mask(q_for_rays(-rays[:1]), 1e-10, det)[0]


def parameters():
    return dict(param1_name='energy', param2_name='rotz', param1_range=(9000, 15000, 3000),
                param2_range=(75, 88, .25), sam_space_group=221, sam_a=4, sam_b=4, sam_c=4,
                sam_alpha=90, sam_beta=90, sam_gamma=90,
                hkls_names=[[-1, 0, 0]], e_bandwidth=10)


def test_optional_filter_subset_energy_and_parallel(monkeypatch):
    def no_grid(*args):
        raise AssertionError('Filtering must not allocate a detector image grid.')
    monkeypatch.setattr(Detector, 'calculate_lab_grid', no_grid)
    kwargs = parameters()
    unfiltered = single_crystal.scan_two_parameters_for_Bragg_condition(**kwargs)
    assert unfiltered == single_crystal.scan_two_parameters_for_Bragg_condition(
        **kwargs, require_detector_hit=False, detector_obj=detector(poni2=10))
    options = dict(require_detector_hit=True, detector_obj=detector())
    filtered = single_crystal.scan_two_parameters_for_Bragg_condition(**kwargs, **options)
    key = next(iter(filtered))
    assert 0 < len(filtered[key]) < len(unfiltered[key])
    assert set(filtered[key]) < set(unfiltered[key])
    assert len({p[0] for p in filtered[key]}) >= 2
    assert filtered == single_crystal.scan_two_parameters_for_Bragg_condition(
        **kwargs, **options, parallel=True, workers=2)
    missed = single_crystal.scan_two_parameters_for_Bragg_condition(
        **kwargs, require_detector_hit=True, detector_obj=detector(poni2=10))
    assert not any(missed.values())
    with pytest.raises(ValueError, match='detector_obj is required'):
        single_crystal.scan_two_parameters_for_Bragg_condition(**kwargs, require_detector_hit=True)


def test_filter_uses_each_scan_energy(monkeypatch):
    wavelengths = []
    def accept(q, wavelength, det):
        wavelengths.append(wavelength)
        return np.ones(len(q), dtype=bool)
    monkeypatch.setattr(experiment, 'detector_hit_mask', accept)
    kwargs = parameters()
    expected = single_crystal.scan_two_parameters_for_Bragg_condition(**kwargs)
    actual = single_crystal.scan_two_parameters_for_Bragg_condition(
        **kwargs, require_detector_hit=True, detector_obj=detector())
    assert actual == expected
    np.testing.assert_allclose(sorted(set(wavelengths)),
                               sorted(utils.energy_to_wavelength(e) for e in (9000, 12000, 15000)))


def test_chain_filter_geometry_pose_and_parallel():
    chain = MotorChain.from_list([dict(name='omega', axis='z'), dict(name='chi', axis='y')])
    kwargs = dict(motor1_name='omega', motor2_name='chi', motor1_range=(82, 86, .5),
                  motor2_range=(-5, 5, 5), sam_space_group=221, sam_a=4, sam_b=4, sam_c=4,
                  sam_alpha=90, sam_beta=90, sam_gamma=90, energy=15000,
                  hkls_names=[[-1, 0, 0]], e_bandwidth=10, motor_chain=chain)
    plain = single_crystal.scan_two_motors_for_Bragg_condition(**kwargs)
    filtered = single_crystal.scan_two_motors_for_Bragg_condition(
        **kwargs, require_detector_hit=True, detector_obj=detector())
    assert any(filtered.values())
    assert all(set(filtered[k]) <= set(plain[k]) for k in plain)
    assert filtered == single_crystal.scan_two_motors_for_Bragg_condition(
        **kwargs, require_detector_hit=True, detector_obj=detector(), parallel=True, workers=2)
    geometry = DiffractometerGeometry.from_dict(dict(
        sample=chain, detector=[dict(name='arm', axis='y')]))
    det = detector()
    before = det.get_transform().copy()
    away = single_crystal.scan_two_motors_for_Bragg_condition(
        **kwargs, geometry=geometry, detector_angles={'arm': 180},
        require_detector_hit=True, detector_obj=det)
    assert not any(away.values())
    np.testing.assert_equal(det.get_transform(), before)


def test_gui_detector_filter_state_geometry_and_result_metadata(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    def fail_dialog(*args):
        raise AssertionError(str(args))
    monkeypatch.setattr(QMessageBox, 'critical', fail_dialog)
    app = QApplication.instance() or QApplication([])
    state = GuiState.from_dict({'state_version': 13, 'single': {}})
    assert not state.single.require_detector_hit
    tab = SingleCrystalTab(SimulationService(), state)
    tab.single_func_combo.setCurrentText('scan_two_parameters_for_Bragg_condition')
    tab.single_param2_combo.setCurrentText('rotz')
    tab.single_line_names.setText('[-1,0,0]')
    assert not tab.require_detector_hit.isChecked()
    tab.require_detector_hit.setChecked(True)
    tab.single_line_rotz.setText('30')
    assert not tab.single_det_group.isHidden()
    assert not tab.single_manual_group.isHidden()
    request = tab._single_run_function_impl()
    assert request.kwargs['require_detector_hit'] is False
    assert request.kwargs['detector_obj'] is None
    restored = GuiState.from_dict(state.to_dict())
    assert restored.single.require_detector_hit
    tab.require_detector_hit.setChecked(False)
    tab.state = restored
    tab.load_from_state()
    assert tab.require_detector_hit.isChecked()
    result = scan_result(request, {'[-1 0 0]': [(9000, 80)]},
                         {'function': 'scan_two_parameters_for_Bragg_condition'})
    saved = load_result(save_result(tmp_path / 'filtered.h5', result))
    assert saved.metadata['require_detector_hit'] is False
    assert saved.metadata['points_are_complete'] is True
    tab.single_geometry_mode_combo.setCurrentText('Custom geometry')
    tab.single_custom_sample_chain.setPlainText('[{"name":"omega","axis":"z"},{"name":"chi","axis":"y"}]')
    tab.single_geometry_motor1_name.setText('omega')
    tab.single_geometry_motor2_name.setText('chi')
    request = tab._single_run_function_impl()
    assert request.kwargs['require_detector_hit'] is False
    assert request.kwargs['detector_obj'] is None
    tab.close()
    app.processEvents()
