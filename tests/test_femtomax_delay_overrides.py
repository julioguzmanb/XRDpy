from copy import deepcopy

import h5py
import numpy as np
import pytest

from trxrdpy.analysis.MaxIV_FemtoMAX import datared, datared_utils
from trxrdpy.analysis.MaxIV_FemtoMAX import single_shot_azimint
from trxrdpy.analysis.common.paths import AnalysisPaths
from trxrdpy.analysis.gui.services.preparation_service import PreparationService


@pytest.fixture
def setup_scans(tmp_path):
    paths = AnalysisPaths(tmp_path, raw_subdir="raw", analysis_subdir="analysis")
    paths.raw_root.mkdir()
    for scan in (10, 11, 12):
        with h5py.File(paths.raw_root / f"scan-{scan}.h5", "w") as handle:
            handle.create_dataset("entry/measurement/pilatus/data", data=np.full((3, 2, 2), float(scan)))
            if scan == 10:
                for channel in (2, 4):
                    handle.create_dataset(f"entry/measurement/oscc_02_maui/Ping_Ch{channel}_value", data=[0, 10e-15, np.nan])
    refs = tmp_path / "refs.csv"
    refs.write_text("scan_start,scan_end,ping2_ref_s,ping4_ref_s\n10,10,0,0\n")
    return paths, refs


def build(paths, refs=None, scans=(10, 11, 12), overrides=None, **kwargs):
    exp = datared.create_h5_files(
        scans, "sample", 300, 1500, 5, 100,
        paths=paths, ping_reference_path=refs,
        scan_delay_overrides_fs=overrides, **kwargs,
    )
    return exp, next(paths.analysis_root.rglob("*.h5"))


def test_mixed_scans_preserve_standard_metadata_and_real_ping_diagnostics(setup_scans):
    paths, refs = setup_scans
    _, path = build(paths, refs, overrides={11: 200000000, 12: 200000000})
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle["meta/selected_delays_fs"][:], [5, 200000000])
        regular = handle["delays/5fs/scans/10"]
        np.testing.assert_array_equal(regular["indices"][:], [0, 1])
        assert regular.attrs["timing_source"] == "ping"
        forced = handle["delays/200000000fs"]
        assert forced.attrs["nshots_total_expected"] == 6
        for scan in (11, 12):
            group = forced[f"scans/{scan}"]
            np.testing.assert_array_equal(group["indices"][:], [0, 1, 2])
            assert group.attrs["timing_source"] == "manual"
            assert group.attrs["assigned_delay_fs"] == 200000000
        assert forced["delays_pings2_fs"].size == 0
        np.testing.assert_array_equal(handle["meta/ping_reference_scans"][:], [10])
    metadata, records = single_shot_azimint._metadata_records(path)
    assert len(records) == 8
    assert {(r.scan, r.delay_fs) for r in records} == {(10, 5), (11, 200000000), (12, 200000000)}
    assert metadata["scan_type"] == "delay"


def test_all_forced_never_load_references_or_read_pings(setup_scans, monkeypatch):
    paths, _ = setup_scans
    def unexpected(*args, **kwargs):
        pytest.fail("Forced timing must not access pings or references")
    monkeypatch.setattr(datared_utils, "load_ping_reference_table", unexpected)
    monkeypatch.setattr(datared_utils.Experiment, "read_corrected_pings_seconds", unexpected)
    exp, path = build(paths, scans=[11, 12, 11], overrides={11: 200000000, 12: 200000001})
    with h5py.File(path) as handle:
        for delay, scan in ((200000000, 11), (200000001, 12)):
            group = handle[f"delays/{delay}fs"]
            assert list(group["scans"]) == [str(scan)]
            assert group.attrs["nshots_total_expected"] == 3
    np.testing.assert_allclose(exp.get_delays(11, unit="ns"), [200] * 3)
    exp.validate_metadata_ping_references(path)


def test_manual_points_extend_explicit_selection_without_neighbor_assignment(setup_scans):
    paths, refs = setup_scans
    _, path = build(paths, refs, overrides={11: 200000000, 12: 200000001}, selected_delays=[0])
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle["meta/selected_delays_fs"][:], [0, 200000000, 200000001])
        assert list(handle["delays/200000000fs/scans"]) == ["11"]


def test_changed_overrides_reject_cached_metadata(setup_scans):
    paths, refs = setup_scans
    exp, path = build(paths, refs, overrides={11: 200000000, 12: 200000000})
    exp.validate_metadata_ping_references(path)
    for overrides in (None, {11: 300000000, 12: 200000000}):
        other = datared_utils.Experiment([10, 11, 12], paths=paths, ping_reference_path=refs, scan_delay_overrides_fs=overrides)
        with pytest.raises(ValueError, match="different scan delay overrides"):
            other.validate_metadata_ping_references(path)


@pytest.mark.parametrize("forced_delay,regular_count", [(1, 2), (0, 5)])
def test_overrides_do_not_change_regular_bin_selection(setup_scans, forced_delay, regular_count):
    paths, refs = setup_scans
    _, path = build(paths, refs, scans=[10, 11], overrides={11: forced_delay}, selected_delays=[0])
    with h5py.File(path) as handle:
        assert handle["delays/0fs"].attrs["nshots_total_expected"] == regular_count
        if forced_delay:
            assert list(handle[f"delays/{forced_delay}fs/scans"]) == ["11"]


def test_old_metadata_without_override_fields_still_loads(setup_scans):
    paths, refs = setup_scans
    exp, path = build(paths, refs, scans=[10])
    with h5py.File(path, "a") as handle:
        del handle["meta/scan_delay_override_scans"]
        del handle["meta/scan_delay_override_values_fs"]
    exp.validate_metadata_ping_references(path)


@pytest.mark.parametrize("parallel", [False, True])
def test_manual_metadata_works_with_image_export(setup_scans, parallel):
    paths, _ = setup_scans
    _, result = datared.generate_2D_imgs(
        [11, 12], "sample", 300, 1500, 5, 100,
        paths=paths, scan_delay_overrides_fs={11: 200000000, 12: 200000000},
        use_parallel=parallel, max_workers=2, chunk_size=1,
    )
    images = list(paths.analysis_root.rglob("*.npy"))
    assert len(images) == 1
    np.testing.assert_allclose(np.load(images[0]), 11.5)


def test_fluence_preserves_scan_associations_and_rejects_other_fixed_delay(setup_scans):
    paths, _ = setup_scans
    common = dict(scans=[11, 12], sample_name="sample", temperature_K=300,
                  excitation_wl_nm=1500, fluence_mJ_cm2=[5, 10], time_window_fs=100,
                  scan_type="fluence", selected_delays=[200000000], paths=paths)
    datared.create_h5_files(**common, scan_delay_overrides_fs={11: 200000000, 12: 200000000})
    path = next(paths.analysis_root.rglob("*.h5"))
    before = path.read_bytes()
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle["meta/fluences_mJ_cm2"][:], [5, 10])
        assert handle["delays/200000000fs/scans/12"].attrs["fluence_mJ_cm2"] == 10
    with pytest.raises(ValueError, match="must match"):
        datared.create_h5_files(**common, scan_delay_overrides_fs={11: 300000000, 12: 200000000})
    assert path.read_bytes() == before


def test_dark_ignores_timing_overrides(setup_scans):
    paths, _ = setup_scans
    _, path = build(paths, scans=[11], overrides={999: 123}, scan_type="dark")
    with h5py.File(path) as handle:
        assert "delays" not in handle
        assert handle["scans/11"].attrs["nshots_expected"] == 3


@pytest.mark.parametrize("overrides", [True, {10: np.nan}, {10: np.inf}, {10: True}, {10.5: 100}, {999: 100}, [(10, 100), (10, 101)], {10: 0.01}, {10: 1e30}])
def test_invalid_overrides_rejected(overrides):
    with pytest.raises(ValueError):
        datared_utils.normalize_scan_delay_overrides(overrides, [10])


def test_gui_exact_unit_conversion_and_all_forced_no_reference(setup_scans):
    paths, _ = setup_scans
    service = PreparationService()
    assert service.parse_femtomax_delay_overrides("{11: 200.123456}", [11], "ns") == {11: 200123456}
    assert service.parse_femtomax_delay_overrides("None", [11]) == {}
    kwargs = service.build_femtomax_common_kwargs(
        metadata_values=dict(sample_name="sample", temperature_K="300", excitation_wl_nm="1500", fluence_mJ_cm2="5", time_window_fs="100"),
        scans_text="[11, 12]", scan_type="delay", selected_delays_text="auto",
        delay_source="avg", require_both=True, nb_shot_threshold_text="",
        overwrite=True, paths=paths, reference_path_text="/missing/references.csv",
        delay_overrides_text="{11: 200, 12: 500}", delay_overrides_unit="ns",
    )
    assert kwargs["scan_delay_overrides_fs"] == {11: 200000000, 12: 500000000}
    assert kwargs["ping_reference_path"] is None
    datared.create_h5_files(**kwargs)


def test_gui_rejects_conflicting_duplicate_literal_keys():
    with pytest.raises(ValueError, match="conflicting"):
        PreparationService().parse_femtomax_delay_overrides("{11: 200, 11: 500}", [11])


def test_minimum_shots_applies_to_combined_manual_group(setup_scans):
    paths, refs = setup_scans
    _, path = build(paths, refs, overrides={11: 200000000, 12: 200000000}, nb_shot_threshold=4)
    with h5py.File(path) as handle:
        assert list(handle["delays"]) == ["200000000fs"]
        assert handle["delays/200000000fs"].attrs["nshots_total_expected"] == 6


def test_manual_diagnostic_labels(setup_scans):
    import matplotlib.pyplot as plt
    paths, _ = setup_scans
    exp, _ = build(paths, scans=[11], overrides={11: 200000000})
    plots = exp.plot_delay_distribution(unit="ns")
    assert "assigned delay" in plots[0][1].get_legend_handles_labels()[1][0]
    plt.close("all")


def test_gui_state_roundtrip_and_old_state_clears_overrides(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.analysis.gui.main_window import AnalysisMainWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(AnalysisMainWindow, "_autosave_gui_state", lambda self: None)
    window = AnalysisMainWindow()
    tab = window._state_widget_roots()["preparation"]
    tab.datared_femto_scans.setText("[11]")
    tab.datared_femto_delay_overrides.setText("{11: 200}")
    tab.datared_femto_delay_overrides_unit.setCurrentText("ns")
    saved = window._collect_gui_state()
    tab.datared_femto_delay_overrides.clear()
    window._apply_gui_state(saved)
    assert tab.datared_femto_delay_overrides.text() == "{11: 200}"
    older = deepcopy(saved)
    del older["tabs"]["preparation"]["datared_femto_delay_overrides"]
    del older["tabs"]["preparation"]["datared_femto_delay_overrides_unit"]
    window._apply_gui_state(older)
    assert tab.datared_femto_delay_overrides.text() == ""
    assert tab.datared_femto_delay_overrides_unit.currentText() == "ns"
    assert tab.datared_femto_scans.text() == "[11]"
    window._allow_close_without_confirmation = True
    window.close()
    window.deleteLater()
    app.processEvents()
