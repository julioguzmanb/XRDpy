from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from trxrdpy.analysis import fitting
from trxrdpy.analysis.common import plot_utils


@pytest.fixture(autouse=True)
def close_figures(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


@pytest.fixture
def fit_csv(tmp_path):
    rows = []
    for group, label, baseline in [("-90_90", "Full", 10.), ("-15_15", "0", 20.)]:
        for x, y, ref in [(np.nan, baseline - 1, True), (np.nan, baseline + 1, True),
                          (-1000, baseline, False), (0, baseline + 2, False), (1000, baseline + 4, False)]:
            rows.append(dict(peak="110", azim_range_str=group, phi_label=label,
                             phi_mode="separate_phi", delay_fs=x,
                             fluence_mJ_cm2=x / 1000 + 1 if np.isfinite(x) else np.nan,
                             hkl_pos=y, is_reference=ref, success=True, time_window_fs=100,
                             series_type="dark" if ref else "delay"))
    path = tmp_path / "fits.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def experiment(path, **kw):
    return dict(sample_name="sample", temperature_K=300, excitation_wl_nm=800,
                fluence_mJ_cm2=1, delay_fs=1000, time_window_fs=100,
                csv_path=str(path), **kw)


@pytest.mark.parametrize("show_sigma", [False, True])
@pytest.mark.parametrize("mode", ["norm_min_max", "relative_variation"])
def test_delay_transform_reference_and_errors(fit_csv, mode, show_sigma):
    out = fitting.plot_time_evolution_multi(
        experiments=[experiment(fit_csv)], phi_window="Full", show=False,
        show_baseline_sigma=show_sigma, **{mode: True})
    series = out["series"][0]
    factor = .25 if mode == "norm_min_max" else 10.
    np.testing.assert_allclose(series["y"], np.array([0, 2, 4]) * factor)
    if show_sigma:
        assert series["baseline_y0"] == 0
        assert series["baseline_sig"] == pytest.approx(np.sqrt(2) * factor)
        baselines = [line for line in out["ax"].lines if line.get_linestyle() == "--"]
        assert baselines
        assert all(np.allclose(line.get_ydata(), 0) for line in baselines)
    assert "Å" not in out["ax"].get_ylabel()
    assert ("%" in out["ax"].get_ylabel()) == (mode == "relative_variation")


def test_explicit_reference_uses_unshifted_delay(fit_csv):
    out = fitting.plot_time_evolution_multi(
        experiments=[experiment(fit_csv, ref_type="delay", ref_value=0, delay_offset_ps=9.6)],
        phi_window="Full", relative_variation=True, show_baseline_sigma=False, show=False)
    np.testing.assert_allclose(out["series"][0]["y"], 100 * (np.array([10, 12, 14]) - 12) / 12)


def test_independent_delay_sectors_and_mixed_modes(fit_csv, tmp_path):
    other = tmp_path / "averaged.csv"
    frame = pd.read_csv(fit_csv)
    frame["phi_mode"] = "phi_avg"
    frame.to_csv(other, index=False)
    out = fitting.plot_time_evolution_multi(
        experiments=[experiment(fit_csv, phi_window="Full"),
                     experiment(other, phi_mode="phi_avg", phi_window=0)],
        phi_window="Full", show=False)
    assert [s["_group"] for s in out["series"]] == ["-90_90", "0"]
    np.testing.assert_allclose(out["series"][1]["y"], [20, 22, 24])
    assert "per-experiment" in out["ax"].get_title()


@pytest.mark.parametrize("cmap", [None, "viridis"])
def test_fluence_sectors_colors_relative(fit_csv, cmap):
    fig, ax, _ = fitting.plot_fluence_evolution_multi(
        experiments=[experiment(fit_csv, phi_window="Full", ref_type="fluence", ref_value=0),
                     experiment(fit_csv, phi_window=(-15, 15), ref_type="fluence", ref_value=0)],
        peak="110", prop="hkl_pos", relative_variation=True,
        show_baseline_sigma=True, cmap=cmap, show=False)
    curves = [line for line in ax.lines if line.get_marker() == "o"]
    np.testing.assert_allclose(curves[0].get_ydata(), [0, 20, 40])
    np.testing.assert_allclose(curves[1].get_ydata(), [0, 10, 20])
    expected = (plot_utils.FitTimeEvolutionMultiPlotter._series_colors(2, cmap=cmap)
                if cmap else plt.rcParams["axes.prop_cycle"].by_key()["color"][:2])
    from matplotlib.colors import to_rgba
    assert [to_rgba(c.get_color()) for c in curves] == [to_rgba(c) for c in expected]
    assert "%" in ax.get_ylabel()


@pytest.mark.parametrize("x_col", ["delay_fs", "fluence_mJ_cm2"])
def test_single_relative_reference_line(fit_csv, x_col):
    frame = pd.read_csv(fit_csv)
    frame["hkl_pos"] = frame["hkl_pos"].astype(int)
    fig, ax = plot_utils.FitTimeEvolutionPlotter().plot(
        frame, peak="110", y="hkl_pos", groups=["-90_90"], x_col=x_col,
        relative_variation=True, ref_type="delay" if x_col == "delay_fs" else "fluence",
        ref_value=0, as_lines=True)
    assert ax.lines[0].get_ydata() == [0, 0]
    expected_ref = 12 if x_col == "delay_fs" else 10
    np.testing.assert_allclose(ax.lines[1].get_ydata(),
                               100 * (np.array([10, 12, 14]) - expected_ref) / expected_ref)
    assert "%" in ax.get_ylabel() and "Å" not in ax.get_ylabel()


def test_invalid_transform_and_missing_reference(fit_csv):
    with pytest.raises(ValueError, match="zero"):
        plot_utils.transform_fit_evolution([1, 2], 0, 1, relative_variation=True)
    with pytest.raises(ValueError, match="zero"):
        plot_utils.transform_fit_evolution([10, 10], 10, 1, norm_min_max=True)
    with pytest.raises(ValueError, match="not both"):
        plot_utils.transform_fit_evolution([10, 20], 10, 1, norm_min_max=True, relative_variation=True)
    with pytest.raises(ValueError, match="selected delay reference"):
        fitting.plot_time_evolution_multi(experiments=[experiment(fit_csv, ref_type="delay", ref_value=999)],
                                         phi_window="Full", relative_variation=True, show=False)


@pytest.mark.parametrize("fluence", [False, True])
def test_single_public_api_normalization(fit_csv, monkeypatch, fluence):
    kwargs = dict(sample_name="sample", temperature_K=300, excitation_wl_nm=800,
                  time_window_fs=100, peak="110", _property="hkl_pos", groups=["Full"],
                  norm_min_max=True, show_baseline_sigma=False)
    if fluence:
        monkeypatch.setattr(fitting.fitting_utils, "resolve_fluence_fitting_csv_path", lambda **kw: str(fit_csv))
        fig, ax, _ = fitting.plot_fluence_evolution(delay_fs=1000, **kwargs)
    else:
        fig, ax = fitting.plot_time_evolution(csv_path=str(fit_csv), fluence_mJ_cm2=1, **kwargs)
    np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 1], [0, .5, 1])
    assert ax.lines[0].get_ydata() == [0, 0]
    assert "Å" not in ax.get_ylabel()


def test_absolute_values_unchanged(fit_csv):
    # Older GUI states may contain a default mode unrelated to an explicit CSV.
    out = fitting.plot_time_evolution_multi(experiments=[experiment(fit_csv, phi_mode="phi_avg")],
                                           phi_window="Full", show=False)
    np.testing.assert_allclose(out["series"][0]["y"], [10, 12, 14])
    assert out["series"][0]["baseline_y0"] == 10
    assert "Å" in out["ax"].get_ylabel()


def test_merged_relative_uses_existing_prepump_reference(fit_csv):
    out = fitting.plot_time_evolution_multi(
        experiments=[{"merge": [experiment(fit_csv), experiment(fit_csv)], "label": "merged"}],
        phi_window="Full", relative_variation=True, show=False)
    np.testing.assert_allclose(out["series"][0]["y"], [0, 0, 20, 20, 40, 40])
    assert out["series"][0]["baseline_y0"] == 0


def test_failed_reference_rejected(fit_csv):
    frame = pd.read_csv(fit_csv)
    frame.loc[frame.is_reference, "success"] = False
    frame.to_csv(fit_csv, index=False)
    with pytest.raises(ValueError, match="reference"):
        fitting.plot_time_evolution_multi(experiments=[experiment(fit_csv)],
                                         phi_window="Full", relative_variation=True, show=False)


def test_gui_state_roundtrip_and_older_defaults(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.analysis.gui.main_window import AnalysisMainWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(AnalysisMainWindow, "_maybe_prompt_restore_autosave", lambda self: None)
    window = AnalysisMainWindow(launch_directory=tmp_path)
    tab = window.fitting_tab
    assert tab.fit_multi_fluence_cmap.text() == ""
    tab.fit_multi_relative_variation.setChecked(True)
    tab.fit_multi_fluence_cmap.setText("viridis")
    tab.fit_fluence_norm_min_max.setChecked(True)
    tab.fit_relative_variation.setChecked(True)
    tab.fit_multi_editor.set_experiments([experiment("one.csv", phi_window="Full"),
                                          experiment("two.csv", phi_window=(-15, 15))])
    tab.fit_multi_editor_fluence.set_experiments([experiment("three.csv", phi_window=30)])
    state = json.loads(json.dumps(window._collect_gui_state()))
    tab.fit_multi_fluence_cmap.clear()
    tab.fit_multi_relative_variation.setChecked(False)
    window._apply_gui_state(state)
    assert tab.fit_multi_relative_variation.isChecked()
    assert tab.fit_multi_fluence_cmap.text() == "viridis"
    assert tab.fit_relative_variation.isChecked()
    assert tab.fit_fluence_norm_min_max.isChecked()
    assert tab.fit_multi_editor.get_experiments()[1]["phi_window"] == [-15, 15]
    assert tab.fit_multi_editor_fluence.get_experiments()[0]["phi_window"] == 30
    calls = []
    monkeypatch.setattr(tab.fitting_service, "plot_fluence_evolution_multi",
                        lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(tab, "_build_analysis_paths", lambda: None)
    tab.set_facility("SACLA")
    tab.fit_multi_series_combo.setCurrentText("Fluence scan")
    tab.fit_multi_fluence_relative_variation.setChecked(True)
    tab._run_time_evolution_multi()
    assert calls[-1]["cmap"] == "viridis"
    assert calls[-1]["relative_variation"] is True
    assert calls[-1]["experiments"][0]["phi_window"] == 30
    tab.fit_multi_fluence_cmap.clear()
    tab._run_time_evolution_multi()
    assert calls[-1]["cmap"] is None
    tab.fit_multi_norm_min_max.setChecked(True)
    assert not tab.fit_multi_relative_variation.isChecked()
    tab.fit_multi_relative_variation.setChecked(True)
    assert not tab.fit_multi_norm_min_max.isChecked()
    window._apply_gui_state({"state_version": 1, "tabs": {"fitting": {
        "fit_multi_norm_min_max": {"type": "QCheckBox", "value": True},
        "fit_multi_editor": {"type": "MultiExperimentEditor", "value": [experiment("old.csv")]},
    }}})
    assert tab.fit_multi_norm_min_max.isChecked()
    assert not tab.fit_multi_relative_variation.isChecked()
    assert not tab.fit_relative_variation.isChecked()
    assert tab.fit_multi_fluence_cmap.text() == ""
    assert "phi_window" not in tab.fit_multi_editor.get_experiments()[0]
    window.close()
    app.processEvents()


@pytest.mark.parametrize("facility", ["FemtoMAX", "SACLA", "ID09"])
@pytest.mark.parametrize("series", ["Delay scan", "Fluence scan"])
def test_viewer_uses_integration_resolution(tmp_path, monkeypatch, facility, series):
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.analysis.gui.main_window import AnalysisMainWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(AnalysisMainWindow, "_maybe_prompt_restore_autosave", lambda self: None)
    w = AnalysisMainWindow(launch_directory=tmp_path)
    viewer = w.viewer_tab
    w.pattern_creation_tab.pattern_npt.setText("2048")
    assert viewer.viewer_npt.text() == "2048"
    w.preparation_tab.datared_single_shot_npt.setText("4096")
    assert viewer.viewer_npt.text() == "4096"
    w._apply_gui_state({"tabs": {
        "pattern": {"pattern_npt": {"type": "QLineEdit", "value": "2048"}},
        "viewer": {"viewer_npt": {"type": "QLineEdit", "value": "1000"}},
    }})
    assert viewer.viewer_npt.text() == "2048"
    assert viewer.viewer_npt.isReadOnly()
    calls = []
    svc = viewer.integration_service
    monkeypatch.setattr(svc, "plot_1d_abs_and_diffs_delay", lambda **kw: calls.append(kw))
    monkeypatch.setattr(svc, "plot_1d_abs_and_diffs_fluence", lambda **kw: calls.append(kw))
    monkeypatch.setattr(svc, "build_id09_kwargs", lambda *a: {})
    monkeypatch.setattr(svc, "ensure_id09_fluence_cache", lambda **kw: None)
    monkeypatch.setattr(viewer, "_build_analysis_paths", lambda: None)
    viewer.state.facility = facility
    viewer.viewer_series_combo.setCurrentText(series)
    viewer.viewer_overwrite_xy.setChecked(True)
    viewer._plot_1d_abs_and_diffs()
    assert calls, w.log_widget.toPlainText() if hasattr(w.log_widget, "toPlainText") else "Viewer did not call backend"
    if facility == "ID09" and series == "Fluence scan":
        assert "npt" not in calls[-1]
    else:
        assert calls[-1]["npt"] == 2048
    assert calls[-1]["overwrite_xy"] is True
    import inspect
    module = svc.get_azimint_module(facility)
    backend = (module.plot_1D_abs_and_diffs_delay if series == "Delay scan"
               else module.plot_1D_abs_and_diffs_fluence)
    kwargs = dict(calls[-1])
    kwargs.pop("facility")
    # Required ID09 acquisition metadata is supplied by its separate builder.
    if facility == "ID09" and series == "Delay scan":
        kwargs.update(dataset=1, scan_nb=1)
    inspect.signature(backend).bind(**kwargs)
    w.close()
    app.processEvents()
