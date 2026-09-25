from pathlib import Path

import h5py
import numpy as np
import pytest

from trxrdpy.analysis.MaxIV_FemtoMAX import datared, datared_utils
from trxrdpy.analysis.common.paths import AnalysisPaths
from trxrdpy.analysis.gui.services.preparation_service import PreparationService


@pytest.mark.parametrize("scan_type", ["delay", "fluence", "dark"])
def test_duplicate_scans_produce_same_metadata_and_images_as_unique(
    tmp_path, monkeypatch, scan_type
):
    raw = tmp_path / "raw"
    raw.mkdir()
    for scan, value in [(12, 2.0), (10, 8.0)]:
        with h5py.File(raw / f"scan-{scan}.h5", "w") as handle:
            handle.create_dataset(
                "entry/measurement/pilatus/data", data=np.full((2, 2, 2), value)
            )
    monkeypatch.setattr(
        datared_utils.Experiment, "read_corrected_pings_seconds",
        lambda self, scan: (np.zeros(2), np.zeros(2)),
    )
    outputs = []
    for name, scans, fluences in [
        ("unique", [12, 10], [5.0, 10.0]),
        ("duplicate", [12, 10, 12], [5.0, 10.0, 5.0]),
    ]:
        paths = AnalysisPaths(tmp_path, raw_subdir="raw", analysis_subdir=name)
        experiment, result = datared.generate_2D_imgs(
            scans, "sample", 300, 1500,
            fluences if scan_type == "fluence" else 5.0, 250,
            scan_type=scan_type, selected_delays=[0], paths=paths,
            ref_provider=lambda scan: (0.0, 0.0), use_parallel=False,
        )
        assert experiment.scans == [12, 10]
        metadata_path = next((tmp_path / name).rglob("*.h5"))
        with h5py.File(metadata_path, "r") as handle:
            np.testing.assert_array_equal(handle["meta/scans"][:], [12, 10])
            if scan_type == "fluence":
                np.testing.assert_array_equal(handle["meta/fluences_mJ_cm2"][:], [5, 10])
            elif scan_type == "dark":
                assert set(handle["scans"]) == {"12", "10"}
        images = {
            path.name: np.load(path) for path in (tmp_path / name).rglob("*.npy")
        }
        assert images
        outputs.append(images)
    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])


def test_repeated_single_dark_scan_keeps_single_scan_path(tmp_path):
    paths = AnalysisPaths(tmp_path)
    meta = datared_utils.ExperimentMeta(
        sample_name="sample", temperature_K=300, scan_type="dark",
        excitation_wl_nm=None, fluence_mJ_cm2=None,
    )
    repeated = datared_utils.Experiment.metadata_h5_path(meta, scans=[12, 12], paths=paths)
    unique = datared_utils.Experiment.metadata_h5_path(meta, scans=[12], paths=paths)
    assert repeated == unique
    assert Path(repeated).name == "sample_300K_dark_scan12.h5"


@pytest.mark.parametrize("scan_type", ["delay", "fluence", "dark"])
def test_gui_filters_scans_and_fluences_together(tmp_path, scan_type):
    references = tmp_path / "references.csv"
    references.write_text("scan_start,scan_end,ping2_ref_s,ping4_ref_s\n10,12,0,0\n")
    kwargs = PreparationService().build_femtomax_common_kwargs(
        metadata_values=dict(sample_name="sample", temperature_K="300",
                             excitation_wl_nm="1500", fluence_mJ_cm2="5", time_window_fs="250"),
        scans_text="[12, 10, 12]", scan_type=scan_type, selected_delays_text="[0]",
        delay_source="avg", require_both=True, nb_shot_threshold_text="",
        overwrite=False, paths=AnalysisPaths(tmp_path),
        fluences_text="[5, 10, 5]", reference_path_text=str(references),
    )
    assert kwargs["scans"] == [12, 10]
    if scan_type == "fluence":
        assert kwargs["fluence_mJ_cm2"] == [5, 10]


def test_fluences_can_already_correspond_to_unique_scans():
    assert datared_utils.normalize_scan_selection([12, 10, 12], [5, 10]) == ([12, 10], [5, 10])


def test_conflicting_duplicate_fluences_are_rejected():
    with pytest.raises(ValueError, match="scan 12 has conflicting fluences"):
        datared_utils.normalize_scan_selection([12, 10, 12], [5, 10, 15])


def test_numpy_scan_numbers_and_empty_selection():
    assert datared_utils.normalize_scan_selection(np.array([12, 10, 12]))[0] == [12, 10]
    assert datared_utils.normalize_scan_selection(np.int64(12))[0] == [12]
    with pytest.raises(ValueError, match="At least one"):
        datared_utils.normalize_scan_selection([])
