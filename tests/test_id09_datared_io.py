from __future__ import annotations

from pathlib import Path

import numpy as np

from trxrdpy.analysis.ESRF_ID09 import datared


class _SelectiveScan:
    def __init__(self, frames):
        self.frames = np.asarray(frames)
        self.selection_calls = []
        self.full_array_calls = 0

    def __getitem__(self, selection):
        self.selection_calls.append(selection)
        return self.frames[selection]

    def __array__(self, dtype=None, copy=None):
        self.full_array_calls += 1
        raise AssertionError("The full ID09 scan must not be materialized.")


def test_mean_selected_frames_reads_only_requested_indices():
    frames = np.arange(6 * 2 * 3, dtype=float).reshape(6, 2, 3)
    scan = _SelectiveScan(frames)
    selection = np.array([False, True, False, True, False, False])

    result = datared._mean_selected_frames(scan, selection)

    np.testing.assert_allclose(result, np.mean(frames[[1, 3]], axis=0))
    np.testing.assert_array_equal(scan.selection_calls[0], [1, 3])
    assert scan.full_array_calls == 0


def test_frame_progress_uses_batches_instead_of_per_frame_reads():
    frames = np.arange(70 * 2 * 2, dtype=float).reshape(70, 2, 2)
    scan = _SelectiveScan(frames)

    result = datared._mean_selected_frames(
        scan,
        np.ones(70, dtype=bool),
        show_progress=True,
    )

    np.testing.assert_allclose(result, np.mean(frames, axis=0))
    assert [len(selection) for selection in scan.selection_calls] == [32, 32, 6]
    assert scan.full_array_calls == 0


def test_get_2d_img_does_not_read_unselected_frames(monkeypatch):
    frames = np.arange(5 * 2 * 2, dtype=float).reshape(5, 2, 2)
    scan = _SelectiveScan(frames)
    monkeypatch.setattr(
        datared,
        "_open_id09_scan",
        lambda **kwargs: (object(), Path("scan.h5"), scan),
    )
    monkeypatch.setattr(
        datared,
        "_delay_tokens_str",
        lambda _scan: np.array(["a", "b", "a", "c", "a"]),
    )

    result = datared.get_2D_img("sample", 1, 2, "a")

    np.testing.assert_allclose(result, np.mean(frames[[0, 2, 4]], axis=0))
    np.testing.assert_array_equal(scan.selection_calls[0], [0, 2, 4])
    assert scan.full_array_calls == 0


def test_multi_delay_export_reuses_open_scan(monkeypatch, tmp_path):
    frames = np.arange(6 * 2 * 2, dtype=float).reshape(6, 2, 2)
    scan = _SelectiveScan(frames)
    open_calls = []

    def fake_open(**kwargs):
        open_calls.append(kwargs)
        return object(), Path("scan.h5"), scan

    class FakeDelayDataset:
        def __init__(self, *args, delay_fs, **kwargs):
            self.delay_fs = int(delay_fs)

        def img_path(self):
            return tmp_path / f"delay_{self.delay_fs}.npy"

    monkeypatch.setattr(datared, "_open_id09_scan", fake_open)
    monkeypatch.setattr(datared, "DelayDataset", FakeDelayDataset)
    monkeypatch.setattr(
        datared,
        "_normalize_delay_selection",
        lambda _scan, delays: ["a", "b"],
    )
    monkeypatch.setattr(
        datared,
        "_delay_tokens_str",
        lambda _scan: np.array(["a", "b", "a", "b", "a", "b"]),
    )
    monkeypatch.setattr(
        datared,
        "delay_token_to_fs",
        lambda token: {"a": 0, "b": 100}[token],
    )

    paths = datared.create_final_2D_images(
        "sample",
        1,
        2,
        100,
        800,
        1.0,
        200,
        delays="all",
        paths=object(),
        show_progress=False,
    )

    assert len(open_calls) == 1
    assert len(paths) == 2
    assert scan.full_array_calls == 0
    np.testing.assert_allclose(
        np.load(tmp_path / "delay_0.npy"),
        np.mean(frames[[0, 2, 4]], axis=0),
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "delay_100.npy"),
        np.mean(frames[[1, 3, 5]], axis=0),
    )
