from __future__ import annotations

import h5py
import numpy as np

from trxrdpy.analysis.MaxIV_FemtoMAX import datared_utils
from trxrdpy.analysis.common.paths import AnalysisPaths


def _write_reference_csv(path, *, ping2: str, ping4: str) -> None:
    path.write_text(
        "scan_start,scan_end,ping2_ref_s,ping4_ref_s\n"
        f"123,123,{ping2},{ping4}\n",
        encoding="utf-8",
    )


def test_ping_reference_csv_precision_reaches_metadata(tmp_path):
    ping2_text = "4.623986123456789e-8"
    ping4_text = "4.820762987654321e-8"
    reference_path = tmp_path / "ping_references.csv"
    _write_reference_csv(
        reference_path,
        ping2=ping2_text,
        ping4=ping4_text,
    )

    experiment = datared_utils.Experiment(
        123,
        ping_reference_path=reference_path,
        paths=AnalysisPaths(tmp_path),
    )
    assert experiment.ref_provider(123) == (
        float(ping2_text),
        float(ping4_text),
    )

    metadata_path = tmp_path / "metadata.h5"
    with h5py.File(metadata_path, "w") as handle:
        experiment._write_ping_reference_metadata(handle.create_group("meta"))

    with h5py.File(metadata_path, "r") as handle:
        meta = handle["meta"]
        assert meta["ping2_reference_s"][0] == float(ping2_text)
        assert meta["ping4_reference_s"][0] == float(ping4_text)
        assert meta["ping2_reference_s"].dtype == np.dtype("float64")
        assert meta["ping4_reference_s"].dtype == np.dtype("float64")
        assert meta.attrs["ping_reference_arithmetic"] == "float64"

    experiment.validate_metadata_ping_references(metadata_path)


def test_metadata_from_before_precision_fix_must_be_recreated(tmp_path):
    reference_path = tmp_path / "ping_references.csv"
    _write_reference_csv(
        reference_path,
        ping2="4.623986123456789e-8",
        ping4="4.820762987654321e-8",
    )
    experiment = datared_utils.Experiment(
        123,
        ping_reference_path=reference_path,
        paths=AnalysisPaths(tmp_path),
    )

    metadata_path = tmp_path / "old_metadata.h5"
    with h5py.File(metadata_path, "w") as handle:
        meta = handle.create_group("meta")
        meta.attrs["ping_reference_sha256"] = experiment.ping_reference_table.sha256

    with np.testing.assert_raises_regex(
        ValueError,
        "predates precision-safe ping correction",
    ):
        experiment.validate_metadata_ping_references(metadata_path)


def test_corrected_pings_do_not_round_reference_to_raw_float32(tmp_path):
    ping2_text = "4.623986123456789e-8"
    ping4_text = "4.820762987654321e-8"
    reference_path = tmp_path / "ping_references.csv"
    _write_reference_csv(
        reference_path,
        ping2=ping2_text,
        ping4=ping4_text,
    )

    raw_ping2 = np.asarray([4.623987e-8], dtype=np.float32)
    raw_ping4 = np.asarray([4.820764e-8], dtype=np.float32)
    raw_path = tmp_path / "scan-123.h5"
    with h5py.File(raw_path, "w") as handle:
        measurement = handle.require_group("entry/measurement")
        measurement.create_dataset("oscc_02_maui/Ping_Ch2_value", data=raw_ping2)
        measurement.create_dataset("oscc_02_maui/Ping_Ch4_value", data=raw_ping4)

    experiment = datared_utils.Experiment(
        123,
        ping_reference_path=reference_path,
        paths=AnalysisPaths(tmp_path),
    )
    corrected2, corrected4 = experiment.read_corrected_pings_seconds(123)

    assert corrected2.dtype == np.dtype("float64")
    assert corrected4.dtype == np.dtype("float64")
    assert corrected2[0] == np.float64(raw_ping2[0]) - np.float64(ping2_text)
    assert corrected4[0] == np.float64(raw_ping4[0]) - np.float64(ping4_text)
