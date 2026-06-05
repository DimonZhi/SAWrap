import gzip
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from UI.helpers_user_datasets import (
    DatasetUploadError,
    load_user_dataset,
    save_uploaded_dataset,
    standardize_uploaded_dataset,
)


def test_standardize_uploaded_dataset_converts_to_training_shape():
    raw = pd.DataFrame(
        {
            "duration": [10, 12, 7, 3, 5, 9],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [64, 58, None, 71, 45, 52],
            "group": ["a", "b", "a", None, "c", "b"],
        }
    )

    standard, manifest = standardize_uploaded_dataset(raw)

    assert list(standard.columns) == ["time", "cens", "age", "group"]
    assert standard["cens"].tolist() == [True, False, True, False, True, False]
    assert standard["age"].isna().sum() == 0
    assert standard["group"].dtype.kind in {"f", "i"}
    assert manifest["time_column"] == "duration"
    assert manifest["event_column"] == "event"
    assert manifest["categorical_features"] == ["group"]


def test_save_and_load_user_dataset_roundtrip(tmp_path: Path):
    content = (
        "time,cens,x,segment\n"
        "1,1,10,a\n"
        "2,0,20,b\n"
        "3,1,30,a\n"
        "4,0,40,b\n"
        "5,1,50,c\n"
        "6,0,60,c\n"
    ).encode()

    manifest = save_uploaded_dataset(tmp_path, "clinic.csv", content)
    X, y, features, categ, sch_nan = load_user_dataset(tmp_path, manifest["id"])

    assert manifest["id"] == "custom_clinic"
    assert features == ["x", "segment"]
    assert X.shape == (6, 2)
    assert y.dtype.names == ("cens", "time")
    assert y["cens"].tolist() == [True, False, True, False, True, False]
    assert y["time"].tolist() == [1, 2, 3, 4, 5, 6]
    assert categ == ["segment"]
    assert sch_nan is None


def test_standardize_uploaded_dataset_requires_time_and_event():
    with pytest.raises(DatasetUploadError):
        standardize_uploaded_dataset(pd.DataFrame({"age": [1, 2, 3, 4, 5]}))


def test_standardize_backblaze_shape_parses_timedelta_and_drops_metadata():
    raw = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"],
            "serial_number": [f"disk_{idx}" for idx in range(6)],
            "model": ["A", "A", "B", "B", "C", "C"],
            "smart_1_raw": [1, 2, None, 4, 5, 6],
            "mostly_empty": [None, None, None, None, None, None],
            "time": ["6 days", "10 days", "12 days", "13 days", "15 days", "20 days"],
            "time_row": [100, 101, 102, 103, 104, 105],
            "event": [0, 1, 0, 1, 0, 1],
        }
    )

    standard, manifest = standardize_uploaded_dataset(raw)

    assert standard["time"].tolist() == [6, 10, 12, 13, 15, 20]
    assert standard["cens"].tolist() == [False, True, False, True, False, True]
    assert "serial_number" not in manifest["features"]
    assert "date" not in manifest["features"]
    assert "time_row" not in manifest["features"]
    assert "mostly_empty" not in manifest["features"]
    assert set(manifest["features"]) == {"model", "smart_1_raw"}


def test_standardize_alibaba_shape_uses_event_time_alias():
    raw = pd.DataFrame(
        {
            "serial_number": [f"disk_{idx}" for idx in range(6)],
            "manufacturer": ["A", "A", "B", "B", "C", "C"],
            "model": [1, 1, 2, 2, 3, 3],
            "smart_1_raw": [10, 20, 30, 40, 50, 60],
            "event": [False, True, False, True, False, True],
            "event_time": [471, 202, 203, 204, 205, 206],
        }
    )

    standard, manifest = standardize_uploaded_dataset(raw)

    assert manifest["time_column"] == "event_time"
    assert manifest["event_column"] == "event"
    assert standard["time"].tolist() == [471, 202, 203, 204, 205, 206]
    assert standard["cens"].tolist() == [False, True, False, True, False, True]
    assert "serial_number" not in manifest["features"]


def test_save_uploaded_dataset_reads_csv_gz_and_zip(tmp_path: Path):
    csv_content = (
        "event_time,event,x\n"
        "1,1,10\n"
        "2,0,20\n"
        "3,1,30\n"
        "4,0,40\n"
        "5,1,50\n"
        "6,0,60\n"
    ).encode()
    gz_manifest = save_uploaded_dataset(tmp_path, "alibaba.csv.gz", gzip.compress(csv_content))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as archive:
        archive.writestr("freddiemac.csv", csv_content)
    zip_manifest = save_uploaded_dataset(tmp_path, "freddiemac.zip", zip_buffer.getvalue())

    assert gz_manifest["id"] == "custom_alibaba_csv"
    assert zip_manifest["id"] == "custom_freddiemac"
    assert (tmp_path / "user_datasets" / gz_manifest["id"] / "data.csv").exists()
    assert (tmp_path / "user_datasets" / zip_manifest["id"] / "data.csv").exists()
