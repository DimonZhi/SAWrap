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
