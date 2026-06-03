from pathlib import Path

import pandas as pd

from UI.app import build_piecewise_model_cfgs


def test_user_dataset_gets_runtime_piecewise_config(tmp_path: Path):
    dataset_dir = tmp_path / "user_datasets" / "custom_train"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.csv").write_text("time,cens,x\n1,1,0\n2,0,1\n", encoding="utf-8")

    cfgs = build_piecewise_model_cfgs(
        base_dir=tmp_path,
        dataset_id="custom_train",
        task_id="classification",
        piecewise_plain_bases=["DecisionTreeClassifier"],
        piecewise_censor_bases=[],
    )

    assert len(cfgs) == 1
    assert cfgs[0]["label"] == "PiecewiseClassifWrapSA(DecisionTreeClassifier, times=8)"
    assert cfgs[0]["id"] == "sklearn.tree.DecisionTreeClassifier"
    assert cfgs[0]["piecewise_family"] == "PiecewiseClassifWrapSA"
    assert not cfgs[0].get("precomputed_only")


def test_precomputed_dataset_keeps_precomputed_piecewise_config(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    pd.DataFrame(
        {
            "METHOD": ["PiecewiseClassifWrapSA(DecisionTreeClassifier, times=8)"],
            "AUC_EVENT_mean": [0.7],
            "LOGLOSS_EVENT_mean": [0.3],
        }
    ).to_excel(tables_dir / "Piecewise_actg.xlsx", index=False)

    cfgs = build_piecewise_model_cfgs(
        base_dir=tmp_path,
        dataset_id="actg",
        task_id="classification",
        piecewise_plain_bases=["DecisionTreeClassifier"],
        piecewise_censor_bases=[],
    )

    assert len(cfgs) == 1
    assert cfgs[0]["label"] == "PiecewiseClassifWrapSA(DecisionTreeClassifier, times=8)"
    assert cfgs[0]["precomputed_only"] is True
