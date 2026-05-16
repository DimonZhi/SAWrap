from pathlib import Path

import pandas as pd

from UI.helpers_tables import get_surv_metrics, list_surv_metrics_from_table


def test_piecewise_metrics_are_read_from_piecewise_table(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    model_label = "PiecewiseClassifWrapSA(DecisionTreeClassifier, times=16)"
    pd.DataFrame(
        {
            "METHOD": [model_label],
            "AUC_EVENT_mean": [0.72],
            "LOGLOSS_EVENT_mean": [0.31],
        }
    ).to_excel(tables_dir / "Piecewise_actg.xlsx", index=False)

    vx, vy, table_path = get_surv_metrics(
        base_dir=tmp_path,
        dataset_id="actg",
        X_tr=None,
        y_tr=None,
        model_cfgs=[],
        model_label=model_label,
        x_metric="AUC_EVENT",
        y_metric="LOGLOSS_EVENT",
        allow_recompute=True,
    )
    metrics, frame = list_surv_metrics_from_table(tmp_path, "actg", model_label=model_label)

    assert table_path.name == "Piecewise_actg.xlsx"
    assert vx == 0.72
    assert vy == 0.31
    assert metrics == ["auc_event", "logloss_event"]
    assert frame["method"].tolist() == [model_label]
