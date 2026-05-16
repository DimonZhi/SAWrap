from pathlib import Path

import pandas as pd

from UI.helpers_piecewise import load_piecewise_classification_summary


def test_load_piecewise_classification_summary_detects_score_gain(tmp_path: Path):
    tables_dir = tmp_path
    pd.DataFrame(
        {
            "METHOD": [
                "ClassifWrapSA(DecisionTreeClassifier)",
                "PiecewiseClassifWrapSA(DecisionTreeClassifier, times=8)",
                "PiecewiseClassifWrapSA(DecisionTreeClassifier, times=16)",
            ],
            "AUC_EVENT_mean": [0.60, 0.95, 0.80],
            "LOGLOSS_EVENT_mean": [0.90, 0.20, 0.40],
            "RMSE_EVENT_mean": [0.50, 0.10, 0.30],
        }
    ).to_excel(tables_dir / "Piecewise_actg.xlsx", index=False)

    summary = load_piecewise_classification_summary(tables_dir)

    assert summary["available"] is True
    assert summary["model_rows"][0]["model"] == "DecisionTreeClassifier"
    assert summary["model_rows"][0]["win_rate"] == "100"
    assert summary["detail_rows"][0]["dataset"] == "ACTG"
    assert summary["detail_rows"][0]["times"] == 16
    assert summary["times_tested"] == "16"
