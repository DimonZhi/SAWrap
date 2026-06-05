from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import UI.app as app_module
import UI.helpers_ai_advice as ai_advice_module


def _write_user_dataset(base_dir: Path, dataset_id: str = "custom_demo") -> Path:
    dataset_dir = base_dir / "user_datasets" / dataset_id
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.csv").write_text(
        (
            "time,cens,x\n"
            "1,1,0\n"
            "2,0,1\n"
            "3,1,2\n"
            "4,0,3\n"
            "5,1,4\n"
            "6,0,5\n"
        ),
        encoding="utf-8",
    )
    (dataset_dir / "manifest.json").write_text(
        f'{{"id":"{dataset_id}","label":"demo"}}',
        encoding="utf-8",
    )
    return dataset_dir


def test_home_context_syncs_ai_picker_with_selected_compare_preset():
    context = app_module.home_context(
        None,
        selected={
            "dataset_id": "custom_demo",
            "preset_id": "cls_auc_logloss",
            "model_ids": [],
            "piecewise_plain_bases": [],
            "piecewise_censor_bases": [],
            "x_metric": "AUC_EVENT",
            "y_metric": "LOGLOSS_EVENT",
        },
    )

    assert context["ai_selected"] == {
        "dataset_id": "custom_demo",
        "task_id": "classification",
    }


def test_ai_advice_uses_existing_metrics_without_recompute(tmp_path: Path, monkeypatch):
    _write_user_dataset(tmp_path)
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "method": ["ElasticNet", "KNeighborsRegressor"],
            "rmse_time_mean": [8.0, 12.0],
            "r2_time_mean": [0.70, 0.40],
        }
    ).to_excel(tables_dir / "custom_demo.xlsx", index=False)

    monkeypatch.setattr(app_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(app_module, "DISABLE_MISSING_RECALC", False)
    monkeypatch.setattr(
        app_module,
        "load_dataset_for_recompute",
        lambda dataset_id: (_ for _ in ()).throw(AssertionError("advice must not recompute models")),
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(ai_advice_module, "_LOCAL_ENV_LOADED", True)

    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post(
        "/ai-advice",
        data={"ai_dataset_id": "custom_demo", "ai_task_id": "regression"},
    )

    assert response.status_code == 200
    assert "ElasticNet" in response.text
    assert "Прогноз времени" in response.text
