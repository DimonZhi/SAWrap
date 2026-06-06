import re
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import UI.app as app_module
import UI.helpers_demo_predict as demo_predict_module
from UI.helpers_demo_predict import DEMO_MODEL_LABEL, _prepare_training_frame, find_demo_candidates


def _without_scripts(html: str) -> str:
    return re.sub(r"<script[\s\S]*?</script>", "", html)


def _write_demo_dataset(base_dir: Path):
    dataset_dir = base_dir / "user_datasets" / "custom_demo"
    dataset_dir.mkdir(parents=True)
    rows = ["time,cens,x1,x2"]
    for index in range(1, 31):
        event = 1 if index % 3 else 0
        rows.append(f"{index},{event},{index / 10:.3f},{(index % 5) / 10:.3f}")
    (dataset_dir / "data.csv").write_text("\n".join(rows), encoding="utf-8")
    (dataset_dir / "manifest.json").write_text(
        '{"id":"custom_demo","label":"demo","features":["x1","x2"]}',
        encoding="utf-8",
    )


def _write_demo_results(base_dir: Path):
    tables_dir = base_dir / "tables"
    tables_dir.mkdir()
    pd.DataFrame(
        {
            "method": ["LogisticRegression", "ElasticNet", "KaplanMeierFitter", DEMO_MODEL_LABEL],
            "ci_mean": [0.6, 0.55, 0.65, 0.5],
            "ibs_mean": [0.2, 0.25, 0.18, 0.3],
            "params": [None, None, None, "{'n_estimators': 2, 'random_state': 123}"],
        }
    ).to_excel(tables_dir / "custom_demo.xlsx", index=False)
    pd.DataFrame(
        {
            "method": [
                "PiecewiseClassifWrapSA(LogisticRegression, times=8)",
                "PiecewiseCensorAwareClassifWrapSA(LogisticRegression, times=16)",
            ],
            "ci_mean": [0.7, 0.72],
            "ibs_mean": [0.21, 0.19],
        }
    ).to_excel(tables_dir / "Piecewise_custom_demo.xlsx", index=False)


def test_prepare_training_frame_imputes_missing_values():
    X = pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0],
            "categorical": ["a", None, "a"],
        }
    )

    prepared = _prepare_training_frame(X)

    assert prepared.isna().sum().sum() == 0
    assert prepared.loc[1, "numeric"] == 2.0
    assert prepared.loc[1, "categorical"] == "a"


def test_load_dataset_for_recompute_finds_case_variant_loader(monkeypatch):
    class FakeDatasets:
        @staticmethod
        def load_Framingham_dataset():
            return "framingham-loaded"

    monkeypatch.setattr(app_module, "ds", FakeDatasets)
    monkeypatch.setattr(app_module, "is_user_dataset", lambda base_dir, dataset_id: False)

    assert app_module.load_dataset_for_recompute("framingham") == "framingham-loaded"


def test_find_demo_candidates_uses_parallel_bootstrap_craid(tmp_path: Path):
    _write_demo_results(tmp_path)

    candidates = find_demo_candidates(
        tmp_path,
        "custom_demo",
        [{"label": DEMO_MODEL_LABEL, "task": "survival"}],
    )

    assert len(candidates) == 1
    assert candidates[0]["method"] == DEMO_MODEL_LABEL


def test_demo_predict_builds_curve_for_parallel_bootstrap_craid(tmp_path: Path, monkeypatch):
    _write_demo_dataset(tmp_path)
    _write_demo_results(tmp_path)
    monkeypatch.setattr(app_module, "BASE_DIR", tmp_path)

    class FakeDemoModel:
        def predict_survival_function(self, sample, times):
            return np.array([[0.95, 0.80, 0.60]]), np.array([1.0, 2.0, 3.0])

        def predict_expected_time(self, sample, times):
            return np.array([2.0])

    monkeypatch.setattr(
        demo_predict_module,
        "_fit_or_load_model",
        lambda base_dir, dataset_id, candidate, model_cfgs, X, y: (FakeDemoModel(), True),
    )

    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post(
        "/demo-predict",
        data={
            "dataset_id": "custom_demo",
            "preset_id": "cls_auc_logloss",
            "feature__x1": "1.5",
            "feature__x2": "0.3",
        },
    )

    assert response.status_code == 200
    html_body = _without_scripts(response.text)
    assert "Демо-прогноз" in html_body
    assert html_body.count("demo-summary-card") == 1

    ajax_response = client.post(
        "/demo-predict",
        headers={"Accept": "application/json"},
        data={
            "dataset_id": "custom_demo",
            "preset_id": "cls_auc_logloss",
            "feature__x1": "1.7",
            "feature__x2": "0.4",
        },
    )

    assert ajax_response.status_code == 200
    payload = ajax_response.json()
    assert payload["ok"] is True
    assert len(payload["traces"]) == 1
    assert len(payload["cards"]) == 1
    assert payload["cards"][0]["method"] == DEMO_MODEL_LABEL
    assert "Демо-прогноз" not in ajax_response.text


def test_demo_block_is_visible_for_precomputed_datasets(tmp_path: Path, monkeypatch):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    pd.DataFrame(
        {
            "method": ["LogisticRegression"],
            "auc_event_mean": [0.8],
            "logloss_event_mean": [0.4],
        }
    ).to_excel(tables_dir / "actg.xlsx", index=False)
    monkeypatch.setattr(app_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(app_module, "DISABLE_MISSING_RECALC", True)
    y = np.array(
        [(True, 1.0), (False, 2.0), (True, 3.0), (False, 4.0), (True, 5.0)],
        dtype=[("cens", "?"), ("time", "<f8")],
    )
    monkeypatch.setattr(
        app_module,
        "load_dataset_for_recompute",
        lambda dataset_id: (
            pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]}),
            y,
            ["x"],
            [],
            None,
        ),
    )

    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post(
        "/compare",
        data={
            "dataset_id": "actg",
            "preset_id": "cls_auc_logloss",
            "model_ids": "sklearn.linear_model.LogisticRegression",
        },
    )

    assert response.status_code == 200
    html_body = _without_scripts(response.text)
    assert "Демо-прогноз" in html_body
    assert "Построить демо-прогноз" in html_body


def test_demo_predict_accepts_precomputed_dataset(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(app_module, "BASE_DIR", tmp_path)
    captured = {}

    def fake_build_demo_prediction(base_dir, dataset_id, raw_values, model_cfgs, load_dataset):
        captured["dataset_id"] = dataset_id
        return {
            "ok": True,
            "dataset_id": dataset_id,
            "traces": [{"name": "best", "x": [1.0], "y": [0.9]}],
            "cards": [
                {
                    "category": "classification",
                    "category_label": "Классификация",
                    "method": "LogisticRegression",
                    "expected_time": "1",
                    "risk": "0.1",
                    "source": "cache",
                }
            ],
            "errors": [],
        }

    monkeypatch.setattr(app_module, "build_demo_prediction", fake_build_demo_prediction)

    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post(
        "/demo-predict",
        headers={"Accept": "application/json"},
        data={"dataset_id": "actg", "preset_id": "cls_auc_logloss", "feature__x": "1.0"},
    )

    assert response.status_code == 200
    assert captured["dataset_id"] == "actg"
    payload = response.json()
    assert payload["ok"] is True
    assert len(payload["traces"]) == 1
