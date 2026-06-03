import re
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import UI.app as app_module


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
            "method": ["LogisticRegression", "ElasticNet", "KaplanMeierFitter"],
            "ci_mean": [0.6, 0.55, 0.65],
            "ibs_mean": [0.2, 0.25, 0.18],
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


def test_demo_predict_builds_curves_for_available_model_families(tmp_path: Path, monkeypatch):
    _write_demo_dataset(tmp_path)
    _write_demo_results(tmp_path)
    monkeypatch.setattr(app_module, "BASE_DIR", tmp_path)

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
    assert "Классификация" in html_body
    assert "Регрессия" in html_body
    assert "Анализ выживаемости" in html_body
    assert "Piecewise" in html_body
    assert "Piecewise censor-aware" in html_body
    assert html_body.count("demo-summary-card") == 5
    assert (tmp_path / "model_store" / "custom_demo").exists()

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
    assert len(payload["traces"]) == 5
    assert "Демо-прогноз" not in ajax_response.text


def test_demo_block_is_hidden_for_precomputed_datasets(tmp_path: Path, monkeypatch):
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
    assert "Демо-прогноз" not in html_body
    assert "Построить демо-прогноз" not in html_body
