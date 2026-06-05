import json
from pathlib import Path

import pandas as pd

import UI.helpers_ai_advice as ai_advice_module
from UI.helpers_ai_advice import (
    build_ai_advice,
    build_openrouter_chat_answer,
    build_openrouter_interpretation,
)


def _write_dataset_table(base_dir: Path, dataset_id: str, df: pd.DataFrame) -> Path:
    table_dir = base_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / f"{dataset_id}.xlsx"
    df.to_excel(table_path, index=False)
    return table_path


def test_build_ai_advice_recommends_best_classification_model(tmp_path: Path):
    _write_dataset_table(
        tmp_path,
        "toy",
        pd.DataFrame(
            {
                "method": ["WeakModel", "StrongModel"],
                "auc_event_mean": [0.62, 0.91],
                "logloss_event_mean": [0.55, 0.22],
                "rmse_event_mean": [0.42, 0.18],
            }
        ),
    )

    advice = build_ai_advice(tmp_path, "toy", "classification")

    assert advice["has_result"] is True
    assert advice["recommended_method"] == "StrongModel"
    assert advice["top_models"][0]["score"] == 100
    assert advice["top_models"][0]["metrics"][0]["position"] == 1
    assert "AUC_EVENT" in advice["available_metrics"]
    assert advice["llm"]["enabled"] is False


def test_build_ai_advice_includes_global_piecewise_variants(tmp_path: Path):
    _write_dataset_table(
        tmp_path,
        "actg",
        pd.DataFrame(
            {
                "method": ["ClassifWrapSA(DecisionTreeClassifier)", "StrongBase"],
                "auc_event_mean": [0.62, 0.82],
                "logloss_event_mean": [0.55, 0.30],
                "rmse_event_mean": [0.42, 0.28],
            }
        ),
    )
    pd.DataFrame(
        {
            "METHOD": [
                "PiecewiseClassifWrapSA(DecisionTreeClassifier, times=8)",
                "PiecewiseClassifWrapSA(DecisionTreeClassifier, times=16)",
            ],
            "AUC_EVENT_mean": [0.95, 0.70],
            "LOGLOSS_EVENT_mean": [0.18, 0.50],
            "RMSE_EVENT_mean": [0.12, 0.40],
        }
    ).to_excel(tmp_path / "tables" / "Piecewise_actg.xlsx", index=False)

    advice = build_ai_advice(tmp_path, "actg", "classification")

    assert advice["has_result"] is True
    assert advice["piecewise_included"] is True
    assert advice["recommended_method"] == "PiecewiseClassifWrapSA(DecisionTreeClassifier, times=8)"
    assert advice["piecewise_variants"][0]["times"] == 8
    assert any("times=8" in reason for reason in advice["why"])


def test_build_ai_advice_respects_lower_is_better_for_survival(tmp_path: Path):
    _write_dataset_table(
        tmp_path,
        "surv",
        pd.DataFrame(
            {
                "METHOD": ["LowIbsModel", "HighIbsModel"],
                "ci_mean": [0.80, 0.80],
                "ibs_mean": [0.10, 0.30],
                "auprc_mean": [0.70, 0.70],
            }
        ),
    )

    advice = build_ai_advice(tmp_path, "surv", "survival")

    assert advice["has_result"] is True
    assert advice["recommended_method"] == "LowIbsModel"
    ibs_metric = next(
        metric for metric in advice["top_models"][0]["metrics"]
        if metric["key"] == "IBS"
    )
    assert ibs_metric["position"] == 1
    assert ibs_metric["direction_label"] == "ниже лучше"


def test_build_ai_advice_reports_missing_table(tmp_path: Path):
    advice = build_ai_advice(tmp_path, "missing", "survival")

    assert advice["has_result"] is False
    assert "Не найдена таблица" in advice["error"]


def test_build_ai_advice_reports_missing_task_metrics(tmp_path: Path):
    _write_dataset_table(
        tmp_path,
        "toy",
        pd.DataFrame({"method": ["Model"], "some_metric_mean": [1.0]}),
    )

    advice = build_ai_advice(tmp_path, "toy", "regression")

    assert advice["has_result"] is False
    assert "нет подходящих метрик" in advice["error"]
    assert "RMSE_TIME" in advice["missing_metrics"]


def test_build_ai_advice_does_not_fall_back_to_other_task(tmp_path: Path):
    _write_dataset_table(
        tmp_path,
        "custom_demo",
        pd.DataFrame(
            {
                "method": ["WeakModel", "StrongModel"],
                "auc_event_mean": [0.62, 0.91],
                "logloss_event_mean": [0.55, 0.22],
            }
        ),
    )

    advice = build_ai_advice(tmp_path, "custom_demo", "survival")

    assert advice["has_result"] is False
    assert advice["requested_task_id"] == "survival"
    assert advice["task_id"] == "survival"
    assert "Сначала рассчитай этот пресет" in advice["error"]


def test_build_ai_advice_uses_any_model_with_selected_task_metrics(tmp_path: Path):
    _write_dataset_table(
        tmp_path,
        "custom_demo",
        pd.DataFrame(
            {
                "method": ["KNeighborsClassifier", "KaplanMeierFitter"],
                "auc_event_mean": [0.70, 0.99],
                "logloss_event_mean": [0.20, 0.05],
            }
        ),
    )

    advice = build_ai_advice(tmp_path, "custom_demo", "classification")

    assert advice["has_result"] is True
    assert advice["recommended_method"] == "KaplanMeierFitter"
    assert [model["method"] for model in advice["top_models"]] == [
        "KaplanMeierFitter",
        "KNeighborsClassifier",
    ]


def test_build_ai_advice_keeps_selected_regression_task(tmp_path: Path):
    _write_dataset_table(
        tmp_path,
        "custom_demo",
        pd.DataFrame(
            {
                "method": ["KNeighborsClassifier", "ElasticNet", "RandomForestRegressor"],
                "auc_event_mean": [0.95, 0.50, 0.40],
                "logloss_event_mean": [0.10, 0.90, 0.80],
                "rmse_time_mean": [100.0, 12.0, 8.0],
                "r2_time_mean": [0.10, 0.55, 0.80],
            }
        ),
    )

    advice = build_ai_advice(tmp_path, "custom_demo", "regression")

    assert advice["has_result"] is True
    assert advice["task_id"] == "regression"
    assert advice["recommended_method"] == "RandomForestRegressor"
    assert [model["method"] for model in advice["top_models"]] == [
        "RandomForestRegressor",
        "ElasticNet",
        "KNeighborsClassifier",
    ]


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return (
            b'{"choices":[{"message":{"content":"LLM says: choose StrongModel because metrics are best."}}]}'
        )


def test_build_openrouter_interpretation_uses_fake_transport(tmp_path: Path):
    _write_dataset_table(
        tmp_path,
        "toy",
        pd.DataFrame(
            {
                "method": ["WeakModel", "StrongModel"],
                "auc_event_mean": [0.62, 0.91],
                "logloss_event_mean": [0.55, 0.22],
                "rmse_event_mean": [0.42, 0.18],
            }
        ),
    )
    advice = build_ai_advice(tmp_path, "toy", "classification")
    captured = {}

    def fake_opener(request, timeout, context):
        captured["auth"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        captured["context"] = context
        return _FakeResponse()

    llm = build_openrouter_interpretation(
        advice,
        api_key="test-key",
        model="test/model",
        timeout=3,
        opener=fake_opener,
    )

    assert llm["enabled"] is True
    assert llm["model"] == "test/model"
    assert llm["text"] == "LLM says: choose StrongModel because metrics are best."
    assert captured["auth"] == "Bearer test-key"
    assert captured["timeout"] == 3
    assert captured["context"] is not None


def test_build_openrouter_interpretation_reports_missing_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(ai_advice_module, "_LOCAL_ENV_LOADED", True)
    advice = {
        "dataset_id": "toy",
        "task_label": "Анализ выживаемости",
        "recommended_method": "Model",
        "score": 50,
        "top_models": [],
    }

    llm = build_openrouter_interpretation(advice)

    assert llm["enabled"] is False
    assert "OPENROUTER_API_KEY" in llm["error"]


def test_build_openrouter_interpretation_loads_local_env(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setattr(ai_advice_module, "_LOCAL_ENV_LOADED", False)
    (tmp_path / ".env").write_text(
        'OPENROUTER_API_KEY="env-test-key"\nOPENROUTER_MODEL=test/env-model\n',
        encoding="utf-8",
    )
    captured = {}

    def fake_opener(request, timeout, context):
        captured["auth"] = request.headers["Authorization"]
        return _FakeResponse()

    llm = build_openrouter_interpretation(
        {
            "dataset_id": "toy",
            "task_label": "Классификация события",
            "recommended_method": "Model",
            "score": 50,
            "top_models": [],
        },
        opener=fake_opener,
    )

    assert llm["enabled"] is True
    assert llm["model"] == "test/env-model"
    assert captured["auth"] == "Bearer env-test-key"


def test_build_openrouter_chat_answer_includes_question_and_result_context(tmp_path: Path):
    _write_dataset_table(
        tmp_path,
        "toy",
        pd.DataFrame(
            {
                "method": ["WeakModel", "StrongModel"],
                "auc_event_mean": [0.62, 0.91],
                "logloss_event_mean": [0.55, 0.22],
                "rmse_event_mean": [0.42, 0.18],
            }
        ),
    )
    advice = build_ai_advice(tmp_path, "toy", "classification")
    captured = {}

    def fake_opener(request, timeout, context):
        captured["payload"] = request.data.decode("utf-8")
        return _FakeResponse()

    answer = build_openrouter_chat_answer(
        advice,
        "Почему выбрана StrongModel?",
        history=[{"role": "user", "content": "Какая модель лучшая?"}],
        api_key="test-key",
        opener=fake_opener,
    )

    assert answer["text"] == "LLM says: choose StrongModel because metrics are best."
    payload = json.loads(captured["payload"])
    message_text = "\n".join(message["content"] for message in payload["messages"])
    assert "Почему выбрана StrongModel?" in message_text
    assert "StrongModel" in message_text
    assert "Датасет: toy" in message_text
