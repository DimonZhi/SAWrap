import json
from pathlib import Path

import pandas as pd

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
