from __future__ import annotations

import json
import os
import re
import ssl
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    import certifi
except ImportError:
    certifi = None


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
_LOCAL_ENV_LOADED = False
PIECEWISE_RE = re.compile(
    r"^(Piecewise(?:CensorAware)?ClassifWrapSA)\(([^,]+),\s*times=(\d+)\)$"
)

AI_TASKS = [
    {"id": "classification", "label": "Классификация события"},
    {"id": "regression", "label": "Прогноз времени"},
    {"id": "survival", "label": "Анализ выживаемости"},
]


TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "classification": {
        "label": "Классификация события",
        "goal": "оценить вероятность наступления события",
        "metrics": [
            {"key": "AUC_EVENT", "label": "AUC события", "direction": "higher", "weight": 0.45},
            {"key": "LOGLOSS_EVENT", "label": "LogLoss события", "direction": "lower", "weight": 0.35},
            {"key": "RMSE_EVENT", "label": "RMSE события", "direction": "lower", "weight": 0.20},
        ],
    },
    "regression": {
        "label": "Прогноз времени",
        "goal": "предсказать ожидаемое время до события",
        "metrics": [
            {"key": "RMSE_TIME", "label": "RMSE времени", "direction": "lower", "weight": 0.30},
            {"key": "R2_TIME", "label": "R2 времени", "direction": "higher", "weight": 0.25},
            {"key": "MAPE_TIME", "label": "MAPE времени", "direction": "lower", "weight": 0.15},
            {"key": "MEDAPE_TIME", "label": "MEDAPE времени", "direction": "lower", "weight": 0.15},
            {"key": "SPEARMAN_TIME", "label": "Spearman времени", "direction": "higher", "weight": 0.10},
            {"key": "RMSLE_TIME", "label": "RMSLE времени", "direction": "lower", "weight": 0.05},
        ],
    },
    "survival": {
        "label": "Анализ выживаемости",
        "goal": "оценить кривую выживаемости и риск во времени",
        "metrics": [
            {"key": "CI", "label": "C-index", "direction": "higher", "weight": 0.45},
            {"key": "IBS", "label": "IBS", "direction": "lower", "weight": 0.35},
            {"key": "AUPRC", "label": "AUPRC", "direction": "higher", "weight": 0.20},
        ],
    },
}

def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _load_local_env_once() -> None:
    global _LOCAL_ENV_LOADED
    if _LOCAL_ENV_LOADED:
        return
    _LOCAL_ENV_LOADED = True

    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].strip()
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            value = value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)


def _env_value(name: str, default: str | None = None) -> str | None:
    _load_local_env_once()
    return os.getenv(name, default)


def _metric_col(metric_key: str) -> str:
    return _norm(metric_key) + "_mean"


def _find_table_path(base_dir: Path, dataset_id: str) -> Path:
    candidates = [
        base_dir / "tables" / f"{dataset_id}.xlsx",
        base_dir / "tables" / f"{dataset_id.upper()}.xlsx",
        base_dir / "tables" / f"{dataset_id.lower()}.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _piecewise_match(method_name: Any):
    return PIECEWISE_RE.match(str(method_name).strip())


def _filter_rows_with_metrics(df: pd.DataFrame, metrics: list[dict[str, Any]]) -> pd.DataFrame:
    if df.empty:
        return df
    has_metric = pd.Series(False, index=df.index)
    for metric in metrics:
        col = _metric_col(metric["key"])
        if col not in df.columns:
            continue
        has_metric = has_metric | pd.to_numeric(df[col], errors="coerce").notna()
    if has_metric.any():
        return df.loc[has_metric].copy()
    return df


def _piecewise_rows_for_advice(
    base_dir: Path,
    dataset_id: str,
    task_id: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if task_id != "classification":
        return pd.DataFrame(), []

    try:
        from .helpers_tables import get_piecewise_table_path, select_global_piecewise_time
    except Exception:
        return pd.DataFrame(), []

    table_path = get_piecewise_table_path(base_dir, dataset_id)
    if not table_path.exists():
        return pd.DataFrame(), []

    try:
        piecewise_df = pd.read_excel(table_path)
    except Exception:
        return pd.DataFrame(), []

    piecewise_df = piecewise_df.rename(columns={column: _norm(column) for column in piecewise_df.columns})
    if "method" not in piecewise_df.columns or piecewise_df.empty:
        return pd.DataFrame(), []

    piecewise_df["method"] = piecewise_df["method"].astype(str).str.strip()
    selected_cache: dict[tuple[str, str], int | None] = {}
    keep_indices = []
    variants: list[dict[str, Any]] = []

    for index, method_name in piecewise_df["method"].items():
        match = _piecewise_match(method_name)
        if not match:
            continue

        family, base_model, times_raw = match.groups()
        times = int(times_raw)
        cache_key = (family, base_model)
        if cache_key not in selected_cache:
            selected_cache[cache_key] = select_global_piecewise_time(
                base_dir,
                family,
                base_model,
                "classification",
            )

        selected_times = selected_cache[cache_key]
        if selected_times is None or times != selected_times:
            continue

        keep_indices.append(index)
        variants.append(
            {
                "method": method_name,
                "family": family,
                "base_model": base_model,
                "times": times,
                "selection": "global",
                "table_path": str(table_path),
            }
        )

    if not keep_indices:
        return pd.DataFrame(), []

    return piecewise_df.loc[keep_indices].copy(), variants


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "нет данных"
    if not np.isfinite(number):
        return "нет данных"
    if abs(number) >= 100:
        return f"{number:.1f}".rstrip("0").rstrip(".")
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _normalize_metric(series: pd.Series, direction: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    low = float(valid.min())
    high = float(valid.max())
    if high == low:
        return pd.Series(np.where(values.notna(), 1.0, np.nan), index=series.index, dtype=float)

    if direction == "higher":
        return (values - low) / (high - low)
    return (high - values) / (high - low)


def _metric_position(df: pd.DataFrame, col: str, direction: str, method: str) -> tuple[int | None, int]:
    values = pd.to_numeric(df[col], errors="coerce")
    total = int(values.notna().sum())
    if total == 0:
        return None, 0

    ascending = direction != "higher"
    ranks = values.rank(ascending=ascending, method="min", na_option="bottom")
    row = df[df["method"].astype(str) == method]
    if row.empty:
        return None, total

    rank_value = ranks.loc[row.index[0]]
    if pd.isna(rank_value):
        return None, total
    return int(rank_value), total


def _active_metrics(df: pd.DataFrame, task_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    metrics = TASK_CONFIGS[task_id]["metrics"]
    active = []
    missing = []
    for metric in metrics:
        col = _metric_col(metric["key"])
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            active.append(metric)
        else:
            missing.append(metric["key"])
    return active, missing


def _score_models(df: pd.DataFrame, metrics: list[dict[str, Any]]) -> pd.DataFrame:
    scored = df.copy()
    weighted_sum = pd.Series(0.0, index=scored.index)
    weight_sum = pd.Series(0.0, index=scored.index)

    for metric in metrics:
        col = _metric_col(metric["key"])
        normalized = _normalize_metric(scored[col], metric["direction"])
        weight = float(metric["weight"])
        present = normalized.notna()
        weighted_sum = weighted_sum.add(normalized.fillna(0.0) * weight, fill_value=0.0)
        weight_sum = weight_sum.add(present.astype(float) * weight, fill_value=0.0)

    raw_score = weighted_sum.divide(weight_sum.replace(0.0, np.nan)).fillna(0.0)
    coverage = weight_sum.divide(sum(float(metric["weight"]) for metric in metrics)).fillna(0.0)
    scored["ai_score"] = raw_score * (0.8 + 0.2 * coverage)
    return scored.sort_values(["ai_score", "method"], ascending=[False, True])


def _model_metrics(df: pd.DataFrame, row: pd.Series, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    method = str(row["method"])
    result = []
    for metric in metrics:
        col = _metric_col(metric["key"])
        value = row.get(col)
        position, total = _metric_position(df, col, metric["direction"], method)
        result.append(
            {
                "key": metric["key"],
                "label": metric["label"],
                "value": _format_number(value),
                "direction": metric["direction"],
                "direction_label": "выше лучше" if metric["direction"] == "higher" else "ниже лучше",
                "position": position,
                "total": total,
            }
        )
    return result


def _build_reason(best_row: pd.Series, top_metrics: list[dict[str, Any]], task_id: str) -> list[str]:
    config = TASK_CONFIGS[task_id]
    strong = [
        metric for metric in top_metrics
        if metric["position"] is not None and metric["position"] <= 3
    ]
    if not strong:
        strong = top_metrics[:2]

    strong_text = ", ".join(
        f"{metric['label']} = {metric['value']} ({metric['direction_label']})"
        for metric in strong[:3]
    )
    score = round(float(best_row["ai_score"]) * 100)

    reasons = [
        f"Модель лучше всего подходит, чтобы {config['goal']}: итоговая оценка по доступным метрикам {score}/100.",
        f"Главные аргументы: {strong_text}.",
        "Рекомендация основана на предрассчитанных таблицах и сравнении моделей внутри выбранного датасета.",
    ]
    match = _piecewise_match(str(best_row.get("method", "")))
    if match:
        family, base_model, times = match.groups()
        reasons.append(
            f"Это Piecewise-вариант: {family} строит интервальные классификаторы поверх {base_model}, "
            f"а times={times} выбран глобально для этой пары по всем датасетам."
        )
    return reasons


def _compact_metric(metric: dict[str, Any]) -> str:
    position = (
        f"{metric['position']} из {metric['total']}"
        if metric.get("position") is not None
        else "нет позиции"
    )
    return (
        f"{metric['label']}: {metric['value']} "
        f"({metric['direction_label']}, позиция {position})"
    )


def _piecewise_fact_lines(advice: dict[str, Any], limit: int = 10) -> list[str]:
    variants = advice.get("piecewise_variants") or []
    if not variants:
        return []
    lines = []
    for variant in variants[:limit]:
        lines.append(
            f"- {variant['method']}: глобально выбран times={variant['times']} "
            f"для {variant['family']} + {variant['base_model']}."
        )
    return lines


def _build_openrouter_messages(advice: dict[str, Any]) -> list[dict[str, str]]:
    top_lines = []
    for model in advice.get("top_models", [])[:3]:
        metrics_text = "; ".join(_compact_metric(metric) for metric in model.get("metrics", []))
        top_lines.append(
            f"{model['position']}. {model['method']} — score {model['score']}/100. Метрики: {metrics_text}."
        )

    user_content = "\n".join(
        [
            f"Датасет: {advice.get('dataset_id')}",
            f"Задача: {advice.get('task_label')}",
            f"Рекомендованная модель: {advice.get('recommended_method')}",
            f"Итоговая оценка: {advice.get('score')}/100",
            f"Piecewise-варианты учтены: {'да' if advice.get('piecewise_included') else 'нет'}",
            "Piecewise-контекст:",
            *(_piecewise_fact_lines(advice) or ["- нет"]),
            "Топ моделей:",
            *top_lines,
            "Напиши интерпретацию на русском языке: какая модель подходит, почему, для какой задачи, и как читать результат.",
        ]
    )

    return [
        {
            "role": "system",
            "content": (
                "Ты ML-аналитик в веб-сервисе сравнения survival-моделей. "
                "Объясняй строго по переданным метрикам, не выдумывай новые результаты. "
                "Пиши кратко: 3-5 предложений, с понятным выводом для защиты проекта. "
                "Если нужны формулы, пиши их в LaTeX delimiters \\( ... \\) или \\[ ... \\]."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _advice_facts(advice: dict[str, Any]) -> str:
    top_lines = []
    for model in advice.get("top_models", [])[:3]:
        metrics_text = "; ".join(_compact_metric(metric) for metric in model.get("metrics", []))
        top_lines.append(
            f"{model['position']}. {model['method']} — score {model['score']}/100. Метрики: {metrics_text}."
        )

    reasons = "\n".join(f"- {reason}" for reason in advice.get("why", []))
    missing = ", ".join(advice.get("missing_metrics", []) or []) or "нет"
    available = ", ".join(advice.get("available_metrics", []) or []) or "нет"

    return "\n".join(
        [
            f"Датасет: {advice.get('dataset_id')}",
            f"Задача: {advice.get('task_label')}",
            f"Рекомендованная модель: {advice.get('recommended_method')}",
            f"Итоговая оценка: {advice.get('score')}/100",
            f"Доступные метрики: {available}",
            f"Отсутствующие метрики: {missing}",
            f"Piecewise-варианты учтены: {'да' if advice.get('piecewise_included') else 'нет'}",
            "Piecewise-контекст:",
            *(_piecewise_fact_lines(advice) or ["- нет"]),
            "Локальное объяснение:",
            reasons,
            "Топ моделей:",
            *top_lines,
        ]
    )


def _normalize_chat_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    normalized = []
    for item in (history or [])[-8:]:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content[:1200]})
    return normalized


def _build_openrouter_chat_messages(
    advice: dict[str, Any],
    question: str,
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "Ты ML-аналитик внутри SAWrap. Отвечай на уточняющие вопросы только по переданным "
                "метрикам и результатам. Если данных недостаточно, прямо скажи, каких метрик или "
                "экспериментов не хватает. Не меняй победителя без объяснения по метрикам. "
                "Ответ должен быть на русском, краткий и прикладной. "
                "Если нужны формулы, пиши их в LaTeX delimiters \\( ... \\) или \\[ ... \\]."
            ),
        },
        {
            "role": "user",
            "content": (
                "Контекст результата:\n"
                f"{_advice_facts(advice)}\n\n"
                "Используй этот контекст для последующего диалога."
            ),
        },
    ]
    messages.extend(_normalize_chat_history(history))
    messages.append({"role": "user", "content": question[:1600]})
    return messages


def _ssl_context():
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _call_openrouter(
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = 420,
    temperature: float = 0.2,
    timeout: float = 20.0,
    opener=urlopen,
) -> dict[str, Any]:
    selected_model = model or _env_value("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
    key = api_key or _env_value("OPENROUTER_API_KEY")

    result = {
        "enabled": bool(key),
        "provider": "OpenRouter",
        "model": selected_model,
        "text": None,
        "error": None,
    }
    if not key:
        result["error"] = "OPENROUTER_API_KEY не задан, используется локальное объяснение."
        return result

    payload = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": _env_value("OPENROUTER_HTTP_REFERER", "https://github.com/DimonZhi/SAWrap"),
        "X-OpenRouter-Title": _env_value("OPENROUTER_APP_TITLE", "SAWrap"),
    }

    request = Request(
        _env_value("OPENROUTER_API_URL", OPENROUTER_API_URL),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with opener(request, timeout=timeout, context=_ssl_context()) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        result["error"] = f"OpenRouter вернул HTTP {exc.code}: {detail[:240]}"
        return result
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        result["error"] = f"OpenRouter недоступен: {exc}"
        return result

    try:
        text = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        result["error"] = "OpenRouter вернул неожиданный формат ответа."
        return result

    result["text"] = str(text).strip()
    return result


def build_openrouter_interpretation(
    advice: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 20.0,
    opener=urlopen,
) -> dict[str, Any]:
    return _call_openrouter(
        _build_openrouter_messages(advice),
        api_key=api_key,
        model=model,
        max_tokens=420,
        temperature=0.2,
        timeout=timeout,
        opener=opener,
    )


def build_openrouter_chat_answer(
    advice: dict[str, Any],
    question: str,
    history: list[dict[str, Any]] | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 20.0,
    opener=urlopen,
) -> dict[str, Any]:
    if not question.strip():
        return {
            "enabled": bool(api_key or _env_value("OPENROUTER_API_KEY")),
            "provider": "OpenRouter",
            "model": model or _env_value("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
            "text": None,
            "error": "Вопрос пустой.",
        }

    return _call_openrouter(
        _build_openrouter_chat_messages(advice, question, history),
        api_key=api_key,
        model=model,
        max_tokens=520,
        temperature=0.2,
        timeout=timeout,
        opener=opener,
    )


def build_ai_advice(base_dir: Path, dataset_id: str, task_id: str, use_llm: bool = False) -> dict[str, Any]:
    if task_id not in TASK_CONFIGS:
        return {
            "has_result": False,
            "error": "Неизвестная задача для интерпретации.",
            "dataset_id": dataset_id,
            "task_id": task_id,
        }

    table_path = _find_table_path(base_dir, dataset_id)
    if not table_path.exists():
        return {
            "has_result": False,
            "error": f"Не найдена таблица результатов для датасета {dataset_id}.",
            "dataset_id": dataset_id,
            "task_id": task_id,
        }

    df = pd.read_excel(table_path)
    df = df.rename(columns={column: _norm(column) for column in df.columns})
    if "method" not in df.columns or df.empty:
        return {
            "has_result": False,
            "error": "В таблице нет столбца method или строк с моделями.",
            "dataset_id": dataset_id,
            "task_id": task_id,
        }

    df["method"] = df["method"].astype(str).str.strip()
    base_df = df[df["method"].ne("")].copy()

    piecewise_rows, piecewise_variants = _piecewise_rows_for_advice(base_dir, dataset_id, task_id)
    df = base_df.copy()
    if not piecewise_rows.empty:
        df = pd.concat([df, piecewise_rows], ignore_index=True, sort=False)
        df = df.drop_duplicates(subset=["method"], keep="last").copy()

    active_metrics, missing_metrics = _active_metrics(df, task_id)

    if not active_metrics:
        return {
            "has_result": False,
            "error": (
                f"Для выбранной задачи «{TASK_CONFIGS[task_id]['label']}» в таблице нет подходящих метрик. "
                "Сначала рассчитай этот пресет для датасета."
            ),
            "dataset_id": dataset_id,
            "task_id": task_id,
            "requested_task_id": task_id,
            "missing_metrics": missing_metrics,
        }

    df = _filter_rows_with_metrics(df, active_metrics)
    scored = _score_models(df, active_metrics)
    top = scored.head(3)
    best = top.iloc[0]
    best_metrics = _model_metrics(df, best, active_metrics)

    top_models = []
    for position, (_, row) in enumerate(top.iterrows(), start=1):
        top_models.append(
            {
                "position": position,
                "method": str(row["method"]),
                "score": round(float(row["ai_score"]) * 100),
                "metrics": _model_metrics(df, row, active_metrics),
            }
        )

    advice = {
        "has_result": True,
        "dataset_id": dataset_id,
        "task_id": task_id,
        "requested_task_id": task_id,
        "task_label": TASK_CONFIGS[task_id]["label"],
        "recommended_method": str(best["method"]),
        "score": round(float(best["ai_score"]) * 100),
        "summary": f"Для датасета {dataset_id} в задаче «{TASK_CONFIGS[task_id]['label']}» лучше всего выглядит {best['method']}.",
        "why": _build_reason(best, best_metrics, task_id),
        "top_models": top_models,
        "available_metrics": [metric["key"] for metric in active_metrics],
        "missing_metrics": missing_metrics,
        "table_path": str(table_path),
        "fallback_task": False,
        "fallback_note": None,
        "piecewise_included": bool(piecewise_variants),
        "piecewise_variants": piecewise_variants,
        "llm": {
            "enabled": False,
            "provider": "OpenRouter",
            "model": _env_value("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
            "text": None,
            "error": None,
        },
    }

    if use_llm:
        advice["llm"] = build_openrouter_interpretation(advice)

    return advice
