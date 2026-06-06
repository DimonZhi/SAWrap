from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import joblib
import numpy as np
import pandas as pd

from survivors.external import ClassifWrapSA, RegrWrapSA, SAWrapSA

from .helpers_ai_advice import TASK_CONFIGS, _score_models
from .helpers_runtime_piecewise import PIECEWISE_RUNTIME_CLASSES
from .helpers_tables import (
    get_piecewise_table_path,
    get_surv_table_path,
    import_class,
    normalize_surv_df,
)


MODEL_STORE_DIR = "model_store"
DEMO_CANDIDATE_LIMIT = 1
DEMO_MODEL_LABEL = "ParallelBootstrapCRAID"
PIECEWISE_RE = re.compile(
    r"^(Piecewise(?:CensorAware)?ClassifWrapSA)\(([^,]+),\s*times=(\d+)\)$"
)


def _norm(value) -> str:
    return str(value).strip().lower()


def _metric_col(metric_key: str) -> str:
    return _norm(metric_key) + "_mean"


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "model"


def _format_number(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "нет данных"
    if not np.isfinite(number):
        return "нет данных"
    if abs(number) >= 100:
        return f"{number:.1f}".rstrip("0").rstrip(".")
    return f"{number:.3f}".rstrip("0").rstrip(".")


def build_demo_input_context(dataset_id: str, X: pd.DataFrame | None) -> dict | None:
    if X is None or X.empty:
        return None

    features = []
    for name in X.columns:
        values = pd.to_numeric(X[name], errors="coerce")
        median = values.median()
        if pd.isna(median):
            median = 0.0
        features.append(
            {
                "name": str(name),
                "label": str(name),
                "value": float(median),
            }
        )

    return {
        "dataset_id": dataset_id,
        "features": features,
    }


def _prepare_training_frame(X: pd.DataFrame) -> pd.DataFrame:
    prepared = X.copy()
    for column in prepared.columns:
        if not prepared[column].isna().any():
            continue
        numeric = pd.to_numeric(prepared[column], errors="coerce")
        if numeric.notna().any():
            fill_value = numeric.median()
            if pd.isna(fill_value):
                fill_value = 0.0
            prepared[column] = numeric.fillna(fill_value)
            continue
        mode = prepared[column].dropna().mode()
        fill_value = mode.iloc[0] if not mode.empty else ""
        prepared[column] = prepared[column].fillna(fill_value)
    return prepared


def _active_metrics(df: pd.DataFrame, task_id: str) -> list[dict]:
    metrics = TASK_CONFIGS[task_id]["metrics"]
    active = []
    for metric in metrics:
        col = _metric_col(metric["key"])
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            active.append(metric)
    return active


def _candidate_metrics(df: pd.DataFrame, task_id: str) -> list[dict]:
    active = _active_metrics(df, task_id)
    if active:
        return active

    seen = set()
    fallback = []
    for config in TASK_CONFIGS.values():
        for metric in config["metrics"]:
            if metric["key"] in seen:
                continue
            seen.add(metric["key"])
            col = _metric_col(metric["key"])
            if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
                fallback.append(metric)
    return fallback


def _load_result_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return normalize_surv_df(pd.read_excel(path))
    except Exception:
        return pd.DataFrame()


def _best_row_for_task(df: pd.DataFrame, labels: set[str], task_id: str) -> tuple[pd.Series | None, list[dict]]:
    if df.empty or "method" not in df.columns:
        return None, []

    work = df[df["method"].astype(str).isin(labels)].copy()
    if work.empty:
        return None, []

    metrics = _candidate_metrics(work, task_id)
    if not metrics:
        return None, []

    scored = _score_models(work, metrics)
    if scored.empty:
        return None, metrics
    return scored.iloc[0], metrics


def _best_piecewise_rows(df: pd.DataFrame) -> list[tuple[str, str, pd.Series, list[dict]]]:
    if df.empty or "method" not in df.columns:
        return []
    work = df[df["method"].astype(str).str.match(PIECEWISE_RE)].copy()
    if work.empty:
        return []

    work["__piecewise_family__"] = work["method"].astype(str).str.extract(PIECEWISE_RE, expand=True)[0]
    result = []
    labels = {
        "PiecewiseClassifWrapSA": "Piecewise",
        "PiecewiseCensorAwareClassifWrapSA": "Piecewise censor-aware",
    }
    for family, label in labels.items():
        family_df = work[work["__piecewise_family__"] == family].copy()
        if family_df.empty:
            continue
        metrics = _candidate_metrics(family_df, "classification")
        if not metrics:
            continue
        scored = _score_models(family_df, metrics)
        if scored.empty:
            continue
        result.append((family, label, scored.iloc[0], metrics))
    return result


def _best_demo_candidates(candidates: list[dict], limit: int = DEMO_CANDIDATE_LIMIT) -> list[dict]:
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (float(item[1].get("score") or 0.0), -item[0]),
        reverse=True,
    )
    return [candidate for _, candidate in ranked[:limit]]


def _forced_demo_candidate(df: pd.DataFrame, method_label: str = DEMO_MODEL_LABEL) -> dict | None:
    if df.empty or "method" not in df.columns:
        return None

    work = df[df["method"].astype(str).str.strip() == method_label].copy()
    if work.empty:
        return None

    metrics = _candidate_metrics(work, "survival")
    if not metrics:
        return None

    scored = _score_models(work, metrics)
    if scored.empty:
        return None

    row = scored.iloc[0]
    return {
        "category": "survival",
        "category_label": "Анализ выживаемости",
        "method": str(row["method"]),
        "score": float(row.get("ai_score", 0.0)),
        "params": row.get("params"),
        "task_id": "survival",
        "metrics": [metric["key"] for metric in metrics],
    }


def find_demo_candidates(base_dir: Path, dataset_id: str, model_cfgs: list[dict]) -> list[dict]:
    regular_df = _load_result_table(get_surv_table_path(base_dir, dataset_id))
    forced_candidate = _forced_demo_candidate(regular_df)
    if forced_candidate is not None:
        return [forced_candidate]

    piecewise_df = _load_result_table(get_piecewise_table_path(base_dir, dataset_id))

    labels_by_task = {
        task: {cfg["label"] for cfg in model_cfgs if cfg.get("task") == task}
        for task in ("classification", "regression", "survival")
    }

    specs = [
        ("classification", "Классификация", regular_df, labels_by_task["classification"], "classification"),
        ("regression", "Регрессия", regular_df, labels_by_task["regression"], "regression"),
        ("survival", "Анализ выживаемости", regular_df, labels_by_task["survival"], "survival"),
    ]

    candidates = []
    for category, label, df, labels, task_id in specs:
        row, metrics = _best_row_for_task(df, labels, task_id)
        if row is None:
            continue
        candidates.append(
            {
                "category": category,
                "category_label": label,
                "method": str(row["method"]),
                "score": float(row.get("ai_score", 0.0)),
                "params": row.get("params"),
                "task_id": task_id,
                "metrics": [metric["key"] for metric in metrics],
            }
        )

    for family, label, row, metrics in _best_piecewise_rows(piecewise_df):
        candidates.append(
            {
                "category": _safe_slug(family),
                "category_label": label,
                "method": str(row["method"]),
                "score": float(row.get("ai_score", 0.0)),
                "params": row.get("params"),
                "task_id": "classification",
                "metrics": [metric["key"] for metric in metrics],
            }
        )

    return _best_demo_candidates(candidates)


def _parse_params(value) -> dict:
    if isinstance(value, dict):
        return value
    if value is None or pd.isna(value):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _model_cfg_by_label(model_cfgs: list[dict], label: str) -> dict | None:
    return next((cfg for cfg in model_cfgs if cfg["label"] == label), None)


def _model_cfg_for_piecewise(model_cfgs: list[dict], method_label: str) -> dict | None:
    match = PIECEWISE_RE.match(method_label)
    if not match:
        return None
    _, base_model, _ = match.groups()
    return next((cfg for cfg in model_cfgs if cfg["label"] == base_model), None)


def _demo_fit_params(cfg: dict, params: dict) -> dict:
    fit_params = dict(params)
    model_id = str(cfg.get("id", ""))
    if model_id in {"survivors.tree.CRAID", "survivors.ensemble.ParallelBootstrapCRAID"}:
        fit_params["n_jobs"] = 1
    return fit_params


def _build_wrapped_model(candidate: dict, model_cfgs: list[dict]):
    method_label = candidate["method"]
    params = _parse_params(candidate.get("params"))
    match = PIECEWISE_RE.match(method_label)

    if match:
        family, _, times = match.groups()
        cfg = _model_cfg_for_piecewise(model_cfgs, method_label)
        if cfg is None:
            raise ValueError(f"Нет базовой модели для {method_label}")
        Est = import_class(cfg["id"])
        params = _demo_fit_params(cfg, params)
        return PIECEWISE_RUNTIME_CLASSES[family](Est(**params), times=int(times))

    cfg = _model_cfg_by_label(model_cfgs, method_label)
    if cfg is None:
        raise ValueError(f"Модель не найдена: {method_label}")
    Est = import_class(cfg["id"])
    params = _demo_fit_params(cfg, params)
    raw_model = Est(**params)
    if cfg["task"] == "classification":
        return ClassifWrapSA(raw_model)
    if cfg["task"] == "regression":
        return RegrWrapSA(raw_model)
    if cfg["task"] == "survival":
        return SAWrapSA(raw_model)
    raise ValueError(f"Неизвестная задача: {cfg['task']}")


def _model_cache_path(base_dir: Path, dataset_id: str, candidate: dict) -> Path:
    params = json.dumps(_parse_params(candidate.get("params")), sort_keys=True, default=str)
    params_hash = hashlib.sha1(params.encode("utf-8")).hexdigest()[:12]
    return (
        base_dir
        / MODEL_STORE_DIR
        / _safe_slug(dataset_id)
        / f"{candidate['category']}__{_safe_slug(candidate['method'])}__{params_hash}.joblib"
    )


def _legacy_model_cache_path(base_dir: Path, dataset_id: str, candidate: dict) -> Path:
    params = json.dumps(_parse_params(candidate.get("params")), sort_keys=True, default=str)
    return (
        base_dir
        / MODEL_STORE_DIR
        / _safe_slug(dataset_id)
        / f"{candidate['category']}__{_safe_slug(candidate['method'])}__{_safe_slug(params)}.joblib"
    )


def _load_cached_model(path: Path):
    try:
        if path.exists():
            return joblib.load(path)
    except OSError:
        return None
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return None


def _fit_or_load_model(base_dir: Path, dataset_id: str, candidate: dict, model_cfgs: list[dict], X: pd.DataFrame, y):
    cache_path = _model_cache_path(base_dir, dataset_id, candidate)
    for candidate_cache_path in (cache_path, _legacy_model_cache_path(base_dir, dataset_id, candidate)):
        cached_model = _load_cached_model(candidate_cache_path)
        if cached_model is not None:
            return cached_model, True

    model = _build_wrapped_model(candidate, model_cfgs)
    model.fit(X, y, time_col="time", event_col="cens")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, cache_path)
        meta = {
            "dataset_id": dataset_id,
            "category": candidate["category"],
            "method": candidate["method"],
            "params": _parse_params(candidate.get("params")),
            "features": list(X.columns),
        }
        cache_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return model, False


def _build_manual_frame(X: pd.DataFrame, raw_values: dict[str, str]) -> pd.DataFrame:
    row = {}
    for column in X.columns:
        values = pd.to_numeric(X[column], errors="coerce")
        median = values.median()
        if pd.isna(median):
            median = 0.0
        raw = raw_values.get(str(column), "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float(median)
        row[column] = value
    return pd.DataFrame([row], columns=X.columns)


def build_demo_prediction(
    base_dir: Path,
    dataset_id: str,
    raw_values: dict[str, str],
    model_cfgs: list[dict],
    load_dataset: Callable[[str], tuple],
) -> dict:
    loaded = load_dataset(dataset_id)
    if loaded is None:
        return {"ok": False, "error": "Датасет недоступен для обучения демо-моделей."}

    X, y, *_ = loaded
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = _prepare_training_frame(X)
    candidates = find_demo_candidates(base_dir, dataset_id, model_cfgs)
    if not candidates:
        return {
            "ok": False,
            "error": f"Нет предрасчитанных результатов для {DEMO_MODEL_LABEL}.",
        }

    sample = _build_manual_frame(X, raw_values)
    times = np.linspace(0.0, float(np.nanmax(y["time"])), 128)
    traces = []
    cards = []
    errors = []

    for candidate in candidates:
        try:
            model, from_cache = _fit_or_load_model(base_dir, dataset_id, candidate, model_cfgs, X, y)
            survival, curve_times = model.predict_survival_function(sample, times=times)
            survival_values = np.asarray(survival, float)[0]
            expected_time = model.predict_expected_time(sample, times=times)
            expected_value = float(np.asarray(expected_time, float).ravel()[0])
            risk = float(1.0 - survival_values[-1]) if survival_values.size else float("nan")
        except Exception as exc:
            errors.append(f"{candidate['category_label']}: {exc}")
            continue

        trace_name = f"{candidate['category_label']}: {candidate['method']}"
        traces.append(
            {
                "name": trace_name,
                "category": candidate["category"],
                "x": [float(value) for value in np.asarray(curve_times, float)],
                "y": [float(value) for value in survival_values],
            }
        )
        cards.append(
            {
                "category": candidate["category"],
                "category_label": candidate["category_label"],
                "method": candidate["method"],
                "expected_time": _format_number(expected_value),
                "risk": _format_number(risk),
                "source": "cache" if from_cache else "fresh",
            }
        )

    if not traces:
        return {
            "ok": False,
            "error": "Не удалось построить ни один демо-прогноз.",
            "errors": errors,
        }

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "traces": traces,
        "cards": cards,
        "errors": errors,
    }
