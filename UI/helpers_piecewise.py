from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .helpers_ai_advice import TASK_CONFIGS, _score_models
from .helpers_tables import select_global_piecewise_time


PIECEWISE_FILES = {
    "ACTG": "Piecewise_actg.xlsx",
    "Framingham": "Piecewise_framingham.xlsx",
    "GBSG": "Piecewise_gbsg.xlsx",
    "PBC": "Piesewise_pbc.xlsx",
    "Rott2": "Piecewise_rott2.xlsx",
    "Smarto": "Piecewise_smarto.xlsx",
    "Support2": "Piecewise_support2.xlsx",
}

PIECEWISE_RE = re.compile(
    r"^(Piecewise(?:CensorAware)?ClassifWrapSA)\(([^,]+),\s*times=(\d+)\)$"
)
BASE_CLASSIFIER_RE = re.compile(r"^ClassifWrapSA\((.+)\)$")


def _fmt(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}"


def _variant_label(raw_name: str) -> str:
    match = PIECEWISE_RE.match(raw_name)
    if not match:
        return raw_name
    family, model, times = match.groups()
    short_family = "Censor-aware" if family == "PiecewiseCensorAwareClassifWrapSA" else "Piecewise"
    return f"{short_family}, times={times}"


def _read_piecewise_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_excel(path)
    except Exception:
        return None
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "method" not in frame.columns:
        return None
    return frame


def _best_piecewise_rows(tables_dir: Path) -> list[dict[str, Any]]:
    metrics = TASK_CONFIGS["classification"]["metrics"]
    rows: list[dict[str, Any]] = []
    piecewise_base_dir = tables_dir.parent if tables_dir.name == "tables" else tables_dir
    selected_times_cache: dict[tuple[str, str], int | None] = {}

    for dataset_label, filename in PIECEWISE_FILES.items():
        frame = _read_piecewise_table(tables_dir / filename)
        if frame is None:
            continue

        scored = _score_models(frame, metrics)
        for method_name in frame["method"].astype(str):
            base_match = BASE_CLASSIFIER_RE.match(method_name)
            if not base_match:
                continue

            model_name = base_match.group(1)
            base_rows = scored[scored["method"].astype(str) == method_name]
            if base_rows.empty:
                continue
            base_row = base_rows.iloc[0]

            for _, candidate in scored.iterrows():
                candidate_name = str(candidate["method"])
                piece_match = PIECEWISE_RE.match(candidate_name)
                if not piece_match or piece_match.group(2) != model_name:
                    continue
                times = int(piece_match.group(3))
                cache_key = (piece_match.group(1), model_name)
                if cache_key not in selected_times_cache:
                    selected_times_cache[cache_key] = select_global_piecewise_time(
                        piecewise_base_dir,
                        piece_match.group(1),
                        model_name,
                        "classification",
                    )
                selected_times = selected_times_cache[cache_key]
                if selected_times is None or times != selected_times:
                    continue
                base_score = float(base_row["ai_score"]) * 100.0
                piece_score = float(candidate["ai_score"]) * 100.0
                rows.append(
                    {
                        "score": piece_score,
                        "times": times,
                        "family": piece_match.group(1),
                        "method": candidate_name,
                        "row": candidate,
                        "dataset": dataset_label,
                        "model": model_name,
                        "base_score": base_score,
                        "piecewise_score": piece_score,
                        "delta_score": piece_score - base_score,
                        "variant": candidate_name,
                        "variant_label": _variant_label(candidate_name),
                        "base_auc": float(base_row.get("auc_event_mean", 0.0)),
                        "piece_auc": float(candidate.get("auc_event_mean", 0.0)),
                        "base_logloss": float(base_row.get("logloss_event_mean", 0.0)),
                        "piece_logloss": float(candidate.get("logloss_event_mean", 0.0)),
                        "base_rmse": float(base_row.get("rmse_event_mean", 0.0)),
                        "piece_rmse": float(candidate.get("rmse_event_mean", 0.0)),
                    }
                )

    return rows


def load_piecewise_classification_summary(tables_dir: Path) -> dict[str, Any]:
    rows = _best_piecewise_rows(tables_dir)
    if not rows:
        return {
            "available": False,
            "summary_cards": [],
            "model_rows": [],
            "detail_rows": [],
        }

    frame = pd.DataFrame(rows)
    positive = int((frame["delta_score"] > 0).sum())
    total = int(len(frame))
    decision_rows = frame[frame["model"] == "DecisionTreeClassifier"].copy()

    model_rows = []
    for (family, model), group in frame.groupby(["family", "model"]):
        best_times = ", ".join(str(int(value)) for value in sorted(group["times"].unique()))
        model_rows.append(
            {
                "model": model,
                "family": family,
                "piecewise_model": f"{family}({model})",
                "datasets": int(group["dataset"].nunique()),
                "base_score": _fmt(group["base_score"].mean()),
                "piecewise_score": _fmt(group["piecewise_score"].mean()),
                "delta_score": _fmt(group["delta_score"].mean()),
                "win_rate": _fmt((group["delta_score"] > 0).mean() * 100, 0),
                "best_times": best_times,
                "is_strong": model == "DecisionTreeClassifier",
            }
        )
    model_rows.sort(key=lambda row: float(row["delta_score"]), reverse=True)

    detail_rows = []
    if not decision_rows.empty:
        for _, row in decision_rows.sort_values("delta_score", ascending=False).iterrows():
            detail_rows.append(
                {
                    "dataset": row["dataset"],
                    "base_score": _fmt(row["base_score"]),
                    "piecewise_score": _fmt(row["piecewise_score"]),
                    "delta_score": _fmt(row["delta_score"]),
                    "times": int(row["times"]),
                    "variant_label": row["variant_label"],
                    "auc_delta": _fmt((row["piece_auc"] - row["base_auc"]) * 100, 2),
                    "logloss_reduction": _fmt(
                        (row["base_logloss"] - row["piece_logloss"]) / abs(row["base_logloss"]) * 100
                        if row["base_logloss"]
                        else None,
                        1,
                    ),
                }
            )

    dt_avg_delta = float(decision_rows["delta_score"].mean()) if not decision_rows.empty else 0.0
    dt_win_rate = float((decision_rows["delta_score"] > 0).mean() * 100) if not decision_rows.empty else 0.0
    best_case = frame.sort_values("delta_score", ascending=False).iloc[0]

    return {
        "available": True,
        "summary_cards": [
            {
                "label": "DecisionTreeClassifier",
                "value": f"+{_fmt(dt_avg_delta)}",
                "caption": "средний прирост classification score при глобальном выборе times",
            },
            {
                "label": "Стабильность",
                "value": f"{_fmt(dt_win_rate, 0)}%",
                "caption": "датасетов, где Piecewise улучшил DecisionTreeClassifier",
            },
            {
                "label": "Лучший кейс",
                "value": f"+{_fmt(float(best_case['delta_score']))}",
                "caption": f"{best_case['dataset']}, {best_case['model']}, times={int(best_case['times'])}",
            },
            {
                "label": "Все классификаторы",
                "value": f"{positive}/{total}",
                "caption": "пар dataset/model с улучшением score",
            },
        ],
        "model_rows": model_rows,
        "detail_rows": detail_rows,
        "times_tested": "глобальный автовыбор одного times на Piecewise-модель",
        "score_note": "Score = AUC_EVENT 45%, LOGLOSS_EVENT 35%, RMSE_EVENT 20%.",
    }
