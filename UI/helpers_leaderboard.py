from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


OVERALL_LEADERBOARD_SHEET = "OVERALL_ALL"

TASK_LABELS = {
    "ALL": "Все задачи",
    "CLASSIFICATION": "Классификация",
    "REGRESSION": "Регрессия",
    "SURVIVAL": "Выживаемость",
}

IMAGE_TITLES = {
    "Classif_vs_Regr": "Классификация vs Регрессия",
    "Classif_vs_SA": "Классификация vs Выживаемость",
    "Regr_vs_SA": "Регрессия vs Выживаемость",
}


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_overall_leaderboard_rows(table_path: Path) -> list[dict[str, Any]]:
    if not table_path.exists():
        return []

    df = pd.read_excel(table_path, sheet_name=OVERALL_LEADERBOARD_SHEET)
    if df.empty:
        return []

    df = df.rename(columns=lambda name: str(name).strip())
    if "Overall_position" in df.columns:
        df = df.sort_values("Overall_position")

    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        position = _as_int(row.get("Overall_position"))
        rows.append(
            {
                "position": position,
                "method": str(row.get("Method", "—")),
                "task": TASK_LABELS.get(str(row.get("Task", "")).upper(), str(row.get("Task", "—"))),
                "datasets": _as_int(row.get("Datasets")),
                "avg_rank_sum": _format_number(row.get("Avg_RankSum")),
                "avg_position": _format_number(row.get("Avg_Position")),
                "is_top_three": position is not None and position <= 3,
            }
        )
    return rows


def _figure_meta(image_path: Path) -> tuple[str, str]:
    stem = image_path.stem
    title = IMAGE_TITLES.get(stem, stem.replace("_", " "))
    return title, "Источник: UI/images"


def load_leaderboard_images(image_dir: Path, limit: int = 3) -> list[dict[str, str]]:
    if not image_dir.exists():
        return []
    if limit <= 0:
        return []

    figures: list[dict[str, str]] = []
    image_paths = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    ]
    image_paths.sort(key=lambda path: path.stat().st_mtime)

    for image_path in image_paths[-limit:]:
        title, caption = _figure_meta(image_path)
        figures.append(
            {
                "title": title,
                "caption": caption,
                "src": f"/images/{image_path.name}",
            }
        )
    return figures
