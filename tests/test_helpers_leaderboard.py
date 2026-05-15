from pathlib import Path
import os

import pandas as pd

from UI.helpers_leaderboard import (
    _as_int,
    _format_number,
    load_leaderboard_images,
    load_overall_leaderboard_rows,
)


def test_format_number_and_int_helpers_handle_display_values():
    assert _format_number(3.0) == "3"
    assert _format_number(3.456) == "3.46"
    assert _format_number(None) == "—"

    assert _as_int("7.0") == 7
    assert _as_int("bad") is None


def test_load_overall_leaderboard_rows_sorts_and_marks_top_three(tmp_path: Path):
    table_path = tmp_path / "leaderboard.xlsx"
    df = pd.DataFrame(
        [
            {
                " Task ": "ALL",
                "Method": "SecondModel",
                "Datasets": 7,
                "Avg_RankSum": 2.25,
                "Avg_Position": 2.0,
                "Overall_position": 2,
            },
            {
                " Task ": "SURVIVAL",
                "Method": "FirstModel",
                "Datasets": 7,
                "Avg_RankSum": 1.5,
                "Avg_Position": 1.0,
                "Overall_position": 1,
            },
        ]
    )
    with pd.ExcelWriter(table_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="OVERALL_ALL", index=False)

    rows = load_overall_leaderboard_rows(table_path)

    assert [row["method"] for row in rows] == ["FirstModel", "SecondModel"]
    assert rows[0]["task"] == "Выживаемость"
    assert rows[0]["avg_position"] == "1"
    assert rows[0]["is_top_three"] is True


def test_load_overall_leaderboard_rows_returns_empty_for_missing_file(tmp_path: Path):
    assert load_overall_leaderboard_rows(tmp_path / "missing.xlsx") == []


def test_load_leaderboard_images_returns_latest_supported_images(tmp_path: Path):
    old_image = tmp_path / "Classif_vs_Regr.png"
    ignored_file = tmp_path / "notes.txt"
    newest_image = tmp_path / "custom_chart.jpg"
    for path in [old_image, ignored_file, newest_image]:
        path.write_bytes(b"test")

    os.utime(old_image, (100, 100))
    os.utime(ignored_file, (200, 200))
    os.utime(newest_image, (300, 300))

    figures = load_leaderboard_images(tmp_path, limit=2)

    assert [figure["src"] for figure in figures] == [
        "/images/Classif_vs_Regr.png",
        "/images/custom_chart.jpg",
    ]
    assert figures[0]["title"] == "Классификация vs Регрессия"
    assert figures[1]["title"] == "custom chart"
