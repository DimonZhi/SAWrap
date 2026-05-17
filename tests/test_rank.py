import pandas as pd

from rank import (
    aggregate_overall,
    compute_global_piecewise_times,
    compute_rank_block,
    extract_piecewise_rows,
    rank_dataset_blocks,
)


def test_compute_rank_block_respects_metric_direction():
    df = pd.DataFrame(
        {
            "CI_mean": [0.70, 0.85, 0.60],
            "IBS_mean": [0.20, 0.10, 0.30],
        },
        index=["middle", "best", "worst"],
    )

    block, present, missing = compute_rank_block(
        df,
        ["CI_mean", "IBS_mean", "MISSING_mean"],
        "survival",
    )

    assert present == ["CI_mean", "IBS_mean"]
    assert missing == ["MISSING_mean"]
    assert list(block.index) == ["best", "middle", "worst"]
    assert list(block["survival_position"]) == [1, 2, 3]


def test_rank_dataset_blocks_filters_methods_and_reports_missing_metrics():
    df = pd.DataFrame(
        {
            "METHOD": [" Alpha ", "Beta", "Ignored"],
            "AUC_EVENT_mean": [0.9, 0.8, 1.0],
            "LOGLOSS_EVENT_mean": [0.2, 0.4, 0.1],
            "CI_mean": [0.7, 0.6, 0.9],
        }
    )

    ranked, diag = rank_dataset_blocks(
        df,
        dataset_name="toy",
        allowed_methods=["Alpha", "Beta"],
    )

    assert ranked["Method"].tolist() == ["Alpha", "Beta"]
    assert ranked["Dataset"].tolist() == ["toy", "toy"]
    assert diag["Rows_after_model_filter"] == 2
    assert "RMSE_EVENT_mean" in diag["classification_missing"]
    assert "IBS_mean" in diag["survival_missing"]


def test_extract_piecewise_rows_keeps_best_available_times_and_prefixes():
    df = pd.DataFrame(
        {
            "METHOD": [
                "PiecewiseClassifWrapSA(LogisticRegression, times=16)",
                "PiecewiseClassifWrapSA(LogisticRegression, times=8)",
                "PiecewiseCensorAwareClassifWrapSA(RandomForestClassifier, times=16)",
                "PiecewiseCensorAwareClassifWrapSA(RandomForestClassifier, times=4)",
                "LogisticRegression",
            ],
            "AUC_EVENT_mean": [0.70, 0.90, 0.85, 0.75, 0.6],
            "LOGLOSS_EVENT_mean": [0.50, 0.20, 0.30, 0.45, 0.6],
            "RMSE_EVENT_mean": [0.40, 0.20, 0.25, 0.35, 0.5],
        }
    )

    piecewise, method_col = extract_piecewise_rows(
        df,
        dataset_name="toy",
        target_method_col="Method",
    )

    assert method_col == "Method"
    assert sorted(piecewise["Method"].tolist()) == sorted([
        "PiecewiseClassifWrapSA(LogisticRegression, times=8)",
        "PiecewiseCensorAwareClassifWrapSA(RandomForestClassifier, times=16)",
    ])


def test_global_piecewise_times_prefers_one_times_across_datasets(tmp_path):
    pd.DataFrame(
        {
            "METHOD": [
                "PiecewiseClassifWrapSA(LogisticRegression, times=8)",
                "PiecewiseClassifWrapSA(LogisticRegression, times=16)",
            ],
            "AUC_EVENT_mean": [0.90, 0.70],
        }
    ).to_excel(tmp_path / "Piecewise_framingham.xlsx", index=False)
    pd.DataFrame(
        {
            "METHOD": [
                "PiecewiseClassifWrapSA(LogisticRegression, times=16)",
            ],
            "AUC_EVENT_mean": [0.80],
        }
    ).to_excel(tmp_path / "Piecewise_gbsg.xlsx", index=False)

    selected = compute_global_piecewise_times(tmp_path)

    assert selected[("PiecewiseClassifWrapSA", "LogisticRegression")] == 16


def test_aggregate_overall_orders_by_mean_task_median():
    per_dataset_tables = {
        "ds1": pd.DataFrame(
            {
                "Method": ["A", "B"],
                "classification_position": [1, 2],
                "regression_position": [2, 1],
                "survival_position": [1, 2],
                "overall_position": [1, 2],
                "overall_rank_sum": [4, 5],
            }
        ),
        "ds2": pd.DataFrame(
            {
                "Method": ["A", "B"],
                "classification_position": [1, 2],
                "regression_position": [1, 2],
                "survival_position": [1, 2],
                "overall_position": [1, 2],
                "overall_rank_sum": [3, 6],
            }
        ),
    }

    _, _, _, _, all_overall = aggregate_overall(per_dataset_tables)

    assert all_overall["Method"].tolist() == ["A", "B"]
    assert all_overall.loc[0, "Overall_position"] == 1
    assert all_overall.loc[0, "Datasets"] == 2
