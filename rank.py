import os
import pandas as pd
import numpy as np

FILES = [
    #"framingham.xlsx",
    "gbsg.xlsx",
    "rott2.xlsx",
    "smarto.xlsx",
    "pbc.xlsx",
    #"support2.xlsx",
    "actg.xlsx"
]

MODELS = [
    "LogisticRegression",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "ElasticNet",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "SVR",
    "KNeighborsRegressor",
    "KaplanMeierFitter",
    "CoxPHSurvivalAnalysis",
    "RandomSurvivalForest",
    "SurvivalTree",
    "GradientBoostingSurvivalAnalysis",
    "CRAID",
    "SVC",
    "ParallelBootstrapCRAID",
]

CLASSIFICATION_METRICS = [
    "AUPRC_mean",
    "AUC_EVENT_mean",
    "LOGLOSS_EVENT_mean",
    "RMSE_EVENT_mean",
]

REGRESSION_METRICS = [
    "RMSE_TIME_mean",
    "R2_TIME_mean",
    "MAPE_TIME_mean",
    "MEDAPE_TIME_mean",
    "RMSLE_TIME_mean",
    "SPEARMAN_TIME_mean",
]

SURVIVAL_METRICS = [
    "CI_mean",
    "IBS_mean",
]

HIGHER_BETTER = {
    "CI_mean",
    "AUPRC_mean",
    "R2_TIME_mean",
    "SPEARMAN_TIME_mean",
    "AUC_EVENT_mean",
}

OUTPUT_EXCEL = "leaderboards_by_task.xlsx"

def _find_method_col(df):
    for c in df.columns:
        if str(c).strip().upper() == "METHOD":
            return c
    return None

def compute_rank_block(df_metrics, metrics, block_name):
    present = [m for m in metrics if m in df_metrics.columns]
    missing = [m for m in metrics if m not in df_metrics.columns]
    if not present:
        return None, present, missing

    block_df = pd.DataFrame(index=df_metrics.index)
    for m in present:
        block_df[m] = df_metrics[m]
        rank_col = m + "_rank"
        if m in HIGHER_BETTER:
            block_df[rank_col] = df_metrics[m].rank(ascending=False, method="average", na_option="bottom")
        else:
            block_df[rank_col] = df_metrics[m].rank(ascending=True, method="average", na_option="bottom")

    rank_cols = [m + "_rank" for m in present]
    block_df[block_name + "_rank_sum"] = block_df[rank_cols].sum(axis=1, numeric_only=True)
    block_df = block_df.sort_values(block_name + "_rank_sum")
    block_df[block_name + "_position"] = np.arange(1, len(block_df) + 1)

    ordered_cols = []
    for m in present:
        ordered_cols.append(m)
        ordered_cols.append(m + "_rank")
    ordered_cols.append(block_name + "_rank_sum")
    ordered_cols.append(block_name + "_position")

    block_df = block_df[ordered_cols].copy()
    return block_df, present, missing


def rank_dataset_blocks(df, dataset_name):
    method_col = _find_method_col(df)
    if method_col is None:
        raise ValueError(f"[{dataset_name}] miss method.")
    df = df.copy()
    df[method_col] = df[method_col].astype(str).str.strip()
    all_metrics = list(dict.fromkeys(CLASSIFICATION_METRICS + REGRESSION_METRICS + SURVIVAL_METRICS))
    df = df[df[method_col].isin(MODELS)].copy()
    df = df.set_index(method_col)

    results = pd.DataFrame(index=df.index)

    cls, cls_present, cls_missing = compute_rank_block(df, CLASSIFICATION_METRICS, "classification")
    reg, reg_present, reg_missing = compute_rank_block(df, REGRESSION_METRICS, "regression")
    surv, surv_present, surv_missing = compute_rank_block(df, SURVIVAL_METRICS, "survival")

    if cls is not None:
        results = results.join(cls, how="left")
    if reg is not None:
        results = results.join(reg, how="left")
    if surv is not None:
        results = results.join(surv, how="left")

    block_sums = []
    if "classification_rank_sum" in results.columns:
        block_sums.append(results["classification_rank_sum"])
    if "regression_rank_sum" in results.columns:
        block_sums.append(results["regression_rank_sum"])
    if "survival_rank_sum" in results.columns:
        block_sums.append(results["survival_rank_sum"])

    if block_sums:
        results["overall_rank_sum"] = pd.concat(block_sums, axis=1).sum(axis=1)
        results = results.sort_values("overall_rank_sum")
        results["overall_position"] = np.arange(1, len(results) + 1)

    results.insert(0, "Dataset", dataset_name)
    results = results.reset_index().rename(columns={method_col: "Method", "index": "Method"})

    diag = {
        "Dataset": dataset_name,
        "Rows_after_model_filter": int(results.shape[0]),
        "classification_present": ", ".join(cls_present),
        "classification_missing": ", ".join(cls_missing),
        "regression_present": ", ".join(reg_present),
        "regression_missing": ", ".join(reg_missing),
        "survival_present": ", ".join(surv_present),
        "survival_missing": ", ".join(surv_missing),
    }
    return results, diag

def aggregate_overall(per_dataset_tables):
    long_rows = []
    for ds, tbl in per_dataset_tables.items():
        cols = [
            "Method",
            "classification_rank_sum","classification_position",
            "regression_rank_sum","regression_position",
            "survival_rank_sum","survival_position",
            "overall_rank_sum","overall_position"
        ]
        take = [c for c in cols if c in tbl.columns]
        t = tbl[take].copy()
        t["Dataset"] = ds
        long_rows.append(t)

    long = pd.concat(long_rows, ignore_index=True)

    def overall_table(sum_col, pos_col, sheet_label):
        if sum_col not in long.columns:
            return None
        g = long.groupby("Method", as_index=False).agg(
            Datasets=("Dataset", "nunique"),
            Avg_RankSum=(sum_col, "mean"),
        )
        if pos_col in long.columns:
            g["Avg_Position"] = long.groupby("Method")[pos_col].mean().values
            g = g.sort_values(["Avg_RankSum", "Avg_Position"], ascending=True)
        else:
            g = g.sort_values(["Avg_RankSum"], ascending=True)
        g["Overall_position"] = np.arange(1, len(g) + 1)
        g.insert(0, "Task", sheet_label)
        return g.reset_index(drop=True)

    cls_overall = overall_table("classification_rank_sum", "classification_position", "CLASSIFICATION")
    reg_overall = overall_table("regression_rank_sum", "regression_position", "REGRESSION")
    surv_overall = overall_table("survival_rank_sum", "survival_position", "SURVIVAL")
    all_overall = overall_table("overall_rank_sum", "overall_position", "ALL")

    return long, cls_overall, reg_overall, surv_overall, all_overall

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    paths = [os.path.join(script_dir, f) for f in FILES]

    print("Script dir:", script_dir)
    print("Looking for files:")
    for p in paths:
        print(" ", os.path.basename(p), "|", "OK" if os.path.exists(p) else "MISSING")

    per_dataset = {}
    diagnostics = []

    for path in paths:
        ds = os.path.splitext(os.path.basename(path))[0]

        if not os.path.exists(path):
            diagnostics.append({"Dataset": ds, "Status": "missing_file", "Path": path})
            continue

        try:
            df = pd.read_excel(path)
        except Exception as e:
            diagnostics.append({"Dataset": ds, "Status": "read_error", "Path": path, "Error": str(e)})
            continue

        try:
            tbl, diag = rank_dataset_blocks(df, ds)
            per_dataset[ds] = tbl
            diag["Status"] = "ok"
            diagnostics.append(diag)
        except Exception as e:
            diagnostics.append({"Dataset": ds, "Status": "process_error", "Path": path, "Error": str(e)})

    diag_df = pd.DataFrame(diagnostics)

    if not per_dataset:
        print("No datasets processed. Diagnostics:")
        print(diag_df)
        return

    long, cls_overall, reg_overall, surv_overall, all_overall = aggregate_overall(per_dataset)

    out_path = os.path.join(script_dir, OUTPUT_EXCEL)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        diag_df.to_excel(writer, sheet_name="DIAGNOSTICS", index=False)
        long.to_excel(writer, sheet_name="LONG_PER_DATASET", index=False)

        if cls_overall is not None:
            cls_overall.to_excel(writer, sheet_name="OVERALL_CLASSIFICATION", index=False)
        if reg_overall is not None:
            reg_overall.to_excel(writer, sheet_name="OVERALL_REGRESSION", index=False)
        if surv_overall is not None:
            surv_overall.to_excel(writer, sheet_name="OVERALL_SURVIVAL", index=False)
        if all_overall is not None:
            all_overall.to_excel(writer, sheet_name="OVERALL_ALL", index=False)

        for ds, tbl in per_dataset.items():
            tbl.to_excel(writer, sheet_name=ds[:31], index=False)

    print("Saved:", out_path)

if __name__ == "__main__":
    main()