import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import root_mean_squared_error, r2_score, roc_auc_score, log_loss
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from lifelines import KaplanMeierFitter

from survivors.external import ClassifWrapSA, RegrWrapSA, SAWrapSA
from survivors.experiments import grid as exp
from survivors.datasets import other as ds_other
from survivors.tree import CRAID
from survivors.ensemble import ParallelBootstrapCRAID

warnings.filterwarnings("ignore")

DATASETS = {
    #"framingham": lambda: ds_other.load_Framingham_dataset(),
    #"actg": lambda: ds_other.load_actg_dataset(),
    ##"flchain": lambda: ds_other.load_flchain_dataset(),
    #"gbsg": lambda: ds_other.load_gbsg_dataset(),
    #"rott2": lambda: ds_other.load_rott2_dataset(),
    #"smarto": lambda: ds_other.load_smarto_dataset(),
    "support2": lambda: ds_other.load_support2_dataset(),
    ##"wuhan": lambda: ds_other.load_wuhan_dataset(invert_death=False),
}

def find_sf_at_truetime(pred_sf, event_time, bins):
    idx_pred = np.clip(np.searchsorted(bins, event_time), 0, len(bins) - 1)
    proba = np.take_along_axis(pred_sf, idx_pred[:, np.newaxis], axis=1).squeeze()
    return proba

rmse_exp_time = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: root_mean_squared_error(y_tst["time"], pred_time)
r2_exp_time = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: r2_score(y_tst["time"], pred_time)
mape_exp_time = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: np.mean(np.abs((y_tst["time"] - pred_time) / np.maximum(y_tst["time"], 1))) * 100
medape_exp_time = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: np.median(np.abs((y_tst["time"] - pred_time) / np.maximum(y_tst["time"], 1))) * 100
spearman_exp_time = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: spearmanr(y_tst["time"], pred_time)[0]
rmsle_exp_time = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: np.sqrt(np.mean((np.log1p(np.clip(y_tst["time"], 0, None)) - np.log1p(np.clip(pred_time, 0, None)))**2))

auc_event = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: roc_auc_score(y_tst["cens"].astype(int), 1 - np.mean(pred_sf, axis=1))
log_loss_event = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: log_loss(y_tst["cens"], 1 - np.mean(pred_sf, axis=1))
rmse_event = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: root_mean_squared_error(y_tst["cens"], 1 - np.mean(pred_sf, axis=1))

L_METRICS = ["CI", "IBS", "AUPRC", "RMSE_TIME", "R2_TIME", "MAPE_TIME", "MEDAPE_TIME", "SPEARMAN_TIME", "RMSLE_TIME", "AUC_EVENT", "LOGLOSS_EVENT", "RMSE_EVENT"]

CLASS_PARAM_GRIDS = {
    "logistic_regression": {
        "penalty": ["l2"],
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"],
        "class_weight": [None, "balanced"],
        "max_iter": [1000],
    },
    "svc": {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 10],
        "class_weight": [None, "balanced"],
    },
    "knn_classifier": {"n_neighbors": [5, 10, 20], "weights": ["uniform", "distance"]},
    "decision_tree_classifier": {
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5],
        "criterion": ["gini", "entropy"],
    },
    "random_forest_classifier": {
        "n_estimators": [100, 300],
        "max_depth": [10, 30],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5],
    },
    "gradient_boosting_classifier": {
        "n_estimators": [100, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "subsample": [0.7, 1.0],
    },
}

REGR_PARAM_GRIDS = {
    "elastic_net": {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8], "max_iter": [1000, 5000]},
    "decision_tree_regressor": {
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5],
        "criterion": ["squared_error", "friedman_mse"],
    },
    "random_forest_regressor": {
        "n_estimators": [100, 300],
        "max_depth": [10, 30],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5],
    },
    "gradient_boosting_regressor": {
        "n_estimators": [100, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "subsample": [0.7, 1.0],
    },
    "svr": {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10], "epsilon": [0.1, 0.2]},
    "knn_regressor": {"n_neighbors": [5, 10, 20], "weights": ["uniform", "distance"]},
}

EXTERNAL_SURV_PARAM_GRIDS = {
    "km": {},
    "cox_ph": {"alpha": [100, 10, 1, 0.1, 0.01, 0.001], "ties": ["breslow"]},
    "random_survival_forest": {"n_estimators": [50], "max_depth": [5, 20], "min_samples_leaf": [0.001, 0.01, 0.1, 0.25], "random_state": [123]},
    "survival_tree": {"max_depth": [None, 20, 30], "min_samples_leaf": [1, 10, 20], "max_features": [None, "sqrt"], "random_state": [123]},
    "gbsa": {"loss": ["coxph"], "learning_rate": [0.01, 0.05, 0.1, 0.5], "n_estimators": [50], "min_samples_leaf": [1, 10, 50, 100], "max_features": ["sqrt"], "random_state": [123]},
}

def build_experiment(categ):
    experim = exp.Experiments(folds=5, mode="CV+SAMPLE")

    experim.add_new_metric("RMSE_TIME", rmse_exp_time)
    experim.add_new_metric("R2_TIME", r2_exp_time)
    experim.add_new_metric("MAPE_TIME", mape_exp_time)
    experim.add_new_metric("MEDAPE_TIME", medape_exp_time)
    experim.add_new_metric("SPEARMAN_TIME", spearman_exp_time)
    experim.add_new_metric("RMSLE_TIME", rmsle_exp_time)
    experim.add_new_metric("AUC_EVENT", auc_event)
    experim.add_new_metric("LOGLOSS_EVENT", log_loss_event)
    experim.add_new_metric("RMSE_EVENT", rmse_event)
    experim.set_metrics(L_METRICS)

    experim.add_method(ClassifWrapSA(LogisticRegression()), CLASS_PARAM_GRIDS["logistic_regression"])
    experim.add_method(ClassifWrapSA(SVC()), CLASS_PARAM_GRIDS["svc"])
    experim.add_method(ClassifWrapSA(KNeighborsClassifier()), CLASS_PARAM_GRIDS["knn_classifier"])
    experim.add_method(ClassifWrapSA(DecisionTreeClassifier()), CLASS_PARAM_GRIDS["decision_tree_classifier"])
    experim.add_method(ClassifWrapSA(RandomForestClassifier()), CLASS_PARAM_GRIDS["random_forest_classifier"])
    experim.add_method(ClassifWrapSA(GradientBoostingClassifier()), CLASS_PARAM_GRIDS["gradient_boosting_classifier"])

    experim.add_method(RegrWrapSA(ElasticNet()), REGR_PARAM_GRIDS["elastic_net"])
    experim.add_method(RegrWrapSA(DecisionTreeRegressor()), REGR_PARAM_GRIDS["decision_tree_regressor"])
    experim.add_method(RegrWrapSA(RandomForestRegressor()), REGR_PARAM_GRIDS["random_forest_regressor"])
    experim.add_method(RegrWrapSA(GradientBoostingRegressor()), REGR_PARAM_GRIDS["gradient_boosting_regressor"])
    experim.add_method(RegrWrapSA(SVR()), REGR_PARAM_GRIDS["svr"])
    experim.add_method(RegrWrapSA(KNeighborsRegressor()), REGR_PARAM_GRIDS["knn_regressor"])

    experim.add_method(SAWrapSA(KaplanMeierFitter()), EXTERNAL_SURV_PARAM_GRIDS["km"])
    experim.add_method(CoxPHSurvivalAnalysis, EXTERNAL_SURV_PARAM_GRIDS["cox_ph"])
    experim.add_method(RandomSurvivalForest, EXTERNAL_SURV_PARAM_GRIDS["random_survival_forest"])
    experim.add_method(SurvivalTree, EXTERNAL_SURV_PARAM_GRIDS["survival_tree"])
    experim.add_method(GradientBoostingSurvivalAnalysis, EXTERNAL_SURV_PARAM_GRIDS["gbsa"])

    INTERNAL_SURV_PARAM_GRIDS = {
        "CRAID": {"depth": [10], "criterion": ["wilcoxon", "logrank"], "l_reg": [0, 0.01, 0.1], "min_samples_leaf": [0.05, 0.01, 0.001], "categ": [categ]},
        "ParallelBootstrapCRAID": {"n_estimators": [50], "depth": [7], "size_sample": [0.3, 0.7], "l_reg": [0, 0.01, 0.1], "criterion": ["tarone-ware", "wilcoxon"], "min_samples_leaf": [0.05, 0.01], "ens_metric_name": ["IBS_REMAIN"], "max_features": ["sqrt"], "categ": [categ]},
    }

    experim.add_method(CRAID, INTERNAL_SURV_PARAM_GRIDS["CRAID"])
    experim.add_method(ParallelBootstrapCRAID, INTERNAL_SURV_PARAM_GRIDS["ParallelBootstrapCRAID"])

    return experim

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_dataset(dataset_name):
    loader = DATASETS[dataset_name]
    print(f"\n===== {dataset_name} =====", flush=True)
    experim = None
    success = False
    try:
        X, y, features, categ, _ = loader()
        experim = build_experiment(categ)
        experim.run_effective(X, y, verbose=1, stratify_best=[])
        success = True
    except Exception as e:
        print("[ERROR]", dataset_name, repr(e), flush=True)
    finally:
        if experim is not None:
            try:
                df_results = experim.get_best_by_mode()
                out = OUT_DIR / f"{dataset_name}.xlsx"
                df_results.to_excel(out, index=False)
                print("saved", out, flush=True)
            except Exception as e:
                success = False
                print("[ERROR] save failed", dataset_name, repr(e), flush=True)
    return dataset_name, success


def parse_args():
    parser = argparse.ArgumentParser(description="Run survival experiments across datasets.")
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=1,
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="datasets",
        action="append",
        choices=sorted(DATASETS),
        help="Limit the run to a specific dataset. Repeat the flag to pass several datasets.",
    )
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("--processes must be at least 1")
    return args


def main():
    args = parse_args()
    dataset_names = args.datasets or list(DATASETS)
    max_workers = min(args.processes, len(dataset_names))
    failed = []

    if max_workers == 1:
        for dataset_name in dataset_names:
            _, success = run_dataset(dataset_name)
            if not success:
                failed.append(dataset_name)
        return 1 if failed else 0

    print(
        f"Running {len(dataset_names)} datasets with {max_workers} processes",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_dataset = {
            executor.submit(run_dataset, dataset_name): dataset_name
            for dataset_name in dataset_names
        }
        for future in as_completed(future_to_dataset):
            dataset_name = future_to_dataset[future]
            try:
                _, success = future.result()
            except Exception as e:
                success = False
                print("[ERROR]", dataset_name, repr(e), flush=True)
            if not success:
                failed.append(dataset_name)

    if failed:
        print("[DONE WITH ERRORS]", ", ".join(sorted(set(failed))), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
