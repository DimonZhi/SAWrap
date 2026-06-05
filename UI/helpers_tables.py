import re
import os

import pandas as pd
from pathlib import Path
import numpy as np
import importlib

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from survivors.experiments import grid as exp
from survivors.external import SAWrapSA, ClassifWrapSA, RegrWrapSA
from sklearn.metrics import root_mean_squared_error, r2_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from .helpers_ai_advice import TASK_CONFIGS, _score_models
from .helpers_runtime_piecewise import PIECEWISE_RUNTIME_CLASSES
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented*")


PIECEWISE_TABLE_FILES = {
    "actg": "Piecewise_actg.xlsx",
    "framingham": "Piecewise_framingham.xlsx",
    "gbsg": "Piecewise_gbsg.xlsx",
    "pbc": "Piesewise_pbc.xlsx",
    "rott2": "Piecewise_rott2.xlsx",
    "smarto": "Piecewise_smarto.xlsx",
    "support2": "Piecewise_support2.xlsx",
}
PIECEWISE_RE = re.compile(
    r"^(Piecewise(?:CensorAware)?ClassifWrapSA)\(([^,]+),\s*times=(\d+)\)$"
)


def _norm(s: str) -> str:
    return str(s).strip().lower()


def is_piecewise_model_label(model_label: str) -> bool:
    label = str(model_label).strip()
    return label.startswith("PiecewiseClassifWrapSA(") or label.startswith("PiecewiseCensorAwareClassifWrapSA(")


def _piecewise_match(model_label: str):
    return PIECEWISE_RE.match(str(model_label).strip())


def _active_task_metrics(df: pd.DataFrame, task_id: str) -> list[dict]:
    task_config = TASK_CONFIGS.get(task_id) or TASK_CONFIGS["classification"]
    metrics = []
    for metric in task_config["metrics"]:
        col = _norm(metric["key"]) + "_mean"
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            metrics.append(metric)
    if metrics:
        return metrics

    fallback = []
    for metric in TASK_CONFIGS["classification"]["metrics"]:
        col = _norm(metric["key"]) + "_mean"
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            fallback.append(metric)
    return fallback


def select_global_piecewise_time(
    base_dir: Path,
    family: str,
    base_model: str,
    task_id: str = "classification",
) -> int | None:
    scores_by_time: dict[int, list[float]] = {}
    datasets_with_model = 0

    for dataset_id in PIECEWISE_TABLE_FILES:
        table_path = get_piecewise_table_path(base_dir, dataset_id)
        df = load_surv_df(table_path)
        if df is None or df.empty or "method" not in df.columns:
            continue

        metrics = _active_task_metrics(df, task_id)
        if not metrics:
            continue

        scored = _score_models(df, metrics)
        dataset_scores: dict[int, float] = {}
        for _, row in scored.iterrows():
            method = str(row.get("method", ""))
            match = _piecewise_match(method)
            if not match:
                continue
            row_family, row_base_model, row_times = match.groups()
            if row_family == family and row_base_model == base_model:
                dataset_scores[int(row_times)] = float(row["ai_score"])

        if not dataset_scores:
            continue

        datasets_with_model += 1
        for times, score in dataset_scores.items():
            scores_by_time.setdefault(times, []).append(score)

    if not scores_by_time or datasets_with_model == 0:
        return None

    full_coverage = [
        (float(np.mean(scores)), len(scores), times)
        for times, scores in scores_by_time.items()
        if len(scores) == datasets_with_model
    ]
    candidates = full_coverage or [
        (float(np.mean(scores)), len(scores), times)
        for times, scores in scores_by_time.items()
    ]
    _, _, best_times = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
    return int(best_times)


def select_best_piecewise_variant(
    base_dir: Path,
    dataset_id: str,
    family: str,
    base_model: str,
    task_id: str,
) -> tuple[str | None, Path]:
    table_path = get_piecewise_table_path(base_dir, dataset_id)
    best_times = select_global_piecewise_time(base_dir, family, base_model, task_id)
    if best_times is None:
        return None, table_path

    label = f"{family}({base_model}, times={best_times})"
    df = load_surv_df(table_path)
    if df is None or df.empty or "method" not in df.columns:
        return None, table_path
    if df[df["method"].astype(str) == label].empty:
        return None, table_path
    return label, table_path

def import_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    if module_name.startswith("survivors."):
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def get_surv_table_path(base_dir: Path, dataset_id: str) -> Path:
    cands = [
        base_dir / "tables" / f"{dataset_id}.xlsx",
        base_dir / "tables" / f"{dataset_id.upper()}.xlsx",
        base_dir / "tables" / f"{dataset_id.lower()}.xlsx",
    ]
    for p in cands:
        if p.exists():
            return p.resolve()
    return cands[0].resolve()


def get_piecewise_table_path(base_dir: Path, dataset_id: str) -> Path:
    key = str(dataset_id).strip().lower()
    filename = PIECEWISE_TABLE_FILES.get(key, f"Piecewise_{key}.xlsx")
    nested_path = base_dir / "tables" / filename
    direct_path = base_dir / filename
    if nested_path.exists():
        return nested_path.resolve()
    if direct_path.exists():
        return direct_path.resolve()
    return nested_path.resolve()


def get_metric_table_path(base_dir: Path, dataset_id: str, model_label: str | None = None) -> Path:
    if model_label and is_piecewise_model_label(model_label):
        return get_piecewise_table_path(base_dir, dataset_id)
    return get_surv_table_path(base_dir, dataset_id)

def normalize_surv_df(df: pd.DataFrame):
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    df = df.rename(columns={c: _norm(c) for c in df.columns})
    if "method" not in df.columns:
        df["method"] = pd.NA
    df["__method_norm__"] = df["method"].map(_norm)
    return df

def load_surv_df(table_path: Path):
    if not table_path.exists():
        return None
    df = pd.read_excel(table_path)
    return normalize_surv_df(df)

def lookup_metric(df: pd.DataFrame, model_label: str, metric_key: str):
    if df is None:
        return None
    if "__method_norm__" not in df.columns:
        return None
    row = df[df["__method_norm__"] == _norm(model_label)]
    if row.empty:
        return None
    col = _norm(metric_key) + "_mean"
    if col not in df.columns:
        return None
    v = row.iloc[0].get(col)
    if v is None or pd.isna(v):
        return None
    try:
        return float(v)
    except Exception:
        return None

def _ensure_cols(df: pd.DataFrame, cols: list):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _copy_surv_y(y):
    if isinstance(y, np.ndarray):
        return y.copy()
    if isinstance(y, pd.DataFrame):
        return y.copy(deep=True)
    if isinstance(y, dict):
        return {key: np.asarray(value).copy() for key, value in y.items()}
    return y


def _is_sparse_stratify_error(exc: ValueError) -> bool:
    text = str(exc)
    return (
        "least populated classes" in text
        or "minimum number of groups for any class" in text
    )


def _run_effective_without_holdout_stratify(experim, X, y, verbose=0, stratify_best=None):
    stratify_best = stratify_best or []
    if experim.mode != "CV+SAMPLE":
        experim.run(X, y, dir_path=None, verbose=verbose)
        return None

    y_run = _copy_surv_y(y)
    y_run["time"] = exp.bins_scheme(y_run["time"], scheme=experim.bins_sch)
    experim.bins_sch = ""

    folds = 20
    X_TR, X_HO = train_test_split(
        X,
        stratify=None,
        test_size=0.33,
        random_state=42,
    )
    X_tr, y_tr, X_HO, y_HO, bins_HO = exp.prepare_sample(X, y_run, X_TR.index, X_HO.index)
    old_mode = experim.mode
    try:
        experim.mode = "CV"
        experim.run(X_tr, y_tr, dir_path=None, verbose=verbose)
        try:
            experim.sample_table = experim.eval_on_sample_by_best_params(
                X,
                y_run,
                folds=folds,
                stratify=stratify_best,
            )
        except ValueError as exc:
            if not _is_sparse_stratify_error(exc):
                raise
            experim.sample_table = experim.get_cv_result(stratify=stratify_best)
    finally:
        experim.mode = old_mode
    return None


def _run_effective_with_fallback(experim, X, y, verbose=0, stratify_best=None):
    bins_sch = getattr(experim, "bins_sch", "")
    mode = getattr(experim, "mode", None)
    try:
        return experim.run_effective(
            X,
            _copy_surv_y(y),
            verbose=verbose,
            stratify_best=stratify_best or [],
        )
    except ValueError as exc:
        if not _is_sparse_stratify_error(exc):
            raise
        experim.bins_sch = bins_sch
        if mode is not None:
            experim.mode = mode
        return _run_effective_without_holdout_stratify(
            experim,
            X,
            y,
            verbose=verbose,
            stratify_best=stratify_best or [],
        )

def _matching_model_cfgs_for_table(base_dir: Path, dataset_id: str, model_cfgs: list, table_path: Path) -> list:
    target = Path(table_path).resolve()
    return [
        cfg
        for cfg in (model_cfgs or [])
        if get_metric_table_path(base_dir, dataset_id, cfg.get("label")).resolve() == target
    ]


def _build_method_obj(mcfg: dict):
    Est = import_class(mcfg["id"])
    est_obj = Est()
    family = mcfg.get("piecewise_family")
    if family:
        Wrapper = PIECEWISE_RUNTIME_CLASSES[family]
        return Wrapper(est_obj, times=int(mcfg.get("times") or 8))

    t = mcfg.get("task")
    if t == "survival":
        return SAWrapSA(est_obj)
    if t == "classification":
        return ClassifWrapSA(est_obj)
    return RegrWrapSA(est_obj)

def find_sf_at_truetime(pred_sf, event_time, bins):
        idx_pred = np.clip(np.searchsorted(bins, event_time), 0, len(bins) - 1)
        proba = np.take_along_axis(pred_sf, idx_pred[:, np.newaxis], axis=1).squeeze()
        return proba

def supplement_surv_table_missing(
    base_dir: Path,
    dataset_id: str,
    X_tr,
    y_tr,
    model_cfgs: list,
    need_metrics: list,
    table_path: Path | None = None,
):
    table_path = Path(table_path) if table_path is not None else get_surv_table_path(base_dir, dataset_id)
    model_cfgs = _matching_model_cfgs_for_table(base_dir, dataset_id, model_cfgs, table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    #классификация
    auc_event      = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: roc_auc_score(y_tst["cens"].astype(int), find_sf_at_truetime(pred_sf, y_tst["time"], bins))
    log_loss_event = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: log_loss(y_tst["cens"], find_sf_at_truetime(pred_sf, y_tst["time"], bins))
    rmse_event     = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: root_mean_squared_error(y_tst["cens"], find_sf_at_truetime(pred_sf, y_tst["time"], bins))

    #регрессия
    rmse_exp_time = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: root_mean_squared_error(y_tst["time"], pred_time)
    r2_exp_time   = lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins: r2_score(y_tst["time"], pred_time)

    if table_path.exists():
        df = pd.read_excel(table_path)
    else:
        df = pd.DataFrame(columns=["method"])

    df = normalize_surv_df(df)
    if df is None:
        df = pd.DataFrame(columns=["method"])
        df = normalize_surv_df(df)

    need_cols = ["method", "__method_norm__", "params"] + [_norm(mk) + "_mean" for mk in need_metrics]
    df = _ensure_cols(df, need_cols)

    cfg2missing = {}
    for mcfg in (model_cfgs or []):
        ml = mcfg["label"]
        mn = _norm(ml)
        row = df[df["__method_norm__"] == mn]

        miss = []
        if row.empty:
            miss = list(need_metrics)
        else:
            for mk in need_metrics:
                col = _norm(mk) + "_mean"
                v = row.iloc[0].get(col)
                if v is None or pd.isna(v):
                    miss.append(mk)

        if miss:
            cfg2missing[mn] = miss

    if not cfg2missing:
        out = df.drop(columns=["__method_norm__"], errors="ignore")
        out.to_excel(table_path, index=False)
        return table_path

    for mcfg in model_cfgs:
        ml = mcfg["label"]
        mn = _norm(ml)
        if mn not in cfg2missing:
            continue

        miss = cfg2missing[mn]
        if not miss:
            continue

        hit = (df["__method_norm__"] == mn)
        if hit.any():
            idx = df[hit].index[0]
        else:
            new_row = {"method": ml, "__method_norm__": mn}
            for mk2 in need_metrics:
                new_row[_norm(mk2) + "_mean"] = pd.NA
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            idx = df.index[-1]

        experim = exp.Experiments(folds=5, mode="CV+SAMPLE")
        experim.add_new_metric("RMSE_TIME", rmse_exp_time)
        experim.add_new_metric("R2_TIME", r2_exp_time)
        experim.add_new_metric("AUC_EVENT", auc_event)
        experim.add_new_metric("LOGLOSS_EVENT", log_loss_event)
        experim.add_new_metric("RMSE_EVENT", rmse_event)
        experim.set_metrics(list(miss))

        best_metric = miss[0]
        if isinstance(best_metric, str):
            best_metric = _norm(best_metric).upper()
        experim.add_metric_best(best_metric)

        grid = mcfg.get("param_grid")
        if not grid:
            kwargs = mcfg.get("kwargs") or {}
            grid = {k: [v] for k, v in kwargs.items()}

        method_obj = _build_method_obj(mcfg)
        experim.add_method(method_obj, grid)

        _run_effective_with_fallback(experim, X_tr, y_tr, verbose=1, stratify_best=[])

        df_best = experim.get_best_by_mode()
        df_best = df_best.rename(columns={c: _norm(c) for c in df_best.columns})

        if df_best is None or len(df_best) == 0:
            continue
        src = df_best.iloc[[0]]                                                         

        if "params" in src.columns:
            df.at[idx, "params"] = str(src.iloc[0].get("params"))

        for mk in miss:
            mean_col = _norm(mk) + "_mean"
            if mean_col not in src.columns:
                continue
            v = src.iloc[0].get(mean_col)
            if v is None or pd.isna(v):
                continue
            df.at[idx, mean_col] = float(v)

    out = df.drop(columns=["__method_norm__"], errors="ignore")
    out.to_excel(table_path, index=False)
    return table_path


def get_surv_metrics(
    base_dir: Path,
    dataset_id: str,
    X_tr,
    y_tr,
    model_cfgs: list,
    model_label: str,
    x_metric: str,
    y_metric: str,
    allow_recompute: bool = True,
):
    table_path = get_metric_table_path(base_dir, dataset_id, model_label)
    df = load_surv_df(table_path)

    vx = lookup_metric(df, model_label, x_metric)
    vy = lookup_metric(df, model_label, y_metric)
    if (vx is not None) and (vy is not None):
        return vx, vy, table_path

    need = []
    if vx is None:
        need.append(x_metric)
    if vy is None and _norm(y_metric) != _norm(x_metric):
        need.append(y_metric)
    if (not allow_recompute) or (not need) or (X_tr is None) or (y_tr is None) or (not model_cfgs):
        return vx, vy, table_path
    supplement_surv_table_missing(
        base_dir=base_dir,
        dataset_id=dataset_id,
        X_tr=X_tr,
        y_tr=y_tr,
        model_cfgs=model_cfgs,
        need_metrics=need,
        table_path=table_path,
    )

    df = load_surv_df(table_path)
    vx = lookup_metric(df, model_label, x_metric)
    vy = lookup_metric(df, model_label, y_metric)
    return vx, vy, table_path

def list_surv_metrics_from_table(base_dir: Path, dataset_id: str, model_label: str | None = None):
    if model_label:
        table_paths = [get_metric_table_path(base_dir, dataset_id, model_label)]
    else:
        table_paths = [get_surv_table_path(base_dir, dataset_id)]
        piecewise_path = get_piecewise_table_path(base_dir, dataset_id)
        if piecewise_path.exists():
            table_paths.append(piecewise_path)

    frames = []
    metric_keys = set()
    for table_path in table_paths:
        if not table_path.exists():
            continue
        df = pd.read_excel(table_path)
        df = df.rename(columns={c: _norm(c) for c in df.columns})
        metric_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_mean")]
        for c in metric_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                metric_keys.add(c[:-5])
        frames.append(df)

    if not frames:
        return [], pd.DataFrame(columns=["method"])

    return sorted(metric_keys), pd.concat(frames, ignore_index=True, sort=False)
