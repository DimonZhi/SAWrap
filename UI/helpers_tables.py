import pandas as pd
from pathlib import Path
import numpy as np
import importlib
from survivors.experiments import grid as exp
from survivors.external import SAWrapSA, ClassifWrapSA, RegrWrapSA
from sklearn.metrics import root_mean_squared_error, r2_score, roc_auc_score, log_loss
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented*")




def _norm(s: str) -> str:
    return str(s).strip().lower()

def import_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
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
):
    table_path = get_surv_table_path(base_dir, dataset_id)
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

    need_cols = ["method", "__method_norm__"] + [_norm(mk) + "_mean" for mk in need_metrics]
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

        # 1) гарантируем idx (строка в df под этот метод) СРАЗУ
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

        Est = import_class(mcfg["id"])
        grid = mcfg.get("param_grid")
        if not grid:
            kwargs = mcfg.get("kwargs") or {}
            grid = {k: [v] for k, v in kwargs.items()}

        est_obj = Est()
        t = mcfg.get("task")
        if t == "survival":
            method_obj = SAWrapSA(est_obj)
        elif t == "classification":
            method_obj = ClassifWrapSA(est_obj)
        else:
            method_obj = RegrWrapSA(est_obj)
        experim.add_method(method_obj, grid)

        experim.run_effective(X_tr, y_tr, verbose=0, stratify_best=[])

        df_best = experim.get_best_by_mode()
        df_best = df_best.rename(columns={c: _norm(c) for c in df_best.columns})

        # мы запускали один метод → берём первую строку без матчинга по названию
        if df_best is None or len(df_best) == 0:
            continue
        src = df_best.iloc[[0]]                                                         


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
):
    table_path = get_surv_table_path(base_dir, dataset_id)
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
    supplement_surv_table_missing(
        base_dir=base_dir,
        dataset_id=dataset_id,
        X_tr=X_tr,
        y_tr=y_tr,
        model_cfgs=model_cfgs,
        need_metrics=need,
    )

    df = load_surv_df(table_path)
    vx = lookup_metric(df, model_label, x_metric)
    vy = lookup_metric(df, model_label, y_metric)
    return vx, vy, table_path

def list_surv_metrics_from_table(base_dir: Path, dataset_id: str):
    table_path = get_surv_table_path(base_dir, dataset_id)
    df = pd.read_excel(table_path)
    df = df.rename(columns={c: _norm(c) for c in df.columns})
    metric_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_mean")]
    metric_keys = []
    for c in metric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            metric_keys.append(c[:-5])
    metric_keys = sorted(metric_keys)
    return metric_keys, df
