import numpy as np
from typing import Iterable, Dict, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    log_loss,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from survivors.metrics import (
    concordance_index,
    ibs_remain,
    auprc
)

def _to_binary_labels(y) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype.names is None:
        return y.astype(int)

    names = y.dtype.names
    if "event" in names:
        return y["event"].astype(int)
    if "cens" in names:
        return y["cens"].astype(int)


    raise ValueError(f"Не могу извлечь бинарные метки из y с полями {names}")

def _split_time_event(y) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    if isinstance(y, tuple) and len(y) == 2:
        time, event = y
        return np.asarray(time, float), np.asarray(event, bool)

    if y.dtype.names is None:
        raise ValueError("Ожидался структурированный y или кортеж (time, event)")

    names = y.dtype.names
    if "time" not in names:
        raise ValueError(f"В y нет поля 'time', имеем поля: {names}")

    time = y["time"].astype(float)
    if "event" in names:
        event = y["event"].astype(bool)
    elif "cens" in names:
        event = (~y["cens"]).astype(bool)
    else:
        raise ValueError(f"В y нет ни 'event', ни 'cens', имеем поля: {names}")

    return time, event

def _get_survival_from_model(model, X, bins) -> np.ndarray:
    if not hasattr(model, "predict_survival_function"):
        raise ValueError("У модели нет метода predict_survival_function(X)")

    S = model.predict_survival_function(X, bins)
    if isinstance(S, tuple) and len(S) == 2:
        S, _ = S
    return np.asarray(S, float)
def _get_risk_from_model(model, X, bins) -> np.ndarray:
    risk = model.predict_expected_time(X, bins)
    return risk

# --- метрики классификации -------------------------------------------------

def eval_classification_proba(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = _to_binary_labels(y_true)
    y_proba = np.asarray(y_proba, float)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
    }

def eval_classification_model(
    model,
    X,
    y_true,
    threshold: float = 0.5,
    proba_column: int = 1,
) -> Dict[str, float]:
    proba = model.predict_proba(X)
    y_proba = np.asarray(proba[:, proba_column], float)
    return eval_classification_proba(y_true, y_proba, threshold=threshold)

# --- метрики регрессии -------------------------------------------------
def eval_regression(y_true, y_pred) -> Dict[str, float]:
    """
    Метрики регрессии. Без C-index (по твоему требованию).
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def eval_regression_model(model, X, y_true) -> Dict[str, float]:
    return eval_regression(y_true, model.predict(X))

# --- метрики аналиаз выживаемости -------------------------------------------------
def cindex_survival(y, risk) -> float:
    time, event = _split_time_event(y)
    risk = np.asarray(risk, float)
    # return float(concordance_index(time, risk, event.astype(int)))
    return float(concordance_index(time, risk))


def cindex_survival_model(model, X, y, bins) -> float:
    risk = _get_risk_from_model(model, X, bins)
    return cindex_survival(y, risk)


def eval_survival_curves(y_train, y_test, pred_sf, bins):
    return {
        "ibs_remain": float(ibs_remain(y_train, y_test, pred_sf, bins)),
        "auprc": float(auprc(y_train, y_test, pred_sf, bins)),
    }


def eval_survival_model(model, X_train, y_train, X_test, y_test, bins):
    risk = _get_risk_from_model(model, X_test, bins)
    ci = cindex_survival(y_test, risk)
    S = _get_survival_from_model(model, X_test, bins)

    curves = eval_survival_curves(y_train, y_test, S, bins)

    res = {"c_index": ci}
    res.update(curves)
    return res

def print_metrics(metrics: Dict[str, float], prefix: str = None):
    if prefix:
        print(prefix)
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")
