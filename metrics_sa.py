import numpy as np
from typing import Iterable, Dict, Optional

from sklearn.metrics import (
    accuracy_score,  # классовая
    roc_auc_score,   # вероятностная
    f1_score,        # классовая
    log_loss,        # вероятностная
    brier_score_loss # вероятностная
)

def cindex(y_true: Iterable[float], y_scores: Iterable[float]) -> float:
    y_true = np.asarray(y_true, float)
    y_scores = np.asarray(y_scores, float)

    n = len(y_true)
    if n < 2:
        return np.nan

    concordant = 0.0
    comparable = 0.0

    for i in range(n - 1):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue  
            comparable += 1
            diff_true = y_true[i] - y_true[j]
            diff_score = y_scores[i] - y_scores[j]

            if diff_true * diff_score > 0:
                concordant += 1.0
            elif diff_score == 0.0:
                concordant += 0.5

    return concordant / comparable if comparable > 0 else np.nan
def eval_binary_proba(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, int)
    y_proba = np.asarray(y_proba, float)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1_score": f1_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "c_index": cindex(y_true, y_proba),
    }
    return metrics


def eval_model_proba(
    model,
    X,
    y_true,
    threshold: float = 0.5,
) -> Dict[str, float]:
    proba = model.predict_proba(X)
    y_proba = proba[:, 1]
    return eval_binary_proba(y_true, y_proba, threshold)
def print_metrics(metrics: Dict[str, float], prefix: Optional[str] = None) -> None:
    if prefix:
        print(prefix)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
