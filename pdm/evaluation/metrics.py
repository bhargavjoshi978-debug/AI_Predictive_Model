"""Evaluation metrics for classification, anomaly detection, RUL, and early warning."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute multi-class classification metrics: accuracy, macro precision, recall, F1, and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "confusion_matrix": cm,
    }


def compute_anomaly_metrics(y_true_binary: np.ndarray, scores: np.ndarray, threshold: float = 1.0) -> Dict[str, Any]:
    """Compute anomaly detection metrics: ROC-AUC, recall at 1% FPR, and false alarm rate per 1,000 healthy windows."""
    # Ensure binary target
    y_true_bin = (np.asarray(y_true_binary) > 0).astype(int)

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true_bin, scores))
        fpr, tpr, thresholds = roc_curve(y_true_bin, scores)
        # Recall at 1% FPR (0.01)
        idx_1pct = np.where(fpr <= 0.01)[0]
        rec_at_1pct_fpr = float(tpr[idx_1pct[-1]]) if len(idx_1pct) > 0 else 0.0
    except Exception:
        roc_auc = 0.5
        fpr, tpr, thresholds = np.array([0, 1]), np.array([0, 1]), np.array([0, 1])
        rec_at_1pct_fpr = 0.0

    # False alarms on healthy windows
    healthy_mask = (y_true_bin == 0)
    n_healthy = int(np.count_nonzero(healthy_mask))
    if n_healthy > 0:
        false_alarms = int(np.count_nonzero(scores[healthy_mask] >= threshold))
        fa_rate_pct = float(false_alarms / n_healthy)
        fa_per_1000 = float((false_alarms / n_healthy) * 1000.0)
    else:
        false_alarms = 0
        fa_rate_pct = 0.0
        fa_per_1000 = 0.0

    # Anomaly recall
    anom_mask = (y_true_bin == 1)
    n_anom = int(np.count_nonzero(anom_mask))
    recall_anom = float(np.count_nonzero(scores[anom_mask] >= threshold) / n_anom) if n_anom > 0 else 0.0

    return {
        "roc_auc": roc_auc,
        "recall_at_1pct_fpr": rec_at_1pct_fpr,
        "anomaly_recall": recall_anom,
        "false_alarm_rate_pct": fa_rate_pct,
        "false_alarms_per_1000": fa_per_1000,
        "fpr": fpr,
        "tpr": tpr,
    }


def compute_rul_metrics(y_true_rul: np.ndarray, y_pred_rul: np.ndarray) -> Dict[str, float]:
    """Compute RUL regression metrics: MAE, RMSE, and NASA asymmetric score."""
    mask = ~np.isnan(y_true_rul) & ~np.isnan(y_pred_rul)
    if not np.any(mask):
        return {"rul_mae": float("nan"), "rul_rmse": float("nan"), "nasa_score": float("nan")}

    yt = np.asarray(y_true_rul)[mask]
    yp = np.asarray(y_pred_rul)[mask]

    diff = yp - yt  # error = predicted - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # NASA asymmetric score:
    # d < 0 (early prediction): exp(-d/13) - 1
    # d >= 0 (late prediction, more dangerous): exp(d/10) - 1
    penalties = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    # Clip large penalties to prevent overflow
    penalties = np.clip(penalties, 0.0, 1e6)
    nasa_score = float(np.sum(penalties))

    return {
        "rul_mae": mae,
        "rul_rmse": rmse,
        "nasa_score": nasa_score,
    }


def compute_early_warning_metrics(
    alert_history: List[Dict[str, Any]],
    onset_step: int,
    failure_step: Optional[int],
) -> Dict[str, Optional[float]]:
    """Compute lead time to failure and detection delay from alert history."""
    first_warn: Optional[int] = None
    first_crit: Optional[int] = None

    for item in alert_history:
        state = item.get("state")
        t = item.get("t", 0)
        if state == "WARNING" and first_warn is None:
            first_warn = t
        elif state == "CRITICAL" and first_crit is None:
            first_crit = t

    detection_delay: Optional[float] = None
    if first_warn is not None and onset_step > 0:
        detection_delay = float(first_warn - onset_step)

    lead_time: Optional[float] = None
    if failure_step is not None and first_crit is not None:
        lead_time = float(failure_step - first_crit)

    return {
        "first_warning_step": first_warn,
        "first_critical_step": first_crit,
        "detection_delay": detection_delay,
        "lead_time": lead_time,
    }
