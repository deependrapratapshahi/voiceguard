"""
Classification metrics shared by training and evaluation scripts.

All metrics come directly from scikit-learn -- nothing here fabricates
or estimates a number. If a metric cannot be computed (e.g. ROC-AUC
needs both classes present), that is reported explicitly as `None`
rather than a made-up value.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray | None = None
) -> dict:
    """
    y_true, y_pred: 0/1 integer arrays (real=0, synthetic=1)
    y_scores: predicted probability of the positive (synthetic) class,
              required for ROC-AUC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0:
        return {
            "n_samples": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "note": "No samples available to compute metrics.",
        }

    metrics = {
        "n_samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }

    unique_classes = set(np.unique(y_true).tolist())
    if y_scores is not None and unique_classes == {0, 1}:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            metrics["roc_auc"] = None
    elif y_scores is None:
        metrics["note"] = "roc_auc not computed: no prediction scores provided."
    else:
        metrics["note"] = "roc_auc not computed: test set does not contain both classes."

    return metrics


def format_metrics_report(metrics: dict) -> str:
    """Human-readable summary for CLI output."""
    if metrics.get("n_samples", 0) == 0:
        return "No samples available to compute metrics."

    lines = [
        f"Samples:   {metrics['n_samples']}",
        f"Accuracy:  {metrics['accuracy']:.3f}",
        f"Precision: {metrics['precision']:.3f}",
        f"Recall:    {metrics['recall']:.3f}",
        f"F1 score:  {metrics['f1']:.3f}",
    ]
    if metrics.get("roc_auc") is not None:
        lines.append(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    else:
        lines.append(f"ROC-AUC:   not available ({metrics.get('note', 'unknown reason')})")
    return "\n".join(lines)
