"""
Tests for ml/common/metrics.py. Only needs numpy + scikit-learn (both
already required by the base backend), so these run everywhere.
"""
import numpy as np

from ml.common.metrics import compute_classification_metrics, format_metrics_report


def test_metrics_on_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_classification_metrics(y_true, y_pred, y_scores)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_metrics_on_imperfect_predictions():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 1, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.4, 0.9, 0.3, 0.7, 0.6])
    m = compute_classification_metrics(y_true, y_pred, y_scores)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0
    assert m["roc_auc"] is not None
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_metrics_on_empty_input_does_not_crash():
    m = compute_classification_metrics(np.array([]), np.array([]))
    assert m["n_samples"] == 0
    assert m["accuracy"] is None
    assert "note" in m


def test_metrics_without_scores_skips_roc_auc_gracefully():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    m = compute_classification_metrics(y_true, y_pred, y_scores=None)
    assert m["roc_auc"] is None
    assert "note" in m
    assert m["accuracy"] is not None  # other metrics still computed


def test_metrics_single_class_test_set_skips_roc_auc_gracefully():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 0])
    y_scores = np.array([0.9, 0.8, 0.4])
    m = compute_classification_metrics(y_true, y_pred, y_scores)
    assert m["roc_auc"] is None
    assert "note" in m


def test_metrics_never_exceed_valid_bounds():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50)
    y_pred = rng.integers(0, 2, size=50)
    y_scores = rng.random(50)
    m = compute_classification_metrics(y_true, y_pred, y_scores)
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= m[key] <= 1.0
    if m["roc_auc"] is not None:
        assert 0.0 <= m["roc_auc"] <= 1.0


def test_format_metrics_report_handles_empty():
    m = compute_classification_metrics(np.array([]), np.array([]))
    report = format_metrics_report(m)
    assert "No samples" in report


def test_format_metrics_report_includes_all_fields():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8])
    m = compute_classification_metrics(y_true, y_pred, y_scores)
    report = format_metrics_report(m)
    assert "Accuracy" in report
    assert "Precision" in report
    assert "Recall" in report
    assert "F1 score" in report
    assert "ROC-AUC" in report
