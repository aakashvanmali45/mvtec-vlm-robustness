"""Tests for src/metrics.py."""

import math

from src.metrics import compute_metrics


def test_perfect_classifier():
    """All predictions correct; metrics should all be 1.0."""
    results = [
        {"true_label": "good",      "predicted_label": "good",      "prob_good": 0.9, "prob_defective": 0.1},
        {"true_label": "defective", "predicted_label": "defective", "prob_good": 0.1, "prob_defective": 0.9},
        {"true_label": "good",      "predicted_label": "good",      "prob_good": 0.8, "prob_defective": 0.2},
        {"true_label": "defective", "predicted_label": "defective", "prob_good": 0.2, "prob_defective": 0.8},
    ]
    m = compute_metrics(results)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["auroc"] == 1.0
    assert m["n_total"] == 4
    assert m["n_good"] == 2
    assert m["n_defective"] == 2


def test_constant_predictor():
    """Model predicts 'defective' for everything; balanced_accuracy should be 0.5."""
    results = [
        {"true_label": "good",      "predicted_label": "defective", "prob_good": 0.4, "prob_defective": 0.6},
        {"true_label": "good",      "predicted_label": "defective", "prob_good": 0.3, "prob_defective": 0.7},
        {"true_label": "defective", "predicted_label": "defective", "prob_good": 0.2, "prob_defective": 0.8},
        {"true_label": "defective", "predicted_label": "defective", "prob_good": 0.1, "prob_defective": 0.9},
    ]
    m = compute_metrics(results)
    assert m["balanced_accuracy"] == 0.5
    assert m["recall"] == 1.0  # catches all defects trivially
    assert m["accuracy"] == 0.5  # 2/4 correct


def test_auroc_nan_when_single_class():
    """AUROC should be NaN if only one true class is present."""
    results = [
        {"true_label": "good", "predicted_label": "good",      "prob_good": 0.9, "prob_defective": 0.1},
        {"true_label": "good", "predicted_label": "defective", "prob_good": 0.4, "prob_defective": 0.6},
    ]
    m = compute_metrics(results)
    assert math.isnan(m["auroc"])


def test_empty_input_raises():
    """Empty results list should raise ValueError."""
    import pytest
    with pytest.raises(ValueError):
        compute_metrics([])