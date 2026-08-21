"""Threshold calibration for zero-shot VLM classification.

Optimizes a decision threshold on a small calibration set instead of
fine-tuning the model, providing a training-free alternative to few-shot
LoRA adaptation. Serves as a baseline that isolates whether few-shot
adaptation improvements stem from better representations or better
threshold placement.
"""

from typing import Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def find_optimal_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    n_thresholds: int = 201,
) -> tuple[float, float]:
    """Find the threshold that maximizes balanced accuracy on the given set.

    Args:
        scores: Predicted probability of the positive class (defective) per sample.
        labels: True binary labels (1 = defective, 0 = good).
        n_thresholds: How many candidate thresholds to sweep in [0, 1].

    Returns:
        (optimal_threshold, achieved_balanced_accuracy).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if len(set(labels.tolist())) < 2:
        raise ValueError(
            "Calibration set must contain both classes; got only one class."
        )

    candidates = np.linspace(0.0, 1.0, n_thresholds)
    best_thresh = 0.5
    best_score = -1.0

    for t in candidates:
        preds = (scores > t).astype(np.int64)
        try:
            bacc = balanced_accuracy_score(labels, preds)
        except ValueError:
            continue
        if bacc > best_score:
            best_score = bacc
            best_thresh = float(t)

    return best_thresh, float(best_score)


def apply_threshold_to_results(
    results: list[dict],
    threshold: float,
    positive_class: str = "defective",
) -> list[dict]:
    """Return a new results list with predictions recomputed at the given threshold.

    Args:
        results: Output of zero_shot_classify_category (each dict has
            prob_good, prob_defective, true_label).
        threshold: Cutoff on prob_defective. If prob_defective > threshold,
            predict defective; else predict good.
        positive_class: Which class is scored against the threshold.

    Returns:
        New list of dicts with updated 'predicted_label' fields.
    """
    prob_key = f"prob_{positive_class}"
    negative_class = "good" if positive_class == "defective" else "defective"

    updated = []
    for r in results:
        new_r = dict(r)
        new_r["predicted_label"] = (
            positive_class if r[prob_key] > threshold else negative_class
        )
        updated.append(new_r)
    return updated