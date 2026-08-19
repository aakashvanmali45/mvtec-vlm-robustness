"""Classification metrics for MVTec-AD zero-shot and few-shot evaluation."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    results: list[dict[str, Any]],
    positive_class: str = "defective",
) -> dict[str, Any]:
    """Compute binary classification metrics from per-image predictions.

    Args:
        results: List of dicts, each containing keys 'true_label',
            'predicted_label', 'prob_good', 'prob_defective'.
        positive_class: Which label is treated as positive for
            precision/recall/AUROC.

    Returns:
        Dictionary with: n_total, n_good, n_defective, accuracy,
        balanced_accuracy, precision, recall, f1, auroc, confusion_matrix.
        AUROC is float('nan') if only one class is present in y_true.
    """
    if not results:
        raise ValueError("results is empty")

    y_true = [r["true_label"] for r in results]
    y_pred = [r["predicted_label"] for r in results]

    prob_key = f"prob_{positive_class}"
    y_score = [r[prob_key] for r in results]

    y_true_bin = [1 if y == positive_class else 0 for y in y_true]
    y_pred_bin = [1 if y == positive_class else 0 for y in y_pred]

    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    balanced_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )

    if len(set(y_true_bin)) > 1:
        auroc = roc_auc_score(y_true_bin, y_score)
    else:
        auroc = float("nan")

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])

    return {
        "n_total": len(results),
        "n_good": y_true.count("good"),
        "n_defective": y_true.count("defective"),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "confusion_matrix": cm.tolist(),
    }