"""
common/metrics.py
=================
Shared evaluation metrics for all four model tracks.

Model owners should import these functions rather than implementing their own,
so that all experiments use identical metric definitions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Top-1 overall accuracy.

    Parameters
    ----------
    predictions : (N,) integer tensor of predicted class IDs
    targets     : (N,) integer tensor of ground-truth class IDs

    Returns
    -------
    accuracy : float in [0, 1]
    """
    correct = (predictions == targets).sum().item()
    return correct / max(1, len(targets))


def per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int = 100,
) -> Dict[int, float]:
    """
    Per-class accuracy (mean accuracy over classes = mAcc).

    Parameters
    ----------
    predictions : (N,) integer tensor
    targets     : (N,) integer tensor
    n_classes   : number of classes

    Returns
    -------
    per_class_acc : dict {class_id: accuracy}
    """
    per_class: Dict[int, float] = {}
    for cls in range(n_classes):
        mask = targets == cls
        if mask.sum() == 0:
            per_class[cls] = float("nan")
            continue
        per_class[cls] = (
            ((predictions == targets) & mask).sum().item() / mask.sum().item()
        )
    return per_class


def mean_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int = 100,
) -> float:
    """
    Mean per-class accuracy (mAcc), also called class-averaged accuracy.

    Parameters
    ----------
    predictions : (N,) integer tensor
    targets     : (N,) integer tensor
    n_classes   : number of classes

    Returns
    -------
    mAcc : float in [0, 1]
    """
    per_class = per_class_accuracy(predictions, targets, n_classes)
    valid = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def confusion_matrix_np(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int = 100,
) -> np.ndarray:
    """
    Compute the (n_classes, n_classes) confusion matrix as a NumPy array.

    C[i, j] = number of samples with true class i predicted as class j.

    Parameters
    ----------
    predictions : (N,) integer tensor
    targets     : (N,) integer tensor
    n_classes   : number of classes

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
    """
    preds_np = predictions.cpu().numpy().astype(int)
    tgts_np = targets.cpu().numpy().astype(int)
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(tgts_np, preds_np):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            C[t, p] += 1
    return C


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int = 100,
) -> Dict[str, float]:
    """
    Convenience wrapper: compute OA and mAcc in one call.

    Parameters
    ----------
    predictions : (N,) integer tensor
    targets     : (N,) integer tensor
    n_classes   : number of classes

    Returns
    -------
    metrics : dict {"overall_accuracy": float, "mean_class_accuracy": float}
    """
    return {
        "overall_accuracy": accuracy(predictions, targets),
        "mean_class_accuracy": mean_class_accuracy(predictions, targets, n_classes),
    }
