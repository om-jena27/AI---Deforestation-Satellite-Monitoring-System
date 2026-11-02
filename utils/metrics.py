"""
utils/metrics.py
================
Evaluation metrics for binary segmentation masks.

IoU, Dice/F1, Precision, Recall, Accuracy — all computed from flat
numpy arrays (pred and target must be the same shape).
"""

from __future__ import annotations

import numpy as np


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-6,
) -> dict:
    """
    Compute binary segmentation metrics.

    Parameters
    ----------
    pred   : (H, W) uint8 or bool  — predicted mask  {0, 1}
    target : (H, W) uint8 or bool  — ground-truth mask {0, 1}
    eps    : smoothing term to avoid division by zero

    Returns
    -------
    dict with: accuracy, precision, recall, f1, dice, iou, tp, fp, fn, tn
    """
    p = pred.astype(bool).ravel()
    t = target.astype(bool).ravel()

    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())

    accuracy  = (tp + tn) / (tp + fp + fn + tn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    dice      = f1  # identical to F1 for binary case
    iou       = tp / (tp + fp + fn + eps)

    return {
        "accuracy":  float(accuracy),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "dice":      float(dice),
        "iou":       float(iou),
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "tn":        tn,
    }
