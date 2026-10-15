"""Evaluation metrics when labels are available."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_binary(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": fpr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "note": "Metrics require ground-truth labels. Scores are not claimed as calibrated probabilities.",
    }


def unsupervised_summary(scores: list[float], top_n: int = 10) -> dict[str, Any]:
    arr = np.asarray(scores, dtype=float)
    return {
        "n": int(len(arr)),
        "score_mean": float(arr.mean()) if len(arr) else 0.0,
        "score_std": float(arr.std()) if len(arr) else 0.0,
        "score_min": float(arr.min()) if len(arr) else 0.0,
        "score_max": float(arr.max()) if len(arr) else 0.0,
        "top_n_threshold": float(np.sort(arr)[-top_n]) if len(arr) >= top_n else None,
        "note": "Unsupervised run: no precision/recall without labels. Scores are ranking signals only.",
    }
