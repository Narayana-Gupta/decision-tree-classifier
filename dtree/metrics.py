"""Evaluation metrics (thin wrapper over scikit-learn)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

AVERAGES = ("macro", "weighted", "micro")


def evaluate(y_true, y_pred, labels=None, average: str = "macro") -> dict:
    """Accuracy, precision, recall, F1, confusion matrix and a per-class report."""
    if average not in AVERAGES:
        raise ValueError(f"average must be one of {AVERAGES}")
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    labels = list(labels) if labels is not None else []
    for v in list(pd.unique(y_true)) + list(pd.unique(y_pred)):
        if v not in labels:
            labels.append(v)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=average,
                                                 zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None,
                                                     zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "average": average,
        "labels": labels,
        "confusion_matrix": pd.DataFrame(cm, index=labels, columns=labels),
        "per_class": pd.DataFrame({"precision": pc, "recall": rc, "f1-score": fc, "support": sc},
                                  index=pd.Index(labels, name="class")),
    }
