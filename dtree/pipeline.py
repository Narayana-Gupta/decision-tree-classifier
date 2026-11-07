"""Train, evaluate and compare the models on prepared data."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .c45 import C45Classifier
from .id3 import ID3Classifier
from .metrics import evaluate
from .preprocessing import PreparedData, PreprocessConfig, clean_for_modelling, prepare_split

SKLEARN_NAME = "sklearn CART (entropy)"


@dataclass
class ModelConfig:
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    prune: bool = True
    confidence_factor: float = 0.25
    continuous_penalty: bool = True
    include_sklearn: bool = False
    average: str = "macro"
    random_state: int = 42


@dataclass
class ModelResult:
    name: str
    model: object
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    metrics: dict
    fit_time: float
    predict_time: float
    n_nodes: int
    n_leaves: int
    depth: int


def _onehot(train: pd.DataFrame, test: pd.DataFrame):
    both = pd.concat([train, test], keys=["train", "test"])
    enc = pd.get_dummies(both, dtype=float)
    return enc.xs("train"), enc.xs("test")


def train_and_evaluate(data: PreparedData, cfg: ModelConfig) -> dict[str, ModelResult]:
    results: dict[str, ModelResult] = {}
    common = dict(max_depth=cfg.max_depth, min_samples_split=cfg.min_samples_split,
                  min_samples_leaf=cfg.min_samples_leaf)
    specs = [
        ("ID3", ID3Classifier(**common), data.X_train_id3, data.X_test_id3),
        ("C4.5", C45Classifier(**common, prune=cfg.prune, confidence_factor=cfg.confidence_factor,
                               continuous_penalty=cfg.continuous_penalty),
         data.X_train_c45, data.X_test_c45),
    ]
    for name, model, Xtr, Xte in specs:
        model.fit(Xtr, data.y_train)
        t0 = time.perf_counter()
        proba = model.predict_proba(Xte)
        pred = model.classes_[np.argmax(proba, axis=1)] if len(proba) else np.array([], dtype=object)
        pt = time.perf_counter() - t0
        results[name] = ModelResult(name, model, pred, proba,
                                    evaluate(data.y_test, pred, data.classes, cfg.average),
                                    model.fit_time_, pt, model.n_nodes_, model.n_leaves_, model.depth_)

    if cfg.include_sklearn:
        Xtr, Xte = _onehot(data.X_train_imputed, data.X_test_imputed)
        sk = DecisionTreeClassifier(criterion="entropy", random_state=cfg.random_state, **common)
        t0 = time.perf_counter()
        sk.fit(Xtr, data.y_train.to_numpy(dtype=object))
        ft = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = sk.predict(Xte)
        pt = time.perf_counter() - t0
        results[SKLEARN_NAME] = ModelResult(
            SKLEARN_NAME, sk, np.asarray(pred, dtype=object), sk.predict_proba(Xte),
            evaluate(data.y_test, pred, data.classes, cfg.average), ft, pt,
            int(sk.tree_.node_count), int(sk.get_n_leaves()), int(sk.get_depth()))
    return results


def comparison_table(results: dict[str, ModelResult]) -> pd.DataFrame:
    rows = {}
    for name, r in results.items():
        rows[name] = {
            "accuracy": r.metrics["accuracy"], "precision": r.metrics["precision"],
            "recall": r.metrics["recall"], "f1": r.metrics["f1"],
            "nodes": r.n_nodes, "leaves": r.n_leaves, "depth": r.depth,
            "fit time (ms)": r.fit_time * 1000, "predict time (ms)": r.predict_time * 1000,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def agreement(results: dict[str, ModelResult], a: str = "ID3", b: str = "C4.5") -> float:
    pa, pb = results[a].y_pred, results[b].y_pred
    return float(np.mean(pa == pb)) if len(pa) else float("nan")


def cross_validate(df: pd.DataFrame, prep_cfg: PreprocessConfig, model_cfg: ModelConfig,
                   k: int = 5) -> tuple[pd.DataFrame, str]:
    """k-fold CV of every model; preprocessing is re-fitted inside each fold."""
    X, y, _ = clean_for_modelling(df, prep_cfg)
    if k < 2 or k > len(X):
        raise ValueError(f"k must be between 2 and {len(X)}")
    if y.value_counts().min() >= k:
        splitter, kind = StratifiedKFold(k, shuffle=True, random_state=prep_cfg.random_state), "stratified"
    else:
        splitter, kind = KFold(k, shuffle=True, random_state=prep_cfg.random_state), "plain (a class has < k rows)"
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X, y), start=1):
        data = prepare_split(X, y, tr, te, prep_cfg)
        for name, r in train_and_evaluate(data, model_cfg).items():
            rows.append({"fold": fold, "model": name, "accuracy": r.metrics["accuracy"],
                         "precision": r.metrics["precision"], "recall": r.metrics["recall"],
                         "f1": r.metrics["f1"], "leaves": r.n_leaves, "fit time (ms)": r.fit_time * 1000})
    return pd.DataFrame(rows), f"{k}-fold {kind} cross-validation"


def cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    cols = ["accuracy", "precision", "recall", "f1", "leaves"]
    g = cv.groupby("model", sort=False)[cols]
    mean, std = g.mean(), g.std(ddof=1).fillna(0)
    out = pd.DataFrame(index=mean.index)
    for c in cols:
        fmt = "{:.1f} ± {:.1f}" if c == "leaves" else "{:.3f} ± {:.3f}"
        out[c] = [fmt.format(m, s) for m, s in zip(mean[c], std[c])]
    return out
