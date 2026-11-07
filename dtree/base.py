"""Shared machinery for the from-scratch ID3 and C4.5 implementations.

Both algorithms grow a tree top-down with the same recursive procedure and
differ only in how a split is chosen (``_find_best_split``), which attribute
types they accept, and what happens after growth (C4.5 prunes).

Every training instance carries a weight.  Normally the weight is 1; C4.5
uses fractional weights to distribute instances whose split attribute is
missing across all branches (Quinlan, 1993).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

LEFT, RIGHT = "<=", ">"
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Entropy helpers
# --------------------------------------------------------------------------- #
def entropy(counts) -> float:
    """Shannon entropy (bits) of a vector of (possibly weighted) class counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def entropy_rows(matrix) -> np.ndarray:
    """Row-wise entropy of a 2-D array of class counts."""
    m = np.asarray(matrix, dtype=float)
    tot = m.sum(axis=1, keepdims=True)
    safe_tot = np.where(tot > 0, tot, 1.0)
    p = m / safe_tot
    logp = np.log2(np.where(p > 0, p, 1.0))
    return -(p * logp).sum(axis=1)


def to_str_object(series: pd.Series) -> pd.Series:
    """Convert values to ``str`` while keeping missing values as ``NaN`` (object dtype)."""
    arr = series.to_numpy(dtype=object)
    out = np.empty(len(arr), dtype=object)
    for i, v in enumerate(arr):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            out[i] = np.nan
        elif isinstance(v, float) and v.is_integer():
            out[i] = str(int(v))
        else:
            out[i] = str(v).strip()
    return pd.Series(out, index=series.index, dtype=object, name=series.name)


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def fmt_num(x: float) -> str:
    """Compact number formatting used in labels (4 significant digits)."""
    if x is None:
        return ""
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.4g}"


# --------------------------------------------------------------------------- #
# Tree structures
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    feature: str
    kind: str                     # "categorical" | "numeric"
    score: float                  # the value the algorithm maximises (IG or gain ratio)
    info_gain: float
    threshold: float | None = None
    values: np.ndarray | None = None
    split_info: float | None = None


@dataclass
class Node:
    node_id: int
    depth: int
    class_counts: np.ndarray
    feature: str | None = None
    kind: str | None = None
    threshold: float | None = None
    score: float = 0.0
    info_gain: float = 0.0
    split_info: float | None = None
    children: dict = field(default_factory=dict)
    branch_weights: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> float:
        return float(self.class_counts.sum())

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def distribution(self) -> np.ndarray:
        total = self.class_counts.sum()
        if total <= 0:
            return np.full(len(self.class_counts), 1.0 / len(self.class_counts))
        return self.class_counts / total

    def make_leaf(self) -> None:
        self.children = {}
        self.branch_weights = {}
        self.feature = self.kind = self.threshold = self.split_info = None
        self.score = self.info_gain = 0.0


def branch_label(node: Node, key) -> str:
    """Human-readable condition for the edge ``node -> children[key]``."""
    if node.kind == "numeric":
        op = "≤" if key == LEFT else ">"
        return f"{op} {fmt_num(node.threshold)}"
    return f"= {key}"


# --------------------------------------------------------------------------- #
# Base classifier
# --------------------------------------------------------------------------- #
class BaseDecisionTree:
    """Common fit / predict / inspect logic.  Subclasses implement the split rule."""

    algorithm = "base"
    criterion = ""
    criterion_short = ""
    handles_missing = False

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    # ------------------------------------------------------------------ API
    def get_params(self) -> dict:
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        t0 = time.perf_counter()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.reset_index(drop=True)
        X.columns = [str(c) for c in X.columns]
        y = np.asarray(pd.Series(y).to_numpy(dtype=object))
        if len(y) != len(X):
            raise ValueError("X and y have different numbers of rows")
        if len(y) == 0:
            raise ValueError("cannot fit on an empty dataset")
        if any(is_missing(v) for v in y):
            raise ValueError("the target contains missing values")

        self.feature_names_ = list(X.columns)
        self.feature_types_ = self._infer_types(X)
        X = self._coerce(X)

        self.classes_ = np.array(sorted(set(y.tolist()), key=str), dtype=object)
        self.n_classes_ = len(self.classes_)
        index = {c: i for i, c in enumerate(self.classes_)}
        self._y = np.array([index[v] for v in y], dtype=int)

        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)

        self._cols, self._missing = {}, {}
        for c in self.feature_names_:
            if self.feature_types_[c] == "numeric":
                arr = X[c].to_numpy(dtype=float)
                miss = np.isnan(arr)
            else:
                arr = X[c].to_numpy(dtype=object)
                miss = np.array([is_missing(v) for v in arr], dtype=bool)
            self._cols[c], self._missing[c] = arr, miss
        bad = [c for c, m in self._missing.items() if m.any()]
        if bad and not self.handles_missing:
            raise ValueError(f"{self.algorithm} cannot handle missing values (found in: "
                             f"{', '.join(bad)}). Impute them first.")

        self._ids = itertools.count()
        self.n_pruned_ = 0
        self.root_ = self._build(np.arange(len(X)), w, 0, frozenset(self.feature_names_))
        self._post_fit()
        del self._cols, self._missing, self._y
        self.fit_time_ = time.perf_counter() - t0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self._prepare_predict(X)
        rows = X.to_dict("records")
        if not rows:
            return np.empty((0, self.n_classes_))
        return np.vstack([self._proba_row(self.root_, r) for r in rows])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)] if len(proba) else np.array([], dtype=object)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=object)))

    # ------------------------------------------------------------ inspection
    def iter_nodes(self):
        stack = [self.root_]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(list(node.children.values())))

    @property
    def n_nodes_(self) -> int:
        return sum(1 for _ in self.iter_nodes())

    @property
    def n_leaves_(self) -> int:
        return sum(1 for n in self.iter_nodes() if n.is_leaf)

    @property
    def depth_(self) -> int:
        return max(n.depth for n in self.iter_nodes())

    @property
    def feature_importances_(self) -> pd.Series:
        """Total information gain contributed by each feature, weighted by node size."""
        imp = dict.fromkeys(self.feature_names_, 0.0)
        root_n = self.root_.n_samples or 1.0
        for n in self.iter_nodes():
            if not n.is_leaf:
                imp[n.feature] += n.n_samples / root_n * max(n.info_gain, 0.0)
        s = pd.Series(imp, dtype=float)
        if s.sum() > 0:
            s = s / s.sum()
        return s.sort_values(ascending=False)

    def rules(self) -> pd.DataFrame:
        """All root-to-leaf paths as IF … THEN rules."""
        out = []

        def walk(node, conds):
            if node.is_leaf:
                d = node.distribution()
                k = int(np.argmax(d))
                out.append({"rule": " AND ".join(conds) if conds else "(always)",
                            "prediction": self.classes_[k],
                            "support": round(node.n_samples, 2),
                            "confidence": round(float(d[k]), 4)})
                return
            for key, child in node.children.items():
                walk(child, conds + [f"{node.feature} {branch_label(node, key)}"])

        walk(self.root_, [])
        return pd.DataFrame(out, columns=["rule", "prediction", "support", "confidence"])

    def explain(self, row) -> tuple[list[str], dict]:
        """Trace one instance down the tree.  Returns (steps, class -> probability)."""
        frame = row.to_frame().T if isinstance(row, pd.Series) else pd.DataFrame([row])
        rec = self._prepare_predict(frame).to_dict("records")[0]
        steps, node = [], self.root_
        while not node.is_leaf:
            v = rec.get(node.feature)
            if is_missing(v):
                if self.handles_missing:
                    steps.append(f"{node.feature} is missing → combine all {len(node.children)} "
                                 "branches, weighted by their training frequency")
                else:
                    steps.append(f"{node.feature} is missing → stop and use this node's majority class")
                break
            key = self._branch_for(node, v)
            if key not in node.children:
                steps.append(f"{node.feature} = {v} was never seen here during training → "
                             "use this node's class distribution")
                break
            suffix = f"   (value: {fmt_num(v)})" if node.kind == "numeric" else ""
            steps.append(f"{node.feature} {branch_label(node, key)}{suffix}")
            node = node.children[key]
        proba = self._proba_row(self.root_, rec)
        return steps, {c: float(p) for c, p in zip(self.classes_, proba)}

    # -------------------------------------------------------------- internals
    def _infer_types(self, X: pd.DataFrame) -> dict:
        raise NotImplementedError

    def _find_best_split(self, idx, w, counts, available):
        raise NotImplementedError

    def _post_fit(self) -> None:
        """Hook run after the tree is grown (C4.5 prunes here)."""

    def _coerce(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.feature_names_:
            if self.feature_types_[c] == "numeric":
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
            else:
                X[c] = to_str_object(X[c])
        return X

    def _prepare_predict(self, X) -> pd.DataFrame:
        if not hasattr(self, "root_"):
            raise RuntimeError("the model has not been fitted yet")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
        X = X.copy()
        X.columns = [str(c) for c in X.columns]
        missing_cols = [c for c in self.feature_names_ if c not in X.columns]
        if missing_cols:
            raise ValueError(f"missing feature columns: {', '.join(missing_cols)}")
        return self._coerce(X[self.feature_names_].reset_index(drop=True))

    @staticmethod
    def _branch_for(node: Node, value):
        if node.kind == "numeric":
            return LEFT if float(value) <= node.threshold else RIGHT
        return str(value)

    def _proba_row(self, node: Node, row: dict) -> np.ndarray:
        if node.is_leaf:
            return node.distribution()
        v = row.get(node.feature)
        if is_missing(v):
            if self.handles_missing:
                acc = np.zeros(self.n_classes_)
                for key, child in node.children.items():
                    acc += node.branch_weights.get(key, 0.0) * self._proba_row(child, row)
                s = acc.sum()
                return acc / s if s > 0 else node.distribution()
            return node.distribution()
        key = self._branch_for(node, v)
        child = node.children.get(key)
        return node.distribution() if child is None else self._proba_row(child, row)

    def _build(self, idx, w, depth, available) -> Node:
        counts = np.bincount(self._y[idx], weights=w, minlength=self.n_classes_)
        node = Node(next(self._ids), depth, counts)
        if (np.count_nonzero(counts > _EPS) <= 1
                or not available
                or (self.max_depth is not None and depth >= self.max_depth)
                or counts.sum() < self.min_samples_split):
            return node

        split = self._find_best_split(idx, w, counts, available)
        if split is None:
            return node

        node.feature, node.kind, node.threshold = split.feature, split.kind, split.threshold
        node.score, node.info_gain, node.split_info = split.score, split.info_gain, split.split_info

        col = self._cols[split.feature][idx]
        miss = self._missing[split.feature][idx]
        known_idx, known_w, known_vals = idx[~miss], w[~miss], col[~miss]
        miss_idx, miss_w = idx[miss], w[miss]
        total_known = known_w.sum()

        if split.kind == "numeric":
            masks = {LEFT: known_vals <= split.threshold, RIGHT: known_vals > split.threshold}
        else:
            masks = {v: known_vals == v for v in split.values}

        child_available = available - {split.feature} if split.kind == "categorical" else available
        for key, m in masks.items():
            bw = known_w[m].sum()
            if bw <= 0:
                continue
            frac = bw / total_known
            node.branch_weights[key] = frac
            child_idx = np.concatenate([known_idx[m], miss_idx])
            child_w = np.concatenate([known_w[m], miss_w * frac])
            node.children[key] = self._build(child_idx, child_w, depth + 1, child_available)

        if len(node.children) < 2:
            node.make_leaf()
        return node

    # --- split statistics shared by subclasses ---------------------------------
    def _categorical_table(self, feature, idx, w):
        """(values, class-count table [V x K], missing weight) for a categorical feature."""
        col = self._cols[feature][idx]
        miss = self._missing[feature][idx]
        kv, kw, ky = col[~miss], w[~miss], self._y[idx][~miss]
        if len(kv) == 0:
            return None
        values, inv = np.unique(kv.astype(str), return_inverse=True)
        k = self.n_classes_
        table = np.bincount(inv * k + ky, weights=kw, minlength=len(values) * k).reshape(len(values), k)
        return values.astype(object), table, float(w[miss].sum())

    def _numeric_best_threshold(self, feature, idx, w):
        """Best binary threshold (by information gain on the known values)."""
        col = self._cols[feature][idx]
        miss = self._missing[feature][idx]
        v, yy, ww = col[~miss], self._y[idx][~miss], w[~miss]
        if len(v) < 2:
            return None
        order = np.argsort(v, kind="mergesort")
        v, yy, ww = v[order], yy[order], ww[order]
        n_distinct = int(np.count_nonzero(v[1:] > v[:-1]) + 1)
        if n_distinct < 2:
            return None
        onehot = np.zeros((len(v), self.n_classes_))
        onehot[np.arange(len(v)), yy] = ww
        cum = np.cumsum(onehot, axis=0)
        total = cum[-1]
        cand = np.nonzero(v[1:] > v[:-1])[0]
        left = cum[cand]
        right = total - left
        wl, wr = left.sum(axis=1), right.sum(axis=1)
        ok = (wl >= self.min_samples_leaf) & (wr >= self.min_samples_leaf)
        if not ok.any():
            return None
        cand, left, right, wl, wr = cand[ok], left[ok], right[ok], wl[ok], wr[ok]
        w_known = float(total.sum())
        cond = (wl * entropy_rows(left) + wr * entropy_rows(right)) / w_known
        gains = entropy(total) - cond
        b = int(np.argmax(gains))
        threshold = (v[cand[b]] + v[cand[b] + 1]) / 2.0
        return {"threshold": float(threshold), "gain_known": float(gains[b]),
                "parts": [float(wl[b]), float(wr[b])], "w_known": w_known,
                "n_distinct": n_distinct, "known_counts": total}
