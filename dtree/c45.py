"""C4.5 (Quinlan, 1993).

Improvements over ID3 implemented here:

* **Gain ratio**  GR(S, A) = IG(S, A) / SplitInfo(S, A) corrects ID3's bias
  toward many-valued attributes.  As in Quinlan's program, only attributes
  whose gain is at least the average gain of all candidates are eligible,
  which stops tiny split-info values from inflating the ratio.
* **Continuous attributes** are split with a binary test ``A <= t``; the
  threshold is the midpoint between adjacent sorted values that maximises
  information gain.  The same attribute may be tested again further down.
  Optionally the Release-8 MDL penalty ``log2(N_distinct - 1) / |S_known|``
  is subtracted from the gain of continuous splits.
* **Missing values**: gain is computed on the cases with a known value and
  scaled by the known fraction F; split info treats "missing" as an extra
  outcome.  During training a case with a missing value is sent down every
  branch with a fractional weight proportional to the branch's size; during
  prediction the class distributions of all branches are combined the same way.
* **Error-based (pessimistic) pruning** with confidence factor CF (default
  0.25): a subtree is replaced by a leaf when the leaf's upper-confidence
  error estimate is not worse than the subtree's.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .base import _EPS, BaseDecisionTree, Split, entropy, entropy_rows


def added_errors(n: float, e: float, cf: float) -> float:
    """Extra errors predicted by C4.5's pessimistic estimate (as in Quinlan / Weka).

    Returns U_CF(e, n) * n - e, where U_CF is the upper limit of the binomial
    confidence interval for the error rate at confidence ``cf``.
    """
    if n <= 0:
        return 0.0
    if cf > 0.5 or cf <= 0:
        raise ValueError("confidence factor must be in (0, 0.5]")
    if e < 1:
        base = n * (1 - cf ** (1.0 / n))
        if e == 0:
            return base
        return base + e * (added_errors(n, 1.0, cf) - base)
    if e + 0.5 >= n:
        return max(n - e, 0.0)
    z = NormalDist().inv_cdf(1 - cf)
    f = (e + 0.5) / n
    r = (f + z * z / (2 * n) + z * math.sqrt(f / n - f * f / n + z * z / (4 * n * n))) / (1 + z * z / n)
    return r * n - e


class C45Classifier(BaseDecisionTree):
    algorithm = "C4.5"
    criterion = "Gain ratio"
    criterion_short = "GR"
    handles_missing = True

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=2,
                 prune=True, confidence_factor=0.25, continuous_penalty=True):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        if not 0 < confidence_factor <= 0.5:
            raise ValueError("confidence_factor must be in (0, 0.5]")
        self.prune = prune
        self.confidence_factor = confidence_factor
        self.continuous_penalty = continuous_penalty

    def get_params(self):
        p = super().get_params()
        p.update(prune=self.prune, confidence_factor=self.confidence_factor,
                 continuous_penalty=self.continuous_penalty)
        return p

    def _infer_types(self, X):
        return {c: ("numeric" if is_numeric_dtype(X[c]) and not is_bool_dtype(X[c]) else "categorical")
                for c in X.columns}

    # ------------------------------------------------------------- splitting
    def _find_best_split(self, idx, w, counts, available):
        total = float(counts.sum())
        candidates = []
        for f in self.feature_names_:
            if f not in available:
                continue
            if self.feature_types_[f] == "numeric":
                r = self._numeric_best_threshold(f, idx, w)
                if r is None:
                    continue
                w_known = r["w_known"]
                gain = (w_known / total) * r["gain_known"]
                if self.continuous_penalty:
                    gain -= math.log2(max(r["n_distinct"] - 1, 1)) / w_known
                parts = r["parts"]
                extra = {"threshold": r["threshold"]}
            else:
                res = self._categorical_table(f, idx, w)
                if res is None:
                    continue
                values, table, _ = res
                branch_w = table.sum(axis=1)
                if len(values) < 2 or (branch_w >= self.min_samples_leaf).sum() < 2:
                    continue
                w_known = float(branch_w.sum())
                known_h = entropy(table.sum(axis=0))
                cond = float((branch_w * entropy_rows(table)).sum() / w_known)
                gain = (w_known / total) * (known_h - cond)
                parts = list(branch_w)
                extra = {"values": values}

            w_missing = total - w_known
            split_info = entropy(parts + ([w_missing] if w_missing > _EPS else []))
            if gain <= _EPS or split_info <= _EPS:
                continue
            kind = self.feature_types_[f]
            candidates.append(Split(f, kind, score=gain / split_info, info_gain=gain,
                                    split_info=split_info, **extra))
        if not candidates:
            return None
        avg_gain = float(np.mean([c.info_gain for c in candidates]))
        eligible = [c for c in candidates if c.info_gain >= avg_gain - _EPS]
        return max(eligible, key=lambda c: c.score)

    # --------------------------------------------------------------- pruning
    def _post_fit(self):
        if self.prune:
            self._prune(self.root_)

    def _prune(self, node) -> float:
        n = node.n_samples
        e = n - float(node.class_counts.max())
        leaf_est = e + added_errors(n, e, self.confidence_factor)
        if node.is_leaf:
            return leaf_est
        subtree_est = sum(self._prune(child) for child in node.children.values())
        if leaf_est <= subtree_est + 0.1:
            self.n_pruned_ += sum(1 for _ in self._iter_subtree(node)) - 1
            node.make_leaf()
            return leaf_est
        return subtree_est

    @staticmethod
    def _iter_subtree(node):
        stack = [node]
        while stack:
            cur = stack.pop()
            yield cur
            stack.extend(cur.children.values())
