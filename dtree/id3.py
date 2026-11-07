"""ID3 (Quinlan, 1986).

* Split criterion: information gain  IG(S, A) = H(S) - sum_v |S_v|/|S| * H(S_v)
* Every attribute is treated as categorical and produces a multi-way split
  (one branch per value).  An attribute is used at most once on any path.
* No native support for continuous attributes (discretise them first) or for
  missing values (impute them first).  The preprocessing module does both.
* No pruning; growth stops when a node is pure, no attributes remain, or a
  user limit (max depth / min samples) is reached.
"""
from __future__ import annotations

from .base import _EPS, BaseDecisionTree, Split, entropy, entropy_rows


class ID3Classifier(BaseDecisionTree):
    algorithm = "ID3"
    criterion = "Information gain"
    criterion_short = "IG"
    handles_missing = False

    def _infer_types(self, X):
        return {c: "categorical" for c in X.columns}

    def _find_best_split(self, idx, w, counts, available):
        parent_h = entropy(counts)
        total = counts.sum()
        best = None
        for f in self.feature_names_:
            if f not in available:
                continue
            res = self._categorical_table(f, idx, w)
            if res is None:
                continue
            values, table, _ = res
            branch_w = table.sum(axis=1)
            if len(values) < 2 or (branch_w >= self.min_samples_leaf).sum() < 2:
                continue
            ig = parent_h - float((branch_w * entropy_rows(table)).sum() / total)
            if best is None or ig > best.score + _EPS:
                best = Split(f, "categorical", score=ig, info_gain=ig, values=values)
        if best is None or best.score <= _EPS:
            return None
        return best
