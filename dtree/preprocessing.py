"""Loading, profiling and preprocessing of a classification CSV.

Pipeline (all statistics are learned on the *training* split only, then
applied unchanged to the test split and to new data):

1. Clean: trim names/values, treat ``?``, ``NA``, ``null``… as missing,
   convert columns whose values are all numeric to numbers.
2. Drop rows whose target is missing (and, if requested, rows with any
   missing feature).
3. Train/test split (stratified when every class has enough rows).
4. Impute: median for numeric, mode for categorical features.
5. Discretise numeric features for ID3 (equal-frequency or equal-width bins).

C4.5 receives numeric features unchanged and, with the ``native`` strategy,
keeps missing values so the algorithm can handle them itself.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split

from .base import fmt_num, to_str_object

NA_VALUES = ["?", "NA", "N/A", "n/a", "na", "NaN", "nan", "null", "NULL", "None", "none", "-", ""]
MISSING_STRATEGIES = {
    "impute": "Impute (median / mode)",
    "drop": "Drop rows with missing values",
    "native": "C4.5 native handling (ID3 still imputes)",
}
TARGET_HINTS = ("target", "class", "label", "outcome", "y", "status", "species", "result")


# --------------------------------------------------------------------------- #
# Loading & profiling
# --------------------------------------------------------------------------- #
def load_csv(source, drop_empty: bool = True) -> pd.DataFrame:
    """Read a CSV from a path, bytes, or file-like object and lightly clean it."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    kwargs = dict(na_values=NA_VALUES, keep_default_na=True, skipinitialspace=True)
    try:
        df = pd.read_csv(source, sep=None, engine="python", **kwargs)
    except (pd.errors.ParserError, csv.Error, UnicodeDecodeError):
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, encoding_errors="replace", **kwargs)
    return clean_frame(df, drop_empty=drop_empty)


def clean_frame(df: pd.DataFrame, drop_empty: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("the CSV has duplicate column names")
    if drop_empty:
        df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all").reset_index(drop=True)
    if df.empty:
        raise ValueError("the CSV contains no data")
    for c in df.columns:
        if is_numeric_dtype(df[c]) or is_bool_dtype(df[c]):
            continue
        s = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.replace({"": np.nan})
        conv = pd.to_numeric(s, errors="coerce")
        nonnull = int(s.notna().sum())
        if nonnull > 0 and int(conv.notna().sum()) == nonnull:
            df[c] = conv
        else:
            df[c] = to_str_object(s)
    return df


def column_kind(s: pd.Series) -> str:
    return "numeric" if is_numeric_dtype(s) and not is_bool_dtype(s) else "categorical"


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        kind = column_kind(s)
        nunique = int(s.nunique(dropna=True))
        notes = []
        if nunique <= 1:
            notes.append("constant")
        if looks_like_id(s):
            notes.append("looks like an ID")
        if kind == "categorical" and nunique > max(20, 0.5 * n):
            notes.append("high cardinality")
        sample = ", ".join(fmt_num(v) if isinstance(v, (int, float, np.number)) else str(v)
                           for v in s.dropna().unique()[:4])
        rows.append({"column": c, "type": kind, "missing": int(s.isna().sum()),
                     "missing %": round(100 * s.isna().mean(), 1), "unique": nunique,
                     "examples": sample, "notes": "; ".join(notes)})
    return pd.DataFrame(rows)


def looks_like_id(s: pd.Series) -> bool:
    n = len(s)
    if n < 5:
        return False
    nunique = s.nunique(dropna=True)
    name = str(s.name).lower()
    if any(tok in name for tok in ("id", "serial", "index", "s.no", "sno")) and nunique >= 0.9 * n:
        return True
    if nunique == n and column_kind(s) == "categorical":
        return True
    if nunique == n and column_kind(s) == "numeric":
        vals = np.sort(s.dropna().to_numpy(dtype=float))
        return bool(np.all(np.diff(vals) == 1))
    return False


def suggest_target(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in TARGET_HINTS:
            return c
    return df.columns[-1]


def suggest_exclusions(df: pd.DataFrame, target: str) -> list[str]:
    return [c for c in df.columns if c != target and (looks_like_id(df[c]) or df[c].nunique(dropna=True) <= 1)]


def target_issues(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """(errors, warnings) about using ``target`` as the class attribute."""
    errors, warnings = [], []
    s = df[target].dropna()
    k = s.nunique()
    if k < 2:
        errors.append(f"'{target}' has fewer than two distinct classes.")
    if column_kind(df[target]) == "numeric" and k > 20:
        warnings.append(f"'{target}' is numeric with {k} distinct values — it looks continuous. "
                        "This is a classification system; each value will be treated as a class.")
    elif k > 50:
        warnings.append(f"'{target}' has {k} classes; trees and confusion matrices will be large.")
    if df[target].isna().any():
        warnings.append(f"{int(df[target].isna().sum())} rows have no target value and will be dropped.")
    if k >= 2 and s.value_counts().min() < 2:
        warnings.append("Some classes have a single example, so the split cannot be stratified.")
    return errors, warnings


def label_series(s: pd.Series) -> pd.Series:
    """Target values as clean strings (``1.0`` → ``"1"``)."""
    return to_str_object(s)


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
class Preprocessor:
    """Learns imputation values and bin edges on training data."""

    def __init__(self, n_bins: int = 4, binning: str = "quantile", native_missing: bool = False):
        if binning not in ("quantile", "uniform"):
            raise ValueError("binning must be 'quantile' or 'uniform'")
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins, self.binning, self.native_missing = n_bins, binning, native_missing

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.features_ = list(X.columns)
        self.numeric_ = [c for c in X.columns if column_kind(X[c]) == "numeric"]
        self.categorical_ = [c for c in X.columns if c not in self.numeric_]
        self.medians_, self.modes_, self.bin_edges_, self.categories_ = {}, {}, {}, {}
        for c in self.numeric_:
            vals = pd.to_numeric(X[c], errors="coerce").dropna().to_numpy(dtype=float)
            self.medians_[c] = float(np.median(vals)) if len(vals) else 0.0
            self.bin_edges_[c] = self._edges(vals)
        for c in self.categorical_:
            s = to_str_object(X[c]).dropna()
            self.modes_[c] = s.mode().iloc[0] if len(s) else "missing"
            self.categories_[c] = sorted(s.unique().tolist(), key=str)
        return self

    def _edges(self, x: np.ndarray) -> np.ndarray:
        uniq = np.unique(x)
        if len(uniq) < 2:
            return np.array([])
        k = min(self.n_bins, len(uniq))
        if self.binning == "quantile":
            edges = np.quantile(x, np.linspace(0, 1, k + 1))[1:-1]
        else:
            edges = np.linspace(x.min(), x.max(), k + 1)[1:-1]
        return np.unique(np.round(edges, 10))

    def bin_labels(self, c: str) -> list[str]:
        e = self.bin_edges_[c]
        if len(e) == 0:
            return ["all values"]
        labels = [f"< {fmt_num(e[0])}"]
        labels += [f"[{fmt_num(a)}, {fmt_num(b)})" for a, b in zip(e[:-1], e[1:])]
        labels.append(f"≥ {fmt_num(e[-1])}")
        return labels

    def impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features_].copy()
        for c in self.numeric_:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float).fillna(self.medians_[c])
        for c in self.categorical_:
            X[c] = to_str_object(X[c]).fillna(self.modes_[c])
        return X

    def transform_id3(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.impute(X)
        for c in self.numeric_:
            labels = np.array(self.bin_labels(c), dtype=object)
            codes = np.digitize(X[c].to_numpy(dtype=float), self.bin_edges_[c], right=False)
            X[c] = pd.Series(labels[codes], index=X.index, dtype=object)
        return X

    def transform_c45(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.native_missing:
            return self.impute(X)
        X = X[self.features_].copy()
        for c in self.numeric_:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        for c in self.categorical_:
            X[c] = to_str_object(X[c])
        return X

    def bins_table(self) -> pd.DataFrame:
        rows = [{"feature": c, "cut points": ", ".join(fmt_num(v) for v in self.bin_edges_[c]) or "—",
                 "ID3 categories": " | ".join(self.bin_labels(c))} for c in self.numeric_]
        return pd.DataFrame(rows, columns=["feature", "cut points", "ID3 categories"])


# --------------------------------------------------------------------------- #
# End-to-end preparation
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessConfig:
    target: str
    features: list[str]
    test_size: float = 0.3
    random_state: int = 42
    missing_strategy: str = "impute"
    binning: str = "quantile"
    n_bins: int = 4
    stratify: bool = True


@dataclass
class PreparedData:
    config: PreprocessConfig
    classes: list
    feature_types: dict
    preprocessor: Preprocessor
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_id3: pd.DataFrame
    X_test_id3: pd.DataFrame
    X_train_c45: pd.DataFrame
    X_test_c45: pd.DataFrame
    X_train_imputed: pd.DataFrame
    X_test_imputed: pd.DataFrame
    log: list = field(default_factory=list)

    def transform_new(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the fitted preprocessing to new rows → (ID3 view, C4.5 view)."""
        missing = [c for c in self.preprocessor.features_ if c not in X.columns]
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")
        X = X[self.preprocessor.features_]
        return self.preprocessor.transform_id3(X), self.preprocessor.transform_c45(X)


def clean_for_modelling(df: pd.DataFrame, cfg: PreprocessConfig) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Select columns, drop unusable rows, return (X, y, log)."""
    log = []
    if not cfg.features:
        raise ValueError("select at least one feature")
    if cfg.target in cfg.features:
        raise ValueError("the target cannot also be a feature")
    absent = [c for c in [cfg.target, *cfg.features] if c not in df.columns]
    if absent:
        raise ValueError(f"columns not found: {', '.join(absent)}")
    if cfg.missing_strategy not in MISSING_STRATEGIES:
        raise ValueError(f"unknown missing-value strategy '{cfg.missing_strategy}'")

    data = df[[*cfg.features, cfg.target]].copy()
    log.append(f"Selected {len(cfg.features)} feature(s) and target '{cfg.target}' from {len(df)} rows.")
    no_target = data[cfg.target].isna()
    if no_target.any():
        log.append(f"Dropped {int(no_target.sum())} row(s) with a missing target.")
        data = data[~no_target]
    n_missing = int(data[cfg.features].isna().sum().sum())
    if cfg.missing_strategy == "drop" and n_missing:
        keep = data[cfg.features].notna().all(axis=1)
        log.append(f"Dropped {int((~keep).sum())} row(s) containing missing feature values.")
        data = data[keep]
    elif n_missing:
        log.append(f"Found {n_missing} missing feature value(s); strategy: "
                   f"{MISSING_STRATEGIES[cfg.missing_strategy].lower()}.")
    else:
        log.append("No missing feature values.")
    data = data.reset_index(drop=True)
    y = label_series(data[cfg.target])
    X = data[cfg.features]
    if len(X) < 4:
        raise ValueError("fewer than 4 usable rows remain after cleaning")
    if y.nunique() < 2:
        raise ValueError("the target needs at least two classes after cleaning")
    return X, y, log


def can_stratify(y: pd.Series, test_size: float) -> bool:
    counts = y.value_counts()
    n_test = math.ceil(test_size * len(y))
    n_train = len(y) - n_test
    return counts.min() >= 2 and n_test >= len(counts) and n_train >= len(counts)


def prepare_split(X: pd.DataFrame, y: pd.Series, train_idx, test_idx, cfg: PreprocessConfig,
                  log: list[str] | None = None) -> PreparedData:
    log = list(log or [])
    X_tr, X_te = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_tr, y_te = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    pre = Preprocessor(cfg.n_bins, cfg.binning, native_missing=cfg.missing_strategy == "native").fit(X_tr)
    if pre.numeric_:
        log.append(f"ID3: discretised {len(pre.numeric_)} numeric feature(s) into ≤{cfg.n_bins} "
                   f"{'equal-frequency' if cfg.binning == 'quantile' else 'equal-width'} bins "
                   "(cut points learned on the training set).")
        log.append(f"C4.5: kept {len(pre.numeric_)} numeric feature(s) continuous (binary threshold splits).")
    if X_tr.isna().any().any() or X_te.isna().any().any():
        if cfg.missing_strategy == "native":
            log.append("ID3: imputed missing values (median / mode). C4.5: kept them as missing "
                       "(fractional instances).")
        else:
            log.append("Imputed missing values with the training median (numeric) / mode (categorical).")
    classes = sorted(y.unique().tolist(), key=str)
    return PreparedData(
        config=cfg, classes=classes,
        feature_types={c: ("numeric" if c in pre.numeric_ else "categorical") for c in X.columns},
        preprocessor=pre, X_train_raw=X_tr, X_test_raw=X_te, y_train=y_tr, y_test=y_te,
        X_train_id3=pre.transform_id3(X_tr), X_test_id3=pre.transform_id3(X_te),
        X_train_c45=pre.transform_c45(X_tr), X_test_c45=pre.transform_c45(X_te),
        X_train_imputed=pre.impute(X_tr), X_test_imputed=pre.impute(X_te),
        log=log,
    )


def prepare(df: pd.DataFrame, cfg: PreprocessConfig) -> PreparedData:
    if not 0.05 <= cfg.test_size <= 0.6:
        raise ValueError("test_size must be between 0.05 and 0.6")
    X, y, log = clean_for_modelling(df, cfg)
    strat = y if cfg.stratify and can_stratify(y, cfg.test_size) else None
    tr, te = train_test_split(np.arange(len(X)), test_size=cfg.test_size,
                              random_state=cfg.random_state, stratify=strat)
    log.append(f"Split {len(X)} rows into {len(tr)} training / {len(te)} test rows "
               f"({'stratified' if strat is not None else 'not stratified'}, seed {cfg.random_state}).")
    return prepare_split(X, y, tr, te, cfg, log)
