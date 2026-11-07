import io
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dtree.metrics import evaluate
from dtree.pipeline import ModelConfig, comparison_table, cross_validate, train_and_evaluate
from dtree.preprocessing import (PreprocessConfig, Preprocessor, load_csv, prepare, profile,
                                 suggest_exclusions, suggest_target)
from dtree.visualize import render, tree_to_dot

DATA = Path(__file__).resolve().parents[1] / "data"


def loan():
    df = load_csv(DATA / "loan_approval.csv")
    feats = [c for c in df.columns if c not in ("loan_status", "applicant_id")]
    return df, feats


def test_load_csv_cleans_values():
    raw = b"a; b ;label\n1; x ;yes\n?; y ;no\n3;  ;yes\n4;x;no\n"
    df = load_csv(io.BytesIO(raw))
    assert list(df.columns) == ["a", "b", "label"]
    assert df["a"].isna().sum() == 1 and df["a"].dtype.kind == "f"
    assert df["b"].isna().sum() == 1
    assert df.loc[0, "b"] == "x"


def test_profile_and_suggestions():
    df, _ = loan()
    assert suggest_target(df) == "loan_status"
    assert "applicant_id" in suggest_exclusions(df, "loan_status")
    prof = profile(df)
    assert prof.loc[prof.column == "credit_score", "missing"].item() > 0


def test_preprocessor_learns_on_train_only():
    tr = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, 8], "c": list("aabbaabb")})
    te = pd.DataFrame({"x": [100.0, np.nan], "c": [np.nan, "z"]})
    p = Preprocessor(n_bins=4).fit(tr)
    out = p.transform_id3(te)
    assert out.loc[0, "x"] == p.bin_labels("x")[-1]            # beyond training range → last bin
    assert out.loc[1, "x"] == p.transform_id3(pd.DataFrame({"x": [4.5], "c": ["a"]})).loc[0, "x"]
    assert out.loc[0, "c"] in ("a", "b")                        # imputed with training mode
    native = Preprocessor(native_missing=True).fit(tr).transform_c45(te)
    assert native["x"].isna().sum() == 1


def test_prepare_split_and_views():
    df, feats = loan()
    cfg = PreprocessConfig("loan_status", feats, missing_strategy="native")
    d = prepare(df, cfg)
    assert len(d.y_train) + len(d.y_test) == len(df)
    assert not d.X_train_id3.isna().any().any()
    assert d.X_train_c45.isna().any().any()
    assert d.X_train_id3["credit_score"].dtype == object
    assert d.X_train_c45["credit_score"].dtype.kind == "f"


@pytest.mark.parametrize("strategy", ["impute", "drop", "native"])
def test_train_and_evaluate(strategy):
    df, feats = loan()
    d = prepare(df, PreprocessConfig("loan_status", feats, missing_strategy=strategy))
    res = train_and_evaluate(d, ModelConfig(include_sklearn=True))
    assert set(res) == {"ID3", "C4.5", "sklearn CART (entropy)"}
    for r in res.values():
        for k in ("accuracy", "precision", "recall", "f1"):
            assert 0 <= r.metrics[k] <= 1
        assert r.metrics["confusion_matrix"].to_numpy().sum() == len(d.y_test)
    assert res["C4.5"].metrics["accuracy"] > 0.7
    table = comparison_table(res)
    assert list(table.index) == list(res)


def test_cross_validate():
    df = load_csv(DATA / "iris.csv")
    cfg = PreprocessConfig("species", [c for c in df.columns if c != "species"])
    cv, desc = cross_validate(df, cfg, ModelConfig(), k=4)
    assert len(cv) == 8 and "stratified" in desc
    assert cv.groupby("model")["accuracy"].mean().min() > 0.8


def test_metrics_values():
    m = evaluate(["a", "a", "b", "b"], ["a", "b", "b", "b"], labels=["a", "b"])
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"].loc["a", "b"] == 1
    assert m["per_class"].loc["b", "precision"] == pytest.approx(2 / 3)


def test_errors_for_bad_config():
    df, feats = loan()
    with pytest.raises(ValueError):
        prepare(df, PreprocessConfig("loan_status", []))
    with pytest.raises(ValueError):
        prepare(df, PreprocessConfig("loan_status", feats + ["loan_status"]))
    one_class = df.assign(loan_status="Approved")
    with pytest.raises(ValueError):
        prepare(one_class, PreprocessConfig("loan_status", feats))


def test_dot_export_renders():
    df, feats = loan()
    d = prepare(df, PreprocessConfig("loan_status", feats))
    res = train_and_evaluate(d, ModelConfig(max_depth=3))
    dot = tree_to_dot(res["C4.5"].model, max_depth=2, rankdir="LR")
    assert dot.startswith("digraph") and "credit_score" in dot
    if shutil.which("dot"):
        svg = render(dot, "svg")
        assert svg and b"<svg" in svg
