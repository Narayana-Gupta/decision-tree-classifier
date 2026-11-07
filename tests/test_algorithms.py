import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from dtree import C45Classifier, ID3Classifier, entropy
from dtree.c45 import added_errors

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def tennis():
    df = pd.read_csv(DATA / "play_tennis.csv")
    return df.drop(columns=["Day", "PlayTennis"]), df["PlayTennis"]


@pytest.fixture
def weather():
    df = pd.read_csv(DATA / "weather_numeric.csv")
    return df.drop(columns="Play"), df["Play"]


@pytest.fixture
def iris():
    df = pd.read_csv(DATA / "iris.csv")
    return df.drop(columns="species"), df["species"]


def test_entropy_known_values():
    assert entropy([9, 5]) == pytest.approx(0.9403, abs=1e-4)
    assert entropy([7, 7]) == pytest.approx(1.0)
    assert entropy([10, 0]) == 0.0
    assert entropy([]) == 0.0


# ------------------------------------------------------------------ ID3
def test_id3_reproduces_mitchell_tree(tennis):
    X, y = tennis
    t = ID3Classifier().fit(X, y)
    root = t.root_
    assert root.feature == "Outlook"
    assert root.info_gain == pytest.approx(0.2467, abs=1e-4)
    assert set(root.children) == {"Sunny", "Overcast", "Rain"}
    assert root.children["Overcast"].is_leaf
    assert root.children["Sunny"].feature == "Humidity"
    assert root.children["Rain"].feature == "Wind"
    assert (t.n_nodes_, t.n_leaves_, t.depth_) == (8, 5, 2)
    assert t.score(X, y) == 1.0


def test_id3_unseen_value_falls_back_to_node_majority(tennis):
    X, y = tennis
    t = ID3Classifier().fit(X, y)
    row = pd.DataFrame([{"Outlook": "Snow", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}])
    assert t.predict(row)[0] == "Yes"          # 9 Yes vs 5 No at the root
    steps, proba = t.explain(row.iloc[0])
    assert "never seen" in steps[0]
    assert sum(proba.values()) == pytest.approx(1.0)


def test_id3_rejects_missing_values(tennis):
    X, y = tennis
    X = X.copy()
    X.loc[0, "Wind"] = np.nan
    with pytest.raises(ValueError, match="cannot handle missing"):
        ID3Classifier().fit(X, y)


def test_id3_max_depth(tennis):
    X, y = tennis
    t = ID3Classifier(max_depth=1).fit(X, y)
    assert t.depth_ == 1 and t.n_leaves_ == 3


def test_rules_cover_every_leaf(tennis):
    X, y = tennis
    t = ID3Classifier().fit(X, y)
    rules = t.rules()
    assert len(rules) == t.n_leaves_
    assert rules["support"].sum() == pytest.approx(len(X))


# ------------------------------------------------------------------ C4.5
def test_c45_numeric_thresholds_on_quinlan_weather(weather):
    X, y = weather
    t = C45Classifier(prune=False, min_samples_leaf=1).fit(X, y)
    assert t.root_.feature == "Outlook"
    sunny = t.root_.children["sunny"]
    assert sunny.feature == "Humidity" and sunny.kind == "numeric"
    assert 70 < sunny.threshold < 85
    assert t.score(X, y) == 1.0


def test_c45_prefers_gain_ratio_over_id_column(tennis):
    X, y = tennis
    X = X.assign(Day=[f"D{i}" for i in range(len(X))])
    id3 = ID3Classifier().fit(X, y)
    c45 = C45Classifier(prune=False).fit(X, y)  # C4.5 default: >= 2 cases in two branches
    assert id3.root_.feature == "Day"           # information gain's bias toward many values
    assert c45.root_.feature == "Outlook"       # gain ratio + minimum-cases rule avoid it
    # gain ratio alone already shrinks the ID column's advantage a lot
    unrestricted = C45Classifier(prune=False, min_samples_leaf=1).fit(X, y)
    assert unrestricted.root_.score < id3.root_.info_gain / 3


def test_c45_missing_values_train_and_predict(iris):
    X, y = iris
    rng = np.random.default_rng(0)
    X = X.copy()
    X = X.mask(rng.random(X.shape) < 0.1)
    t = C45Classifier().fit(X, y)
    proba = t.predict_proba(pd.DataFrame([{c: np.nan for c in X.columns}]))
    assert proba.shape == (1, 3)
    assert proba.sum() == pytest.approx(1.0)
    assert t.score(X.fillna(X.median()), y) > 0.8


def test_c45_pruning_never_grows_tree(iris):
    X, y = iris
    a, b, ya, yb = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    full = C45Classifier(prune=False, min_samples_leaf=1).fit(a, ya)
    pruned = C45Classifier(prune=True, min_samples_leaf=1).fit(a, ya)
    assert pruned.n_leaves_ <= full.n_leaves_
    assert pruned.score(b, yb) >= 0.85


def test_c45_matches_sklearn_accuracy_ballpark(iris):
    from sklearn.tree import DecisionTreeClassifier
    X, y = iris
    a, b, ya, yb = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    ours = C45Classifier().fit(a, ya).score(b, yb)
    sk = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(a, ya).score(b, yb)
    assert abs(ours - sk) < 0.12


def test_added_errors():
    # E = 0: U = 1 - CF^(1/N)
    assert added_errors(6, 0, 0.25) == pytest.approx(6 * (1 - 0.25 ** (1 / 6)))
    # monotone in e and bounded by n - e
    assert added_errors(20, 2, 0.25) > 0
    assert added_errors(4, 4, 0.25) == 0
    assert added_errors(0, 0, 0.25) == 0.0


def test_feature_importances_sum_to_one(weather):
    X, y = weather
    imp = C45Classifier(prune=False, min_samples_leaf=1).fit(X, y).feature_importances_
    assert math.isclose(imp.sum(), 1.0)
    assert set(imp.index) == set(X.columns)


def test_parameter_validation():
    with pytest.raises(ValueError):
        ID3Classifier(max_depth=0)
    with pytest.raises(ValueError):
        C45Classifier(confidence_factor=0.9)
    with pytest.raises(RuntimeError):
        ID3Classifier().predict(pd.DataFrame({"a": ["x"]}))
