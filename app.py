"""Decision Tree Classification System — Streamlit front end.

Run with:  streamlit run app.py

Contributors: Karthik Naidu Allu (2023002399), Boddu Narayana Gupta (2023002459)
"""
from __future__ import annotations

import hashlib
import html
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dtree import CONTRIBUTORS, __version__
from dtree import plots
from dtree.metrics import AVERAGES
from dtree.pipeline import (SKLEARN_NAME, ModelConfig, agreement, comparison_table,
                            cross_validate, cv_summary, train_and_evaluate)
from dtree.preprocessing import (MISSING_STRATEGIES, PreprocessConfig, load_csv, prepare,
                                 profile, suggest_exclusions, suggest_target, target_issues)
from dtree.theme import (ANTIQUE_GOLD, ANTIQUE_WHITE, CREAM, DARK_GREEN, GITAM_GREEN,
                         LIGHT_GREEN, TEXT, model_color)
from dtree.visualize import render, tree_to_dot

DATA_DIR = Path(__file__).parent / "data"
SAMPLES = {
    "Play Tennis (categorical, ID3 classic)": ("play_tennis.csv", "PlayTennis"),
    "Weather (numeric + categorical, C4.5 classic)": ("weather_numeric.csv", "Play"),
    "Iris (continuous, 3 classes)": ("iris.csv", "species"),
    "Loan approval (mixed types, missing values)": ("loan_approval.csv", "loan_status"),
}
TREE_MODELS = ("ID3", "C4.5")

st.set_page_config(page_title="Decision Tree Classification System", page_icon="🌳",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Styling (GITAM palette, Inter = GITAM's digital secondary typeface)
# --------------------------------------------------------------------------- #
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th {{ font-family: 'Inter', sans-serif; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: {DARK_GREEN}; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid #E8D7B8; }}
section[data-testid="stSidebar"] h3 {{ font-size: 0.95rem; margin-top: 0.6rem; color: {GITAM_GREEN}; }}

.hero {{ position: relative; overflow: hidden; background: {GITAM_GREEN}; color: {ANTIQUE_WHITE};
        border-radius: 18px; padding: 26px 30px 22px; margin-bottom: 18px; }}
.hero::after {{ content: ""; position: absolute; right: -40px; top: -30px; width: 360px; height: 260px;
        background: repeating-linear-gradient(115deg, rgba(0,71,64,.75) 0 7px, transparent 7px 16px);
        border-radius: 0 0 0 100%; }}
.hero > * {{ position: relative; z-index: 1; }}
.hero .eyebrow {{ font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; opacity: .85; }}
.hero h1 {{ color: {ANTIQUE_WHITE} !important; font-size: 2.05rem; font-weight: 700; margin: 6px 0 4px; padding: 0; }}
.hero p {{ color: {ANTIQUE_WHITE}; opacity: .92; max-width: 780px; margin: 0 0 14px; font-size: .98rem; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
.chips .lbl {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; opacity: .85; margin-right: 4px; }}
.chip {{ background: {ANTIQUE_WHITE}; color: {DARK_GREEN}; border-radius: 999px; padding: 5px 12px;
        font-size: .86rem; font-weight: 600; }}
.chip span {{ font-weight: 400; color: {ANTIQUE_GOLD}; margin-left: 6px; }}

.card {{ background: #FFFFFF; border: 1px solid #EADBC0; border-left: 4px solid {GITAM_GREEN};
        border-radius: 12px; padding: 12px 16px; margin-bottom: 10px; }}
.card.gold {{ border-left-color: {ANTIQUE_GOLD}; }}
.card h4 {{ margin: 0 0 4px; font-size: 1rem; }}
.card p, .card li {{ margin: 0; font-size: .92rem; }}
.pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600;
        background: {ANTIQUE_WHITE}; color: {DARK_GREEN}; margin-right: 6px; }}
div[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid #EADBC0; border-radius: 12px;
        padding: 10px 14px; }}
div[data-testid="stMetricValue"] {{ color: {DARK_GREEN}; }}
button[data-baseweb="tab"] p {{ font-weight: 600; }}
.path-step {{ font-family: 'Inter', sans-serif; padding: 6px 10px; margin: 4px 0; background: #FFFFFF;
        border: 1px solid #EADBC0; border-radius: 8px; font-size: .9rem; }}
.footer {{ margin-top: 36px; padding: 16px 20px; border-top: 3px solid {GITAM_GREEN}; color: {TEXT};
        font-size: .86rem; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; }}
.footer b {{ color: {DARK_GREEN}; }}
</style>
""", unsafe_allow_html=True)


def contributor_chips() -> str:
    return "".join(f'<span class="chip">{n}<span>{r}</span></span>' for n, r in CONTRIBUTORS)


st.markdown(f"""
<div class="hero">
  <div class="eyebrow">GITAM (Deemed to be University) · Machine Learning Project</div>
  <h1>Decision Tree Classification System</h1>
  <p>Upload a classification CSV, choose the target attribute, preprocess the data and compare
     <b>ID3</b> (information gain) with <b>C4.5</b> (gain ratio, continuous thresholds,
     missing-value handling and error-based pruning).</p>
  <div class="chips"><span class="lbl">Contributors</span>{contributor_chips()}</div>
</div>
""", unsafe_allow_html=True)


def footer():
    names = " &nbsp;·&nbsp; ".join(f"<b>{n}</b> ({r})" for n, r in CONTRIBUTORS)
    st.markdown(f"""<div class="footer"><div>Developed by {names}</div>
    <div>Python · Pandas · NumPy · scikit-learn · Matplotlib · Graphviz · Streamlit &nbsp;|&nbsp; v{__version__}</div>
    </div>""", unsafe_allow_html=True)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def read_dataset(raw: bytes) -> pd.DataFrame:
    return load_csv(raw)


@st.cache_data(show_spinner=False)
def render_cached(dot: str, fmt: str):
    return render(dot, fmt)


# --------------------------------------------------------------------------- #
# Sidebar — dataset and settings
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.markdown("### 1 · Dataset")
source = sb.radio("Source", ["Upload CSV", "Sample dataset"], horizontal=True, label_visibility="collapsed")
raw, data_name, sample_target = None, None, None
if source == "Upload CSV":
    up = sb.file_uploader("Classification dataset (.csv)", type=["csv"])
    if up is not None:
        raw, data_name = up.getvalue(), up.name
else:
    choice = sb.selectbox("Sample dataset", list(SAMPLES))
    fname, sample_target = SAMPLES[choice]
    raw, data_name = (DATA_DIR / fname).read_bytes(), fname

if raw is None:
    st.markdown(f"""
<div class="card"><h4>Get started</h4>
<p>Upload a CSV in the sidebar (one row per example, one column per attribute, including a class column),
or pick a sample dataset. The system then lets you choose the target, preprocesses the data,
builds ID3 and C4.5 trees, and evaluates and compares them.</p></div>
<div class="card gold"><h4>What the system does</h4><ul>
<li><b>Preprocessing</b> — cleans values, imputes or drops missing data, discretises numeric attributes for ID3, stratified train/test split.</li>
<li><b>Models</b> — ID3 and C4.5 implemented from scratch in NumPy/Pandas (optional scikit-learn CART reference).</li>
<li><b>Outputs</b> — Graphviz trees, IF–THEN rules, predictions with decision paths, confusion matrices, accuracy / precision / recall / F1, side-by-side comparison and k-fold cross-validation.</li>
</ul></div>""", unsafe_allow_html=True)
    footer()
    st.stop()

try:
    df = read_dataset(raw)
except Exception as exc:  # noqa: BLE001 — surface any parse problem to the user
    st.error(f"Could not read **{data_name}**: {exc}")
    footer()
    st.stop()

sb.caption(f"**{data_name}** · {len(df):,} rows × {df.shape[1]} columns")

sb.markdown("### 2 · Target & attributes")
cols = list(df.columns)
default_target = sample_target if sample_target in cols else suggest_target(df)
target = sb.selectbox("Target (class) attribute", cols, index=cols.index(default_target))
candidates = [c for c in cols if c != target]
excluded_default = suggest_exclusions(df, target)
features = sb.multiselect("Attributes used for training", candidates,
                          default=[c for c in candidates if c not in excluded_default],
                          help="ID-like and constant columns are left out by default.")

sb.markdown("### 3 · Preprocessing")
test_size = sb.slider("Test set size", 0.1, 0.5, 0.3, 0.05)
seed = int(sb.number_input("Random seed", 0, 10_000, 42, 1))
missing_strategy = sb.selectbox("Missing values", list(MISSING_STRATEGIES),
                                format_func=MISSING_STRATEGIES.get, index=0)
binning = sb.selectbox("ID3 discretisation", ["quantile", "uniform"],
                       format_func={"quantile": "Equal-frequency bins", "uniform": "Equal-width bins"}.get)
n_bins = sb.slider("Bins per numeric attribute (ID3)", 2, 10, 4)

sb.markdown("### 4 · Tree settings")
max_depth_in = sb.slider("Maximum depth (0 = unlimited)", 0, 20, 0)
min_split = int(sb.number_input("Min. samples to split a node", 2, 500, 2, 1))
min_leaf = int(sb.number_input("Min. samples per branch", 1, 200, 2, 1,
                                help="C4.5 uses 2 by default (at least two branches must reach this size)."))
prune = sb.toggle("C4.5 error-based pruning", value=True)
cf = sb.slider("C4.5 confidence factor (CF)", 0.05, 0.50, 0.25, 0.05, disabled=not prune,
               help="Smaller values prune more aggressively.")
penalty = sb.toggle("C4.5 MDL penalty for numeric splits", value=True)
average = sb.selectbox("Averaging for precision / recall / F1", list(AVERAGES), index=0)
include_sklearn = sb.toggle("Add scikit-learn CART (entropy) as a reference", value=False)
auto = sb.toggle("Retrain automatically when settings change", value=True)
train_clicked = sb.button("Train ID3 & C4.5", type="primary")

# --------------------------------------------------------------------------- #
# Validation and training
# --------------------------------------------------------------------------- #
errors, warnings = target_issues(df, target)
if not features:
    errors.append("Select at least one attribute to train on.")

prep_cfg = PreprocessConfig(target=target, features=features, test_size=test_size, random_state=seed,
                            missing_strategy=missing_strategy, binning=binning, n_bins=n_bins)
model_cfg = ModelConfig(max_depth=max_depth_in or None, min_samples_split=min_split,
                        min_samples_leaf=min_leaf, prune=prune, confidence_factor=cf,
                        continuous_penalty=penalty, include_sklearn=include_sklearn,
                        average=average, random_state=seed)
signature = hashlib.sha1(raw + repr((asdict(prep_cfg), asdict(model_cfg))).encode()).hexdigest()
state = st.session_state

if not errors and (train_clicked or (auto and state.get("signature") != signature)):
    try:
        with st.spinner("Preprocessing and growing trees…"):
            prepared = prepare(df, prep_cfg)
            results = train_and_evaluate(prepared, model_cfg)
        state["prepared"], state["results"], state["signature"] = prepared, results, signature
        state["data_key"] = hashlib.sha1(raw).hexdigest()
        state.pop("cv", None)
    except ValueError as exc:
        errors.append(str(exc))

stale = state.get("signature") != signature
have_results = "results" in state and state.get("data_key") == hashlib.sha1(raw).hexdigest()
for e in errors:
    st.error(e)
for w in warnings:
    st.warning(w)
if have_results and stale and not errors:
    st.info("Settings changed since the last run — press **Train ID3 & C4.5** in the sidebar to refresh results.")

tabs = st.tabs(["Data", "Preprocessing", "Decision trees", "Predictions", "Evaluation", "Comparison", "About"])

# --------------------------------------------------------------------------- #
# Tab: Data
# --------------------------------------------------------------------------- #
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Classes in target", int(df[target].nunique()))
    st.subheader("Preview")
    st.dataframe(df.head(200), hide_index=True)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Attribute profile")
        st.dataframe(profile(df), hide_index=True)
    with right:
        st.subheader("Target")
        show_fig(plots.plot_class_distribution(df[target].dropna().astype(str), f"Distribution of '{target}'"))

if not have_results:
    for t in tabs[1:6]:
        with t:
            st.info("Fix the issues above (or press **Train ID3 & C4.5**) to see results.")
    with tabs[6]:
        st.write("See the README for algorithm details.")
    footer()
    st.stop()

prepared = state["prepared"]
results = state["results"]
pre = prepared.preprocessor

# --------------------------------------------------------------------------- #
# Tab: Preprocessing
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Steps applied")
    for i, line in enumerate(prepared.log, 1):
        st.markdown(f"{i}. {line}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training rows", len(prepared.y_train))
    c2.metric("Test rows", len(prepared.y_test))
    c3.metric("Classes", len(prepared.classes))

    st.subheader("How each attribute is handled")
    rows = []
    for f in pre.features_:
        numeric = f in pre.numeric_
        miss = int(prepared.X_train_raw[f].isna().sum() + prepared.X_test_raw[f].isna().sum())
        native = prepared.config.missing_strategy == "native"
        fill = pre.medians_.get(f) if numeric else pre.modes_.get(f)
        rows.append({
            "attribute": f, "type": "numeric" if numeric else "categorical", "missing": miss,
            "ID3": (f"discretised into {len(pre.bin_labels(f))} bins" if numeric else "used as-is")
                   + (f"; missing → {fill:.4g}" if numeric and miss else (f"; missing → '{fill}'" if miss else "")),
            "C4.5": ("binary threshold split" if numeric else "multi-way split")
                    + ("; missing handled natively" if native and miss else
                       (f"; missing → {fill:.4g}" if numeric and miss else (f"; missing → '{fill}'" if miss else ""))),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    if pre.numeric_:
        st.subheader("ID3 discretisation (learned on training data)")
        st.dataframe(pre.bins_table(), hide_index=True)

    with st.expander("Training data as seen by ID3"):
        st.dataframe(prepared.X_train_id3.assign(**{target: prepared.y_train}).head(200), hide_index=True)
    with st.expander("Training data as seen by C4.5"):
        st.dataframe(prepared.X_train_c45.assign(**{target: prepared.y_train}).head(200), hide_index=True)
    show_fig(plots.plot_class_distribution(prepared.y_train, "Class balance in the training split"))

# --------------------------------------------------------------------------- #
# Tab: Decision trees
# --------------------------------------------------------------------------- #
with tabs[2]:
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 3])
    show_depth = ctrl1.slider("Display depth (0 = full tree)", 0, 15, 0, key="show_depth")
    orient = ctrl2.radio("Orientation", ["Top → bottom", "Left → right"], horizontal=True, key="orient")
    ctrl3.caption("Inner nodes show the attribute tested, the split score and the class counts; leaves are "
                  "coloured by predicted class, with stronger colour meaning a purer leaf.")
    sub = st.tabs(list(TREE_MODELS))
    for name, t in zip(TREE_MODELS, sub):
        r = results[name]
        model = r.model
        with t:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Nodes", r.n_nodes)
            m2.metric("Leaves", r.n_leaves)
            m3.metric("Depth", r.depth)
            m4.metric("Training time", f"{r.fit_time * 1000:.1f} ms")
            if name == "C4.5":
                m5.metric("Nodes pruned", model.n_pruned_ if prune else "off")
            else:
                m5.metric("Criterion", "Info gain")
            dot = tree_to_dot(model, max_depth=show_depth or None,
                              rankdir="LR" if orient.startswith("Left") else "TB")
            if r.n_nodes > 150 and not show_depth:
                st.caption(f"This tree has {r.n_nodes} nodes — reduce the display depth for a readable view.")
            st.graphviz_chart(dot)
            d1, d2, d3, _ = st.columns([1, 1, 1, 3])
            d1.download_button("DOT source", dot, file_name=f"{name.lower().replace('.', '')}_tree.dot",
                               mime="text/vnd.graphviz", key=f"dot_{name}")
            svg = render_cached(dot, "svg")
            png = render_cached(dot, "png")
            if svg:
                d2.download_button("SVG", svg, file_name=f"{name.lower().replace('.', '')}_tree.svg",
                                   mime="image/svg+xml", key=f"svg_{name}")
            if png:
                d3.download_button("PNG", png, file_name=f"{name.lower().replace('.', '')}_tree.png",
                                   mime="image/png", key=f"png_{name}")
            left, right = st.columns([3, 2])
            with left:
                rules = model.rules()
                st.markdown(f"**Decision rules** ({len(rules)})")
                st.dataframe(rules, hide_index=True)
                txt = "\n".join(f"IF {row.rule} THEN {target} = {row.prediction}  "
                                f"[support {row.support}, confidence {row.confidence:.2f}]"
                                for row in rules.itertuples())
                st.download_button("Download rules (.txt)", txt, file_name=f"{name.lower().replace('.', '')}_rules.txt",
                                   key=f"rules_{name}")
            with right:
                show_fig(plots.plot_feature_importance(model.feature_importances_,
                                                       f"{name} attribute importance", model_color(name)))

# --------------------------------------------------------------------------- #
# Tab: Predictions
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Test-set predictions")
    pred_df = prepared.X_test_raw.copy()
    pred_df.insert(0, f"actual {target}", prepared.y_test.to_numpy())
    for name, r in results.items():
        pred_df.insert(1 + list(results).index(name), f"{name} prediction", r.y_pred)
    view = st.radio("Show", ["All rows", "Rows where ID3 and C4.5 disagree", "Rows misclassified by either model"],
                    horizontal=True, key="pred_filter")
    actual = prepared.y_test.to_numpy(dtype=object)
    if view.startswith("Rows where"):
        mask = results["ID3"].y_pred != results["C4.5"].y_pred
    elif view.startswith("Rows misclassified"):
        mask = (results["ID3"].y_pred != actual) | (results["C4.5"].y_pred != actual)
    else:
        mask = np.ones(len(pred_df), dtype=bool)
    st.caption(f"{int(mask.sum())} of {len(pred_df)} test rows")
    st.dataframe(pred_df[mask], hide_index=True)
    st.download_button("Download predictions (.csv)", pred_df.to_csv(index=False).encode(),
                       file_name="test_predictions.csv", mime="text/csv")

    st.subheader("Predict a new instance")
    st.caption("Enter attribute values; choose “(missing)” to see how each algorithm deals with an unknown value.")
    with st.form("single_prediction"):
        inputs = {}
        grid = st.columns(3)
        for i, f in enumerate(pre.features_):
            col = grid[i % 3]
            if f in pre.numeric_:
                vals = pd.to_numeric(prepared.X_train_raw[f], errors="coerce").dropna()
                lo, hi = float(vals.min()), float(vals.max())
                missing_box = col.checkbox(f"{f}: missing", key=f"miss_{f}")
                v = col.number_input(f, value=float(pre.medians_[f]), key=f"in_{f}",
                                     help=f"training range {lo:.4g} – {hi:.4g}")
                inputs[f] = np.nan if missing_box else v
            else:
                opts = pre.categories_[f] + ["(missing)"]
                default = opts.index(pre.modes_[f]) if pre.modes_[f] in opts else 0
                v = col.selectbox(f, opts, index=default, key=f"in_{f}")
                inputs[f] = np.nan if v == "(missing)" else v
        submitted = st.form_submit_button("Predict with both models", type="primary")
    if submitted:
        row = pd.DataFrame([inputs])
        x_id3, x_c45 = prepared.transform_new(row)
        cols_out = st.columns(len(TREE_MODELS))
        for name, col, x in zip(TREE_MODELS, cols_out, (x_id3, x_c45)):
            model = results[name].model
            steps, proba = model.explain(x.iloc[0])
            best = max(proba, key=proba.get)
            with col:
                st.markdown(f'<div class="card {"gold" if name == "C4.5" else ""}"><h4>{name} → '
                            f'{html.escape(str(best))}</h4><p>confidence {proba[best]:.1%}</p></div>', unsafe_allow_html=True)
                st.markdown("**Decision path**")
                for s_ in steps or ["(the tree is a single leaf)"]:
                    st.markdown(f'<div class="path-step">{html.escape(s_)}</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({"class": list(proba), "probability": list(proba.values())})
                             .sort_values("probability", ascending=False), hide_index=True)
        if any(pd.isna(v) for v in inputs.values()):
            st.caption("ID3 cannot use missing values, so the value was imputed for ID3 (training median / mode). "
                       + ("C4.5 kept it missing and combined branches." if prepared.config.missing_strategy == "native"
                          else "C4.5 also received the imputed value (choose the native strategy to compare)."))

    st.subheader("Batch prediction")
    batch = st.file_uploader("CSV with the same attribute columns (target column optional)", type=["csv"],
                             key="batch_upload")
    if batch is not None:
        try:
            new = load_csv(batch.getvalue(), drop_empty=False)
            x_id3, x_c45 = prepared.transform_new(new)
            out = new.copy()
            out["ID3 prediction"] = results["ID3"].model.predict(x_id3)
            out["C4.5 prediction"] = results["C4.5"].model.predict(x_c45)
            st.dataframe(out.head(500), hide_index=True)
            st.download_button("Download batch predictions (.csv)", out.to_csv(index=False).encode(),
                               file_name="batch_predictions.csv", mime="text/csv")
        except ValueError as exc:
            st.error(f"Could not score this file: {exc}")

# --------------------------------------------------------------------------- #
# Tab: Evaluation
# --------------------------------------------------------------------------- #
with tabs[4]:
    normalize = st.toggle("Show confusion matrices as row percentages", value=False, key="cm_norm")
    st.caption(f"Precision, recall and F1 use **{average}** averaging over {len(prepared.classes)} classes; "
               f"evaluated on {len(prepared.y_test)} held-out test rows.")
    names = list(results)
    cols_eval = st.columns(len(names))
    for name, col in zip(names, cols_eval):
        m = results[name].metrics
        with col:
            st.markdown(f"#### {name}")
            a, b = st.columns(2)
            a.metric("Accuracy", f"{m['accuracy']:.3f}")
            b.metric("Precision", f"{m['precision']:.3f}")
            a.metric("Recall", f"{m['recall']:.3f}")
            b.metric("F1-score", f"{m['f1']:.3f}")
            show_fig(plots.plot_confusion_matrix(m["confusion_matrix"], f"{name} confusion matrix", normalize))
            st.markdown("**Per-class report**")
            st.dataframe(m["per_class"].round(3))

# --------------------------------------------------------------------------- #
# Tab: Comparison
# --------------------------------------------------------------------------- #
with tabs[5]:
    table = comparison_table(results)
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    st.subheader("Side-by-side")
    st.dataframe(table.style.format({**{c: "{:.3f}" for c in metric_cols},
                                     "fit time (ms)": "{:.1f}", "predict time (ms)": "{:.1f}"})
                 .highlight_max(subset=metric_cols, color=ANTIQUE_WHITE)
                 .highlight_min(subset=["nodes", "leaves", "depth"], color="#DDEFE9"))
    show_fig(plots.plot_metric_comparison(table, title="Test-set performance"))

    i3, c4 = table.loc["ID3"], table.loc["C4.5"]
    agree = agreement(results)
    winner = "C4.5" if c4["f1"] > i3["f1"] else ("ID3" if i3["f1"] > c4["f1"] else None)
    diff = abs(c4["f1"] - i3["f1"])
    points = []
    if winner:
        points.append(f"<b>{winner}</b> has the higher F1-score ({max(c4['f1'], i3['f1']):.3f} vs "
                      f"{min(c4['f1'], i3['f1']):.3f}, a difference of {diff:.3f}).")
    else:
        points.append(f"Both trees reach the same F1-score ({c4['f1']:.3f}).")
    points.append(f"Accuracy: ID3 {i3['accuracy']:.3f} · C4.5 {c4['accuracy']:.3f}.")
    smaller = "C4.5" if c4["leaves"] < i3["leaves"] else "ID3"
    points.append(f"Tree size: ID3 has {int(i3['leaves'])} leaves (depth {int(i3['depth'])}), C4.5 has "
                  f"{int(c4['leaves'])} leaves (depth {int(c4['depth'])}) — {smaller} is the simpler model.")
    points.append(f"The two models agree on {agree:.0%} of the test rows.")
    if pre.numeric_:
        points.append(f"{len(pre.numeric_)} numeric attribute(s) were binned for ID3 but split on learned "
                      "thresholds by C4.5, which usually loses less information.")
    if prepared.config.missing_strategy == "native" and prepared.X_train_raw.isna().any().any():
        points.append("Missing values were imputed for ID3 and handled with fractional instances by C4.5.")
    if len(prepared.y_test) < 30:
        points.append("The test set is small, so these differences are noisy — use cross-validation below.")
    st.markdown('<div class="card"><h4>Summary</h4><ul>' + "".join(f"<li>{p}</li>" for p in points)
                + "</ul></div>", unsafe_allow_html=True)

    st.subheader("k-fold cross-validation")
    st.caption("Repeats preprocessing and training on k different train/test partitions for a more reliable comparison.")
    k = st.slider("Folds (k)", 3, 10, 5, key="cv_k")
    if st.button("Run cross-validation", key="run_cv"):
        try:
            with st.spinner(f"Running {k}-fold cross-validation…"):
                cv, desc = cross_validate(df, prep_cfg, model_cfg, k)
            state["cv"] = (cv, desc, k)
        except ValueError as exc:
            st.error(str(exc))
    if "cv" in state:
        cv, desc, kk = state["cv"]
        st.markdown(f'<span class="pill">{desc}</span>', unsafe_allow_html=True)
        st.dataframe(cv_summary(cv))
        cv_metric = st.selectbox("Metric to plot", metric_cols, index=3, key="cv_metric")
        show_fig(plots.plot_cv_results(cv, cv_metric))
        with st.expander("Per-fold results"):
            st.dataframe(cv.round(4), hide_index=True)

# --------------------------------------------------------------------------- #
# Tab: About
# --------------------------------------------------------------------------- #
with tabs[6]:
    a, b = st.columns(2)
    with a:
        st.markdown("#### ID3 (Quinlan, 1986)")
        st.latex(r"H(S) = -\sum_{c} p_c \log_2 p_c")
        st.latex(r"IG(S,A) = H(S) - \sum_{v \in A} \frac{|S_v|}{|S|} H(S_v)")
        st.markdown("""
- Chooses the attribute with the highest **information gain**; one branch per value.
- Works on categorical attributes only — numeric attributes are discretised first.
- Cannot use missing values (they are imputed), no pruning.
""")
    with b:
        st.markdown("#### C4.5 (Quinlan, 1993)")
        st.latex(r"SplitInfo(S,A) = -\sum_{v} \frac{|S_v|}{|S|}\log_2\frac{|S_v|}{|S|} \qquad "
                 r"GR(S,A) = \frac{IG(S,A)}{SplitInfo(S,A)}")
        st.markdown("""
- Chooses the highest **gain ratio** among attributes with at least average gain.
- Numeric attributes: binary test `A ≤ t` at the best midpoint threshold.
- Missing values: gain scaled by the known fraction; cases are split into all branches with fractional weights.
- **Error-based pruning** with confidence factor CF replaces subtrees whose estimated error is not lower than a leaf's.
""")
    st.markdown("#### Evaluation")
    st.markdown("Accuracy, precision, recall and F1 (macro / weighted / micro average), confusion matrices, "
                "and k-fold cross-validation. Metrics are computed with scikit-learn; the trees themselves are "
                "implemented from scratch in the `dtree` package.")
    st.markdown("#### Contributors")
    st.markdown(f'<div class="chips" style="margin-bottom:8px">{contributor_chips()}</div>', unsafe_allow_html=True)
    with st.expander("Current settings"):
        st.json({"preprocessing": asdict(prep_cfg), "model": asdict(model_cfg)})

footer()
