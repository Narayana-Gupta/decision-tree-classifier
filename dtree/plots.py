"""Matplotlib charts styled with the GITAM palette."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .theme import (ANTIQUE_WHITE, CLASS_COLORS, DARK_GREEN, GITAM_GREEN,  # noqa: E402
                    LIGHT_GREEN, TEXT, model_color)

CMAP = LinearSegmentedColormap.from_list(
    "gitam", ["#FFFFFF", ANTIQUE_WHITE, LIGHT_GREEN, GITAM_GREEN, DARK_GREEN])

_installed = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Inter", "Forma DJR Display", "DejaVu Sans") if f in _installed), "sans-serif")

RC = {
    "font.family": FONT, "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT,
    "axes.edgecolor": "#B8C9C5", "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.titleweight": "semibold", "axes.titlesize": 12, "axes.titlecolor": DARK_GREEN,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "grid.color": "#E7EEEC", "grid.linewidth": 0.8, "figure.dpi": 110,
}


def _fig(w, h):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def _finish(fig):
    fig.tight_layout()
    return fig


def plot_class_distribution(y: pd.Series, title: str = "Class distribution"):
    counts = pd.Series(y).value_counts().sort_values()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.4, max(1.8, 0.42 * len(counts) + 0.9)))
        colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(counts))][::-1]
        bars = ax.barh([str(c) for c in counts.index], counts.values, color=colors, height=0.6)
        total = counts.sum()
        for b, v in zip(bars, counts.values):
            ax.text(b.get_width() + total * 0.01, b.get_y() + b.get_height() / 2,
                    f"{v}  ({v / total:.0%})", va="center", fontsize=9)
        ax.set_xlim(0, counts.max() * 1.25)
        ax.set_xlabel("rows")
        ax.set_title(title, loc="left")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    return _finish(fig)


def plot_confusion_matrix(cm: pd.DataFrame, title: str = "Confusion matrix", normalize: bool = False):
    values = cm.to_numpy(dtype=float)
    shown = values / np.where(values.sum(1, keepdims=True) == 0, 1, values.sum(1, keepdims=True)) \
        if normalize else values
    n = len(cm)
    size = min(2.6 + 0.65 * n, 9)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(size + 0.8, size))
        im = ax.imshow(shown, cmap=CMAP, vmin=0, vmax=1 if normalize else max(shown.max(), 1))
        ax.set_xticks(range(n), [str(c) for c in cm.columns], rotation=35 if n > 3 else 0,
                      ha="right" if n > 3 else "center")
        ax.set_yticks(range(n), [str(c) for c in cm.index])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title(title, loc="left")
        ax.spines[:].set_visible(False)
        hi = shown.max() if shown.size else 1
        for i in range(n):
            for j in range(n):
                txt = f"{shown[i, j]:.0%}\n({int(values[i, j])})" if normalize else f"{int(values[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10 if n <= 6 else 8,
                        color="white" if shown[i, j] > 0.55 * hi else TEXT,
                        fontweight="semibold" if i == j else "normal")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.outline.set_visible(False)
    return _finish(fig)


def plot_metric_comparison(table: pd.DataFrame, metrics=("accuracy", "precision", "recall", "f1"),
                           title: str = "Model comparison"):
    models = list(table.index)
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        for i, m in enumerate(models):
            vals = [float(table.loc[m, k]) for k in metrics]
            bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width * 0.92,
                          label=m, color=model_color(m))
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center",
                        va="bottom", fontsize=8)
        ax.set_xticks(x, [k.capitalize() if k != "f1" else "F1-score" for k in metrics])
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("score")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=len(models), loc="upper left", bbox_to_anchor=(0, 1.13))
        ax.set_title(title, loc="right")
    return _finish(fig)


def plot_feature_importance(importance: pd.Series, title: str, color: str = GITAM_GREEN, top: int = 15):
    s = importance[importance > 0].head(top).sort_values()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.4, max(1.8, 0.36 * len(s) + 0.9)))
        if s.empty:
            ax.text(0.5, 0.5, "The tree has no splits (single leaf).", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
        else:
            ax.barh([str(i) for i in s.index], s.values, color=color, height=0.6)
            for i, v in enumerate(s.values):
                ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)
            ax.set_xlim(0, min(1.0, s.max() * 1.25 + 0.02))
            ax.set_xlabel("share of total information gain")
            ax.grid(axis="x")
            ax.set_axisbelow(True)
        ax.set_title(title, loc="left")
    return _finish(fig)


def plot_cv_results(cv: pd.DataFrame, metric: str = "f1", title: str | None = None):
    models = list(dict.fromkeys(cv["model"]))
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.8, 3.4))
        for i, m in enumerate(models):
            vals = cv.loc[cv["model"] == m, metric].to_numpy(dtype=float)
            ax.bar(i, vals.mean(), 0.55, color=model_color(m), alpha=0.9, label=m)
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1) if len(vals) > 1 else 0,
                        color=DARK_GREEN, capsize=6, lw=1.4)
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(i + np.asarray(jitter), vals, s=18, color="white", edgecolor=DARK_GREEN,
                       zorder=3, linewidth=1)
            ax.text(i, min(vals.max() + 0.04, 1.06), f"{vals.mean():.3f} ± "
                    f"{(vals.std(ddof=1) if len(vals) > 1 else 0):.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(models)), models)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(metric if metric != "f1" else "F1-score")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_title(title or f"Cross-validated {metric if metric != 'f1' else 'F1-score'} (mean ± sd)",
                     loc="left")
    return _finish(fig)
