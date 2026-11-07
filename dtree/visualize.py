"""Graphviz rendering of fitted trees (DOT source + optional SVG/PNG)."""
from __future__ import annotations

import html
import shutil
import subprocess

import numpy as np

from .base import BaseDecisionTree, branch_label, fmt_num
from .theme import (ANTIQUE_GOLD, CLASS_COLORS, CREAM, DARK_GREEN, GITAM_GREEN, TEXT,
                    blend_with_white, readable_text)

try:  # the python ``graphviz`` package is optional; the ``dot`` binary is enough
    import graphviz as _graphviz
except ImportError:  # pragma: no cover
    _graphviz = None


def _esc(s) -> str:
    return html.escape(str(s), quote=False)


def _n(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def class_color_map(classes) -> dict:
    return {c: CLASS_COLORS[i % len(CLASS_COLORS)] for i, c in enumerate(classes)}


def tree_to_dot(tree: BaseDecisionTree, max_depth: int | None = None, rankdir: str = "TB",
                show_counts: bool = True) -> str:
    """DOT source for ``tree``.  ``max_depth`` collapses deeper subtrees for display."""
    classes = list(tree.classes_)
    colors = class_color_map(classes)
    k = len(classes)
    crit = tree.criterion_short
    lines = [
        "digraph DecisionTree {",
        f'  graph [rankdir={rankdir}, bgcolor="transparent", nodesep=0.3, ranksep=0.45, pad=0.25, '
        'fontname="Inter"];',
        f'  node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11, color="{GITAM_GREEN}", '
        f'penwidth=1.3, margin="0.14,0.07", fontcolor="{TEXT}"];',
        f'  edge [fontname="Inter", fontsize=10, color="{ANTIQUE_GOLD}", fontcolor="{DARK_GREEN}", '
        'penwidth=1.2, arrowsize=0.7];',
    ]

    def counts_text(node):
        return " / ".join(str(int(round(c))) if abs(c - round(c)) < 1e-9 else f"{c:.1f}"
                          for c in node.class_counts)

    def add(node, depth):
        nid = f"n{node.node_id}"
        dist = node.distribution()
        top = int(np.argmax(dist))
        cls = classes[top]
        collapsed = (not node.is_leaf) and max_depth is not None and depth >= max_depth
        if node.is_leaf or collapsed:
            purity = (dist[top] - 1 / k) / (1 - 1 / k) if k > 1 else 1.0
            fill = blend_with_white(colors[cls], 0.30 + 0.70 * max(purity, 0.0))
            fc = readable_text(fill)
            rows = [f"<B>{_esc(cls)}</B>", f"n = {_n(node.n_samples)} · {dist[top]:.0%}"]
            if show_counts and k <= 6:
                rows.append(f'<FONT POINT-SIZE="9">[{_esc(counts_text(node))}]</FONT>')
            if collapsed:
                rows.append(f'<FONT POINT-SIZE="9"><I>+ {sum(1 for _ in _walk(node)) - 1} hidden nodes</I></FONT>')
            style = ', style="rounded,filled,dashed"' if collapsed else ""
            label = "<BR/>".join(rows)
            lines.append(f'  {nid} [label=<{label}>, fillcolor="{fill}", fontcolor="{fc}", '
                         f'color="{colors[cls]}"{style}];')
            return
        rows = [f'<B>{_esc(node.feature)}</B>',
                f'<FONT POINT-SIZE="9">{crit} = {node.score:.3f}'
                + (f" · IG = {node.info_gain:.3f}" if crit != "IG" else "") + "</FONT>",
                f"n = {_n(node.n_samples)}"]
        if show_counts and k <= 6:
            rows.append(f'<FONT POINT-SIZE="9">[{_esc(counts_text(node))}]</FONT>')
        label = "<BR/>".join(rows)
        lines.append(f'  {nid} [label=<{label}>, fillcolor="{CREAM}"];')
        for key, child in node.children.items():
            lines.append(f'  {nid} -> n{child.node_id} [label=<{_esc(branch_label(node, key))}>];')
            add(child, depth + 1)

    add(tree.root_, 0)
    # legend
    legend = "".join(
        f'<TR><TD BGCOLOR="{colors[c]}" WIDTH="12"> </TD><TD ALIGN="LEFT">{_esc(c)}</TD></TR>'
        for c in classes[:12])
    lines.append(f'  legend [shape=plaintext, style="", label=<<TABLE BORDER="0" CELLBORDER="0" '
                 f'CELLSPACING="3"><TR><TD COLSPAN="2" ALIGN="LEFT"><B>Classes</B></TD></TR>{legend}</TABLE>>];')
    lines.append("}")
    return "\n".join(lines)


def _walk(node):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(cur.children.values())


def render(dot: str, fmt: str = "svg") -> bytes | None:
    """Render DOT to ``fmt`` bytes, or ``None`` when Graphviz is not installed."""
    if _graphviz is not None:
        try:
            return _graphviz.Source(dot).pipe(format=fmt)
        except Exception:  # fall back to the binary below
            pass
    exe = shutil.which("dot")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, f"-T{fmt}"], input=dot.encode("utf-8"), capture_output=True,
                             timeout=60, check=True)
        return out.stdout
    except (subprocess.SubprocessError, OSError):
        return None
