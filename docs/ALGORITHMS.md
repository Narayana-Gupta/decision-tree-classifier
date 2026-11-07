# Algorithms and design

*Decision Tree Classification System — Karthik Naidu Allu (2023002399), Boddu Narayana Gupta (2023002459)*

## 1. Common tree-growing procedure

Both algorithms use the same top-down, recursive procedure (`dtree/base.py`, `BaseDecisionTree._build`):

1. Compute the (weighted) class counts of the cases reaching the node.
2. Make the node a **leaf** if it is pure, no attributes remain, the maximum depth is reached, or it has
   fewer than `min_samples_split` cases.
3. Otherwise ask the algorithm for the best split (`_find_best_split`). If no split improves the
   criterion, the node becomes a leaf.
4. Partition the cases and recurse into each branch. A categorical attribute is removed from the
   candidates below the node; a numeric attribute stays available (it can be split again at a different
   threshold).

Each case carries a **weight** (1 by default). C4.5 uses fractional weights for missing values (§3.3).
Leaves store their class distribution, so the model also produces class probabilities.

Entropy of a set *S* with class proportions *p<sub>c</sub>*:

> H(S) = − Σ<sub>c</sub> p<sub>c</sub> log<sub>2</sub> p<sub>c</sub>

## 2. ID3 (`dtree/id3.py`)

Information gain of attribute *A* with values *v*:

> IG(S, A) = H(S) − Σ<sub>v</sub> |S<sub>v</sub>| / |S| · H(S<sub>v</sub>)

* The attribute with the highest gain is chosen; ties go to the attribute that appears first in the data.
* Splits are multi-way (one branch per value seen at the node).
* ID3 only handles categorical attributes, so numeric attributes are **discretised** during preprocessing
  into equal-frequency (quantile) or equal-width bins whose cut points are learned on the training set.
* ID3 cannot use missing values; they are imputed with the training median / mode.
* No pruning.
* Prediction for a value that never reached a node during training (e.g. a new category) returns that
  node's class distribution.

Check: on Mitchell's Play-Tennis data the root is *Outlook* with IG = 0.2467 and the tree has 5 leaves,
exactly as in the textbook (`tests/test_algorithms.py`).

## 3. C4.5 (`dtree/c45.py`)

### 3.1 Gain ratio

> SplitInfo(S, A) = − Σ<sub>v</sub> |S<sub>v</sub>| / |S| · log<sub>2</sub>(|S<sub>v</sub>| / |S|)  
> GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)

Information gain favours attributes with many values (an ID column gets maximal gain). Dividing by the split
information penalises such attributes. Following Quinlan's program, only attributes whose gain is **at least
the average gain** of all candidates are eligible, so a tiny split information cannot inflate the ratio of
an almost useless attribute. A split must also send at least `min_samples_leaf` cases (default 2, like C4.5's
`-m 2`) to at least two branches.

### 3.2 Continuous attributes

Cases with a known value are sorted; every boundary between two distinct adjacent values is a candidate
threshold *t* (the midpoint). The gain of the binary test `A ≤ t` is computed for all boundaries at once with
cumulative class counts (O(n log n) per attribute and node), and the threshold with the best gain is kept.
With the optional **MDL penalty** (C4.5 Release 8), log<sub>2</sub>(N<sub>distinct</sub> − 1) / |S<sub>known</sub>| is subtracted
from the gain, which discourages choosing a continuous attribute just because it has many cut points.

On Quinlan's weather data the tree splits *sunny* days on *Humidity ≤ 77.5*.

### 3.3 Missing values

Let *F* be the fraction (by weight) of cases at the node whose value of *A* is known.

* **Gain** is computed on the known cases and multiplied by *F*.
* **Split information** treats "missing" as one extra outcome.
* **Training**: a case with a missing value goes down *every* branch with weight
  *w · |S<sub>v</sub>| / |S<sub>known</sub>|*.
* **Prediction**: when the tested value is missing, the class distributions returned by all branches are
  averaged with the same branch proportions.

These rules are active with the *C4.5 native handling* strategy; ID3 always receives imputed data.

### 3.4 Error-based pruning

After growth, the tree is pruned bottom-up (subtree replacement). For a node covering *N* cases of which *E*
are misclassified by its majority class, the pessimistic error estimate is *N · U<sub>CF</sub>(E, N)*, the upper
limit of a binomial confidence interval at confidence factor *CF* (default 0.25; normal approximation with
continuity correction, and the exact form 1 − CF<sup>1/N</sup> when *E = 0*). A subtree is replaced by a leaf when

> estimated errors(leaf) ≤ Σ estimated errors(leaves of the subtree) + 0.1

Smaller *CF* prunes more. The number of removed nodes is shown in the app.

## 4. Preprocessing (`dtree/preprocessing.py`)

| Step | Detail |
|---|---|
| Parsing | Separator sniffing; `?`, `NA`, `N/A`, `null`, `None`, `-` and blanks are missing; whitespace trimmed; columns whose non-missing values are all numeric become numeric. |
| Profiling | Type, missing count, cardinality; flags ID-like, constant and high-cardinality columns (ID-like and constant columns are excluded by default). |
| Target | Converted to clean strings (`1.0` → `"1"`); rows with a missing target are dropped. |
| Split | `train_test_split`, stratified whenever every class has ≥ 2 rows and both sides can hold every class. |
| Imputation | Median (numeric) / mode (categorical) from the training split. |
| Discretisation (ID3) | Equal-frequency or equal-width bins from the training split; labels such as `< 5.1`, `[5.1, 5.8)`, `≥ 6.4`. Values outside the training range fall into the outer bins. |

Fitting every statistic on the training split only prevents information from the test split leaking into
the models. Cross-validation repeats the whole preprocessing inside each fold for the same reason.

## 5. Evaluation (`dtree/metrics.py`, `dtree/pipeline.py`)

* Confusion matrix *C* with *C<sub>ij</sub>* = test cases of actual class *i* predicted as *j*.
* Accuracy = Σ<sub>i</sub> C<sub>ii</sub> / Σ<sub>ij</sub> C<sub>ij</sub>.
* Per class: precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2PR / (P + R).
* Averaging: **macro** (unweighted mean over classes), **weighted** (by support), **micro** (global counts).
* Comparison also reports nodes, leaves, depth, training and prediction time, the share of test cases on
  which ID3 and C4.5 agree, and k-fold cross-validated mean ± standard deviation of every metric.
* The optional reference model is scikit-learn's `DecisionTreeClassifier(criterion="entropy")` on one-hot
  encoded, imputed data — a binary-split CART tree, useful as a sanity check.

## 6. Visualisation (`dtree/visualize.py`, `dtree/plots.py`)

Trees are exported to Graphviz DOT with HTML-like labels. Inner nodes show the attribute, the split score
(IG for ID3; GR and IG for C4.5), the number of cases and the class counts; leaves are filled with the class
colour, with opacity rising with purity. Deep trees can be collapsed at a display depth. Rendering to
SVG/PNG uses the `graphviz` Python package or, if absent, the `dot` executable.

Colours come from the GITAM palette (`dtree/theme.py`): GITAM Green and Antique Gold for ID3 and C4.5,
Wide Blue for the reference model, and a white → Antique White → Light Green → GITAM Green → Dark Green
scale for confusion matrices, keeping the tertiary colours to accents as the guidelines require.

## 7. Limitations

* Pure-Python recursion: comfortable up to tens of thousands of rows; very large files will be slow.
* C4.5 is implemented without subtree raising and without grouping of categorical values (C4.5's `-s`).
* ID3's results depend on the number of bins chosen for numeric attributes.
* Very small test sets (e.g. the 14-row samples) give noisy metrics; use cross-validation to compare.
