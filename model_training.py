"""Stage 2I — Baselines + multinomial Logistic Regression + LightGBM
multiclass, all producing a canonical (n_rows, 3) probability matrix in
FROZEN column order (DOWN=0, NEUTRAL=1, UP=2), regardless of what order the
underlying estimator's own ``classes_`` happens to report.

Every fit function here operates on plain arrays/DataFrames the caller has
already sliced (outer_train / outer_test, or internal_train / internal_val
for the models that need it) — no full-dataset access, matching the
fold-locality discipline established in walk_forward.py/feature_selection.py.

MISSING-CLASS POLICY (class_checks.check_class_coverage) is enforced before
fitting Baseline C (Logistic Regression) and the LightGBM model — both are
real model fits. The two deterministic rule-based baselines (A, B) do not
"fit" in the statistical sense (Baseline A's majority vote is a trivial
count, not a model needing balanced classes) and are exempt per the frozen
spec's "MISSING-CLASS POLICY ... For model-fitting splits" framing, but
Baseline A's majority-vote source (y_train) is still validated to be
non-empty.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from class_checks import check_class_coverage

CLASS_ORDER = (0, 1, 2)  # DOWN, NEUTRAL, UP — frozen mapping (target_engineering.py)


class ModelTrainingError(ValueError):
    pass


def align_probs_to_class_order(
    proba: np.ndarray, classes_: np.ndarray, class_order: tuple = CLASS_ORDER
) -> np.ndarray:
    """Reindex a fitted estimator's predict_proba() output columns from
    whatever order its own ``classes_`` attribute reports into the frozen
    canonical column order (DOWN, NEUTRAL, UP).

    THIS IS NOT OPTIONAL: sklearn/LightGBM estimators order predict_proba
    columns by ``np.unique(y_train)`` — i.e. by whatever classes happened
    to be present (and in what order they first sorted) in THAT PARTICULAR
    fold's training data, not by any fixed convention. A fold whose
    training data is missing one class, or a caller who assumes
    "column 0 = class 0" without checking, silently produces a
    mislabeled probability matrix. This function is the single place that
    conversion happens, so every caller in this module (and the OOF builder
    in Stage 2J) goes through it rather than re-deriving the same fragile
    assumption independently.
    """
    classes_ = np.asarray(classes_)
    n_rows = proba.shape[0]
    aligned = np.zeros((n_rows, len(class_order)), dtype=float)
    for out_col, cls in enumerate(class_order):
        matches = np.nonzero(classes_ == cls)[0]
        if len(matches) == 1:
            aligned[:, out_col] = proba[:, matches[0]]
        elif len(matches) > 1:
            raise ModelTrainingError(f"classes_ contains duplicate entries for class {cls}: {classes_}")
        # else: class not present in this fold's classes_ -> column stays 0.0
        # (can only happen if a caller bypassed check_class_coverage).
    return aligned


# ---------------------------------------------------------------------------
# Baseline A — majority-class
# ---------------------------------------------------------------------------

def majority_class_baseline_probs(y_train: np.ndarray, n_test_rows: int) -> np.ndarray:
    """Deterministic one-hot baseline: every test row gets prob=1.0 on
    whichever class was most frequent in y_train (ties broken by lowest
    class value, i.e. DOWN < NEUTRAL < UP, for reproducibility)."""
    y_train = np.asarray(y_train)
    if len(y_train) == 0:
        raise ModelTrainingError("majority_class_baseline_probs: y_train is empty")
    counts = {c: int(np.sum(y_train == c)) for c in CLASS_ORDER}
    majority_class = max(CLASS_ORDER, key=lambda c: (counts[c], -c))
    proba = np.zeros((n_test_rows, 3), dtype=float)
    proba[:, CLASS_ORDER.index(majority_class)] = 1.0
    return proba


# ---------------------------------------------------------------------------
# Baseline B — causal persistence
# ---------------------------------------------------------------------------

def persistence_baseline_probs(close_with_lookback: pd.Series, test_start_pos: int, n_test_rows: int) -> np.ndarray:
    """Deterministic one-hot baseline: predicts a continuation of the most
    recently realized bar-over-bar price direction as of each test row t —
    i.e. sign(close[t] - close[t-1]) mapped directly to DOWN/NEUTRAL/UP,
    with 0.0 exactly (no change) mapped to NEUTRAL. This is a documented
    design choice (the frozen spec names the baseline "causal persistence"
    without pinning an exact formula) rather than a persistence of the
    lagging *target label* itself, since the target label for row t-1 is
    not actually knowable at row t (it depends on close[t-1+horizon_bars],
    which may still be in the future relative to t) — using it would
    itself be a subtle look-ahead bug. Bar-over-bar price direction is
    unambiguously known at row t.

    ``close_with_lookback`` must be indexed 0..N-1 by row position and
    include at least one row before ``test_start_pos`` (so row
    test_start_pos's own predecessor is available); ``n_test_rows`` test
    rows starting at ``test_start_pos`` are scored.
    """
    close = np.asarray(close_with_lookback, dtype=float)
    if test_start_pos < 1:
        raise ModelTrainingError("persistence_baseline_probs: test_start_pos must be >= 1 (needs a t-1 row)")
    proba = np.zeros((n_test_rows, 3), dtype=float)
    for i in range(n_test_rows):
        t = test_start_pos + i
        delta = close[t] - close[t - 1]
        if delta > 0:
            cls = 2  # UP
        elif delta < 0:
            cls = 0  # DOWN
        else:
            cls = 1  # NEUTRAL
        proba[i, CLASS_ORDER.index(cls)] = 1.0
    return proba


# ---------------------------------------------------------------------------
# Baseline C — multinomial Logistic Regression
# ---------------------------------------------------------------------------

def fit_logistic_regression_probs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    min_class_count: int,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
):
    """Fit a scaler + multinomial LogisticRegression on (X_train, y_train)
    ONLY (scaler.fit is never called on X_test), transform+predict on
    X_test, and return probabilities in the frozen (DOWN, NEUTRAL, UP)
    column order via align_probs_to_class_order."""
    check_class_coverage(y_train, min_class_count, context="outer_train (Baseline C: Logistic Regression)")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_train_clean = X_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test_clean = X_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)  # fit ONLY on train
    X_test_scaled = scaler.transform(X_test_clean)         # test uses train's fitted scaler

    # multi_class="multinomial" is intentionally NOT passed: recent
    # scikit-learn versions removed the kwarg (it raises TypeError) because
    # the default solver ("lbfgs") already fits a genuine multinomial model
    # for a >2-class problem — there is no other behavior to select.
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    raw_proba = model.predict_proba(X_test_scaled)
    return align_probs_to_class_order(raw_proba, model.classes_)


# ---------------------------------------------------------------------------
# Primary model — LightGBM multiclass
# ---------------------------------------------------------------------------

def fit_lightgbm_multiclass_probs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    min_class_count: int,
    n_estimators: int,
    lgbm_params: Optional[dict] = None,
    random_state: int = 42,
):
    """Fit the FINAL fold model: LightGBM multiclass on ALL of outer_train
    (no internal validation split here — that already happened in Stage 2H
    to pick n_estimators=best_iteration), predict outer_test, return
    probabilities in frozen (DOWN, NEUTRAL, UP) column order."""
    check_class_coverage(y_train, min_class_count, context="outer_train (LightGBM primary model)")
    if n_estimators < 1:
        raise ModelTrainingError(f"n_estimators must be >= 1, got {n_estimators}")

    import lightgbm as lgb

    params = dict(
        objective="multiclass", num_class=3,
        num_leaves=31, learning_rate=0.05, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, n_jobs=-1, verbosity=-1,
    )
    if lgbm_params:
        params.update(lgbm_params)
    params["n_estimators"] = n_estimators  # fixed, from Stage 2H's best_iteration — not re-tuned here

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    raw_proba = model.predict_proba(X_test)
    return align_probs_to_class_order(raw_proba, model.classes_)
