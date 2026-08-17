"""Stage 2H — Fold-local feature selection: correlation pruning + LightGBM
importance.

Frozen contract (Stage 1B/1C, restated in the Stage 2H task):
    Both correlation pruning and importance selection must be fold-local.
    Outer test cannot influence them.
    Feature-importance model: train on internal_train; internal_validation
    only for early stopping. Freeze selected features.

FOLD-LOCALITY IS ENFORCED STRUCTURALLY, not just by convention: every
function in this module takes only the exact rows the caller has already
sliced out (internal_train / internal_validation) as plain DataFrame/array
arguments. Nothing in this module ever receives, stores, or reaches for a
reference to the full dataset or outer_test — there is no global state and
no "pass the whole df and an index mask" pattern that could accidentally
leak a wider view in. tests/test_feature_selection.py includes a direct
locality probe: perturbing rows OUTSIDE what was passed in must not change
the result, by construction (the function literally never sees them).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from class_checks import check_class_coverage
from model_training import align_probs_to_class_order


class FeatureSelectionError(ValueError):
    pass


def prune_correlated_features(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95,
) -> List[str]:
    """Drop features whose absolute pairwise Pearson correlation (computed
    ONLY on the rows in ``train_df``) exceeds ``threshold``.

    Upper-triangle rule: for each pair (i, j) with i < j (column order in
    ``feature_cols``), if |corr(i, j)| > threshold, column j is marked for
    removal (column i, appearing first, is kept). This mirrors the
    convention already used elsewhere in the codebase so pruning behavior
    is predictable and order-dependent in a documented way, not arbitrary.
    """
    if not (0 < threshold <= 1):
        raise FeatureSelectionError(f"threshold must be in (0, 1], got {threshold}")
    if not feature_cols:
        return []

    X = (
        train_df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = {col for col in upper.columns if (upper[col] > threshold).any()}
    return [c for c in feature_cols if c not in to_drop]


@dataclass(frozen=True)
class ImportanceSelectionResult:
    selected_features: List[str]
    best_iteration: int
    importances: Dict[str, float]
    # (n_val, 3) probabilities, canonical (DOWN, NEUTRAL, UP) column order,
    # from THIS ALREADY-FITTED importance model predicting X_val. Exposed
    # as a free byproduct (no extra training cost) for Stage 2K's "fold 1:
    # use fold 1 internal validation predictions" threshold-selection rule.
    # Fit on the full correlation-pruned candidate set (not narrowed to
    # selected_features) since that is the actual model that was fit here —
    # documented as a deliberate approximation of "the final model's
    # predictions on internal_validation" rather than a separate refit.
    internal_val_probs: np.ndarray


def select_features_by_lgbm_importance(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    candidate_features: List[str],
    top_k: int,
    min_class_count: int,
    random_state: int = 42,
    early_stopping_rounds: int = 50,
    lgbm_params: Optional[dict] = None,
) -> ImportanceSelectionResult:
    """Train a LightGBM multiclass model on ``(X_train, y_train)`` using
    ``(X_val, y_val)`` ONLY for early stopping (never for gradient
    updates), then rank ``candidate_features`` by the fitted model's
    feature_importances_ and keep the top ``top_k``.

    MISSING-CLASS POLICY: both the training and validation splits are
    checked via class_checks.check_class_coverage before fitting anything;
    insufficient class representation raises MissingClassError (fails
    loudly) rather than silently training a degenerate model.
    """
    if top_k < 1:
        raise FeatureSelectionError(f"top_k must be >= 1, got {top_k}")
    if not candidate_features:
        raise FeatureSelectionError("candidate_features must be non-empty")

    check_class_coverage(y_train, min_class_count, context="internal_train (feature-importance model)")
    check_class_coverage(y_val, min_class_count, context="internal_validation (feature-importance model)")

    import lightgbm as lgb

    params = dict(
        objective="multiclass",
        num_class=3,
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=500,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    if lgbm_params:
        params.update(lgbm_params)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train[candidate_features], y_train,
        eval_set=[(X_val[candidate_features], y_val)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(-1)],
    )
    best_iteration = int(model.best_iteration_) if model.best_iteration_ else int(params["n_estimators"])

    importances = dict(zip(candidate_features, (float(v) for v in model.feature_importances_)))
    ranked = sorted(candidate_features, key=lambda f: (-importances[f], f))
    selected = ranked[: min(top_k, len(ranked))]

    raw_val_proba = model.predict_proba(X_val[candidate_features])
    internal_val_probs = align_probs_to_class_order(raw_val_proba, model.classes_)

    return ImportanceSelectionResult(
        selected_features=selected, best_iteration=best_iteration, importances=importances,
        internal_val_probs=internal_val_probs,
    )


def select_stable_features(
    fold_selected_features: List[List[str]],
    candidate_features: List[str],
    min_fold_stability: float,
) -> List[str]:
    """Final production feature set: keep a candidate feature iff it was
    selected in >= ``min_fold_stability`` fraction of folds (Stage 2H's
    "after all folds" step). ``>=`` is inclusive — a feature selected in
    exactly the threshold fraction of folds is kept, not dropped.
    """
    if not (0 < min_fold_stability <= 1):
        raise FeatureSelectionError(f"min_fold_stability must be in (0, 1], got {min_fold_stability}")
    n = len(fold_selected_features)
    if n == 0:
        return list(candidate_features)

    appearance: Counter = Counter()
    for feats in fold_selected_features:
        for f in feats:
            appearance[f] += 1

    return [f for f in candidate_features if appearance.get(f, 0) / n >= min_fold_stability]
