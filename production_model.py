"""Stage 2N — Production model training.

Frozen contract: after historical OOF evaluation, take the median
best_iteration across outer folds and the stability-selected feature set,
refit LightGBM on ALL historical (labeled) data with those fixed
hyperparameters. This model is for FUTURE inference only -- it must NEVER
be used to produce reported historical performance (that is exclusively
the OOF table's job, per Stage 2J's frozen "sole source" rule), because it
has seen every historical labeled row during fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from class_checks import check_class_coverage
from feature_selection import select_stable_features
from model_training import align_probs_to_class_order

CLASS_ORDER = (0, 1, 2)


class ProductionModelError(ValueError):
    pass


@dataclass
class ProductionModelResult:
    model: object  # fitted lgb.LGBMClassifier
    selected_features: List[str]
    n_estimators: int
    lgbm_params: dict
    n_training_rows: int
    class_counts: Dict[int, int]


def select_median_best_iteration(fold_best_iterations: List[int]) -> int:
    """Median across outer folds' best_iteration, rounded to the nearest
    integer (round-half-to-even via numpy.round, then floored at 1 -- a
    LightGBM model needs at least 1 tree)."""
    if not fold_best_iterations:
        raise ProductionModelError("fold_best_iterations must be non-empty")
    med = float(np.median(np.asarray(fold_best_iterations, dtype=float)))
    return max(1, int(round(med)))


def train_production_model(
    dataset: pd.DataFrame,
    candidate_features: List[str],
    fold_selected_features: List[List[str]],
    fold_best_iterations: List[int],
    min_fold_stability: float,
    min_class_count: int,
    lgbm_params: Optional[dict] = None,
    random_state: int = 42,
) -> ProductionModelResult:
    if "target_class" not in dataset.columns:
        raise ProductionModelError("dataset is missing required column: target_class")

    stable_features = select_stable_features(fold_selected_features, candidate_features, min_fold_stability)
    if not stable_features:
        raise ProductionModelError("no features survived stability selection; cannot train a production model")

    n_estimators = select_median_best_iteration(fold_best_iterations)

    labeled = dataset[dataset["target_class"].notna()]
    if labeled.empty:
        raise ProductionModelError("dataset has no labeled rows (target_class all NaN)")
    y = labeled["target_class"].astype(int).to_numpy()
    check_class_coverage(y, min_class_count, context="production model (all historical labeled data)")
    X = labeled[stable_features]

    params = dict(
        objective="multiclass", num_class=3, num_leaves=31, learning_rate=0.05,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, n_jobs=-1, verbosity=-1,
    )
    if lgbm_params:
        params.update({k: v for k, v in lgbm_params.items() if k != "n_estimators"})
    # n_estimators is FIXED to the median across-fold best_iteration -- this
    # is the one hyperparameter this stage is explicitly allowed to set
    # itself (per the frozen contract); any n_estimators in lgbm_params is
    # deliberately overridden, not merged, and never re-tuned here.
    params["n_estimators"] = n_estimators

    import lightgbm as lgb
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    return ProductionModelResult(
        model=model, selected_features=stable_features, n_estimators=n_estimators,
        lgbm_params=params, n_training_rows=len(labeled),
        class_counts={c: int((y == c).sum()) for c in CLASS_ORDER},
    )


def predict_proba_production(result: ProductionModelResult, X: pd.DataFrame) -> np.ndarray:
    """Future-inference prediction: canonical (DOWN, NEUTRAL, UP) column
    order via align_probs_to_class_order, same as every other estimator in
    this codebase -- never assumes model.classes_ positional ordering."""
    raw = result.model.predict_proba(X[result.selected_features])
    return align_probs_to_class_order(raw, result.model.classes_)
