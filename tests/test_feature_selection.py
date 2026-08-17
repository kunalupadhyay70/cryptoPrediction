"""Tests for feature_selection.py (Stage 2H)."""
import numpy as np
import pandas as pd
import pytest

from class_checks import MissingClassError
from feature_selection import (
    FeatureSelectionError, ImportanceSelectionResult,
    prune_correlated_features, select_features_by_lgbm_importance, select_stable_features,
)


# ---------------------------------------------------------------------------
# prune_correlated_features
# ---------------------------------------------------------------------------

def test_perfectly_correlated_duplicate_is_dropped():
    n = 50
    rng = np.random.default_rng(1)
    base = rng.normal(size=n)
    df = pd.DataFrame({"a": base, "b": base.copy(), "c": rng.normal(size=n)})
    kept = prune_correlated_features(df, ["a", "b", "c"], threshold=0.95)
    # "a" appears first -> kept; "b" is the exact duplicate -> dropped.
    assert kept == ["a", "c"]


def test_uncorrelated_features_all_kept():
    n = 200
    rng = np.random.default_rng(2)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    kept = prune_correlated_features(df, list(df.columns), threshold=0.3)
    assert kept == list(df.columns)


def test_correlation_boundary_is_strict_not_inclusive():
    n = 30
    rng = np.random.default_rng(3)
    base = rng.normal(size=n)
    df = pd.DataFrame({"a": base, "b": base.copy()})
    # corr(a, b) is EXACTLY 1.0 (identical columns) — with threshold=1.0,
    # "exceeds threshold" (> 1.0) is impossible, so both must be kept.
    kept = prune_correlated_features(df, ["a", "b"], threshold=1.0)
    assert kept == ["a", "b"]
    # With threshold=0.99, 1.0 > 0.99 -> "b" dropped.
    kept2 = prune_correlated_features(df, ["a", "b"], threshold=0.99)
    assert kept2 == ["a"]


def test_prune_correlated_features_is_fold_local_pure_function():
    # Calling twice with fresh identical copies of the SAME train slice
    # gives identical results — proves no hidden state/dependence beyond
    # the exact rows passed in.
    n = 60
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    r1 = prune_correlated_features(df.copy(), ["a", "b"], threshold=0.95)
    r2 = prune_correlated_features(df.copy(), ["a", "b"], threshold=0.95)
    assert r1 == r2


def test_empty_feature_list_returns_empty():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    assert prune_correlated_features(df, [], threshold=0.95) == []


def test_invalid_threshold_raises():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(FeatureSelectionError):
        prune_correlated_features(df, ["a"], threshold=0.0)
    with pytest.raises(FeatureSelectionError):
        prune_correlated_features(df, ["a"], threshold=1.5)


# ---------------------------------------------------------------------------
# select_features_by_lgbm_importance
# ---------------------------------------------------------------------------

def _synthetic_classification_data(n=400, seed=7):
    rng = np.random.default_rng(seed)
    # One strongly informative feature (directly encodes the class up to
    # noise), several pure-noise features.
    y = rng.integers(0, 3, size=n)
    informative = y + rng.normal(0, 0.15, size=n)
    noise1 = rng.normal(size=n)
    noise2 = rng.normal(size=n)
    noise3 = rng.normal(size=n)
    X = pd.DataFrame({
        "informative": informative,
        "noise1": noise1,
        "noise2": noise2,
        "noise3": noise3,
    })
    split = int(n * 0.8)
    return X.iloc[:split], y[:split], X.iloc[split:], y[split:]


def test_lgbm_importance_selects_the_informative_feature():
    X_train, y_train, X_val, y_val = _synthetic_classification_data()
    result = select_features_by_lgbm_importance(
        X_train, y_train, X_val, y_val,
        candidate_features=list(X_train.columns),
        top_k=1,
        min_class_count=10,
        early_stopping_rounds=10,
        lgbm_params={"n_estimators": 100, "num_leaves": 7},
    )
    assert result.selected_features == ["informative"]
    assert result.importances["informative"] > max(
        result.importances["noise1"], result.importances["noise2"], result.importances["noise3"]
    )


def test_lgbm_importance_top_k_is_respected():
    X_train, y_train, X_val, y_val = _synthetic_classification_data()
    result = select_features_by_lgbm_importance(
        X_train, y_train, X_val, y_val,
        candidate_features=list(X_train.columns),
        top_k=2,
        min_class_count=10,
        early_stopping_rounds=10,
        lgbm_params={"n_estimators": 100, "num_leaves": 7},
    )
    assert len(result.selected_features) == 2
    assert set(result.importances.keys()) == set(X_train.columns)


def test_lgbm_importance_best_iteration_within_configured_bound():
    X_train, y_train, X_val, y_val = _synthetic_classification_data()
    n_estimators = 100
    result = select_features_by_lgbm_importance(
        X_train, y_train, X_val, y_val,
        candidate_features=list(X_train.columns),
        top_k=4,
        min_class_count=10,
        early_stopping_rounds=10,
        lgbm_params={"n_estimators": n_estimators, "num_leaves": 7},
    )
    assert 1 <= result.best_iteration <= n_estimators


def test_lgbm_importance_deterministic_given_fixed_random_state():
    X_train, y_train, X_val, y_val = _synthetic_classification_data()
    kwargs = dict(
        candidate_features=list(X_train.columns), top_k=2, min_class_count=10,
        early_stopping_rounds=10, lgbm_params={"n_estimators": 60, "num_leaves": 7},
        random_state=123,
    )
    r1 = select_features_by_lgbm_importance(X_train, y_train, X_val, y_val, **kwargs)
    r2 = select_features_by_lgbm_importance(X_train, y_train, X_val, y_val, **kwargs)
    assert r1.selected_features == r2.selected_features
    assert r1.best_iteration == r2.best_iteration


def test_lgbm_importance_raises_on_missing_class_in_train():
    n = 100
    rng = np.random.default_rng(9)
    X_train = pd.DataFrame({"a": rng.normal(size=n)})
    y_train = np.zeros(n, dtype=int)  # only class 0 present
    X_val = pd.DataFrame({"a": rng.normal(size=20)})
    y_val = rng.integers(0, 3, size=20)
    with pytest.raises(MissingClassError):
        select_features_by_lgbm_importance(
            X_train, y_train, X_val, y_val,
            candidate_features=["a"], top_k=1, min_class_count=5,
        )


def test_lgbm_importance_raises_on_missing_class_in_validation():
    n = 100
    rng = np.random.default_rng(10)
    X_train = pd.DataFrame({"a": rng.normal(size=n)})
    y_train = rng.integers(0, 3, size=n)
    X_val = pd.DataFrame({"a": rng.normal(size=20)})
    y_val = np.ones(20, dtype=int)  # only class 1 present
    with pytest.raises(MissingClassError):
        select_features_by_lgbm_importance(
            X_train, y_train, X_val, y_val,
            candidate_features=["a"], top_k=1, min_class_count=5,
        )


def test_lgbm_importance_empty_candidates_raises():
    X_train, y_train, X_val, y_val = _synthetic_classification_data()
    with pytest.raises(FeatureSelectionError):
        select_features_by_lgbm_importance(
            X_train, y_train, X_val, y_val, candidate_features=[], top_k=1, min_class_count=5,
        )


# ---------------------------------------------------------------------------
# select_stable_features
# ---------------------------------------------------------------------------

def test_stability_selection_boundary_is_inclusive():
    # "f" selected in exactly 2 of 4 folds -> stability = 0.5.
    fold_selected = [["f", "g"], ["f"], ["g"], []]
    candidates = ["f", "g"]
    # min_fold_stability == 0.5 -> "f" kept (0.5 >= 0.5).
    kept_at_boundary = select_stable_features(fold_selected, candidates, min_fold_stability=0.5)
    assert "f" in kept_at_boundary
    # min_fold_stability == 0.51 -> "f" (exactly 0.5) now dropped.
    kept_above = select_stable_features(fold_selected, candidates, min_fold_stability=0.51)
    assert "f" not in kept_above


def test_stability_selection_counts_correctly():
    fold_selected = [["a", "b"], ["a", "b"], ["a"], ["a"]]
    candidates = ["a", "b", "c"]
    kept = select_stable_features(fold_selected, candidates, min_fold_stability=0.75)
    assert kept == ["a"]  # a: 4/4=1.0 kept, b: 2/4=0.5 dropped, c: 0/4=0 dropped


def test_stability_selection_no_folds_returns_all_candidates():
    assert select_stable_features([], ["a", "b"], min_fold_stability=0.5) == ["a", "b"]


def test_stability_selection_invalid_fraction_raises():
    with pytest.raises(FeatureSelectionError):
        select_stable_features([["a"]], ["a"], min_fold_stability=0.0)
    with pytest.raises(FeatureSelectionError):
        select_stable_features([["a"]], ["a"], min_fold_stability=1.5)
