"""Tests for production_model.py (Stage 2N)."""
import numpy as np
import pandas as pd
import pytest

from production_model import (
    ProductionModelError, ProductionModelResult,
    predict_proba_production, select_median_best_iteration, train_production_model,
)
from class_checks import MissingClassError


# ---------------------------------------------------------------------------
# select_median_best_iteration
# ---------------------------------------------------------------------------

def test_median_best_iteration_odd_count_exact_middle():
    assert select_median_best_iteration([10, 30, 20]) == 20


def test_median_best_iteration_even_count_averages_and_rounds():
    # [10, 20, 30, 40] -> median = 25.0 -> round -> 25
    assert select_median_best_iteration([10, 20, 30, 40]) == 25


def test_median_best_iteration_floors_at_one():
    assert select_median_best_iteration([0, 0, 0]) == 1


def test_median_best_iteration_empty_raises():
    with pytest.raises(ProductionModelError):
        select_median_best_iteration([])


# ---------------------------------------------------------------------------
# train_production_model
# ---------------------------------------------------------------------------

def _synthetic_dataset(n=300, seed=11):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    informative = y + rng.normal(0, 0.2, n)
    noise = rng.normal(size=n)
    unlabeled_tail = 10
    target_class = y.astype(float)
    target_class[-unlabeled_tail:] = np.nan  # simulate genuinely unlabeled tail rows
    return pd.DataFrame({"informative": informative, "noise": noise, "target_class": target_class})


def test_train_production_model_uses_stability_selected_features_and_median_iteration():
    dataset = _synthetic_dataset()
    fold_selected = [["informative", "noise"], ["informative"], ["informative"]]  # informative: 3/3, noise: 1/3
    fold_best_iterations = [10, 20, 30]
    result = train_production_model(
        dataset, candidate_features=["informative", "noise"],
        fold_selected_features=fold_selected, fold_best_iterations=fold_best_iterations,
        min_fold_stability=0.5, min_class_count=5, lgbm_params={"num_leaves": 7},
    )
    assert result.selected_features == ["informative"]  # noise: 1/3=0.33 < 0.5 -> dropped
    assert result.n_estimators == 20  # median of [10,20,30]
    assert result.n_training_rows == 290  # 300 - 10 unlabeled tail rows
    assert sum(result.class_counts.values()) == 290


def test_train_production_model_trains_on_all_labeled_rows_not_a_subset():
    dataset = _synthetic_dataset(n=200)
    result = train_production_model(
        dataset, candidate_features=["informative", "noise"],
        fold_selected_features=[["informative"]], fold_best_iterations=[15],
        min_fold_stability=0.5, min_class_count=5,
    )
    n_labeled = int(dataset["target_class"].notna().sum())
    assert result.n_training_rows == n_labeled


def test_train_production_model_raises_on_missing_class():
    n = 50
    dataset = pd.DataFrame({
        "informative": np.random.default_rng(1).normal(size=n),
        "target_class": np.zeros(n),  # only class 0 present
    })
    with pytest.raises(MissingClassError):
        train_production_model(
            dataset, candidate_features=["informative"],
            fold_selected_features=[["informative"]], fold_best_iterations=[10],
            min_fold_stability=0.5, min_class_count=5,
        )


def test_train_production_model_raises_when_no_stable_features_survive():
    dataset = _synthetic_dataset(n=100)
    with pytest.raises(ProductionModelError):
        train_production_model(
            dataset, candidate_features=["informative", "noise"],
            fold_selected_features=[["informative"], []],  # informative: 1/2=0.5
            fold_best_iterations=[10, 20],
            min_fold_stability=0.99,  # nothing survives at this bar
            min_class_count=5,
        )


def test_train_production_model_missing_target_class_column_raises():
    with pytest.raises(ProductionModelError):
        train_production_model(
            pd.DataFrame({"informative": [1.0, 2.0]}), candidate_features=["informative"],
            fold_selected_features=[["informative"]], fold_best_iterations=[10],
            min_fold_stability=0.5, min_class_count=1,
        )


def test_n_estimators_override_in_lgbm_params_is_ignored_not_merged():
    dataset = _synthetic_dataset(n=200)
    result = train_production_model(
        dataset, candidate_features=["informative", "noise"],
        fold_selected_features=[["informative"]], fold_best_iterations=[42],
        min_fold_stability=0.5, min_class_count=5,
        lgbm_params={"n_estimators": 999, "num_leaves": 7},
    )
    assert result.n_estimators == 42
    assert result.model.n_estimators == 42


# ---------------------------------------------------------------------------
# predict_proba_production
# ---------------------------------------------------------------------------

def test_predict_proba_production_canonical_order_and_sums_to_one():
    dataset = _synthetic_dataset(n=300)
    result = train_production_model(
        dataset, candidate_features=["informative", "noise"],
        fold_selected_features=[["informative", "noise"]], fold_best_iterations=[30],
        min_fold_stability=0.5, min_class_count=5, lgbm_params={"num_leaves": 7},
    )
    X_new = pd.DataFrame({"informative": [0.0, 1.0, 2.0], "noise": [0.1, -0.1, 0.2]})
    probs = predict_proba_production(result, X_new)
    assert probs.shape == (3, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
