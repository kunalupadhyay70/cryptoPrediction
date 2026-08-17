"""Tests for model_training.py (Stage 2I)."""
import numpy as np
import pandas as pd
import pytest

from class_checks import MissingClassError
from model_training import (
    CLASS_ORDER, ModelTrainingError, align_probs_to_class_order,
    fit_lightgbm_multiclass_probs, fit_logistic_regression_probs,
    majority_class_baseline_probs, persistence_baseline_probs,
)


# ---------------------------------------------------------------------------
# align_probs_to_class_order — the "never assume positional ordering" guard
# ---------------------------------------------------------------------------

def test_align_probs_reorders_scrambled_classes_correctly():
    # Estimator reports classes_ = [2, 0, 1] (UP, DOWN, NEUTRAL) — i.e.
    # column 0 of raw proba is P(UP), column 1 is P(DOWN), column 2 is P(NEUTRAL).
    raw_proba = np.array([
        [0.7, 0.2, 0.1],   # P(UP)=0.7, P(DOWN)=0.2, P(NEUTRAL)=0.1
        [0.1, 0.1, 0.8],   # P(UP)=0.1, P(DOWN)=0.1, P(NEUTRAL)=0.8
    ])
    classes_ = np.array([2, 0, 1])
    aligned = align_probs_to_class_order(raw_proba, classes_)
    # Canonical column order is (DOWN, NEUTRAL, UP) = (0, 1, 2).
    expected = np.array([
        [0.2, 0.1, 0.7],
        [0.1, 0.8, 0.1],
    ])
    np.testing.assert_array_equal(aligned, expected)


def test_align_probs_identity_when_already_canonical():
    raw_proba = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
    classes_ = np.array([0, 1, 2])
    aligned = align_probs_to_class_order(raw_proba, classes_)
    np.testing.assert_array_equal(aligned, raw_proba)


def test_align_probs_missing_class_gets_zero_column():
    # Only DOWN and UP present (e.g. a fold whose training data happened to
    # have zero NEUTRAL rows and skipped the class-coverage check).
    raw_proba = np.array([[0.3, 0.7]])
    classes_ = np.array([0, 2])
    aligned = align_probs_to_class_order(raw_proba, classes_)
    np.testing.assert_array_equal(aligned, np.array([[0.3, 0.0, 0.7]]))


def test_align_probs_rows_still_sum_to_one_when_fully_covered():
    raw_proba = np.array([[0.5, 0.3, 0.2]])
    classes_ = np.array([1, 2, 0])  # scrambled but complete
    aligned = align_probs_to_class_order(raw_proba, classes_)
    assert aligned.sum(axis=1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Baseline A — majority class
# ---------------------------------------------------------------------------

def test_majority_class_baseline_picks_the_most_frequent_class():
    y_train = np.array([0, 0, 1, 2, 2, 2])  # class 2 (UP) is majority
    proba = majority_class_baseline_probs(y_train, n_test_rows=4)
    assert proba.shape == (4, 3)
    np.testing.assert_array_equal(proba, np.tile([0.0, 0.0, 1.0], (4, 1)))


def test_majority_class_baseline_tie_breaks_to_lower_class_value():
    y_train = np.array([0, 0, 2, 2])  # DOWN and UP tied at 2 each, NEUTRAL=0
    proba = majority_class_baseline_probs(y_train, n_test_rows=1)
    # Tie-break rule: lowest class value wins -> DOWN (0).
    np.testing.assert_array_equal(proba, np.array([[1.0, 0.0, 0.0]]))


def test_majority_class_baseline_rows_are_exactly_one_hot():
    y_train = np.array([1] * 10 + [0] * 3 + [2] * 2)
    proba = majority_class_baseline_probs(y_train, n_test_rows=5)
    assert (proba.sum(axis=1) == 1.0).all()
    assert ((proba == 0.0) | (proba == 1.0)).all()


def test_majority_class_baseline_empty_train_raises():
    with pytest.raises(ModelTrainingError):
        majority_class_baseline_probs(np.array([]), n_test_rows=3)


# ---------------------------------------------------------------------------
# Baseline B — causal persistence
# ---------------------------------------------------------------------------

def test_persistence_baseline_matches_hand_computed_directions():
    # close: [100, 101, 101, 99, 99]
    #   t=1: 101-100=+1 -> UP
    #   t=2: 101-101=0  -> NEUTRAL
    #   t=3: 99-101=-2  -> DOWN
    #   t=4: 99-99=0    -> NEUTRAL
    close = pd.Series([100.0, 101.0, 101.0, 99.0, 99.0])
    proba = persistence_baseline_probs(close, test_start_pos=1, n_test_rows=4)
    expected = np.array([
        [0.0, 0.0, 1.0],  # UP
        [0.0, 1.0, 0.0],  # NEUTRAL
        [1.0, 0.0, 0.0],  # DOWN
        [0.0, 1.0, 0.0],  # NEUTRAL
    ])
    np.testing.assert_array_equal(proba, expected)


def test_persistence_baseline_requires_a_lookback_row():
    close = pd.Series([100.0, 101.0])
    with pytest.raises(ModelTrainingError):
        persistence_baseline_probs(close, test_start_pos=0, n_test_rows=1)


def test_persistence_baseline_only_uses_rows_up_to_t_not_future():
    # Perturbing rows AFTER the scored window must not change results —
    # direct causality probe.
    close_before = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    close_after = close_before.copy()
    close_after.iloc[3:] = [999.0, 999.0]  # shock everything from t=3 onward
    a = persistence_baseline_probs(close_before, test_start_pos=1, n_test_rows=2)  # scores t=1,2
    b = persistence_baseline_probs(close_after, test_start_pos=1, n_test_rows=2)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Baseline C — Logistic Regression
# ---------------------------------------------------------------------------

def _synthetic_classification_data(n=300, seed=11):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    x1 = y + rng.normal(0, 0.3, size=n)
    x2 = rng.normal(size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    split = int(n * 0.8)
    return X.iloc[:split].reset_index(drop=True), y[:split], X.iloc[split:].reset_index(drop=True), y[split:]


def test_logistic_regression_probs_sum_to_one_and_correct_shape():
    X_train, y_train, X_test, y_test = _synthetic_classification_data()
    proba = fit_logistic_regression_probs(X_train, y_train, X_test, min_class_count=5)
    assert proba.shape == (len(X_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_logistic_regression_scaler_is_fit_only_on_train():
    X_train, y_train, X_test, y_test = _synthetic_classification_data()
    # Shift X_test far away from the train distribution; the scaler's
    # mean_/scale_ (derived only from fit_transform on X_train) must be
    # identical whether or not we do this, since fit_logistic_regression_probs
    # never calls scaler.fit on X_test.
    from sklearn.preprocessing import StandardScaler
    expected_scaler = StandardScaler().fit(X_train.to_numpy())

    import model_training as mt
    original_fit_transform = StandardScaler.fit_transform
    captured = {}

    def spy_fit_transform(self, X, y=None):
        captured["mean_from_call"] = np.array(self.fit(X).mean_) if not hasattr(self, "mean_") else None
        return original_fit_transform(self, X, y)

    # Simpler, robust approach: just re-run and inspect via monkeypatching
    # the class to capture the fitted scaler instance.
    fitted_scalers = []
    real_init = StandardScaler.fit

    def spy_fit(self, X, y=None):
        result = real_init(self, X, y)
        fitted_scalers.append(np.array(self.mean_))
        return result

    StandardScaler.fit = spy_fit
    try:
        fit_logistic_regression_probs(X_train, y_train, X_test, min_class_count=5)
    finally:
        StandardScaler.fit = real_init

    assert len(fitted_scalers) == 1
    np.testing.assert_allclose(fitted_scalers[0], expected_scaler.mean_)


def test_logistic_regression_raises_on_missing_class():
    n = 60
    rng = np.random.default_rng(12)
    X_train = pd.DataFrame({"x1": rng.normal(size=n)})
    y_train = np.zeros(n, dtype=int)
    X_test = pd.DataFrame({"x1": rng.normal(size=10)})
    with pytest.raises(MissingClassError):
        fit_logistic_regression_probs(X_train, y_train, X_test, min_class_count=5)


def test_logistic_regression_recovers_informative_signal():
    X_train, y_train, X_test, y_test = _synthetic_classification_data(n=600)
    proba = fit_logistic_regression_probs(X_train, y_train, X_test, min_class_count=5)
    pred = proba.argmax(axis=1)
    accuracy = (pred == y_test).mean()
    # x1 is strongly informative (y + small noise) -> should comfortably
    # beat random-guess accuracy (1/3) on a held-out split.
    assert accuracy > 0.6


# ---------------------------------------------------------------------------
# Primary model — LightGBM multiclass
# ---------------------------------------------------------------------------

def test_lightgbm_probs_sum_to_one_and_correct_shape():
    X_train, y_train, X_test, y_test = _synthetic_classification_data()
    proba = fit_lightgbm_multiclass_probs(
        X_train, y_train, X_test, min_class_count=5, n_estimators=50,
        lgbm_params={"num_leaves": 7},
    )
    assert proba.shape == (len(X_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_lightgbm_raises_on_missing_class():
    n = 60
    rng = np.random.default_rng(13)
    X_train = pd.DataFrame({"x1": rng.normal(size=n)})
    y_train = np.array([0] * 40 + [1] * 20)  # no class 2 at all
    X_test = pd.DataFrame({"x1": rng.normal(size=10)})
    with pytest.raises(MissingClassError):
        fit_lightgbm_multiclass_probs(X_train, y_train, X_test, min_class_count=5, n_estimators=20)


def test_lightgbm_deterministic_given_fixed_random_state():
    X_train, y_train, X_test, y_test = _synthetic_classification_data()
    kwargs = dict(min_class_count=5, n_estimators=40, lgbm_params={"num_leaves": 7}, random_state=99)
    a = fit_lightgbm_multiclass_probs(X_train, y_train, X_test, **kwargs)
    b = fit_lightgbm_multiclass_probs(X_train, y_train, X_test, **kwargs)
    np.testing.assert_array_equal(a, b)


def test_lightgbm_invalid_n_estimators_raises():
    X_train, y_train, X_test, y_test = _synthetic_classification_data()
    with pytest.raises(ModelTrainingError):
        fit_lightgbm_multiclass_probs(X_train, y_train, X_test, min_class_count=5, n_estimators=0)
