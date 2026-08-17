"""Tests for threshold_selection.py (Stage 2K)."""
import numpy as np
import pandas as pd
import pytest

from threshold_selection import (
    CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP, SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL,
    ThresholdSelectionError, apply_thresholds_and_signals, compute_signal,
    select_production_threshold, select_threshold,
)


# ---------------------------------------------------------------------------
# compute_signal — frozen SIGNAL RULE, hand-derived expected outputs
# ---------------------------------------------------------------------------

def test_signal_rule_hand_derived_cases():
    prob_down = np.array([0.2, 0.5, 0.1, 0.34, 0.3])
    prob_neutral = np.array([0.3, 0.2, 0.2, 0.33, 0.4])
    prob_up = np.array([0.5, 0.3, 0.7, 0.33, 0.3])
    sig = compute_signal(prob_down, prob_neutral, prob_up, buy_threshold=0.5, sell_threshold=0.5)
    # row0: prob_up=0.5>=0.5, strict max over down(0.2)/neutral(0.3) -> BUY
    # row1: prob_down=0.5>=0.5, strict max over up(0.3)/neutral(0.2) -> SELL
    # row2: prob_up=0.7>=0.5, strict max -> BUY
    # row3: three-way near-tie, no prob clears its own threshold as strict max -> HOLD
    # row4: prob_up=0.3 < 0.5 threshold, prob_down=0.3 < 0.5 threshold -> HOLD
    expected = np.array([SIGNAL_BUY, SIGNAL_SELL, SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_HOLD], dtype=object)
    np.testing.assert_array_equal(sig, expected)


def test_signal_rule_threshold_boundary_is_inclusive():
    # prob_up exactly equal to buy_threshold -> BUY (>=, not >).
    sig = compute_signal(np.array([0.1]), np.array([0.3]), np.array([0.6]), buy_threshold=0.6, sell_threshold=0.6)
    assert sig[0] == SIGNAL_BUY
    # Just below the threshold -> HOLD.
    sig2 = compute_signal(np.array([0.1]), np.array([0.3]), np.array([0.599]), buy_threshold=0.6, sell_threshold=0.6)
    assert sig2[0] == SIGNAL_HOLD


def test_signal_rule_requires_strict_max_not_just_threshold_clearance():
    # prob_up clears buy_threshold but is not the strict max (prob_neutral higher) -> HOLD.
    sig = compute_signal(np.array([0.1]), np.array([0.5]), np.array([0.4]), buy_threshold=0.3, sell_threshold=0.3)
    assert sig[0] == SIGNAL_HOLD


# ---------------------------------------------------------------------------
# select_threshold — sweep behavior + fallback
# ---------------------------------------------------------------------------

def _make_probs_return(n, seed, up_bias=False):
    rng = np.random.default_rng(seed)
    if up_bias:
        prob_up = rng.uniform(0.4, 0.9, n)
    else:
        prob_up = rng.uniform(0.1, 0.4, n)
    prob_down = (1 - prob_up) * rng.uniform(0.2, 0.5, n)
    prob_neutral = 1 - prob_up - prob_down
    tradable_return = rng.normal(0.002, 0.001, n)  # small positive drift
    return prob_down, prob_neutral, prob_up, tradable_return


def test_select_threshold_falls_back_when_too_few_trades():
    # Tiny sample: no swept threshold can ever clear min_trades_for_tuning=50.
    prob_down, prob_neutral, prob_up, tret = _make_probs_return(20, seed=1, up_bias=True)
    result = select_threshold(
        prob_down, prob_neutral, prob_up, tret,
        min_trades_for_tuning=50, default_buy_threshold=0.42, default_sell_threshold=0.44,
    )
    assert result.buy_used_fallback is True
    assert result.sell_used_fallback is True
    assert result.buy_threshold == pytest.approx(0.42)
    assert result.sell_threshold == pytest.approx(0.44)


def test_select_threshold_tunes_when_enough_trades():
    prob_down, prob_neutral, prob_up, tret = _make_probs_return(500, seed=2, up_bias=True)
    result = select_threshold(
        prob_down, prob_neutral, prob_up, tret,
        min_trades_for_tuning=10, threshold_sweep_min=0.34, threshold_sweep_max=0.80, threshold_sweep_step=0.02,
    )
    assert result.buy_used_fallback is False
    assert 0.34 <= result.buy_threshold <= 0.80
    assert result.buy_n_trades >= 10


def test_select_threshold_excludes_nan_tradable_return_rows():
    prob_down = np.array([0.1, 0.1, 0.1])
    prob_neutral = np.array([0.2, 0.2, 0.2])
    prob_up = np.array([0.7, 0.7, 0.7])
    tret = np.array([0.01, np.nan, 0.01])
    # With only 2 valid (non-NaN) rows and min_trades_for_tuning=2, tuning
    # should still work on the 2 valid rows, not error out on the NaN one.
    result = select_threshold(
        prob_down, prob_neutral, prob_up, tret,
        min_trades_for_tuning=2, threshold_sweep_min=0.34, threshold_sweep_max=0.5, threshold_sweep_step=0.02,
    )
    assert result.buy_n_trades == 2


def test_select_threshold_invalid_sweep_bounds_raises():
    with pytest.raises(ThresholdSelectionError):
        select_threshold(
            np.array([0.1]), np.array([0.2]), np.array([0.7]), np.array([0.01]),
            threshold_sweep_min=0.8, threshold_sweep_max=0.3,
        )


def test_select_threshold_mismatched_lengths_raises():
    with pytest.raises(ThresholdSelectionError):
        select_threshold(
            np.array([0.1, 0.1]), np.array([0.2, 0.2]), np.array([0.7, 0.7]), np.array([0.01]),
        )


# ---------------------------------------------------------------------------
# apply_thresholds_and_signals — past-only wiring across folds
# ---------------------------------------------------------------------------

def _synthetic_oof_table():
    from oof_builder import OOF_COLUMNS
    rows = []
    rng = np.random.default_rng(5)
    run_id = "r1"
    for fold in (1, 2, 3):
        for model in ("baseline_majority_class", "baseline_persistence", "logistic_regression", "lightgbm"):
            n = 60
            prob_up = rng.uniform(0.2, 0.8, n)
            prob_down = (1 - prob_up) * rng.uniform(0.2, 0.6, n)
            prob_neutral = 1 - prob_up - prob_down
            actual = rng.integers(0, 3, n)
            for i in range(n):
                rows.append({
                    "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=5 * i)
                                 + pd.Timedelta(days=fold),
                    "symbol": "BTCUSDT", "interval": "5m", "fold": fold, "model": model,
                    "actual_class": int(actual[i]),
                    "prob_down": float(prob_down[i]), "prob_neutral": float(prob_neutral[i]),
                    "prob_up": float(prob_up[i]),
                    "pred_class": int(np.argmax([prob_down[i], prob_neutral[i], prob_up[i]])),
                    "buy_threshold": np.nan, "sell_threshold": np.nan, "signal": None,
                    "close_t": 100.0, "next_open": 100.1, "exit_close": 100.2,
                    "tradable_return": float(rng.normal(0.001, 0.002)),
                    "atr": 0.5, "volatility_band": 0.002, "vol_regime": 0.0, "run_id": run_id,
                })
    return pd.DataFrame(rows, columns=OOF_COLUMNS)


def _fold1_internal_val_fixture(n=80, seed=9):
    rng = np.random.default_rng(seed)
    prob_up = rng.uniform(0.2, 0.8, n)
    prob_down = (1 - prob_up) * rng.uniform(0.2, 0.6, n)
    prob_neutral = 1 - prob_up - prob_down
    probs = np.column_stack([prob_down, prob_neutral, prob_up])
    tret = rng.normal(0.001, 0.002, n)
    return {
        "logistic_regression": {"probs": probs, "tradable_return": tret},
        "lightgbm": {"probs": probs.copy(), "tradable_return": tret.copy()},
    }


def test_apply_thresholds_fills_baselines_with_direct_mapping_no_threshold():
    oof_df = _synthetic_oof_table()
    result = apply_thresholds_and_signals(oof_df, _fold1_internal_val_fixture(), min_trades_for_tuning=5)
    baseline_rows = result[result["model"].isin(["baseline_majority_class", "baseline_persistence"])]
    assert baseline_rows["buy_threshold"].isna().all()
    assert baseline_rows["sell_threshold"].isna().all()
    expected_sig = baseline_rows["pred_class"].map({CLASS_DOWN: SIGNAL_SELL, CLASS_NEUTRAL: SIGNAL_HOLD, CLASS_UP: SIGNAL_BUY})
    assert (baseline_rows["signal"].to_numpy() == expected_sig.to_numpy()).all()


def test_apply_thresholds_fold1_uses_internal_val_not_outer_test():
    oof_df = _synthetic_oof_table()
    fixture = _fold1_internal_val_fixture()
    result = apply_thresholds_and_signals(oof_df, fixture, min_trades_for_tuning=5)

    # Recompute fold 1's lightgbm threshold directly from the SAME fixture
    # ingredients, independent of the orchestration function, and compare.
    expected = select_threshold(
        fixture["lightgbm"]["probs"][:, 0], fixture["lightgbm"]["probs"][:, 1], fixture["lightgbm"]["probs"][:, 2],
        fixture["lightgbm"]["tradable_return"], min_trades_for_tuning=5,
    )
    fold1_lgbm = result[(result["model"] == "lightgbm") & (result["fold"] == 1)]
    assert fold1_lgbm["buy_threshold"].iloc[0] == pytest.approx(expected.buy_threshold)
    assert fold1_lgbm["sell_threshold"].iloc[0] == pytest.approx(expected.sell_threshold)


def test_apply_thresholds_fold_k_uses_only_prior_folds_not_current_or_future():
    oof_df = _synthetic_oof_table()
    fixture = _fold1_internal_val_fixture()
    result = apply_thresholds_and_signals(oof_df, fixture, min_trades_for_tuning=5)

    # Fold 3's lightgbm threshold must be recomputable from folds 1+2 OOF
    # rows only (not fold 3's own rows). Independently reproduce and compare.
    prior = oof_df[(oof_df["model"] == "lightgbm") & (oof_df["fold"] < 3)]
    expected = select_threshold(
        prior["prob_down"].to_numpy(), prior["prob_neutral"].to_numpy(), prior["prob_up"].to_numpy(),
        prior["tradable_return"].to_numpy(), min_trades_for_tuning=5,
    )
    fold3_lgbm = result[(result["model"] == "lightgbm") & (result["fold"] == 3)]
    assert fold3_lgbm["buy_threshold"].iloc[0] == pytest.approx(expected.buy_threshold)
    assert fold3_lgbm["sell_threshold"].iloc[0] == pytest.approx(expected.sell_threshold)

    # Perturbing fold 3's OWN rows must not change fold 3's threshold
    # (proves current-fold rows are never part of the sweep input).
    oof_perturbed = oof_df.copy()
    fold3_mask = (oof_perturbed["model"] == "lightgbm") & (oof_perturbed["fold"] == 3)
    oof_perturbed.loc[fold3_mask, "tradable_return"] = 999.0
    result2 = apply_thresholds_and_signals(oof_perturbed, fixture, min_trades_for_tuning=5)
    fold3_lgbm2 = result2[(result2["model"] == "lightgbm") & (result2["fold"] == 3)]
    assert fold3_lgbm2["buy_threshold"].iloc[0] == pytest.approx(fold3_lgbm["buy_threshold"].iloc[0])


def test_apply_thresholds_missing_fold1_fixture_entry_raises():
    oof_df = _synthetic_oof_table()
    incomplete_fixture = {"logistic_regression": _fold1_internal_val_fixture()["logistic_regression"]}
    with pytest.raises(ThresholdSelectionError):
        apply_thresholds_and_signals(oof_df, incomplete_fixture, min_trades_for_tuning=5)


def test_apply_thresholds_signal_column_consistent_with_compute_signal():
    oof_df = _synthetic_oof_table()
    fixture = _fold1_internal_val_fixture()
    result = apply_thresholds_and_signals(oof_df, fixture, min_trades_for_tuning=5)
    tunable = result[result["model"].isin(["logistic_regression", "lightgbm"])]
    for (model, fold), group in tunable.groupby(["model", "fold"]):
        recomputed = compute_signal(
            group["prob_down"].to_numpy(), group["prob_neutral"].to_numpy(), group["prob_up"].to_numpy(),
            group["buy_threshold"].iloc[0], group["sell_threshold"].iloc[0],
        )
        np.testing.assert_array_equal(group["signal"].to_numpy(), recomputed)


# ---------------------------------------------------------------------------
# select_production_threshold (Stage 2P)
# ---------------------------------------------------------------------------

def test_select_production_threshold_pools_all_folds_for_the_model():
    oof_df = _synthetic_oof_table()
    result = select_production_threshold(oof_df, "lightgbm", min_trades_for_tuning=5)
    # Independently reproduce by pooling all folds' rows for that model.
    rows = oof_df[oof_df["model"] == "lightgbm"]
    expected = select_threshold(
        rows["prob_down"].to_numpy(), rows["prob_neutral"].to_numpy(), rows["prob_up"].to_numpy(),
        rows["tradable_return"].to_numpy(), min_trades_for_tuning=5,
    )
    assert result.buy_threshold == pytest.approx(expected.buy_threshold)
    assert result.sell_threshold == pytest.approx(expected.sell_threshold)


def test_select_production_threshold_unknown_model_raises():
    oof_df = _synthetic_oof_table()
    with pytest.raises(ThresholdSelectionError):
        select_production_threshold(oof_df, "nonexistent_model")
