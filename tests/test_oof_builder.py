"""Tests for oof_builder.py (Stage 2J)."""
import numpy as np
import pandas as pd
import pytest

from dataset_builder import build_dataset, feature_columns
from oof_builder import (
    ALL_MODELS, OOFError, OOFPipelineConfig, generate_oof_predictions, validate_oof_invariants,
)
from target_engineering import CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP


def _synthetic_dataset(n=300, seed=21):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.003, n)
    close = 100 * np.cumprod(1 + ret)
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    volume = np.abs(rng.normal(1000, 100, n))
    times = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    ohlcv = pd.DataFrame(
        {"open_time": times, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
    # neutral_atr_mult=0.4 chosen (via a quick empirical check) to give this
    # particular synthetic series a reasonably balanced 3-class split, so
    # min_class_count checks below pass reliably across folds rather than
    # depending on getting lucky with a rare-NEUTRAL random draw.
    dataset = build_dataset(ohlcv, horizon_bars=3, neutral_atr_mult=0.4, atr_period=14, lag_periods=2)
    return dataset


@pytest.fixture(scope="module")
def small_config():
    return OOFPipelineConfig(
        n_folds=2,
        min_train_rows=150,
        min_test_rows=40,
        horizon_bars=3,
        embargo_bars=0,
        min_class_count=3,
        correlation_threshold=0.95,
        importance_top_k=5,
        internal_validation_fraction=0.2,
        lgbm_early_stopping_rounds=5,
        random_state=42,
        lgbm_params={"n_estimators": 30, "num_leaves": 7},
    )


@pytest.fixture(scope="module")
def oof_result(small_config):
    dataset = _synthetic_dataset()
    candidate_features = feature_columns(dataset)
    return generate_oof_predictions(
        dataset, candidate_features, symbol="BTCUSDT", interval="5m", run_id="test_run_1", config=small_config
    )


# ---------------------------------------------------------------------------
# Structural contract
# ---------------------------------------------------------------------------

def test_oof_table_has_all_canonical_columns(oof_result):
    oof_df, _ = oof_result
    from oof_builder import OOF_COLUMNS
    assert list(oof_df.columns) == OOF_COLUMNS


def test_oof_table_has_exactly_the_four_frozen_models(oof_result):
    oof_df, _ = oof_result
    assert set(oof_df["model"].unique()) == set(ALL_MODELS)


def test_oof_probabilities_sum_to_one(oof_result):
    oof_df, _ = oof_result
    sums = oof_df[["prob_down", "prob_neutral", "prob_up"]].sum(axis=1)
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-6)


def test_oof_no_duplicate_timestamp_model_run_id(oof_result):
    oof_df, _ = oof_result
    assert not oof_df.duplicated(subset=["timestamp", "model", "run_id"]).any()


def test_oof_every_model_covers_the_same_eligible_rows(oof_result):
    oof_df, _ = oof_result
    per_model = {m: set(zip(g["timestamp"], g["fold"])) for m, g in oof_df.groupby("model")}
    keys = list(per_model.values())
    assert all(k == keys[0] for k in keys)


def test_oof_neutral_rows_are_retained(oof_result):
    oof_df, _ = oof_result
    assert (oof_df["actual_class"] == CLASS_NEUTRAL).sum() > 0
    assert (oof_df["actual_class"] == CLASS_DOWN).sum() > 0
    assert (oof_df["actual_class"] == CLASS_UP).sum() > 0


def test_oof_thresholds_and_signal_left_unset_by_stage_2j(oof_result):
    # Stage 2K's job, not this module's.
    oof_df, _ = oof_result
    assert oof_df["buy_threshold"].isna().all()
    assert oof_df["sell_threshold"].isna().all()
    assert oof_df["signal"].isna().all()


def test_oof_pred_class_is_argmax_of_probabilities(oof_result):
    oof_df, _ = oof_result
    manual_argmax = oof_df[["prob_down", "prob_neutral", "prob_up"]].to_numpy().argmax(axis=1)
    np.testing.assert_array_equal(oof_df["pred_class"].to_numpy(), manual_argmax)


# ---------------------------------------------------------------------------
# Leakage-critical: no train/test overlap; every eligible row appears
# ---------------------------------------------------------------------------

def test_no_row_is_ever_both_train_and_oof_test_within_a_fold(small_config):
    dataset = _synthetic_dataset()
    candidate_features = feature_columns(dataset)
    from walk_forward import generate_walk_forward_splits
    splits = generate_walk_forward_splits(
        n_rows=len(dataset), n_folds=small_config.n_folds,
        min_train_rows=small_config.min_train_rows, min_test_rows=small_config.min_test_rows,
        horizon_bars=small_config.horizon_bars, embargo_bars=small_config.embargo_bars,
    )
    for s in splits:
        assert not (set(s.train_idx.tolist()) & set(s.test_idx.tolist()))


def test_all_labeled_rows_in_the_walk_forward_test_range_appear_in_oof(small_config, oof_result):
    oof_df, _ = oof_result
    dataset = _synthetic_dataset()
    from walk_forward import generate_walk_forward_splits
    splits = generate_walk_forward_splits(
        n_rows=len(dataset), n_folds=small_config.n_folds,
        min_train_rows=small_config.min_train_rows, min_test_rows=small_config.min_test_rows,
        horizon_bars=small_config.horizon_bars, embargo_bars=small_config.embargo_bars,
    )
    single_model_rows = oof_df[oof_df["model"] == "lightgbm"]
    oof_timestamps = set(single_model_rows["timestamp"])
    for s in splits:
        for t in s.test_idx:
            row = dataset.iloc[int(t)]
            if pd.notna(row["target_class"]):
                assert row["open_time"] in oof_timestamps, f"eligible row {t} missing from OOF"


def test_unlabeled_tail_rows_never_appear_in_oof(oof_result):
    oof_df, _ = oof_result
    dataset = _synthetic_dataset()
    unlabeled_timestamps = set(dataset.loc[dataset["target_class"].isna(), "open_time"])
    assert not (set(oof_df["timestamp"]) & unlabeled_timestamps)


# ---------------------------------------------------------------------------
# validate_oof_invariants — direct unit tests with hand-crafted bad tables
# ---------------------------------------------------------------------------

def _minimal_valid_row(**overrides):
    row = {
        "timestamp": pd.Timestamp("2024-01-01", tz="UTC"), "symbol": "BTCUSDT", "interval": "5m",
        "fold": 1, "model": "lightgbm", "actual_class": 1,
        "prob_down": 0.2, "prob_neutral": 0.5, "prob_up": 0.3, "pred_class": 1,
        "buy_threshold": np.nan, "sell_threshold": np.nan, "signal": None,
        "close_t": 100.0, "next_open": 100.1, "exit_close": 100.2, "tradable_return": 0.001,
        "atr": 0.5, "volatility_band": 0.002, "vol_regime": 0.0, "run_id": "r1",
    }
    row.update(overrides)
    return row


def _four_model_table(bad_row_overrides=None, bad_model=None):
    from oof_builder import OOF_COLUMNS
    rows = []
    for model in ALL_MODELS:
        overrides = {"model": model}
        if bad_row_overrides and model == (bad_model or ALL_MODELS[0]):
            overrides.update(bad_row_overrides)
        rows.append(_minimal_valid_row(**overrides))
    return pd.DataFrame(rows, columns=OOF_COLUMNS)


def test_validate_rejects_duplicate_timestamp_model_run_id():
    from oof_builder import OOF_COLUMNS
    df = pd.DataFrame([_minimal_valid_row(), _minimal_valid_row()], columns=OOF_COLUMNS)
    with pytest.raises(OOFError, match="Duplicate"):
        validate_oof_invariants(df)


def test_validate_rejects_bad_probability_sum():
    df = _four_model_table(bad_row_overrides={"prob_down": 0.9})  # sum now 1.7
    with pytest.raises(OOFError, match="prob_down"):
        validate_oof_invariants(df)


def test_validate_rejects_missing_column():
    df = _four_model_table().drop(columns=["atr"])
    with pytest.raises(OOFError, match="missing required columns"):
        validate_oof_invariants(df)


def test_validate_rejects_uncovered_model_timestamp_mismatch():
    from oof_builder import OOF_COLUMNS
    rows = [_minimal_valid_row(model=m) for m in ALL_MODELS]
    # Give the lightgbm row a different timestamp than the others.
    rows[-1] = dict(rows[-1], timestamp=pd.Timestamp("2099-01-01", tz="UTC"))
    df = pd.DataFrame(rows, columns=OOF_COLUMNS)
    with pytest.raises(OOFError, match="does not cover the same eligible"):
        validate_oof_invariants(df)


def test_validate_rejects_invalid_actual_class():
    df = _four_model_table(bad_row_overrides={"actual_class": 7})
    with pytest.raises(OOFError, match="actual_class"):
        validate_oof_invariants(df)


def test_validate_accepts_a_well_formed_table():
    df = _four_model_table()
    validate_oof_invariants(df)  # must not raise


def test_validate_rejects_empty_table():
    from oof_builder import OOF_COLUMNS
    df = pd.DataFrame(columns=OOF_COLUMNS)
    with pytest.raises(OOFError, match="empty"):
        validate_oof_invariants(df)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_missing_required_dataset_column_raises(small_config):
    dataset = _synthetic_dataset().drop(columns=["atr"])
    with pytest.raises(OOFError, match="missing required columns"):
        generate_oof_predictions(
            dataset, feature_columns(dataset), "BTCUSDT", "5m", "r1", small_config
        )
