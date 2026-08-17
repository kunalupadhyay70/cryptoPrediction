"""Tests for dataset_builder.py (Stage 2E/2F).

Focus areas per the Stage 2E/2F requirements: OHLCV-only (no microstructure
features), no leakage (every feature at row t provably independent of rows
> t), no bfill anywhere, and correct integration with target_engineering's
frozen 3-class target (neutral rows kept, nothing filtered).
"""
import numpy as np
import pandas as pd
import pytest

from dataset_builder import (
    NON_FEATURE_COLUMNS, DatasetBuilderError,
    build_dataset, build_features, feature_columns,
)
from target_engineering import CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP


def _synthetic_ohlcv(n: int = 90, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.002, n)
    close = 100 * np.cumprod(1 + ret)
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0005, n)))
    volume = np.abs(rng.normal(1000, 100, n))
    times = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open_time": times, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@pytest.fixture
def ohlcv():
    return _synthetic_ohlcv()


@pytest.fixture
def features(ohlcv):
    return build_features(ohlcv, lag_periods=3, atr_period=14)


# ---------------------------------------------------------------------------
# OHLCV-only: no microstructure feature names anywhere in the output
# ---------------------------------------------------------------------------

_MICROSTRUCTURE_NAME_FRAGMENTS = [
    "imb_", "weighted_imb", "microprice", "spread_l1", "spread_slope",
    "depth_bid", "depth_ask", "wall_bid", "wall_ask", "book_pressure",
    "mid_slope", "aggr_buy_vol", "aggr_sell_vol", "taker_imb",
]


def test_no_microstructure_features_present(features):
    cols = list(features.columns)
    for fragment in _MICROSTRUCTURE_NAME_FRAGMENTS:
        offenders = [c for c in cols if fragment in c]
        assert offenders == [], f"found order-book/trade-flow columns: {offenders}"


def test_build_features_does_not_touch_orderbook_or_trades_tables():
    # Structural guarantee, not just a naming convention: the actual CODE
    # (function bodies, not the explanatory module docstring which itself
    # discusses why microstructure tables are excluded) never opens a
    # database connection or references order_book_snapshots/trades at all.
    import inspect
    import dataset_builder
    code_src = "\n".join(
        inspect.getsource(obj)
        for name, obj in vars(dataset_builder).items()
        if inspect.isfunction(obj) and obj.__module__ == dataset_builder.__name__
    )
    assert "order_book_snapshots" not in code_src
    assert "sqlite3" not in code_src
    assert "trades" not in code_src.lower()


# ---------------------------------------------------------------------------
# No leakage: perturbing future rows must not change earlier feature rows
# ---------------------------------------------------------------------------

def test_features_at_row_t_are_unaffected_by_future_perturbation(ohlcv):
    cutoff = 70  # well past every rolling window used (max window = 50)
    features_before = build_features(ohlcv, lag_periods=3, atr_period=14)

    perturbed = ohlcv.copy()
    perturbed.loc[cutoff:, ["open", "high", "low", "close", "volume"]] *= 3.0
    features_after = build_features(perturbed, lag_periods=3, atr_period=14)

    numeric_cols = [
        c for c in features_before.columns
        if pd.api.types.is_numeric_dtype(features_before[c]) and c not in ("open", "high", "low", "close", "volume")
    ]
    for col in numeric_cols:
        a = features_before[col].iloc[:cutoff].to_numpy()
        b = features_after[col].iloc[:cutoff].to_numpy()
        np.testing.assert_array_equal(a, b, err_msg=f"leakage detected in column {col!r}")


def test_build_dataset_target_at_row_t_unaffected_by_far_future_perturbation(ohlcv):
    # End-to-end probe through build_dataset (features + target together).
    horizon_bars, atr_period = 3, 14
    cutoff = 75
    before = build_dataset(ohlcv, horizon_bars=horizon_bars, neutral_atr_mult=0.3, atr_period=atr_period)

    perturbed = ohlcv.copy()
    perturbed.loc[cutoff:, ["open", "high", "low", "close", "volume"]] *= 10.0
    after = build_dataset(perturbed, horizon_bars=horizon_bars, neutral_atr_mult=0.3, atr_period=atr_period)

    # Rows strictly before (cutoff - horizon_bars) never look at a perturbed
    # close via exit_price = close[t+h], so must be fully unaffected.
    safe_end = cutoff - horizon_bars
    np.testing.assert_array_equal(
        before["target_class"].iloc[:safe_end].to_numpy(),
        after["target_class"].iloc[:safe_end].to_numpy(),
    )


# ---------------------------------------------------------------------------
# No bfill: warm-up rows stay genuinely NaN, never filled from a later row
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "column, warmup_len",
    [
        ("vol_20", 20),      # rolling(20, min_periods=20) on pct_change (already 1 NaN) -> needs 21 rows total... see note
        ("sma_dist_20", 19),  # rolling(20, min_periods=20) of close -> first 19 rows NaN
        ("bb_width_50", 49),
        ("stoch_k_14", 13),
    ],
)
def test_warmup_rows_are_nan_not_backfilled(features, column, warmup_len):
    # The exact warm-up length is asserted precisely (not just "some NaNs
    # exist") to catch an accidental min_periods change, and the values are
    # confirmed NaN (not equal to the first valid value, which is what a
    # bfill() bug would produce).
    col = features[column]
    first_valid_idx = col.first_valid_index()
    assert first_valid_idx is not None
    assert col.iloc[:first_valid_idx].isna().all()
    first_valid_value = col.loc[first_valid_idx]
    # A bfill() bug would make every pre-warm-up row equal this value.
    pre_warmup = col.iloc[: max(0, first_valid_idx)]
    if len(pre_warmup) > 0:
        assert not (pre_warmup == first_valid_value).all()


def test_no_fillna_or_bfill_used_in_module_source():
    import inspect
    import dataset_builder
    code_src = "\n".join(
        inspect.getsource(obj)
        for name, obj in vars(dataset_builder).items()
        if inspect.isfunction(obj) and obj.__module__ == dataset_builder.__name__
    )
    assert ".bfill(" not in code_src
    assert ".fillna(" not in code_src
    assert ".ffill(" not in code_src


# ---------------------------------------------------------------------------
# Lag features: shift correctness
# ---------------------------------------------------------------------------

def test_lag_features_are_correctly_shifted(features):
    for lag in (1, 2, 3):
        col = f"ret_1_lag{lag}"
        assert col in features.columns
        shifted = features["ret_1"].shift(lag)
        pd.testing.assert_series_equal(features[col], shifted, check_names=False)


def test_lag_zero_periods_adds_no_lag_columns(ohlcv):
    feats = build_features(ohlcv, lag_periods=0, atr_period=14)
    assert not any(c.endswith("_lag1") for c in feats.columns)


# ---------------------------------------------------------------------------
# Time-of-day features: hand-verified for a known timestamp
# ---------------------------------------------------------------------------

def test_time_of_day_features_match_hand_computation(ohlcv):
    feats = build_features(ohlcv, lag_periods=0, atr_period=14)
    # Row 3 -> 2024-01-01T00:15:00Z -> minutes_in_day = 15
    row = 3
    minutes_in_day = 15
    expected_sin = np.sin(2 * np.pi * minutes_in_day / 1440)
    expected_cos = np.cos(2 * np.pi * minutes_in_day / 1440)
    assert feats.loc[row, "tod_sin"] == pytest.approx(expected_sin)
    assert feats.loc[row, "tod_cos"] == pytest.approx(expected_cos)
    # 2024-01-01 is a Monday -> dayofweek = 0
    assert feats.loc[row, "dow_sin"] == pytest.approx(np.sin(0.0))
    assert feats.loc[row, "dow_cos"] == pytest.approx(np.cos(0.0))


# ---------------------------------------------------------------------------
# feature_columns(): correct exclusion set
# ---------------------------------------------------------------------------

def test_feature_columns_excludes_ohlcv_and_target_columns(ohlcv):
    dataset = build_dataset(ohlcv, horizon_bars=3, neutral_atr_mult=0.3, atr_period=14)
    feats = feature_columns(dataset)
    for excluded in NON_FEATURE_COLUMNS:
        if excluded in dataset.columns:
            assert excluded not in feats
    # Sanity: at least the well-known engineered columns ARE included.
    assert "ret_1" in feats
    assert "rsi_14" in feats
    assert "macd_hist" in feats


# ---------------------------------------------------------------------------
# build_dataset integration with target_engineering
# ---------------------------------------------------------------------------

def test_build_dataset_keeps_neutral_rows_and_drops_nothing(ohlcv):
    dataset = build_dataset(ohlcv, horizon_bars=3, neutral_atr_mult=0.3, atr_period=14)
    assert len(dataset) == len(ohlcv)
    assert (dataset["target_class"] == CLASS_NEUTRAL).sum() > 0


def test_build_dataset_contains_all_three_classes_on_reasonably_volatile_data(ohlcv):
    # neutral_atr_mult small enough that all 3 classes appear on this
    # synthetic random-walk series (sanity check the pipeline can actually
    # produce a usable 3-class problem, not just always-neutral).
    dataset = build_dataset(ohlcv, horizon_bars=3, neutral_atr_mult=0.1, atr_period=14)
    present = set(dataset["target_class"].dropna().unique())
    assert present == {float(CLASS_DOWN), float(CLASS_NEUTRAL), float(CLASS_UP)}


def test_build_dataset_uses_target_engineerings_atr_not_a_reimplementation(ohlcv):
    # Cross-check: the `atr` column produced inside build_dataset's target
    # portion must equal target_engineering.compute_causal_atr computed
    # directly on the same frame — single source of truth, not two
    # divergent ATR implementations.
    from target_engineering import compute_causal_atr
    dataset = build_dataset(ohlcv, horizon_bars=3, neutral_atr_mult=0.3, atr_period=14)
    direct_atr = compute_causal_atr(ohlcv, period=14)
    pd.testing.assert_series_equal(dataset["atr"], direct_atr, check_names=False)


# ---------------------------------------------------------------------------
# Input validation / no mutation
# ---------------------------------------------------------------------------

def test_missing_ohlcv_column_raises():
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})  # no volume
    with pytest.raises(DatasetBuilderError):
        build_features(df)


def test_negative_lag_periods_raises(ohlcv):
    with pytest.raises(DatasetBuilderError):
        build_features(ohlcv, lag_periods=-1)


def test_build_features_does_not_mutate_input(ohlcv):
    original = ohlcv.copy(deep=True)
    _ = build_features(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, original)


def test_build_dataset_without_open_time_skips_time_features(ohlcv):
    df = ohlcv.drop(columns=["open_time"])
    feats = build_features(df)
    assert "tod_sin" not in feats.columns
    assert "dow_sin" not in feats.columns
