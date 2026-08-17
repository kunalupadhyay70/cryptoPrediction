"""Tests for target_engineering.py (Stage 2D).

Expected values in the main synthetic-series tests below were computed BY
HAND (independently, with a standalone script — not by calling the module
under test) before this test file was written, then transcribed here as
literal constants. This is deliberate per the project's test-quality rule:
tests must catch real bugs (off-by-one horizon/entry alignment, wrong ATR
warm-up length, wrong tie-breaking direction, wrong class mapping), not
just echo back whatever the implementation currently computes.
"""
import numpy as np
import pandas as pd
import pytest

from target_engineering import (
    CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP, CLASS_LABELS,
    TargetEngineeringError, compute_causal_atr, compute_target, target_columns,
)


# ---------------------------------------------------------------------------
# Synthetic series used across most tests, and its independently-computed
# expected values (see module docstring).
# ---------------------------------------------------------------------------

def _synthetic_df() -> pd.DataFrame:
    close = np.array(
        [100, 100.1, 99.9, 100.2, 100.0, 100.1, 99.95, 100.05, 100.0, 100.1,
         103.0, 103.2, 103.1, 103.3, 103.2, 99.0, 98.8, 98.9, 99.0, 99.1],
        dtype=float,
    )
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


ATR_PERIOD = 4
HORIZON_BARS = 3
NEUTRAL_ATR_MULT = 0.3

# Hand-derived ATR(period=4): first 3 rows NaN (warm-up), then rolling mean of TR.
EXPECTED_ATR = [
    np.nan, np.nan, np.nan, 0.25, 0.3, 0.3, 0.2875, 0.2375, 0.2, 0.2,
    0.8875, 0.9125, 0.925, 0.95, 0.25, 1.25, 1.275, 1.25, 1.25, 0.225,
]

# Hand-derived target_class for horizon_bars=3, neutral_atr_mult=0.3:
# NaN for ATR warm-up (rows 0-2) and tail rows without t+h (rows 17-19).
EXPECTED_CLASS = [
    np.nan, np.nan, np.nan, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, np.nan, np.nan, np.nan,
]


@pytest.fixture
def synthetic_result():
    df = _synthetic_df()
    return compute_target(df, horizon_bars=HORIZON_BARS,
                           neutral_atr_mult=NEUTRAL_ATR_MULT, atr_period=ATR_PERIOD)


# ---------------------------------------------------------------------------
# compute_causal_atr — hand-verified values
# ---------------------------------------------------------------------------

def test_causal_atr_matches_hand_computed_values():
    df = _synthetic_df()
    atr = compute_causal_atr(df, period=ATR_PERIOD)
    np.testing.assert_allclose(atr.to_numpy(), EXPECTED_ATR, rtol=0, atol=1e-9, equal_nan=True)


def test_causal_atr_warmup_rows_are_exactly_period_minus_one():
    df = _synthetic_df()
    atr = compute_causal_atr(df, period=ATR_PERIOD)
    assert atr.iloc[: ATR_PERIOD - 1].isna().all()
    assert atr.iloc[ATR_PERIOD - 1 :].notna().all()


def test_causal_atr_at_row_t_does_not_change_if_future_rows_change():
    # Direct leakage probe: perturbing rows AFTER t must not change ATR[t].
    df = _synthetic_df()
    atr_before = compute_causal_atr(df, period=ATR_PERIOD)

    df_perturbed = df.copy()
    df_perturbed.loc[15:, ["open", "high", "low", "close"]] *= 5.0  # huge future shock
    atr_after = compute_causal_atr(df_perturbed, period=ATR_PERIOD)

    # Every row strictly before the perturbation must be bit-for-bit identical.
    np.testing.assert_array_equal(
        atr_before.iloc[:15].to_numpy(), atr_after.iloc[:15].to_numpy()
    )


# ---------------------------------------------------------------------------
# compute_target — full pipeline against hand-derived expected classes
# ---------------------------------------------------------------------------

def test_target_class_matches_hand_computed_values(synthetic_result):
    np.testing.assert_allclose(
        synthetic_result["target_class"].to_numpy(), EXPECTED_CLASS, rtol=0, atol=0, equal_nan=True
    )


def test_class_mapping_is_frozen():
    assert CLASS_DOWN == 0
    assert CLASS_NEUTRAL == 1
    assert CLASS_UP == 2
    assert CLASS_LABELS == {0: "DOWN", 1: "NEUTRAL", 2: "UP"}


def test_target_label_matches_target_class(synthetic_result):
    df = synthetic_result
    for cls, label in zip(df["target_class"], df["target_label"]):
        if pd.isna(cls):
            assert label is None
        else:
            assert label == CLASS_LABELS[int(cls)]


def test_no_rows_are_dropped(synthetic_result):
    original_len = len(_synthetic_df())
    assert len(synthetic_result) == original_len


def test_neutral_rows_remain_in_output(synthetic_result):
    # Row 4 and row 11 are hand-derived NEUTRAL (class 1) rows — confirm
    # they are present with their label intact, not filtered out.
    assert synthetic_result.loc[4, "target_class"] == CLASS_NEUTRAL
    assert synthetic_result.loc[4, "target_label"] == "NEUTRAL"
    assert synthetic_result.loc[11, "target_class"] == CLASS_NEUTRAL


def test_atr_warmup_rows_are_unlabeled(synthetic_result):
    # atr_period=4 -> rows 0,1,2 have no ATR yet -> must be unlabeled,
    # regardless of what their (fully computable) tradable_return would be.
    for i in range(ATR_PERIOD - 1):
        assert pd.isna(synthetic_result.loc[i, "atr"])
        assert pd.isna(synthetic_result.loc[i, "target_class"])
        assert synthetic_result.loc[i, "target_label"] is None


def test_tail_rows_without_horizon_are_unlabeled(synthetic_result):
    n = len(synthetic_result)
    # Last row has no open[t+1] (entry_price NaN); last horizon_bars rows
    # have no close[t+h] (exit_price NaN). Both must yield unlabeled rows.
    for i in range(n - HORIZON_BARS, n):
        assert pd.isna(synthetic_result.loc[i, "target_class"])
    assert pd.isna(synthetic_result.loc[n - 1, "entry_price"])
    assert pd.isna(synthetic_result.loc[n - HORIZON_BARS, "exit_price"])


def test_entry_and_exit_price_alignment(synthetic_result):
    df = _synthetic_df()
    # entry_price[t] == open[t+1]
    for t in range(len(df) - 1):
        assert synthetic_result.loc[t, "entry_price"] == pytest.approx(df.loc[t + 1, "open"])
    # exit_price[t] == close[t+horizon_bars]
    for t in range(len(df) - HORIZON_BARS):
        assert synthetic_result.loc[t, "exit_price"] == pytest.approx(df.loc[t + HORIZON_BARS, "close"])


def test_up_row_has_tradable_return_strictly_above_band(synthetic_result):
    up_rows = synthetic_result[synthetic_result["target_class"] == CLASS_UP]
    assert len(up_rows) > 0
    assert (up_rows["tradable_return"] > up_rows["target_band"]).all()


def test_down_row_has_tradable_return_strictly_below_negative_band(synthetic_result):
    down_rows = synthetic_result[synthetic_result["target_class"] == CLASS_DOWN]
    assert len(down_rows) > 0
    assert (down_rows["tradable_return"] < -down_rows["target_band"]).all()


# ---------------------------------------------------------------------------
# Boundary / tie behavior: tradable_return exactly == band must be NEUTRAL
# ---------------------------------------------------------------------------

def test_exact_tie_at_band_boundary_is_neutral():
    # Construct a series where tradable_return lands EXACTLY (bit-for-bit,
    # not just approximately) on the band boundary, using only powers of
    # two so every intermediate division is exact in IEEE-754 float64 —
    # this proves the strict-inequality (not >=) contract from the frozen
    # spec, rather than merely getting lucky with float rounding.
    #
    # ATR[0] = TR[0] = high[0]-low[0] = 129-127 = 2 (no prev close at row 0)
    # band[0] = neutral_atr_mult(1.0) * atr(2) / close[0](128) = 2/128 = 0.015625 exactly
    # entry_price[0] = open[1] = 128.0, exit_price[0] = close[1] = 130.0
    # tradable_return[0] = 130/128 - 1 = 0.015625 exactly == band[0]
    close = [128.0, 130.0]
    open_ = [126.0, 128.0]
    high = [129.0, 131.0]
    low = [127.0, 129.0]
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})

    out = compute_target(df, horizon_bars=1, neutral_atr_mult=1.0, atr_period=1)

    tr = out.loc[0, "tradable_return"]
    band = out.loc[0, "target_band"]
    assert tr == band  # exact bit-for-bit tie, not just pytest.approx
    assert band == 0.015625  # sanity: confirms the construction landed where intended
    assert out.loc[0, "target_class"] == CLASS_NEUTRAL  # strict > required for UP, tie -> NEUTRAL

    # And one tick above the tie must flip to UP, proving the boundary is
    # live (not e.g. an off-by-one that makes everything NEUTRAL).
    df_above = df.copy()
    df_above.loc[1, "close"] = 130.0 + 2**-40  # smallest float64 nudge that changes the ratio
    out_above = compute_target(df_above, horizon_bars=1, neutral_atr_mult=1.0, atr_period=1)
    assert out_above.loc[0, "tradable_return"] > out_above.loc[0, "target_band"]
    assert out_above.loc[0, "target_class"] == CLASS_UP


# ---------------------------------------------------------------------------
# No bfill/leakage: verify NaNs are genuine NaNs, never silently filled
# ---------------------------------------------------------------------------

def test_atr_warmup_is_never_backfilled(synthetic_result):
    # A naive (buggy) implementation using bfill() would make rows 0-2 equal
    # to row 3's ATR value (0.25). Assert they are NaN, not 0.25.
    for i in range(ATR_PERIOD - 1):
        assert pd.isna(synthetic_result.loc[i, "atr"])
        assert synthetic_result.loc[i, "atr"] != pytest.approx(0.25)


def test_tail_target_is_never_forward_filled_from_last_valid_row(synthetic_result):
    n = len(synthetic_result)
    last_valid_class = synthetic_result.loc[n - HORIZON_BARS - 1, "target_class"]
    for i in range(n - HORIZON_BARS, n):
        cls = synthetic_result.loc[i, "target_class"]
        assert pd.isna(cls)
        # A ffill() bug would propagate last_valid_class into these rows.
        assert cls != last_valid_class or pd.isna(last_valid_class)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_missing_ohlc_column_raises():
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0]})  # no close
    with pytest.raises(TargetEngineeringError):
        compute_target(df, horizon_bars=1, neutral_atr_mult=0.3, atr_period=2)


@pytest.mark.parametrize("bad_horizon", [0, -1])
def test_invalid_horizon_bars_raises(bad_horizon):
    df = _synthetic_df()
    with pytest.raises(TargetEngineeringError):
        compute_target(df, horizon_bars=bad_horizon, neutral_atr_mult=0.3, atr_period=2)


@pytest.mark.parametrize("bad_mult", [0, -0.1])
def test_invalid_neutral_atr_mult_raises(bad_mult):
    df = _synthetic_df()
    with pytest.raises(TargetEngineeringError):
        compute_target(df, horizon_bars=1, neutral_atr_mult=bad_mult, atr_period=2)


@pytest.mark.parametrize("bad_period", [0, -1])
def test_invalid_atr_period_raises(bad_period):
    df = _synthetic_df()
    with pytest.raises(TargetEngineeringError):
        compute_target(df, horizon_bars=1, neutral_atr_mult=0.3, atr_period=bad_period)


def test_compute_target_does_not_mutate_input():
    df = _synthetic_df()
    original = df.copy(deep=True)
    _ = compute_target(df, horizon_bars=HORIZON_BARS, neutral_atr_mult=NEUTRAL_ATR_MULT, atr_period=ATR_PERIOD)
    pd.testing.assert_frame_equal(df, original)


def test_target_columns_lists_all_added_columns(synthetic_result):
    added = set(synthetic_result.columns) - set(_synthetic_df().columns)
    assert added == set(target_columns())


# ---------------------------------------------------------------------------
# Adversarial: huge single-bar move must still classify correctly even with
# a very small ATR (proves the comparison isn't accidentally using an
# absolute threshold or the wrong sign).
# ---------------------------------------------------------------------------

def test_large_move_with_tiny_atr_is_correctly_classified_up_and_down():
    # Extremely tight range for period, followed by a huge jump for the
    # tradable_return, isolated from the ATR window itself.
    close = [100.0, 100.01, 100.0, 100.01, 200.0, 100.0, 50.0]
    open_ = [100.0, 100.0, 100.01, 100.0, 100.01, 200.0, 100.0]
    high = [c + 0.01 for c in close]
    low = [c - 0.01 for c in close]
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})

    out = compute_target(df, horizon_bars=1, neutral_atr_mult=0.3, atr_period=3)

    # Row 3: entry=open[4]=100.01, exit=close[4]=200.0 -> huge positive return -> UP
    assert out.loc[3, "target_class"] == CLASS_UP
    # Row 4: entry=open[5]=200.0, exit=close[5]=100.0 -> huge negative return -> DOWN
    assert out.loc[4, "target_class"] == CLASS_DOWN
