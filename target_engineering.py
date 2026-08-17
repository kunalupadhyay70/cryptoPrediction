"""Stage 2D — Canonical causal, ATR-scaled 3-class target (frozen architecture).

This module is a deliberately ISOLATED implementation of the target defined
by the frozen architecture (Stage 1B/1C) and the Stage 2D task spec. It does
NOT import from, or get imported by, any legacy runtime module
(feature_engineering.py, model.py, main.py) yet — that wiring happens only
in the later runtime-cutover stage. Nothing here mutates any existing file.

TARGET DEFINITION (frozen, reproduced here for a single source of truth):

    For signal row t, with h = horizon_bars:
        entry[t] = open[t+1]
        exit[t]  = close[t+h]
        tradable_return[t] = exit[t] / entry[t] - 1

    Causal ATR (no look-ahead — uses only rows <= t):
        TR[t]  = max(high[t]-low[t], |high[t]-close[t-1]|, |low[t]-close[t-1]|)
        ATR[t] = rolling mean of TR over the trailing `atr_period` bars
                 ending at t (pandas .rolling(atr_period, min_periods=atr_period))

    Band:
        band[t] = neutral_atr_mult * ATR[t] / close[t]

    Class:
        tradable_return[t] >  band[t]  -> UP      (2)
        tradable_return[t] < -band[t]  -> DOWN    (0)
        otherwise                       -> NEUTRAL (1)

Class mapping is frozen: 0 = DOWN, 1 = NEUTRAL, 2 = UP.

LEAKAGE / MISSING-DATA RULES (frozen, enforced by this implementation):
  - Neutral rows MUST remain in the output — never filtered out.
  - Rows are never dropped/filtered here based on future realized movement;
    this module only ever ADDS columns to a copy of the input frame.
  - Tail rows without an available t+h close (the last `horizon_bars` rows,
    and the very last row for entry[t]=open[t+1]) get NaN tradable_return
    and therefore an unlabeled (NaN) target_class — never a fabricated
    label.
  - ATR warm-up rows (the first `atr_period - 1` rows, where the rolling
    window is not yet full) get NaN ATR and therefore an unlabeled (NaN)
    target_class — never a fabricated label, never forward/back-filled.
  - ``bfill()`` (or any future-looking fill) is never used anywhere in this
    module to populate a causal feature. ATR/band/target_class are left as
    genuine NaN wherever the causal computation cannot yet produce a value.

INDEXING ASSUMPTION: this module operates on ROW POSITION (bar count), not
wall-clock time — ``open[t+1]`` and ``close[t+h]`` mean "the row `1`/`h`
positions after row t in the given, already-sorted, contiguous DataFrame".
Gap detection/repair for the underlying OHLCV series is the responsibility
of the data-collection layer (Stage 2C's interval-aware integrity check);
this module assumes it is handed a chronologically sorted, one-row-per-bar
frame and does not itself validate bar spacing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Frozen class mapping.
CLASS_DOWN = 0
CLASS_NEUTRAL = 1
CLASS_UP = 2

CLASS_LABELS = {CLASS_DOWN: "DOWN", CLASS_NEUTRAL: "NEUTRAL", CLASS_UP: "UP"}


class TargetEngineeringError(ValueError):
    """Raised when the input DataFrame is structurally unsuitable for target
    construction (missing required OHLC columns, empty, etc.)."""


_REQUIRED_COLUMNS = ("open", "high", "low", "close")


def compute_causal_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Causal (no look-ahead) Average True Range.

    ``TR[t] = max(high[t]-low[t], |high[t]-close[t-1]|, |low[t]-close[t-1]|)``
    ``ATR[t]`` is the rolling mean of TR over the trailing ``period`` bars
    ending at (and including) row t — i.e. it uses only rows <= t, never a
    future row.

    ``min_periods=period`` is used deliberately (rather than a smaller
    warm-up-tolerant value): the first ``period - 1`` rows have no fully
    populated trailing window and MUST report NaN rather than an
    under-sampled approximation, per the frozen "ATR warm-up rows must
    remain missing" rule.

    Row 0's TR has no ``close[t-1]`` at all; consistent with the standard
    True-Range convention, ``TR[0]`` reduces to ``high[0] - low[0]`` (the
    two ``|.. - close[t-1]|`` terms are NaN and are skipped by pandas'
    default ``skipna=True`` row-wise max — this is a property of a single
    boundary row and does not resurface later as a look-ahead: TR[0] is
    still computed only from row 0's own open/high/low/close).
    """
    for col in _REQUIRED_COLUMNS:
        if col not in df.columns:
            raise TargetEngineeringError(f"compute_causal_atr: missing required column {col!r}")
    if period < 1:
        raise TargetEngineeringError(f"compute_causal_atr: period must be >= 1, got {period}")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(window=period, min_periods=period).mean()
    atr.name = "atr"
    return atr


def compute_target(
    df: pd.DataFrame,
    horizon_bars: int,
    neutral_atr_mult: float,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Compute the causal, ATR-scaled 3-class target on a copy of ``df``.

    Parameters mirror ``config_schema.TargetConfig`` field names exactly
    (``horizon_bars``, ``neutral_atr_mult``, ``atr_period``) so future
    wiring from the canonical config is a direct pass-through, without this
    module importing config_schema itself.

    Returns a COPY of ``df`` (input is never mutated) with these columns
    added:
        entry_price     — open[t+1]
        exit_price      — close[t+horizon_bars]
        tradable_return — exit_price / entry_price - 1
        atr             — causal ATR(atr_period), see compute_causal_atr
        target_band     — neutral_atr_mult * atr / close
        target_class    — float column holding {0.0, 1.0, 2.0} or NaN
                           (NaN = unlabeled: ATR warm-up or tail row).
                           Float (not a plain int dtype) is used
                           deliberately so NaN can represent "unlabeled"
                           without an artificial sentinel class value.
        target_label     — object column holding {"DOWN","NEUTRAL","UP"} or
                            None, mirroring target_class via CLASS_LABELS.

    No rows are ever dropped or filtered by this function — every row of
    the input appears in the output, labeled or not.
    """
    for col in _REQUIRED_COLUMNS:
        if col not in df.columns:
            raise TargetEngineeringError(f"compute_target: missing required column {col!r}")
    if horizon_bars < 1:
        raise TargetEngineeringError(f"compute_target: horizon_bars must be >= 1, got {horizon_bars}")
    if neutral_atr_mult <= 0:
        raise TargetEngineeringError(
            f"compute_target: neutral_atr_mult must be > 0, got {neutral_atr_mult}"
        )
    if atr_period < 1:
        raise TargetEngineeringError(f"compute_target: atr_period must be >= 1, got {atr_period}")

    out = df.copy()
    open_ = out["open"].astype(float)
    close = out["close"].astype(float)

    entry_price = open_.shift(-1)
    exit_price = close.shift(-horizon_bars)
    tradable_return = exit_price / entry_price - 1

    atr = compute_causal_atr(out, period=atr_period)
    target_band = neutral_atr_mult * atr / close

    # A row is only labelable once BOTH the forward-looking return and the
    # backward-looking ATR band are available. Using `&` (not `|`) here is
    # what enforces "ATR warm-up rows remain missing" and "tail rows
    # without t+h remain unlabeled" simultaneously — either condition alone
    # is enough to leave a row unlabeled.
    valid = tradable_return.notna() & target_band.notna()
    up_mask = valid & (tradable_return > target_band)
    down_mask = valid & (tradable_return < -target_band)
    neutral_mask = valid & ~up_mask & ~down_mask  # includes exact-tie rows

    target_class = pd.Series(np.nan, index=out.index, dtype="float64")
    target_class[up_mask] = float(CLASS_UP)
    target_class[down_mask] = float(CLASS_DOWN)
    target_class[neutral_mask] = float(CLASS_NEUTRAL)

    # Built as an explicit object-dtype numpy array (not via Series.map())
    # so that unlabeled rows hold a genuine Python None rather than being
    # silently normalized to a NaN-like missing marker by pandas' newer
    # string-dtype inference (observed on pandas>=3: constructing a Series
    # of str/None via .map() or a bare object array upcasts to the "str"
    # extension dtype, which coerces None to its own float-NaN sentinel).
    label_values = np.empty(len(out), dtype=object)
    label_values[:] = None
    for cls_value, name in CLASS_LABELS.items():
        label_values[(target_class == float(cls_value)).to_numpy()] = name
    target_label = pd.Series(label_values, index=out.index, dtype=object)

    out["entry_price"] = entry_price
    out["exit_price"] = exit_price
    out["tradable_return"] = tradable_return
    out["atr"] = atr
    out["target_band"] = target_band
    out["target_class"] = target_class
    out["target_label"] = target_label
    return out


def target_columns() -> list:
    """Column names added by compute_target(), for callers that need to
    exclude them from a feature-column list (mirrors the pattern used by
    feature_engineering.feature_columns())."""
    return [
        "entry_price", "exit_price", "tradable_return",
        "atr", "target_band", "target_class", "target_label",
    ]
