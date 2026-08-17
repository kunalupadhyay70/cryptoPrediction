"""Stage 2E/2F — Causal, OHLCV-only feature pipeline + target integration.

This module is the FINAL-architecture replacement for
``feature_engineering.py``'s dataset-building responsibility. It is
deliberately isolated (imported by nothing in the legacy runtime, imports
nothing from the legacy runtime) until the Stage 2P cutover.

What this module intentionally does NOT do, versus the legacy
``feature_engineering.py`` (frozen architecture, Stage 2E/2F requirements):
  - No historical order-book / trade-flow microstructure features. V1 is
    OHLCV-only: the repository has no real historical microstructure
    dataset (Stage 0 finding — the legacy order_book_snapshots/trades
    tables are populated only during live collection and are mostly zero
    placeholders historically), so training on them would be training on
    fabricated signal. Live order-book collection code
    (DataCollector.collect_orderbook_snapshot / collect_orderbook_ws)
    remains untouched and isolated for possible future work — this module
    simply never reads those tables.
  - No global-percentile target ("vol_breakout" mode: a percentile computed
    over the ENTIRE dataset, including future rows relative to any given
    training cutoff — a direct leakage channel) and no legacy binary
    dead_zone/regime_conditioned target modes. The only target this module
    produces is the frozen causal 3-class target from ``target_engineering.
    compute_target`` (single source of truth for target construction — this
    module never recomputes ATR/band/class itself; see ``build_dataset``).
  - No ``bfill()`` anywhere, for either features or the target. Warm-up rows
    for any rolling/EMA feature are left as genuine NaN, never back-filled
    with a later (future) value. This module does not use ``ffill()``
    either, to keep the "every feature value at row t depends only on rows
    <= t" invariant simple to state and simple to test — see
    ``tests/test_dataset_builder.py``'s leakage-probe tests, which perturb
    future rows and assert earlier feature rows are bit-for-bit unchanged.

Causal ATR is computed via ``target_engineering.compute_causal_atr`` (not
reimplemented here) so there is exactly one ATR formula in the codebase,
reused for both the ``atr_14`` feature and the target's band computation.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from target_engineering import compute_causal_atr, compute_target, target_columns

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

# Non-feature columns that may be present in the input/output frame and must
# never be treated as model features.
NON_FEATURE_COLUMNS = {
    "open_time", "close_time", "symbol", "interval",
    "open", "high", "low", "close", "volume",
    *target_columns(),
}


class DatasetBuilderError(ValueError):
    """Raised for structurally invalid input (missing OHLCV columns, etc.)."""


def _ema(s: pd.Series, span: int) -> pd.Series:
    # ewm(adjust=False) is causal: value at row t is a function of rows <= t
    # only (an exponentially-weighted recursive average), never a future row.
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = (
        up.ewm(alpha=1 / period, adjust=False).mean()
        / dn.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    )
    return 100 - 100 / (1 + rs)


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std()
    return (s - mu) / (sd + 1e-9)


def build_features(
    df: pd.DataFrame,
    lag_periods: int = 3,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Compute a causal, OHLCV-only feature set on a COPY of ``df``.

    ``df`` must contain at minimum the OHLCV columns; an ``open_time``
    column (tz-aware or naive datetime64) is used for time-of-day /
    day-of-week features if present, and is otherwise skipped (those
    columns simply won't be added).

    Every feature here is one of:
      - a `.shift(k)` / `.pct_change(k)` for k >= 0 (backward-looking only)
      - a `.rolling(window, min_periods=window)` aggregate (backward-looking
        window, ``min_periods=window`` so a partially-full window never
        silently produces an under-sampled value — it stays NaN until the
        window is genuinely full, mirroring target_engineering's ATR
        warm-up rule)
      - a `.ewm(adjust=False)` recursive average (backward-looking by
        construction)
      - a deterministic function of the row's own timestamp (time-of-day /
        day-of-week — no dependency on any other row at all)

    No feature here reads ``.shift(-k)`` for any k, and no fill method is
    ever applied to a feature column. Both are enforced by
    tests/test_dataset_builder.py's leakage-probe and no-bfill tests, not
    just by this docstring.
    """
    for col in _OHLCV_COLUMNS:
        if col not in df.columns:
            raise DatasetBuilderError(f"build_features: missing required column {col!r}")
    if lag_periods < 0:
        raise DatasetBuilderError(f"build_features: lag_periods must be >= 0, got {lag_periods}")

    out = df.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)
    ret1 = close.pct_change()

    # ── Returns & volatility ────────────────────────────────────────────
    for p in (1, 3, 5, 10, 20):
        out[f"ret_{p}"] = close.pct_change(p)
        out[f"vol_{p}"] = ret1.rolling(p, min_periods=p).std()
        out[f"ema_dist_{p}"] = close / _ema(close, p) - 1
        out[f"sma_dist_{p}"] = close / close.rolling(p, min_periods=p).mean() - 1

    # ── Rolling z-scores (regime-neutral) ───────────────────────────────
    out["ret1_z20"] = _rolling_zscore(ret1, 20)
    out["vol5_z20"] = _rolling_zscore(out["vol_5"], 20)

    # ── Momentum / oscillators ──────────────────────────────────────────
    for period in (7, 14, 21):
        out[f"rsi_{period}"] = _rsi(close, period)
    ema12, ema26 = _ema(close, 12), _ema(close, 26)
    out["macd"] = ema12 - ema26
    out["macd_signal"] = _ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    for kp in (14, 21):
        lo = out["low"].rolling(kp, min_periods=kp).min()
        hi = out["high"].rolling(kp, min_periods=kp).max()
        out[f"stoch_k_{kp}"] = 100 * (close - lo) / (hi - lo + 1e-9)

    # ── ATR (single source of truth: target_engineering.compute_causal_atr) ──
    out["atr_14"] = compute_causal_atr(out, period=atr_period)
    out["atr_7"] = compute_causal_atr(out, period=max(1, atr_period // 2))
    out["atr_ratio"] = out["atr_7"] / (out["atr_14"] + 1e-9)

    # ── Bollinger ────────────────────────────────────────────────────────
    for bbp in (20, 50):
        bm = close.rolling(bbp, min_periods=bbp).mean()
        bs = close.rolling(bbp, min_periods=bbp).std()
        out[f"bb_width_{bbp}"] = (2 * bs) / (bm + 1e-9)
        out[f"bb_pctb_{bbp}"] = (close - (bm - 2 * bs)) / (4 * bs + 1e-9)

    # ── Volume ───────────────────────────────────────────────────────────
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    out["vol_ratio_20"] = volume / (vol_ma20 + 1e-9)
    out["vol_z_20"] = _rolling_zscore(volume, 20)

    # ── Candle structure ─────────────────────────────────────────────────
    out["body"] = (out["close"] - out["open"]).abs() / (close + 1e-9)
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / (close + 1e-9)
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / (close + 1e-9)
    out["body_direction"] = np.sign(out["close"] - out["open"])

    # ── Momentum across horizons ─────────────────────────────────────────
    for p in (3, 5, 10, 20):
        out[f"mom_{p}"] = close / close.shift(p) - 1

    # ── Time-of-day / day-of-week (deterministic, no cross-row dependency) ──
    if "open_time" in out.columns:
        ot = pd.to_datetime(out["open_time"], utc=True)
        minutes_in_day = ot.dt.hour * 60 + ot.dt.minute
        out["tod_sin"] = np.sin(2 * np.pi * minutes_in_day / 1440)
        out["tod_cos"] = np.cos(2 * np.pi * minutes_in_day / 1440)
        out["dow_sin"] = np.sin(2 * np.pi * ot.dt.dayofweek / 7)
        out["dow_cos"] = np.cos(2 * np.pi * ot.dt.dayofweek / 7)

    # ── Lagged features (causal: shift(+k) only looks backward) ─────────
    lag_base_cols = ["ret_1", "ret_3", "vol_5", "rsi_14", "macd_hist", "body_direction"]
    if lag_periods > 0:
        lag_dict = {
            f"{col}_lag{lag}": out[col].shift(lag)
            for col in lag_base_cols
            if col in out.columns
            for lag in range(1, lag_periods + 1)
        }
        out = pd.concat([out, pd.DataFrame(lag_dict, index=out.index)], axis=1)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Column names eligible as model features — everything except raw
    OHLCV/identifier columns and target_engineering's target columns."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def build_dataset(
    df: pd.DataFrame,
    horizon_bars: int,
    neutral_atr_mult: float,
    atr_period: int = 14,
    lag_periods: int = 3,
) -> pd.DataFrame:
    """Full Stage 2E dataset: causal OHLCV-only features + the frozen 3-class
    causal target, in one call.

    No rows are filtered by this function — neutral rows, ATR warm-up rows,
    and un-labelable tail rows are all present in the output (labeled rows
    have a non-null ``target_class``; the rest have ``target_class`` NaN).
    Deciding whether/how to drop unlabeled rows before model fitting is a
    later stage's responsibility (Stage 2H/2I), not this module's.
    """
    features = build_features(df, lag_periods=lag_periods, atr_period=atr_period)
    dataset = compute_target(
        features, horizon_bars=horizon_bars, neutral_atr_mult=neutral_atr_mult, atr_period=atr_period
    )
    return dataset
