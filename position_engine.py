"""Stage 2L — FLAT/LONG/SHORT position state machine + fixed-notional
accounting engine producing the canonical equity timeline and trade ledger.

Frozen contract (restated):
  States: FLAT / LONG / SHORT. One position max. No overlapping,
  pyramiding, or early reversal.

  Per-bar event order (7 steps): (1) close any position whose exit is
  scheduled for this bar -> FLAT, (2) state is now known, (3) observe
  features, (4) read this bar's signal, (5) if FLAT, schedule an entry,
  (6) execution happens at open[t+1] (the NEXT bar), (7) a trade opened
  from a signal at bar t exits at close[t+h] (h = horizon_bars).

  notional = E0 * position_fraction, where E0 is the account equity
  immediately before the trade is opened (documented interpretation of
  the frozen "E0" symbol -- NOT a single fixed initial-equity constant,
  since notional must reflect prevailing equity for realistic
  compounding). units = notional / entry_price.
  gross_pnl_i = direction * units * (close[i] - previous_mark), applied
  per held bar (mark-to-market), i.e. NOT a single lump sum at exit.
  costs = (fee_bps + slippage_bps + latency_bps) / 10000, one-way,
  charged on notional at both entry and exit. Funding is charged per
  held bar on notional; symmetric for LONG and SHORT (documented
  simplification -- real funding is asymmetric by side and regime).

This module deliberately does NOT reuse dataset_builder's precomputed
next_open / exit_close OOF columns for execution prices. It walks the
full underlying bar series (open, close) directly and locates entry/exit
bars by fixed positional offset from the signal bar (i+1 for entry,
i+horizon_bars for exit) -- this is a single source of truth for prices
(the bars series itself) and, by construction, produces exit prices that
are numerically identical to dataset_builder's exit_close for any
contiguous bar series (both are just "close of the bar h positions
later"), which is exercised directly in
tests/test_position_engine.py::test_engine_exit_price_matches_oof_exit_close_precomputation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

FLAT, LONG, SHORT = "FLAT", "LONG", "SHORT"
DIRECTION_LONG, DIRECTION_SHORT = 1, -1

TIMELINE_COLUMNS = [
    "timestamp", "position", "position_units", "position_notional",
    "gross_pnl", "transaction_cost", "funding_cost", "net_pnl",
    "equity_before", "equity_after", "net_strategy_return",
]

LEDGER_COLUMNS = [
    "signal_timestamp", "entry_timestamp", "exit_timestamp", "direction",
    "equity_at_entry", "position_notional", "position_units", "entry_price", "exit_price",
    "gross_pnl", "gross_return", "transaction_cost", "funding_cost", "net_pnl", "net_return",
    "holding_bars", "fold", "model",
]


class PositionEngineError(ValueError):
    pass


@dataclass
class _OpenPosition:
    direction: int
    entry_price: float
    entry_timestamp: object
    exit_index: int
    units: float
    notional: float
    signal_timestamp: object
    fold: object
    previous_mark: float
    equity_at_entry: float


@dataclass
class _PendingEntry:
    direction: int
    entry_index: int
    exit_index: int
    signal_timestamp: object
    fold: object


def run_backtest(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    model: str,
    horizon_bars: int,
    initial_equity: float = 10_000.0,
    position_fraction: float = 1.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    latency_bps: float = 1.0,
    funding_bps_per_bar: float = 0.0,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Bar-by-bar walk producing the equity timeline and trade ledger for a
    single model's signals.

    ``bars``: full contiguous OHLC(V) series, columns [timestamp, open,
      close], sorted ascending, unique timestamps. Must be positionally
      contiguous (bar i+1 is the immediate next bar after bar i) -- the
      same assumption already made throughout target_engineering /
      dataset_builder (exit_price = close.shift(-h)).
    ``signals``: one row per (timestamp) for THIS model only, columns
      [timestamp, fold, signal] with signal in {"BUY", "SELL", "HOLD"}
      (or None/NaN, treated as HOLD). Every timestamp must be present in
      ``bars``.
    """
    if horizon_bars < 1:
        raise PositionEngineError(f"horizon_bars must be >= 1, got {horizon_bars}")
    if not (0 < position_fraction <= 1):
        raise PositionEngineError(f"position_fraction must be in (0, 1], got {position_fraction}")
    required_bar_cols = {"timestamp", "open", "close"}
    if not required_bar_cols.issubset(bars.columns):
        raise PositionEngineError(f"bars is missing required columns: {required_bar_cols - set(bars.columns)}")
    required_signal_cols = {"timestamp", "fold", "signal"}
    if not required_signal_cols.issubset(signals.columns):
        raise PositionEngineError(f"signals is missing required columns: {required_signal_cols - set(signals.columns)}")

    bars = bars.sort_values("timestamp").reset_index(drop=True)
    if bars["timestamp"].duplicated().any():
        raise PositionEngineError("bars contains duplicate timestamps")

    ts_to_index = {ts: i for i, ts in enumerate(bars["timestamp"])}
    missing = [ts for ts in signals["timestamp"] if ts not in ts_to_index]
    if missing:
        raise PositionEngineError(f"{len(missing)} signal timestamp(s) not found in bars, e.g. {missing[:3]}")

    signal_by_index = {}
    for _, row in signals.iterrows():
        signal_by_index[ts_to_index[row["timestamp"]]] = row

    cost_rate = (fee_bps + slippage_bps + latency_bps) / 10_000.0
    funding_rate = funding_bps_per_bar / 10_000.0

    equity = float(initial_equity)
    position: Optional[_OpenPosition] = None
    pending: Optional[_PendingEntry] = None

    timeline_rows = []
    ledger_rows = []
    n = len(bars)

    for i in range(n):
        ts = bars.loc[i, "timestamp"]
        o = float(bars.loc[i, "open"])
        c = float(bars.loc[i, "close"])
        equity_before = equity

        gross_pnl = 0.0
        transaction_cost = 0.0
        funding_cost = 0.0
        pos_label, pos_units, pos_notional = FLAT, 0.0, 0.0

        # Step (6): execute a pending entry scheduled for this bar's open.
        if pending is not None and pending.entry_index == i:
            entry_price = o
            notional = equity_before * position_fraction
            units = notional / entry_price
            entry_cost = notional * cost_rate
            transaction_cost += entry_cost
            position = _OpenPosition(
                direction=pending.direction, entry_price=entry_price, entry_timestamp=ts,
                exit_index=pending.exit_index, units=units, notional=notional,
                signal_timestamp=pending.signal_timestamp, fold=pending.fold,
                previous_mark=entry_price, equity_at_entry=equity_before,
            )
            pending = None

        # Mark-to-market any open position (including one just opened this bar).
        if position is not None:
            pos_label = LONG if position.direction == DIRECTION_LONG else SHORT
            pos_units, pos_notional = position.units, position.notional
            gross_pnl = position.direction * position.units * (c - position.previous_mark)
            funding_cost = position.notional * funding_rate
            position.previous_mark = c

        # Step (1)/(7): close a position whose exit is scheduled for this bar.
        if position is not None and i == position.exit_index:
            exit_cost = position.notional * cost_rate
            transaction_cost += exit_cost
            exit_price = c
            gross_pnl_trade = position.direction * position.units * (exit_price - position.entry_price)
            transaction_cost_trade = position.notional * cost_rate * 2.0
            funding_cost_trade = position.notional * funding_rate * horizon_bars
            net_pnl_trade = gross_pnl_trade - transaction_cost_trade - funding_cost_trade
            ledger_rows.append({
                "signal_timestamp": position.signal_timestamp,
                "entry_timestamp": position.entry_timestamp,
                "exit_timestamp": ts,
                "direction": LONG if position.direction == DIRECTION_LONG else SHORT,
                "equity_at_entry": position.equity_at_entry,
                "position_notional": position.notional,
                "position_units": position.units,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl_trade,
                "gross_return": gross_pnl_trade / position.notional,
                "transaction_cost": transaction_cost_trade,
                "funding_cost": funding_cost_trade,
                "net_pnl": net_pnl_trade,
                "net_return": net_pnl_trade / position.notional,
                "holding_bars": horizon_bars,
                "fold": position.fold,
                "model": model,
            })
            position = None  # FLAT from this point on

        net_pnl = gross_pnl - transaction_cost - funding_cost
        equity_after = equity_before + net_pnl
        equity = equity_after

        # Step (3)/(4)/(5): read signal, schedule entry only if FLAT.
        if position is None and pending is None and i in signal_by_index:
            sig_row = signal_by_index[i]
            sig = sig_row["signal"]
            if sig in ("BUY", "SELL"):
                entry_index, exit_index = i + 1, i + horizon_bars
                if exit_index < n:  # not enough future bars -> signal not actionable
                    pending = _PendingEntry(
                        direction=DIRECTION_LONG if sig == "BUY" else DIRECTION_SHORT,
                        entry_index=entry_index, exit_index=exit_index,
                        signal_timestamp=ts, fold=sig_row["fold"],
                    )

        timeline_rows.append({
            "timestamp": ts, "position": pos_label, "position_units": pos_units,
            "position_notional": pos_notional, "gross_pnl": gross_pnl,
            "transaction_cost": transaction_cost, "funding_cost": funding_cost,
            "net_pnl": net_pnl, "equity_before": equity_before, "equity_after": equity_after,
            "net_strategy_return": (net_pnl / equity_before) if equity_before != 0 else 0.0,
        })

    timeline_df = pd.DataFrame(timeline_rows, columns=TIMELINE_COLUMNS)
    ledger_df = pd.DataFrame(ledger_rows, columns=LEDGER_COLUMNS)
    return timeline_df, ledger_df
