"""Tests for position_engine.py (Stage 2L)."""
import numpy as np
import pandas as pd
import pytest

from position_engine import (
    FLAT, LONG, SHORT, LEDGER_COLUMNS, TIMELINE_COLUMNS,
    PositionEngineError, run_backtest,
)


def _bars(n=7):
    # open[i] = 100+i, close[i] = 100+i+0.5 -- simple, hand-computable series.
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "open": [100.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
    })


def _signals(entries, fold=1, n_bars=7):
    # entries: dict {bar_index: "BUY"/"SELL"}
    bars = _bars(n_bars)
    rows = []
    for i, sig in entries.items():
        rows.append({"timestamp": bars.loc[i, "timestamp"], "fold": fold, "signal": sig})
    return pd.DataFrame(rows, columns=["timestamp", "fold", "signal"])


# ---------------------------------------------------------------------------
# Single BUY trade, hand-derived exact accounting (horizon_bars=2)
# ---------------------------------------------------------------------------

def test_single_buy_trade_hand_derived_accounting():
    bars = _bars(7)
    signals = _signals({0: "BUY"})
    horizon_bars = 2
    initial_equity, position_fraction = 10_000.0, 0.5
    fee_bps, slippage_bps, latency_bps, funding_bps = 10.0, 5.0, 5.0, 1.0
    cost_rate = (fee_bps + slippage_bps + latency_bps) / 10_000.0  # 0.002
    funding_rate = funding_bps / 10_000.0  # 0.0001

    timeline, ledger = run_backtest(
        bars, signals, model="lightgbm", horizon_bars=horizon_bars,
        initial_equity=initial_equity, position_fraction=position_fraction,
        fee_bps=fee_bps, slippage_bps=slippage_bps, latency_bps=latency_bps,
        funding_bps_per_bar=funding_bps,
    )

    # Independently (not via the engine) derive expected values from the
    # frozen formulas: signal at bar0 -> entry at open[1]=101, exit at
    # close[0+2]=close[2]=102.5 (i.e. bar index 2).
    entry_price = 101.0
    exit_price = 102.5
    notional = initial_equity * position_fraction  # E0=10000 (equity unchanged before entry) * 0.5
    units = notional / entry_price
    entry_cost = notional * cost_rate
    exit_cost = notional * cost_rate
    gross_pnl_trade = units * (exit_price - entry_price)
    transaction_cost_trade = entry_cost + exit_cost
    funding_cost_trade = notional * funding_rate * horizon_bars
    net_pnl_trade = gross_pnl_trade - transaction_cost_trade - funding_cost_trade

    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert row["direction"] == LONG
    assert row["entry_price"] == pytest.approx(entry_price)
    assert row["exit_price"] == pytest.approx(exit_price)
    assert row["position_units"] == pytest.approx(units)
    assert row["position_notional"] == pytest.approx(notional)
    assert row["gross_pnl"] == pytest.approx(gross_pnl_trade)
    assert row["transaction_cost"] == pytest.approx(transaction_cost_trade)
    assert row["funding_cost"] == pytest.approx(funding_cost_trade)
    assert row["net_pnl"] == pytest.approx(net_pnl_trade)
    assert row["net_return"] == pytest.approx(net_pnl_trade / notional)
    assert row["holding_bars"] == horizon_bars
    assert row["signal_timestamp"] == bars.loc[0, "timestamp"]
    assert row["entry_timestamp"] == bars.loc[1, "timestamp"]
    assert row["exit_timestamp"] == bars.loc[2, "timestamp"]

    # Timeline: bars 0 = FLAT (scheduled only); bars 1,2 = LONG (held bars);
    # bars 3..6 = FLAT again.
    assert timeline.loc[0, "position"] == FLAT
    assert timeline.loc[1, "position"] == LONG
    assert timeline.loc[2, "position"] == LONG
    for i in range(3, 7):
        assert timeline.loc[i, "position"] == FLAT
        assert timeline.loc[i, "net_pnl"] == pytest.approx(0.0)

    assert float(timeline.loc[0:2, "net_pnl"].sum()) == pytest.approx(net_pnl_trade)
    final_equity = initial_equity + net_pnl_trade
    assert timeline.loc[6, "equity_after"] == pytest.approx(final_equity)


# ---------------------------------------------------------------------------
# Structural contracts
# ---------------------------------------------------------------------------

def test_timeline_and_ledger_have_frozen_columns():
    timeline, ledger = run_backtest(_bars(7), _signals({0: "BUY"}), "lightgbm", horizon_bars=2)
    assert list(timeline.columns) == TIMELINE_COLUMNS
    assert list(ledger.columns) == LEDGER_COLUMNS


def test_reconciliation_total_equity_change_equals_ledger_net_pnl_sum():
    # FLAT bars always contribute net_pnl == 0, so total equity delta must
    # equal the sum of ledger net_pnl exactly, regardless of trade count.
    bars = _bars(30)
    signals = _signals({0: "BUY", 10: "SELL", 20: "BUY"}, n_bars=30)
    timeline, ledger = run_backtest(
        bars, signals, "lightgbm", horizon_bars=3, initial_equity=5000.0, position_fraction=0.3,
        fee_bps=8.0, slippage_bps=3.0, latency_bps=2.0, funding_bps_per_bar=0.5,
    )
    total_equity_delta = timeline["equity_after"].iloc[-1] - 5000.0
    assert total_equity_delta == pytest.approx(ledger["net_pnl"].sum())
    # Also: sum of per-bar net_pnl over the whole timeline must equal the
    # same total (definition of equity_after as a running sum).
    assert timeline["net_pnl"].sum() == pytest.approx(total_equity_delta)


def test_short_trade_direction_and_sign_of_gross_pnl():
    bars = _bars(7)
    signals = _signals({0: "SELL"})
    timeline, ledger = run_backtest(bars, signals, "lightgbm", horizon_bars=2, position_fraction=1.0)
    row = ledger.iloc[0]
    assert row["direction"] == SHORT
    # Prices rise monotonically in this fixture -> a SHORT loses money.
    assert row["gross_pnl"] < 0


# ---------------------------------------------------------------------------
# No overlap / no pyramiding / no early reversal
# ---------------------------------------------------------------------------

def test_signals_while_in_position_are_ignored_no_pyramiding():
    bars = _bars(20)
    # BUY signal fires at EVERY bar; horizon_bars=5 means only one trade
    # can be open at a time -> far fewer trades than signals.
    signals = _signals({i: "BUY" for i in range(15)}, n_bars=20)
    timeline, ledger = run_backtest(bars, signals, "lightgbm", horizon_bars=5)
    assert len(ledger) < 15
    # No two trades' [entry_timestamp, exit_timestamp] windows overlap.
    entries = ledger.sort_values("entry_timestamp").reset_index(drop=True)
    for i in range(len(entries) - 1):
        assert entries.loc[i, "exit_timestamp"] <= entries.loc[i + 1, "entry_timestamp"]


def test_reversal_signal_during_open_position_is_ignored_not_early_exit():
    bars = _bars(10)
    signals = _signals({0: "BUY", 1: "SELL", 2: "SELL"})  # both fire while LONG is open
    timeline, ledger = run_backtest(bars, signals, "lightgbm", horizon_bars=4)
    assert len(ledger) == 1
    assert ledger.iloc[0]["direction"] == LONG
    assert ledger.iloc[0]["exit_timestamp"] == bars.loc[4, "timestamp"]  # exits exactly at t+h, not early


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_missing_signal_timestamp_in_bars_raises():
    bars = _bars(5)
    bad_signals = pd.DataFrame([{
        "timestamp": pd.Timestamp("2099-01-01", tz="UTC"), "fold": 1, "signal": "BUY",
    }])
    with pytest.raises(PositionEngineError):
        run_backtest(bars, bad_signals, "lightgbm", horizon_bars=2)


def test_invalid_horizon_bars_raises():
    with pytest.raises(PositionEngineError):
        run_backtest(_bars(5), _signals({}), "lightgbm", horizon_bars=0)


def test_invalid_position_fraction_raises():
    with pytest.raises(PositionEngineError):
        run_backtest(_bars(5), _signals({}), "lightgbm", horizon_bars=1, position_fraction=1.5)


def test_signal_too_close_to_end_of_data_is_not_actionable():
    # horizon_bars=2 needs exit_index < n; a signal at the last bar can
    # never be fully executed and must be silently skipped (not crash,
    # not truncated-trade).
    bars = _bars(5)
    signals = _signals({4: "BUY"})
    timeline, ledger = run_backtest(bars, signals, "lightgbm", horizon_bars=2)
    assert len(ledger) == 0


def test_hold_signal_never_opens_a_position():
    bars = _bars(10)
    signals = _signals({0: "HOLD", 3: "HOLD"})
    timeline, ledger = run_backtest(bars, signals, "lightgbm", horizon_bars=2)
    assert len(ledger) == 0
    assert (timeline["position"] == FLAT).all()
