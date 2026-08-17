"""Tests for metrics.py (Stage 2M)."""
import numpy as np
import pandas as pd
import pytest

from metrics import (
    MetricsError, compute_passive_long_benchmark, compute_predictive_metrics, compute_trading_metrics,
)
from position_engine import LEDGER_COLUMNS, TIMELINE_COLUMNS, run_backtest


# ---------------------------------------------------------------------------
# Predictive metrics — hand-derived on a tiny, exactly-known confusion case
# ---------------------------------------------------------------------------

def test_predictive_metrics_perfect_classifier():
    actual = np.array([0, 1, 2, 0, 1, 2])
    pred = actual.copy()
    probs_down = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    probs_neutral = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    probs_up = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    result = compute_predictive_metrics(actual, pred, probs_down, probs_neutral, probs_up)
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert result.macro_roc_auc == pytest.approx(1.0)
    np.testing.assert_array_equal(result.confusion_matrix, np.diag([2, 2, 2]))
    assert result.class_distribution == {0: 2, 1: 2, 2: 2}
    for c in (0, 1, 2):
        assert result.precision_per_class[c] == pytest.approx(1.0)
        assert result.recall_per_class[c] == pytest.approx(1.0)


def test_predictive_metrics_hand_derived_confusion_matrix():
    # actual: [0,0,1,1,2,2]; pred: [0,1,1,1,2,0] -- picked so the confusion
    # matrix, precision, and recall can be hand-counted.
    actual = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    # Deterministic one-hot probs matching `pred` (as a real baseline would produce).
    probs = np.zeros((6, 3))
    probs[np.arange(6), pred] = 1.0
    result = compute_predictive_metrics(actual, pred, probs[:, 0], probs[:, 1], probs[:, 2])
    # Confusion[actual, pred]: class0 actual (rows0,1) -> preds 0,1 => row0=[1,1,0]
    # class1 actual (rows2,3) -> preds 1,1 => row1=[0,2,0]
    # class2 actual (rows4,5) -> preds 2,0 => row2=[1,0,1]
    expected_cm = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    np.testing.assert_array_equal(result.confusion_matrix, expected_cm)
    # precision class1 = TP/(TP+FP) = 2/(1+2+0)=2/3 (preds==1 count: rows1,2,3 -> 3 total, 2 correct)
    assert result.precision_per_class[1] == pytest.approx(2 / 3)
    # recall class0 = TP/(actual0 count) = 1/2
    assert result.recall_per_class[0] == pytest.approx(0.5)
    assert result.class_distribution == {0: 2, 1: 2, 2: 2}


def test_predictive_metrics_empty_raises():
    with pytest.raises(MetricsError):
        compute_predictive_metrics(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# Trading metrics — hand-derived on a tiny synthetic timeline/ledger
# ---------------------------------------------------------------------------

def _hand_timeline():
    # 4 bars, all on the SAME day, equity_before starts at 1000.
    # net_strategy_return per bar: +1%, -2%, +1%, 0%
    rows = []
    equity = 1000.0
    rets = [0.01, -0.02, 0.01, 0.0]
    ts0 = pd.Timestamp("2024-01-01T00:00:00Z")
    for i, r in enumerate(rets):
        equity_before = equity
        net_pnl = equity_before * r
        equity_after = equity_before + net_pnl
        rows.append({
            "timestamp": ts0 + pd.Timedelta(minutes=5 * i), "position": "FLAT", "position_units": 0.0,
            "position_notional": 0.0, "gross_pnl": net_pnl, "transaction_cost": 0.0, "funding_cost": 0.0,
            "net_pnl": net_pnl, "equity_before": equity_before, "equity_after": equity_after,
            "net_strategy_return": r,
        })
        equity = equity_after
    return pd.DataFrame(rows, columns=TIMELINE_COLUMNS)


def test_trading_metrics_total_return_and_max_drawdown_hand_derived():
    timeline = _hand_timeline()
    empty_ledger = pd.DataFrame(columns=LEDGER_COLUMNS)
    result = compute_trading_metrics(timeline, empty_ledger, n_actionable_signals=0)

    # Equity path: 1000 -> 1010 -> 989.8 -> 999.698 -> 999.698
    expected_final_equity = 1000 * 1.01 * 0.98 * 1.01 * 1.00
    assert result.total_return == pytest.approx(expected_final_equity / 1000 - 1)

    # Max drawdown: peak=1010 (after bar0), trough=989.8 (after bar1) -> dd=(989.8-1010)/1010
    expected_dd = (989.8 - 1010.0) / 1010.0
    assert result.max_drawdown == pytest.approx(expected_dd, abs=1e-6)

    assert result.trade_count == 0
    assert np.isnan(result.win_rate)


def test_trading_metrics_win_rate_and_profit_factor_hand_derived():
    ledger = pd.DataFrame([
        {"signal_timestamp": None, "entry_timestamp": None, "exit_timestamp": None, "direction": "LONG",
         "equity_at_entry": 1000.0, "position_notional": 500.0, "position_units": 5.0,
         "entry_price": 100.0, "exit_price": 110.0, "gross_pnl": 50.0, "gross_return": 0.1,
         "transaction_cost": 2.0, "funding_cost": 0.5, "net_pnl": 47.5, "net_return": 0.095,
         "holding_bars": 3, "fold": 1, "model": "m"},
        {"signal_timestamp": None, "entry_timestamp": None, "exit_timestamp": None, "direction": "SHORT",
         "equity_at_entry": 1047.5, "position_notional": 500.0, "position_units": 5.0,
         "entry_price": 100.0, "exit_price": 105.0, "gross_pnl": -25.0, "gross_return": -0.05,
         "transaction_cost": 2.0, "funding_cost": 0.5, "net_pnl": -27.5, "net_return": -0.055,
         "holding_bars": 3, "fold": 1, "model": "m"},
        {"signal_timestamp": None, "entry_timestamp": None, "exit_timestamp": None, "direction": "LONG",
         "equity_at_entry": 1020.0, "position_notional": 500.0, "position_units": 5.0,
         "entry_price": 100.0, "exit_price": 102.0, "gross_pnl": 10.0, "gross_return": 0.02,
         "transaction_cost": 2.0, "funding_cost": 0.5, "net_pnl": 7.5, "net_return": 0.015,
         "holding_bars": 3, "fold": 2, "model": "m"},
    ], columns=LEDGER_COLUMNS)
    timeline = _hand_timeline()  # values irrelevant to this assertion set
    result = compute_trading_metrics(timeline, ledger, n_actionable_signals=6)

    assert result.trade_count == 3
    assert result.win_rate == pytest.approx(2 / 3)
    # profit_factor = sum(net_pnl>0) / abs(sum(net_pnl<0)) = (47.5+7.5)/27.5
    assert result.profit_factor == pytest.approx(55.0 / 27.5)
    assert result.avg_trade_return == pytest.approx((0.095 - 0.055 + 0.015) / 3)
    assert result.median_trade_return == pytest.approx(0.015)
    assert result.avg_holding_bars == pytest.approx(3.0)
    assert result.signal_utilization == pytest.approx(3 / 6)


def test_trading_metrics_empty_timeline_raises():
    with pytest.raises(MetricsError):
        compute_trading_metrics(pd.DataFrame(columns=TIMELINE_COLUMNS), pd.DataFrame(columns=LEDGER_COLUMNS), 0)


# ---------------------------------------------------------------------------
# Passive-long benchmark
# ---------------------------------------------------------------------------

def _bars(n=10):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "open": [100.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
    })


def test_passive_long_benchmark_matches_direct_single_trade_run_backtest():
    bars = _bars(10)
    timeline, ledger, tm = compute_passive_long_benchmark(
        bars, initial_equity=2000.0, fee_bps=5.0, slippage_bps=2.0, latency_bps=1.0, funding_bps_per_bar=0.2,
    )
    # Independently reproduce via run_backtest with the exact same
    # single-BUY-at-bar-0 signal and horizon_bars=n-1.
    signals = pd.DataFrame([{"timestamp": bars.loc[0, "timestamp"], "fold": None, "signal": "BUY"}])
    expected_timeline, expected_ledger = run_backtest(
        bars, signals, model="passive_long_benchmark", horizon_bars=len(bars) - 1,
        initial_equity=2000.0, position_fraction=1.0,
        fee_bps=5.0, slippage_bps=2.0, latency_bps=1.0, funding_bps_per_bar=0.2,
    )
    pd.testing.assert_frame_equal(timeline, expected_timeline)
    pd.testing.assert_frame_equal(ledger, expected_ledger)
    assert len(ledger) == 1
    assert ledger.iloc[0]["entry_timestamp"] == bars.loc[1, "timestamp"]
    assert ledger.iloc[0]["exit_timestamp"] == bars.loc[len(bars) - 1, "timestamp"]


def test_passive_long_benchmark_costs_are_not_zero():
    bars = _bars(10)
    _, ledger, _ = compute_passive_long_benchmark(bars, fee_bps=5.0, slippage_bps=2.0, latency_bps=1.0, funding_bps_per_bar=1.0)
    assert ledger.iloc[0]["transaction_cost"] > 0
    assert ledger.iloc[0]["funding_cost"] > 0


def test_passive_long_benchmark_too_few_bars_raises():
    with pytest.raises(MetricsError):
        compute_passive_long_benchmark(_bars(1))
