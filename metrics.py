"""Stage 2M — Predictive metrics, trading metrics, and the passive-long
benchmark. Kept strictly separate (frozen rule): predictive metrics score
classification quality on the OOF table; trading metrics score the
backtest engine's timeline/ledger; neither is derived from the other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from position_engine import LEDGER_COLUMNS, TIMELINE_COLUMNS, run_backtest

CLASS_ORDER = (0, 1, 2)  # DOWN, NEUTRAL, UP -- matches prob_down/neutral/up column order.


class MetricsError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Predictive metrics (classification quality; OOF table only)
# ---------------------------------------------------------------------------

@dataclass
class PredictiveMetrics:
    macro_f1: float
    balanced_accuracy: float
    macro_roc_auc: float
    log_loss: float
    confusion_matrix: np.ndarray  # shape (3, 3), rows=actual, cols=pred, order CLASS_ORDER
    precision_per_class: Dict[int, float]
    recall_per_class: Dict[int, float]
    class_distribution: Dict[int, int]
    n_rows: int


def compute_predictive_metrics(
    actual_class: np.ndarray, pred_class: np.ndarray,
    prob_down: np.ndarray, prob_neutral: np.ndarray, prob_up: np.ndarray,
) -> PredictiveMetrics:
    from sklearn.metrics import (
        balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
        precision_score, recall_score, roc_auc_score,
    )

    actual_class = np.asarray(actual_class, dtype=int)
    pred_class = np.asarray(pred_class, dtype=int)
    if len(actual_class) == 0:
        raise MetricsError("actual_class is empty; cannot compute predictive metrics")
    probs = np.column_stack([
        np.asarray(prob_down, dtype=float), np.asarray(prob_neutral, dtype=float), np.asarray(prob_up, dtype=float),
    ])

    macro_f1 = float(f1_score(actual_class, pred_class, average="macro", labels=list(CLASS_ORDER), zero_division=0))
    balanced_acc = float(balanced_accuracy_score(actual_class, pred_class))
    try:
        macro_auc = float(roc_auc_score(actual_class, probs, multi_class="ovr", average="macro", labels=list(CLASS_ORDER)))
    except ValueError:
        # e.g. a class entirely absent from actual_class in this slice.
        macro_auc = float("nan")
    ll = float(log_loss(actual_class, probs, labels=list(CLASS_ORDER)))
    cm = confusion_matrix(actual_class, pred_class, labels=list(CLASS_ORDER))
    precision = precision_score(actual_class, pred_class, average=None, labels=list(CLASS_ORDER), zero_division=0)
    recall = recall_score(actual_class, pred_class, average=None, labels=list(CLASS_ORDER), zero_division=0)
    distribution = {c: int((actual_class == c).sum()) for c in CLASS_ORDER}

    return PredictiveMetrics(
        macro_f1=macro_f1, balanced_accuracy=balanced_acc, macro_roc_auc=macro_auc, log_loss=ll,
        confusion_matrix=cm,
        precision_per_class={c: float(precision[i]) for i, c in enumerate(CLASS_ORDER)},
        recall_per_class={c: float(recall[i]) for i, c in enumerate(CLASS_ORDER)},
        class_distribution=distribution, n_rows=len(actual_class),
    )


# ---------------------------------------------------------------------------
# Trading metrics (timeline + ledger only; never touches OOF probabilities)
# ---------------------------------------------------------------------------

@dataclass
class TradingMetrics:
    total_return: float
    sharpe_daily_compounded: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trade_count: int
    avg_trade_return: float
    median_trade_return: float
    avg_holding_bars: float
    signal_utilization: float


def _daily_compounded_returns(timeline_df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(timeline_df["timestamp"])
    dates = ts.dt.date
    grouped = timeline_df.groupby(dates)["net_strategy_return"].apply(lambda r: float(np.prod(1.0 + r.to_numpy()) - 1.0))
    return grouped


def compute_trading_metrics(
    timeline_df: pd.DataFrame, ledger_df: pd.DataFrame, n_actionable_signals: int,
) -> TradingMetrics:
    if len(timeline_df) == 0:
        raise MetricsError("timeline_df is empty; cannot compute trading metrics")

    equity = timeline_df["equity_after"].to_numpy()
    equity_start = float(timeline_df["equity_before"].iloc[0])
    total_return = float(equity[-1] / equity_start - 1.0)

    daily_returns = _daily_compounded_returns(timeline_df).to_numpy()
    if len(daily_returns) > 1 and np.std(daily_returns, ddof=1) > 1e-12:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(365.0))
    else:
        sharpe = float("nan")

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    n_trades = len(ledger_df)
    if n_trades > 0:
        net_pnl = ledger_df["net_pnl"].to_numpy()
        net_return = ledger_df["net_return"].to_numpy()
        win_rate = float((net_pnl > 0).mean())
        gross_wins = net_pnl[net_pnl > 0].sum()
        gross_losses = -net_pnl[net_pnl < 0].sum()
        profit_factor = float(gross_wins / gross_losses) if gross_losses > 1e-12 else float("inf")
        avg_trade_return = float(np.mean(net_return))
        median_trade_return = float(np.median(net_return))
        avg_holding_bars = float(ledger_df["holding_bars"].mean())
    else:
        win_rate = float("nan")
        profit_factor = float("nan")
        avg_trade_return = float("nan")
        median_trade_return = float("nan")
        avg_holding_bars = float("nan")

    signal_utilization = (n_trades / n_actionable_signals) if n_actionable_signals > 0 else float("nan")

    return TradingMetrics(
        total_return=total_return, sharpe_daily_compounded=sharpe, max_drawdown=max_drawdown,
        win_rate=win_rate, profit_factor=profit_factor, trade_count=n_trades,
        avg_trade_return=avg_trade_return, median_trade_return=median_trade_return,
        avg_holding_bars=avg_holding_bars, signal_utilization=signal_utilization,
    )


# ---------------------------------------------------------------------------
# Passive-long benchmark
# ---------------------------------------------------------------------------

def compute_passive_long_benchmark(
    bars: pd.DataFrame,
    initial_equity: float = 10_000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    latency_bps: float = 1.0,
    funding_bps_per_bar: float = 0.0,
):
    """Passive-long BTCUSDT perpetual futures over the EXACT same evaluation
    window: entry at the first executable next-bar open (bar 1's open, the
    same execution convention the strategy itself uses), exit at the final
    evaluation close, with the SAME entry/exit cost and funding treatment
    as the strategy (never a zero-cost benchmark). Implemented by reusing
    run_backtest with a single BUY signal at bar 0 and
    horizon_bars=len(bars)-1, so entry/exit land exactly on those bars --
    this is the single source of truth for execution mechanics, not a
    second divergent implementation.
    """
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    n = len(bars)
    if n < 2:
        raise MetricsError("need at least 2 bars to run a passive-long benchmark")
    signals = pd.DataFrame([{"timestamp": bars.loc[0, "timestamp"], "fold": None, "signal": "BUY"}])
    timeline, ledger = run_backtest(
        bars, signals, model="passive_long_benchmark", horizon_bars=n - 1,
        initial_equity=initial_equity, position_fraction=1.0,
        fee_bps=fee_bps, slippage_bps=slippage_bps, latency_bps=latency_bps,
        funding_bps_per_bar=funding_bps_per_bar,
    )
    trading_metrics = compute_trading_metrics(timeline, ledger, n_actionable_signals=1)
    return timeline, ledger, trading_metrics
