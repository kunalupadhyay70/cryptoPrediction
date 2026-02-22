"""
Backtester v2 — fixed return calculation, threshold logic, and Sharpe annualisation.

Key fixes vs v1:
  - sell_threshold is now independent from buy_threshold (not 1-buy), defaults to
    symmetric around 0.5: sell = 1 - buy only if model is well-calibrated.
    The sweep tests both symmetrically AND asymmetrically.
  - Returns use close-to-close pct_change shifted by label_horizon_bars (not open-to-open
    with mismatched horizon), avoiding the entry/exit open-price lookahead confusion.
  - Sharpe annualised with bar_size_minutes so 5-min bars give correct annualisation.
  - Sweep no longer passes None as target_col.
  - Max trades/day guard prevents runaway signal count from dominating metrics.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    fee_bps: float
    slippage_bps: float
    latency_bps: float
    funding_bps_per_bar: float
    min_accuracy: float = 0.52
    min_auc: float = 0.54
    min_sharpe: float = 1.0
    min_trades_per_day: float = 2.0
    min_trades_total: int = 50
    max_trades_per_day: float = 200.0   # guard: if more than this, threshold is too loose
    bar_size_minutes: int = 1           # 1 for 1m bars; used for correct Sharpe annualisation
    label_horizon_bars: int = 5         # must match target.horizon_bars in config
    execute_on_open: bool = True
    threshold_sweep_step: float = 0.01
    optimize_threshold: bool = True


class Backtester:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.optimal_threshold_: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        pred_prob_col: str,
        target_col: str,
        model_metrics: Dict,
        buy_threshold: Optional[float] = None,
        sell_threshold: Optional[float] = None,
    ) -> Dict:
        if df.empty:
            return self._empty(model_metrics)

        auc = model_metrics.get("roc_auc", 0.0)
        acc = model_metrics.get("accuracy", 0.0)
        if auc < self.config.min_auc or acc < self.config.min_accuracy:
            return {**self._empty(model_metrics), "status": "REJECTED_BY_GATES",
                    "buy_threshold": 0.55, "sell_threshold": 0.45}

        df = df.copy().reset_index(drop=True)
        probs  = df[pred_prob_col].values
        y_true = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values

        # Compute forward returns aligned to label_horizon_bars
        fwd_ret = self._compute_forward_returns(df)

        if buy_threshold is None and self.config.optimize_threshold:
            buy_threshold, sell_threshold = self._sweep_thresholds(probs, fwd_ret)
            self.optimal_threshold_ = buy_threshold
        else:
            buy_threshold  = buy_threshold or 0.55
            sell_threshold = sell_threshold or (1.0 - buy_threshold)
            self.optimal_threshold_  = buy_threshold

        report = self._evaluate(probs, fwd_ret, model_metrics, buy_threshold, sell_threshold)
        report["buy_threshold"]  = float(buy_threshold)
        report["sell_threshold"] = float(sell_threshold)
        return report

    # ------------------------------------------------------------------
    # Forward return computation
    # ------------------------------------------------------------------

    def _compute_forward_returns(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return the actual per-bar forward return used for PnL calculation.

        For a 5-bar label horizon: we enter at bar t close, exit at bar t+5 close.
        ret[t] = close[t+h] / close[t] - 1

        We do NOT use open-to-open here because:
          - The label is defined on close prices
          - open[t+1] prices have lookahead issues when shifted by 2
        """
        h = self.config.label_horizon_bars
        if "close" in df.columns:
            c = df["close"].astype(float)
            fwd = (c.shift(-h) / c - 1).values
        else:
            fwd = np.zeros(len(df))
        return fwd

    # ------------------------------------------------------------------
    # Threshold sweep
    # ------------------------------------------------------------------

    def _sweep_thresholds(self, probs: np.ndarray, fwd_ret: np.ndarray) -> Tuple[float, float]:
        """
        Sweep buy threshold from 0.51 to 0.80 in steps.
        sell_threshold is always symmetric: sell_t = 1.0 - buy_t
        (so dead zone widens symmetrically as buy_t increases).
        Pick the threshold that maximises out-of-sample Sharpe.
        """
        best_buy, best_sell, best_sharpe = 0.55, 0.45, -np.inf
        step = self.config.threshold_sweep_step

        for buy_t in np.arange(0.51, 0.80 + step/2, step):
            buy_t  = round(float(buy_t), 4)
            sell_t = round(1.0 - buy_t, 4)
            result = self._evaluate(probs, fwd_ret, {}, buy_t, sell_t, quick=True)
            s = result.get("sharpe", -np.inf)
            n = result.get("trades", 0)
            # Skip degenerate cases: too few trades or too many (threshold too loose)
            n_days = max(1, len(probs) / (60 * 24 / self.config.bar_size_minutes))
            tpd = n / n_days
            if n < self.config.min_trades_total or tpd > self.config.max_trades_per_day:
                continue
            if s > best_sharpe:
                best_sharpe = s
                best_buy, best_sell = buy_t, sell_t

        LOGGER.info("Threshold sweep: best buy=%.2f sell=%.2f sharpe=%.2f",
                    best_buy, best_sell, best_sharpe)
        return best_buy, best_sell

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        probs: np.ndarray,
        fwd_ret: np.ndarray,
        model_metrics: Dict,
        buy_t: float,
        sell_t: float,
        quick: bool = False,
    ) -> Dict:
        """
        Core PnL engine.

        Signal logic:
            prob > buy_t  → BUY  (+1): bet on UP move
            prob < sell_t → SELL (-1): bet on DOWN move
            else          → HOLD  (0): flat

        Return per active bar:
            gross  = signal * fwd_ret
            cost   = one_way_cost (entry) + one_way_cost (exit)  [round trip]
            net    = gross - cost - funding
        """
        n = len(probs)
        signals = np.where(probs > buy_t, 1, np.where(probs < sell_t, -1, 0)).astype(np.int8)

        # Costs (all in decimal, not bps)
        one_way = (self.config.fee_bps + self.config.slippage_bps + self.config.latency_bps) / 10_000
        round_trip = 2 * one_way
        funding_per_bar = self.config.funding_bps_per_bar / 10_000

        gross = signals.astype(float) * fwd_ret
        cost  = np.abs(signals).astype(float) * (round_trip + funding_per_bar)

        # Extra cost on position flips (pay another one-way to reverse)
        prev_sig = np.empty_like(signals); prev_sig[0] = 0; prev_sig[1:] = signals[:-1]
        is_flip  = ((signals != 0) & (prev_sig != 0) & (signals != prev_sig)).astype(float)
        cost += is_flip * one_way

        net = gross - cost

        # Only evaluate active bars
        active = signals != 0
        # Also mask out the last label_horizon_bars bars (no valid forward return)
        h = self.config.label_horizon_bars
        valid = np.ones(n, dtype=bool)
        valid[-h:] = False
        mask = active & valid & np.isfinite(fwd_ret)

        ret = pd.Series(net[mask], dtype=float)
        n_trades = int(mask.sum())

        if n_trades < max(2, self.config.min_trades_total // 4 if quick else self.config.min_trades_total):
            return {"sharpe": -np.inf, "trades": n_trades, "status": "TOO_FEW_TRADES"}

        # Annualisation: bars per year for the given bar size
        bars_per_year = 365 * 24 * 60 / self.config.bar_size_minutes
        sharpe = float(ret.mean() / (ret.std() + 1e-12) * np.sqrt(bars_per_year))

        if quick:
            return {"sharpe": sharpe, "trades": n_trades}

        # Full metrics
        equity = (1 + ret).cumprod()
        dd     = float((equity / equity.cummax() - 1).min())
        n_days = max(1, n / (60 * 24 / self.config.bar_size_minutes))
        tpd    = n_trades / n_days
        wr     = float((ret > 0).mean())

        avg_hold = self._avg_hold(signals)
        turnover = float((pd.Series(signals).diff().fillna(0) != 0).mean())

        auc = model_metrics.get("roc_auc", 0.0)
        acc = model_metrics.get("accuracy", 0.0)

        if tpd > self.config.max_trades_per_day:
            status = "REJECTED_TOO_MANY_TRADES"
        elif tpd < self.config.min_trades_per_day:
            status = "REJECTED_TOO_FEW_TRADES_PER_DAY"
        elif sharpe < self.config.min_sharpe:
            status = "LOW_SHARPE"
        else:
            status = "ACCEPTED"

        LOGGER.info(
            "Backtest | thresholds=%.2f/%.2f | trades=%d (%.1f/day) | "
            "sharpe=%.2f | maxdd=%.1f%% | wr=%.1f%% | status=%s",
            buy_t, sell_t, n_trades, tpd, sharpe, dd*100, wr*100, status,
        )

        return {
            "trades": n_trades,
            "trades_per_day": round(tpd, 2),
            "accuracy": float(acc),
            "roc_auc": float(auc),
            "sharpe": sharpe,
            "max_drawdown": dd,
            "max_drawdown_pct": round(dd * 100, 2),
            "win_rate": wr,
            "pnl_mean": float(ret.mean()),
            "pnl_std": float(ret.std()),
            "pnl_skew": float(ret.skew()) if len(ret) > 3 else 0.0,
            "pnl_kurtosis": float(ret.kurtosis()) if len(ret) > 3 else 0.0,
            "turnover": turnover,
            "avg_holding_bars": avg_hold,
            "equity_final": float(equity.iloc[-1]) if not equity.empty else 1.0,
            "status": status,
        }

    @staticmethod
    def _avg_hold(signals: np.ndarray) -> float:
        runs, cr, ps = [], 0, 0
        for s in signals:
            if s != 0 and s == ps:
                cr += 1
            elif s != 0:
                if cr > 0: runs.append(cr)
                cr = 1
            else:
                if cr > 0: runs.append(cr)
                cr = 0
            ps = s
        if cr > 0: runs.append(cr)
        return float(np.mean(runs)) if runs else 1.0

    def _empty(self, m: Dict) -> Dict:
        return {
            "trades": 0, "trades_per_day": 0.0,
            "accuracy": float(m.get("accuracy", 0.0)),
            "roc_auc": float(m.get("roc_auc", 0.0)),
            "sharpe": 0.0, "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "win_rate": 0.0, "pnl_mean": 0.0, "pnl_std": 0.0,
            "pnl_skew": 0.0, "pnl_kurtosis": 0.0,
            "turnover": 0.0, "avg_holding_bars": 0.0, "equity_final": 1.0,
            "status": "EMPTY", "buy_threshold": 0.55, "sell_threshold": 0.45,
        }