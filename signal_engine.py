"""SignalEngine — regime-aware, accepts both set_optimized_threshold and set_optimal_thresholds."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SignalConfig:
    low_vol_buy: float = 0.57
    low_vol_sell: float = 0.43
    high_vol_buy: float = 0.62
    high_vol_sell: float = 0.38
    vol_regime_threshold: float = 0.0015
    optimized_threshold: Optional[float] = None


class SignalEngine:
    def __init__(self, config: SignalConfig) -> None:
        self.config = config

    # Called by main.py after backtest
    def set_optimal_thresholds(self, buy: float, sell: float) -> None:
        self.config.low_vol_buy   = buy
        self.config.low_vol_sell  = sell
        self.config.high_vol_buy  = min(buy  + 0.04, 0.75)
        self.config.high_vol_sell = max(sell - 0.04, 0.25)
        self.config.optimized_threshold = buy

    # Original single-threshold setter (kept for backward compat)
    def set_optimized_threshold(self, threshold: float) -> None:
        self.set_optimal_thresholds(threshold, 1.0 - threshold)

    def dynamic_signal(self, prob_up: float, volatility: float, vol_threshold: float) -> str:
        if volatility > vol_threshold:
            buy_t, sell_t = self.config.high_vol_buy, self.config.high_vol_sell
        else:
            buy_t, sell_t = self.config.low_vol_buy,  self.config.low_vol_sell
        if prob_up >= buy_t:  return "BUY"
        if prob_up <= sell_t: return "SELL"
        return "HOLD"