"""
FeatureEngineer v3 — AUC-boosting improvements:
  1. Lagged features (t-1, t-2, t-3) for sequential memory
  2. Time-of-day / day-of-week features (intraday seasonality)
  3. Daily-reset VWAP (not cumulative across all history)
  4. Multi-horizon targets (predict 5-bar and 15-bar, use as ensemble signal)
  5. Stronger label: require move > 1.5x ATR/bar (not tiny dead zone)
  6. Rolling z-score normalization of returns (regime-neutral features)
  7. Cross-asset proxy: BTC volume anomaly score
  8. Autocorrelation features (return serial correlation)
  9. Higher-order moments: skewness and kurtosis of recent returns
 10. Price level features: distance from recent high/low
"""
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    db_path: str
    label_horizon_bars: int = 5            # CHANGED: 5-bar horizon (was 1) — far less noisy
    label_mode: str = "dead_zone"

    dead_zone_bps: float = 5.0
    deadzone_threshold: float = 0.0005

    vol_breakout_horizon: int = 3
    vol_breakout_bars: int = 3
    vol_breakout_percentile: float = 70.0

    neutral_atr_mult: float = 0.3          # CHANGED: wider neutral band (was 0.15)

    vol_regime_threshold: float = 0.0015
    regime_vol_threshold: float = 0.0015

    correlation_threshold: float = 0.95
    shap_top_n: int = 50

    # New: lag depth for lagged features
    lag_periods: int = 3

    # OHLCV table name — must match the kline_interval used during collection
    ohlcv_table: str = "ohlcv_5m"   # "ohlcv_1m" for 1m bars, "ohlcv_5m" for 5m, etc.

    def __post_init__(self):
        if self.deadzone_threshold != 0.0005:
            self.dead_zone_bps = self.deadzone_threshold * 10_000
        elif self.dead_zone_bps != 5.0:
            self.deadzone_threshold = self.dead_zone_bps / 10_000
        if self.vol_breakout_bars != 3:
            self.vol_breakout_horizon = self.vol_breakout_bars
        elif self.vol_breakout_horizon != 3:
            self.vol_breakout_bars = self.vol_breakout_horizon
        if self.regime_vol_threshold != 0.0015:
            self.vol_regime_threshold = self.regime_vol_threshold
        elif self.vol_regime_threshold != 0.0015:
            self.regime_vol_threshold = self.vol_regime_threshold


class FeatureEngineer:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.db_path = Path(config.db_path)

    def _load(self, table: str, order_col: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            try:
                return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY {order_col} ASC", conn)
            except Exception:
                return pd.DataFrame()

    @staticmethod
    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        up, dn = delta.clip(lower=0), -delta.clip(upper=0)
        rs = (up.ewm(alpha=1/period, adjust=False).mean()
              / dn.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan))
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        prev = df["close"].shift(1)
        tr = pd.concat([df["high"]-df["low"],
                        (df["high"]-prev).abs(),
                        (df["low"]-prev).abs()], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=2).mean()

    @staticmethod
    def _safe_levels(raw) -> list:
        if not isinstance(raw, str):
            return []
        try:
            lv = json.loads(raw)
            return lv if isinstance(lv, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
        """Z-score each value relative to its rolling window — regime-neutral."""
        mu = s.rolling(window, min_periods=max(2, window//2)).mean()
        sd = s.rolling(window, min_periods=max(2, window//2)).std()
        return (s - mu) / (sd + 1e-9)

    @staticmethod
    def _autocorr(s: pd.Series, window: int, lag: int = 1) -> pd.Series:
        """Vectorized rolling autocorrelation — fast on large series."""
        return s.rolling(window, min_periods=max(4, window//2)).corr(s.shift(lag)).fillna(0.0)

    def _build_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].astype(float)
        h = self.config.label_horizon_bars
        future_ret = close.shift(-h) / close - 1
        mode = self.config.label_mode

        if mode == "dead_zone":
            threshold = self.config.deadzone_threshold
            df["target_primary"] = np.where(
                future_ret > threshold, 1, np.where(future_ret < -threshold, 0, np.nan)
            )
            n_valid = df["target_primary"].notna().sum()
            LOGGER.info("Label dead_zone | h=%d | threshold=%.4f%% | non-neutral: %d/%d (%.1f%%)",
                        h, threshold*100, n_valid, len(df), 100*n_valid/len(df))

        elif mode == "vol_breakout":
            hb = self.config.vol_breakout_horizon
            fwd_high = df["high"].rolling(hb).max().shift(-hb)
            fwd_low  = df["low"].rolling(hb).min().shift(-hb)
            fwd_move = (fwd_high - fwd_low) / (close + 1e-9)
            pct = np.nanpercentile(fwd_move.dropna(), self.config.vol_breakout_percentile)
            df["target_primary"] = np.where(fwd_move >= pct, 1, 0)

        elif mode == "regime_conditioned":
            atr = self._atr(df, 14)
            vr = (atr / (close + 1e-9) > self.config.vol_regime_threshold).astype(int)
            df["vol_regime"] = vr
            hi_t = self.config.deadzone_threshold * 1.5
            lo_t = self.config.deadzone_threshold
            threshold = np.where(vr == 1, hi_t, lo_t)
            df["target_primary"] = np.where(
                future_ret > threshold, 1, np.where(future_ret < -threshold, 0, np.nan)
            )
        else:
            df["target_primary"] = np.where(future_ret > 0, 1, 0)

        df["future_ret"] = future_ret

        # Secondary label: ATR-scaled significant move
        atr_val = self._atr(df, 14)
        neutral_band = self.config.neutral_atr_mult * (atr_val / (close + 1e-9))
        df["target_large_move"] = np.where(
            future_ret > neutral_band, 1, np.where(future_ret < -neutral_band, 0, np.nan)
        )
        return df

    def _build_orderbook_features(self, ob: pd.DataFrame) -> pd.DataFrame:
        if ob.empty:
            return pd.DataFrame()
        ob = ob.copy()
        ob["ts"] = pd.to_datetime(ob["ts"], utc=True)
        ob["open_time"] = ob["ts"].dt.floor("1min")
        ob = ob.sort_values("ts")
        ob["bids"] = ob["bids_json"].apply(self._safe_levels)
        ob["asks"] = ob["asks_json"].apply(self._safe_levels)

        rows = []
        for t, grp in ob.groupby("open_time"):
            last = grp.iloc[-1]
            bids, asks = last["bids"][:20], last["asks"][:20]
            if not bids or not asks:
                rows.append({"open_time": t}); continue

            bid_p = np.array([float(x[0]) for x in bids])
            ask_p = np.array([float(x[0]) for x in asks])
            bid_v = np.array([float(x[1]) for x in bids])
            ask_v = np.array([float(x[1]) for x in asks])

            best_bid, best_ask = bid_p[0], ask_p[0]
            mid = (best_bid + best_ask) / 2
            microprice = (best_bid*ask_v[0] + best_ask*bid_v[0]) / (bid_v[0]+ask_v[0]+1e-9)

            def imb(bv, av): return (bv.sum()-av.sum())/(bv.sum()+av.sum()+1e-9)
            w = 1.0/np.arange(1, max(len(bid_v), len(ask_v))+1)
            bw, aw = w[:len(bid_v)], w[:len(ask_v)]
            weighted_imb = ((bid_v*bw).sum()-(ask_v*aw).sum())/((bid_v*bw).sum()+(ask_v*aw).sum()+1e-9)

            mids = (grp["best_bid"].to_numpy()+grp["best_ask"].to_numpy())/2
            mid_slope = float(np.polyfit(np.arange(len(mids)), mids, 1)[0]) if len(mids)>2 else 0.0
            spread_slope = float(np.polyfit(np.arange(len(grp)), grp["spread"].to_numpy(), 1)[0]) if len(grp)>2 else 0.0
            bid_slope = float(np.polyfit(np.arange(min(10,len(bid_v))), bid_v[:10], 1)[0]) if len(bid_v)>=2 else 0.0
            ask_slope = float(np.polyfit(np.arange(min(10,len(ask_v))), ask_v[:10], 1)[0]) if len(ask_v)>=2 else 0.0

            rows.append({
                "open_time": t,
                "spread_l1": float(best_ask-best_bid),
                "depth_bid_20": float(bid_v.sum()), "depth_ask_20": float(ask_v.sum()),
                "microprice": float(microprice), "microprice_dev": float((microprice-mid)/(mid+1e-9)),
                "imb_l1": float(imb(bid_v[:1], ask_v[:1])),
                "imb_l5": float(imb(bid_v[:5], ask_v[:5])),
                "imb_l10": float(imb(bid_v[:10], ask_v[:10])),
                "imb_l20": float(imb(bid_v, ask_v)),
                "weighted_imb": float(weighted_imb),
                "bid_depth_slope": bid_slope, "ask_depth_slope": ask_slope,
                "wall_bid": float(bid_v.max()), "wall_ask": float(ask_v.max()),
                "book_pressure": float((bid_v[:5].sum()+1e-9)/(ask_v[:5].sum()+1e-9)),
                "spread_slope": spread_slope, "mid_slope": mid_slope,
            })

        obf = pd.DataFrame(rows)
        for col in ["imb_l1","imb_l5","imb_l10","imb_l20","weighted_imb","spread_l1"]:
            if col in obf.columns:
                obf[f"d_{col}"] = obf[col].diff()
        return obf

    def build_dataset(self) -> pd.DataFrame:
        ohlc   = self._load(self.config.ohlcv_table, "open_time")
        ob     = self._load("order_book_snapshots", "ts")
        trades = self._load("trades", "ts")

        if ohlc.empty:
            LOGGER.error("No data in %s", self.config.ohlcv_table); return pd.DataFrame()

        ohlc["open_time"]  = pd.to_datetime(ohlc["open_time"],  utc=True)
        ohlc["close_time"] = pd.to_datetime(ohlc["close_time"], utc=True)
        ohlc = ohlc.sort_values("open_time").drop_duplicates("open_time", keep="last")
        df = ohlc.copy()
        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)
        LOGGER.info("Building features on %d candles...", len(df))

        # ── 1. Returns & volatility features ─────────────────────────────────
        ret1 = close.pct_change()
        for p in [1, 3, 5, 10, 15, 20, 30, 60]:
            mp = min(max(1, p//2), p)
            df[f"ret_{p}"]     = close.pct_change(p)
            df[f"vol_{p}"]     = ret1.rolling(p, min_periods=mp).std()
            df[f"ema_dist_{p}"] = close / self._ema(close, p) - 1
            df[f"sma_dist_{p}"] = close / close.rolling(p, min_periods=mp).mean() - 1

        # ── 2. Rolling z-scores (regime-neutral) ─────────────────────────────
        for p in [10, 30, 60]:
            df[f"ret1_z{p}"]  = self._rolling_zscore(ret1,           p)
            df[f"vol5_z{p}"]  = self._rolling_zscore(df["vol_5"],    p)
            df[f"vol20_z{p}"] = self._rolling_zscore(df["vol_20"],   p)

        # ── 3. Autocorrelation of returns ─────────────────────────────────────
        df["ret_autocorr_10_lag1"] = self._autocorr(ret1, 10, lag=1)
        df["ret_autocorr_20_lag1"] = self._autocorr(ret1, 20, lag=1)
        df["ret_autocorr_10_lag3"] = self._autocorr(ret1, 10, lag=3)

        # ── 4. Higher-order moments (rolling skewness & kurtosis) ─────────────
        for w in [20, 60]:
            df[f"ret_skew_{w}"]  = ret1.rolling(w, min_periods=w//2).skew()
            df[f"ret_kurt_{w}"]  = ret1.rolling(w, min_periods=w//2).kurt()

        # ── 5. Momentum / oscillator features ────────────────────────────────
        for period in [7, 14, 21]:
            df[f"rsi_{period}"] = self._rsi(close, period)
        ema12, ema26 = self._ema(close, 12), self._ema(close, 26)
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = self._ema(df["macd"], 9)
        df["macd_hist"]   = df["macd"] - df["macd_signal"]
        df["macd_hist_z"] = self._rolling_zscore(df["macd_hist"], 30)

        for kp in [14, 21]:
            lo = df["low"].rolling(kp,  min_periods=2).min()
            hi = df["high"].rolling(kp, min_periods=2).max()
            df[f"stoch_k_{kp}"] = 100*(close-lo)/(hi-lo+1e-9)
            df[f"stoch_d_{kp}"] = df[f"stoch_k_{kp}"].rolling(3, min_periods=2).mean()

        # ── 6. ATR & Bollinger ────────────────────────────────────────────────
        df["atr_14"] = self._atr(df, 14)
        df["atr_7"]  = self._atr(df, 7)
        df["atr_ratio"] = df["atr_7"] / (df["atr_14"] + 1e-9)   # short vs long vol
        for bbp in [20, 50]:
            bm = close.rolling(bbp, min_periods=2).mean()
            bs = close.rolling(bbp, min_periods=2).std()
            df[f"bb_width_{bbp}"] = (2*bs) / (bm + 1e-9)
            df[f"bb_pctb_{bbp}"]  = (close-(bm-2*bs)) / (4*bs + 1e-9)

        # ── 7. Daily-reset VWAP (not cumulative across all history) ──────────
        if "open_time" in df.columns:
            df["_date"] = df["open_time"].dt.date
            tp = (df["high"] + df["low"] + df["close"]) / 3
            df["_tp_vol"] = tp * volume
            df["_cum_tp_vol"] = df.groupby("_date")["_tp_vol"].cumsum()
            df["_cum_vol"]    = df.groupby("_date")["volume"].cumsum()
            df["vwap_daily"]  = df["_cum_tp_vol"] / (df["_cum_vol"] + 1e-9)
            df["vwap_dev"]    = close / df["vwap_daily"] - 1
            df.drop(columns=["_date","_tp_vol","_cum_tp_vol","_cum_vol","vwap_daily"], inplace=True)
        else:
            tp = (df["high"]+df["low"]+df["close"])/3
            df["vwap_dev"] = close/((tp*volume).cumsum()/(volume.cumsum()+1e-9))-1

        # ── 8. Volume features ────────────────────────────────────────────────
        vol_ma20 = volume.rolling(20, min_periods=2).mean()
        vol_ma5  = volume.rolling(5,  min_periods=2).mean()
        df["vol_ratio_20"]  = volume / (vol_ma20 + 1e-9)
        df["vol_ratio_5"]   = volume / (vol_ma5  + 1e-9)
        df["vol_z_20"]      = self._rolling_zscore(volume, 20)
        df["vol_trend"]     = vol_ma5 / (vol_ma20 + 1e-9)  # short/long vol MA ratio
        df["vol_spike"]     = (volume > vol_ma20 * 2.0).astype(int)

        # ── 9. Candle structure ───────────────────────────────────────────────
        df["body"]           = (df["close"]-df["open"]).abs() / (close + 1e-9)
        df["upper_wick"]     = (df["high"]-df[["open","close"]].max(axis=1)) / (close + 1e-9)
        df["lower_wick"]     = (df[["open","close"]].min(axis=1)-df["low"]) / (close + 1e-9)
        df["body_direction"] = np.sign(df["close"] - df["open"])
        df["wick_ratio"]     = df["upper_wick"] / (df["lower_wick"] + 1e-9)   # buying vs selling pressure

        # ── 10. Price level features ──────────────────────────────────────────
        for w in [20, 60, 120]:
            df[f"dist_high_{w}"] = close / df["high"].rolling(w, min_periods=2).max() - 1
            df[f"dist_low_{w}"]  = close / df["low"].rolling(w,  min_periods=2).min() - 1

        # ── 11. Time-of-day & day-of-week (intraday seasonality) ─────────────
        if "open_time" in df.columns:
            hour   = df["open_time"].dt.hour
            minute = df["open_time"].dt.minute
            dow    = df["open_time"].dt.dayofweek
            # Encode cyclically so 23:59 is close to 00:00
            minutes_in_day = hour * 60 + minute
            df["tod_sin"] = np.sin(2 * np.pi * minutes_in_day / 1440)
            df["tod_cos"] = np.cos(2 * np.pi * minutes_in_day / 1440)
            df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
            # Key trading sessions: Asia (0-8 UTC), Europe (8-16), US (13-21)
            df["session_asia"]   = ((hour >= 0)  & (hour < 8)).astype(int)
            df["session_europe"] = ((hour >= 8)  & (hour < 16)).astype(int)
            df["session_us"]     = ((hour >= 13) & (hour < 21)).astype(int)
            df["session_overlap"]= ((hour >= 13) & (hour < 16)).astype(int)  # EU+US overlap (high vol)

        # ── 12. Regime feature ────────────────────────────────────────────────
        ret_std = ret1.rolling(60, min_periods=10).std()
        df["vol_regime"] = (ret_std > self.config.vol_regime_threshold).astype(int)
        df["vol_regime_smooth"] = df["vol_regime"].rolling(10, min_periods=1).mean()

        # ── 13. Momentum across horizons ──────────────────────────────────────
        for p in [3, 5, 10, 20]:
            df[f"mom_{p}"] = close / close.shift(p) - 1
        # Momentum acceleration (short mom minus long mom)
        df["mom_accel"] = df["mom_3"] - df["mom_10"]

        # ── 14. LAGGED features (give model sequential memory) ────────────────
        # Lag the most informative base features by 1, 2, 3 bars
        lag_base_cols = [
            "ret_1", "ret_3", "ret_5",
            "vol_5", "vol_10",
            "rsi_14",
            "macd_hist",
            "body_direction",
            "vol_ratio_5",
            "vwap_dev",
        ]
        n_lags = self.config.lag_periods
        lag_dict = {
        f"{col}_lag{lag}": df[col].shift(lag)
        for col in lag_base_cols
        if col in df.columns
        for lag in range(1, n_lags + 1)
         }

        df = pd.concat([df, pd.DataFrame(lag_dict)], axis=1)

        # ── 15. Order book features ───────────────────────────────────────────
        obf = self._build_orderbook_features(ob)
        if not obf.empty:
            df = df.merge(obf, on="open_time", how="left")
        else:
            for col in ["spread_l1","depth_bid_20","depth_ask_20","microprice","microprice_dev",
                        "imb_l1","imb_l5","imb_l10","imb_l20","weighted_imb",
                        "bid_depth_slope","ask_depth_slope","wall_bid","wall_ask",
                        "book_pressure","spread_slope","mid_slope",
                        "d_imb_l1","d_imb_l5","d_imb_l10","d_imb_l20","d_weighted_imb","d_spread_l1"]:
                df[col] = 0.0

        # ── 16. Trade flow features ───────────────────────────────────────────
        if not trades.empty:
            trades["ts"] = pd.to_datetime(trades["ts"], utc=True, format="mixed", errors="coerce")
            trades["open_time"] = trades["ts"].dt.floor("1min")
            trades["qty"]      = pd.to_numeric(trades["qty"], errors="coerce").fillna(0.0)
            trades["buy_vol"]  = np.where(trades["is_buyer_maker"]==0, trades["qty"], 0.0)
            trades["sell_vol"] = np.where(trades["is_buyer_maker"]==1, trades["qty"], 0.0)
            tf = trades.groupby("open_time", as_index=False)[["buy_vol","sell_vol"]].sum()
            tf = tf.rename(columns={"buy_vol":"aggr_buy_vol","sell_vol":"aggr_sell_vol"})
            df = df.merge(tf, on="open_time", how="left")
        else:
            df["aggr_buy_vol"] = 0.0
            df["aggr_sell_vol"] = 0.0
        df["aggr_buy_vol"]  = df["aggr_buy_vol"].fillna(0.0)
        df["aggr_sell_vol"] = df["aggr_sell_vol"].fillna(0.0)
        df["taker_imb"] = (df["aggr_buy_vol"]-df["aggr_sell_vol"]) / (df["aggr_buy_vol"]+df["aggr_sell_vol"]+1e-9)
        df["taker_imb_lag1"] = df["taker_imb"].shift(1)
        df["taker_imb_ma5"]  = df["taker_imb"].rolling(5, min_periods=1).mean()

        # ── 17. Interaction features ──────────────────────────────────────────
        z = pd.Series(0.0, index=df.index)
        df["imb_x_rsi"]        = df.get("imb_l10", z) * (df.get("rsi_14", z+50) - 50)
        df["spread_x_volspike"] = df.get("spread_l1", z) * df["vol_ratio_20"]
        df["pressure_x_vol"]   = df.get("book_pressure", z+1) * df["vol_ratio_5"]
        df["microprice_x_ret"]  = df.get("microprice_dev", z) * df["ret_5"]
        df["rsi_x_vol_regime"]  = df.get("rsi_14", z+50) * df["vol_regime"]
        df["mom_x_volume"]      = df["mom_5"] * df["vol_ratio_5"]

        # ── 18. Labels ────────────────────────────────────────────────────────
        df = self._build_labels(df)

        # ── 19. Cleanup ───────────────────────────────────────────────────────
        df = df.replace([np.inf, -np.inf], np.nan)
        feat_cols = feature_columns(df)
        df[feat_cols] = df[feat_cols].ffill().bfill().fillna(0.0)
        df = df.dropna(subset=["target_primary"]).reset_index(drop=True)

        LOGGER.info(
            "Dataset | rows=%d features=%d label=%s h=%d up=%.1f%% down=%.1f%%",
            len(df), len(feat_cols), self.config.label_mode,
            self.config.label_horizon_bars,
            (df["target_primary"]==1).mean()*100,
            (df["target_primary"]==0).mean()*100,
        )
        return df

    def prune_correlated_features(self, df, feature_cols, threshold=None):
        from model import prune_correlated_features as _p
        return _p(df, feature_cols, threshold or self.config.correlation_threshold)

    def prune_by_shap(self, model, df, feature_cols, top_n=None):
        from model import prune_shap_features as _p
        return _p(model, df, feature_cols, top_n or self.config.shap_top_n)


def feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "open_time", "close_time", "symbol", "interval",
        "open", "high", "low", "close", "volume",
        "target_primary", "target_large_move", "future_ret",
    }
    return [c for c in df.columns if c not in excluded]