"""
Main — upgraded pipeline:
1. Paginated historical data collection (30–90 days)
2. Integrity validation
3. Configurable label modes
4. Correlation + SHAP feature pruning
5. Walk-forward cross-validation (5 folds, no leakage)
6. Regime-specific models (high-vol / low-vol)
7. Threshold sweep optimisation (maximise Sharpe)
8. Rich backtest reporting
9. Full artifact persistence
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from backtester import BacktestConfig, Backtester
from data_collector import DataCollector, DataCollectorConfig
from feature_engineering import FeatureConfig, FeatureEngineer, feature_columns
from model import (
    ModelConfig,
    RegimeModelSet,
    StackedEnsembleModel,
    prune_correlated_features,
    prune_shap_features,
)
from signal_engine import SignalConfig, SignalEngine

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_components(cfg: dict):
    collector = DataCollector(
        DataCollectorConfig(
            exchange_name=cfg["exchange"]["name"],
            rest_base_url=cfg["exchange"]["rest_base_url"],
            ws_base_url=cfg["exchange"]["ws_base_url"],
            symbol=cfg["exchange"]["symbol"],
            depth_limit=cfg["exchange"]["depth_limit"],
            trades_limit=cfg["exchange"]["trades_limit"],
            kline_interval=cfg["exchange"]["kline_interval"],
            kline_limit=cfg["exchange"]["kline_limit"],
            db_path=cfg["database"]["path"],
            target_days=cfg["collection"]["target_days"],
            pagination_sleep=cfg["collection"]["pagination_sleep_seconds"],
        )
    )

    tgt = cfg["target"]
    kline_interval = cfg["exchange"]["kline_interval"]
    ohlcv_table    = f"ohlcv_{kline_interval}"

    fe = FeatureEngineer(
        FeatureConfig(
            db_path=cfg["database"]["path"],
            label_horizon_bars=tgt["horizon_bars"],
            label_mode=tgt["label_mode"],
            deadzone_threshold=tgt.get("deadzone_threshold", 0.0005),
            vol_breakout_percentile=tgt.get("vol_breakout_percentile", 70),
            vol_breakout_bars=tgt.get("vol_breakout_bars", 3),
            regime_vol_threshold=tgt.get("regime_vol_threshold", 0.0015),
            ohlcv_table=ohlcv_table,
        )
    )

    val_cfg = cfg.get("validation", {})
    fp_cfg = cfg.get("feature_pruning", {})
    thr = cfg["thresholds"]
    model_cfg = cfg.get("model", {})

    model_config = ModelConfig(
        random_state=model_cfg.get("random_state", 42),
        primary=model_cfg.get("primary", "lightgbm"),
        n_folds=val_cfg.get("n_folds", 5),
        min_train_rows=val_cfg.get("min_train_rows", 5000),
        min_auc=thr.get("min_auc", 0.58),
        max_auc_std=thr.get("min_auc_std", 0.05),
        use_regime_models=model_cfg.get("use_regime_models", True),
        compare_stacking=model_cfg.get("compare_stacking", True),
        shap_top_k=fp_cfg.get("shap_top_k", 50),
        correlation_threshold=fp_cfg.get("correlation_threshold", 0.95),
        min_fold_stability=fp_cfg.get("min_fold_stability", 0.6),
    )

    model = StackedEnsembleModel(model_config)
    regime_model = RegimeModelSet(model_config) if model_cfg.get("use_regime_models", True) else None

    bt_cfg = cfg["backtest"]
    def _parse_bar_minutes(iv: str) -> int:
        iv = iv.strip().lower()
        if iv.endswith("h"): return int(iv[:-1]) * 60
        if iv.endswith("m"): return int(iv[:-1])
        return 1
    bar_mins = bt_cfg.get("bar_size_minutes", _parse_bar_minutes(kline_interval))

    bt = Backtester(
        BacktestConfig(
            fee_bps=bt_cfg["fee_bps"],
            slippage_bps=bt_cfg["slippage_bps"],
            latency_bps=bt_cfg["latency_bps"],
            funding_bps_per_bar=bt_cfg["funding_bps_per_bar"],
            min_auc=thr.get("min_auc", 0.58),
            min_sharpe=thr.get("min_sharpe", 1.5),
            min_trades_per_day=thr.get("min_trades_per_day", 2.0),
            threshold_sweep_step=bt_cfg.get("signal_threshold_sweep_step", 0.01),
            optimize_threshold=cfg["signal"].get("optimize_threshold", True),
            label_horizon_bars=bt_cfg.get("label_horizon_bars", tgt["horizon_bars"]),
            bar_size_minutes=bar_mins,
            max_trades_per_day=bt_cfg.get("max_trades_per_day", 200.0),
        )
    )

    sig_cfg = cfg["signal"]
    se = SignalEngine(
        SignalConfig(
            low_vol_buy=sig_cfg["low_vol_buy"],
            low_vol_sell=sig_cfg["low_vol_sell"],
            high_vol_buy=sig_cfg["high_vol_buy"],
            high_vol_sell=sig_cfg["high_vol_sell"],
            vol_regime_threshold=sig_cfg["vol_regime_threshold"],
        )
    )
    return collector, fe, model, regime_model, bt, se, model_config


def _prepare_dataset(fe: FeatureEngineer, min_rows: int = 100) -> pd.DataFrame:
    ds = fe.build_dataset()
    if ds.empty:
        return ds
    ds = ds.dropna(subset=["target_primary"]).reset_index(drop=True)
    if len(ds) < min_rows:
        LOGGER.warning("Dataset too small: %d rows (need %d)", len(ds), min_rows)
        return pd.DataFrame()
    return ds


def _run_feature_pruning(
    ds: pd.DataFrame,
    model: StackedEnsembleModel,
    model_config: ModelConfig,
    cols: list,
    target_col: str,
) -> list:
    LOGGER.info("Feature pruning: starting with %d features", len(cols))

    # Step 1: Remove highly correlated features
    cols = prune_correlated_features(ds, cols, threshold=model_config.correlation_threshold)

    # Step 2: SHAP importance pruning on a 50% sample
    n = len(ds)
    half = n // 2
    if half > 1000:
        from sklearn.impute import SimpleImputer
        import numpy as np
        imp = SimpleImputer(strategy="median")
        X_half = (
            ds[cols].iloc[:half]
            .replace([float("inf"), float("-inf")], float("nan"))
            .fillna(0.0)
        )
        X_half_imp = pd.DataFrame(imp.fit_transform(X_half), columns=cols)
        y_half = pd.to_numeric(ds[target_col].iloc[:half], errors="coerce").dropna()
        X_half_imp = X_half_imp.loc[y_half.index]
        try:
            shap_model = model._make_primary_model()
            shap_model.fit(X_half_imp, y_half)
            cols = prune_shap_features(shap_model, X_half_imp, cols, top_k=model_config.shap_top_k)
        except Exception as exc:
            LOGGER.warning("SHAP pruning failed: %s", exc)

    LOGGER.info("Feature pruning complete: %d features retained", len(cols))
    return cols


def train_backtest(cfg: dict) -> None:
    LOGGER.info("=" * 80)
    LOGGER.info("TRAIN & BACKTEST MODE (v2 upgraded pipeline)")
    LOGGER.info("=" * 80)

    collector, fe, model, regime_model, bt, se, model_config = build_components(cfg)
    collector.create_version("train_backtest_v2_paginated")

    # ── 1. Data collection ───────────────────────────────────────────────
    LOGGER.info("Collecting %d days of 1-minute data (paginated)...", cfg["collection"]["target_days"])
    result = collector.collect_historical_paginated()
    collector.collect_orderbook_snapshot()

    integrity = result.get("integrity", {})
    if integrity.get("gap_count", 0) > 0:
        LOGGER.warning("Data has %d time gaps — features will forward-fill across gaps", integrity["gap_count"])

    if result["total_stored"] < cfg.get("validation", {}).get("min_train_rows", 5000):
        LOGGER.error(
            "Insufficient data: %d rows. Need >= %d. Increase target_days in config.",
            result["total_stored"], cfg.get("validation", {}).get("min_train_rows", 5000),
        )
        return
    LOGGER.info("Data ready | %d rows | %.1f days", result["total_stored"], integrity.get("span_days", 0))

    # ── 2. Feature engineering ───────────────────────────────────────────
    LOGGER.info("Building features (label_mode=%s)...", cfg["target"]["label_mode"])
    ds = _prepare_dataset(fe, min_rows=1000)
    if ds.empty:
        LOGGER.error("Feature dataset empty. Check DB and config.")
        return

    target_col = cfg["target"]["primary_target_col"]
    cols = feature_columns(ds)
    LOGGER.info("Dataset: %d rows | %d raw features | label balance: %.1f%%",
                len(ds), len(cols), ds[target_col].mean() * 100)

    # ── 3. Feature pruning ───────────────────────────────────────────────
    cols = _run_feature_pruning(ds, model, model_config, cols, target_col)

    # ── 4. Walk-forward cross-validation ────────────────────────────────
    LOGGER.info("Walk-forward CV (%d folds)...", model_config.n_folds)
    metrics = model.fit_evaluate(ds, cols, target_col)

    fold_metrics = metrics.get("fold_metrics", [])
    LOGGER.info(
        "CV Results | Mean AUC: %.4f ± %.4f | Worst fold: %.4f | Folds: %d",
        metrics["roc_auc"], metrics.get("roc_auc_std", 0),
        metrics.get("worst_fold_auc", 0), metrics.get("n_folds_used", 0),
    )
    for fm in fold_metrics:
        LOGGER.info("  Fold %d: AUC=%.4f | Acc=%.4f | train=%d | test=%d",
                    fm["fold"], fm["auc"], fm["accuracy"], fm["train_rows"], fm["test_rows"])

    # ── 5. Regime models ─────────────────────────────────────────────────
    regime_metrics = {}
    if regime_model is not None and "vol_regime" in ds.columns:
        LOGGER.info("Training regime-specific models...")
        regime_metrics = regime_model.fit_evaluate(ds, cols, target_col)
        for r, m in regime_metrics.items():
            LOGGER.info("  Regime '%s': AUC=%.4f ± %.4f", r, m.get("roc_auc", 0), m.get("roc_auc_std", 0))

    # ── 6. Backtest with threshold sweep ─────────────────────────────────
    LOGGER.info("Backtesting with threshold optimisation...")
    ds_bt = ds.copy()
    ds_bt["pred_prob"] = model.predict_proba(ds_bt, cols)
    report = bt.run(ds_bt, "pred_prob", target_col, metrics)

    if bt.optimal_threshold_ is not None:
        se.set_optimal_thresholds(report.get("buy_threshold", 0.55), report.get("sell_threshold", 0.45))

    LOGGER.info(
        "Backtest | Status: %s | Sharpe: %.2f | MaxDD: %.1f%% | "
        "Trades/day: %.1f | WinRate: %.1f%% | Thresholds: %.2f/%.2f",
        report.get("status"),
        report.get("sharpe", 0),
        abs(report.get("max_drawdown", 0)) * 100,
        report.get("trades_per_day", 0),
        report.get("win_rate", 0) * 100,
        report.get("buy_threshold", 0.55),
        report.get("sell_threshold", 0.45),
    )

    # ── 7. Save artifacts ────────────────────────────────────────────────
    Path("artifacts").mkdir(exist_ok=True)

    ds_bt.to_csv("artifacts/model_dataset.csv", index=False)
    pd.DataFrame([{**metrics, **report}]).to_csv("artifacts/backtest_report.csv", index=False)

    if fold_metrics:
        pd.DataFrame(fold_metrics).to_csv("artifacts/fold_metrics.csv", index=False)

    pd.DataFrame({"feature": cols}).to_csv("artifacts/selected_features.csv", index=False)

    if regime_metrics:
        rows = [{"regime": r, **{k: v for k, v in m.items() if k != "fold_metrics"}}
                for r, m in regime_metrics.items()]
        pd.DataFrame(rows).to_csv("artifacts/regime_metrics.csv", index=False)

    with open("artifacts/config_snapshot.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    LOGGER.info("=" * 80)
    LOGGER.info("ARTIFACTS SAVED → artifacts/")
    LOGGER.info("  model_dataset.csv    — full labeled dataset with predictions")
    LOGGER.info("  backtest_report.csv  — combined model + backtest metrics")
    LOGGER.info("  fold_metrics.csv     — per-fold CV results")
    LOGGER.info("  selected_features.csv — pruned feature list")
    LOGGER.info("  regime_metrics.csv   — regime-split model performance")
    LOGGER.info("  config_snapshot.yaml — reproducibility snapshot")
    LOGGER.info("=" * 80)
    LOGGER.info("FINAL: AUC %.4f±%.4f | Sharpe %.2f | Status: %s",
                metrics["roc_auc"], metrics.get("roc_auc_std", 0),
                report.get("sharpe", 0), report.get("status"))
    LOGGER.info("=" * 80)


def live(cfg: dict) -> None:
    LOGGER.info("=" * 80)
    LOGGER.info("LIVE TRADING MODE (v2)")
    LOGGER.info("=" * 80)

    collector, fe, model, regime_model, bt, se, model_config = build_components(cfg)

    # Incremental data refresh
    LOGGER.info("Incremental data refresh...")
    collector.collect_historical_paginated()
    collector.collect_orderbook_snapshot()

    ds = _prepare_dataset(fe, min_rows=5000)
    if ds.empty:
        LOGGER.error("Not enough data. Run train_backtest first.")
        return

    target_col = cfg["target"]["primary_target_col"]
    cols = feature_columns(ds)
    cols = prune_correlated_features(ds, cols, threshold=model_config.correlation_threshold)

    metrics = model.fit_evaluate(ds, cols, target_col)
    LOGGER.info("Bootstrap AUC: %.4f ± %.4f", metrics["roc_auc"], metrics.get("roc_auc_std", 0))

    passes_gate = metrics["roc_auc"] >= cfg["thresholds"]["min_auc"]
    LOGGER.info("Model gate: %s", "PASS" if passes_gate else "FAIL — HOLD only")

    if passes_gate:
        ds_bt = ds.copy()
        ds_bt["pred_prob"] = model.predict_proba(ds_bt, cols)
        report = bt.run(ds_bt, "pred_prob", target_col, metrics)
        if bt.optimal_threshold_:
            se.set_optimal_thresholds(report["buy_threshold"], report["sell_threshold"])
            LOGGER.info("Thresholds: buy=%.2f sell=%.2f", report["buy_threshold"], report["sell_threshold"])

    for step in range(cfg["live"]["max_iterations"]):
        collector.collect_rest_once()

        if (step + 1) % cfg["live"]["emit_every_iterations"] != 0:
            time.sleep(cfg["live"]["poll_seconds"])
            continue

        cur = _prepare_dataset(fe, min_rows=30)
        if cur.empty:
            time.sleep(cfg["live"]["poll_seconds"])
            continue

        latest = cur.tail(1)
        try:
            cols_live = [c for c in cols if c in latest.columns]
            prob = float(model.predict_proba(latest, cols_live)[0])
        except Exception as exc:
            LOGGER.warning("Prediction failed: %s", exc)
            time.sleep(cfg["live"]["poll_seconds"])
            continue

        volatility = float(latest["vol_5"].iloc[0]) if "vol_5" in latest.columns else 0.0
        signal = "HOLD"
        if passes_gate:
            signal = se.dynamic_signal(prob, volatility, cfg["signal"]["vol_regime_threshold"])

        vol_regime = int(latest["vol_regime"].iloc[0]) if "vol_regime" in latest.columns else 0
        LOGGER.info("[%d] %s | prob=%.3f vol=%.5f regime=%s AUC=%.4f",
                    step + 1, signal, prob, volatility,
                    "HIGH" if vol_regime else "LOW", metrics["roc_auc"])

        time.sleep(cfg["live"]["poll_seconds"])

    LOGGER.info("Live loop ended.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTCUSDT 1-Min ML System v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train_backtest --config config.yaml
  python main.py --mode live           --config config.yaml
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["train_backtest", "live"], default="train_backtest")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    if args.mode == "train_backtest":
        train_backtest(cfg)
    else:
        live(cfg)


if __name__ == "__main__":
    main()