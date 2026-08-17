"""
Main — canonical pipeline entrypoint (Stage 2P runtime cutover).

This REPLACES the legacy dict-config-driven pipeline (global-percentile
vol_breakout labels, static regime-threshold signal config, model-type
switching via ModelConfig/StackedEnsembleModel/RegimeModelSet/Backtester/
SignalEngine in model.py/feature_engineering.py/backtester.py/
signal_engine.py) with the frozen Stage 1-2O architecture:

  - config_schema.AppConfig is the single canonical configuration contract
    (loaded via config_schema.load_config -- strict, fails loudly on any
    unknown/missing/out-of-range field).
  - target_engineering / dataset_builder build the causal, ATR-scaled
    3-class target and OHLCV-only features (never the legacy binary
    vol_breakout/dead-zone labels).
  - oof_builder / threshold_selection / position_engine / metrics /
    production_model / experiment_runner ARE the pipeline: purge-aware
    expanding walk-forward, fold-local feature selection, canonical OOF
    generation, past-only threshold selection, the FLAT/LONG/SHORT
    accounting engine, predictive + trading metrics, the passive-long
    benchmark, and production model training/persistence.

`_parse_bar_minutes` (the Stage 0 audit finding: bar duration silently
re-derived from a local regex instead of the canonical interval authority)
is REMOVED in this cutover -- bar duration is never computed here at all;
intervals.py (via config_schema.DataConfig's own validation and
DataCollector's config.kline_interval) remains the single canonical
authority, exactly as it has been since Stage 2A.

model.py, feature_engineering.py, backtester.py, and signal_engine.py are
NOT imported anywhere in this file any more. They are left in place for
this stage (Stage 2P is a runtime CUTOVER, not a deletion pass) but are
dead code from main.py's perspective; removing the files themselves is
Stage 2S's "clean repository" job, not this stage's.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from config_schema import AppConfig, load_config
from data_collector import DataCollector, DataCollectorConfig
from dataset_builder import build_dataset, feature_columns
from experiment_runner import run_experiment
from oof_builder import ALL_MODELS, MODEL_LIGHTGBM, OOFPipelineConfig
from production_model import ProductionModelResult, predict_proba_production
from threshold_selection import compute_signal, select_production_threshold

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_collector(cfg: AppConfig) -> DataCollector:
    return DataCollector(
        DataCollectorConfig(
            exchange_name=cfg.data.exchange,
            rest_base_url=cfg.data.rest_base_url,
            ws_base_url=cfg.data.ws_base_url,
            symbol=cfg.data.symbol,
            depth_limit=cfg.data.depth_limit,
            trades_limit=cfg.data.trades_limit,
            kline_interval=cfg.data.interval,
            kline_limit=cfg.data.kline_limit,
            db_path=cfg.data.db_path,
            target_days=cfg.data.target_days,
            pagination_sleep=cfg.data.pagination_sleep_seconds,
            incremental=cfg.data.incremental,
            integrity_check=cfg.data.integrity_check,
        )
    )


def _lgbm_params(cfg: AppConfig) -> dict:
    lg = cfg.models.lightgbm
    return {
        "num_leaves": lg.num_leaves, "learning_rate": lg.learning_rate, "n_estimators": lg.n_estimators,
        "min_child_samples": lg.min_child_samples, "subsample": lg.subsample,
        "colsample_bytree": lg.colsample_bytree, "reg_alpha": lg.reg_alpha, "reg_lambda": lg.reg_lambda,
    }


def _oof_config_from_app_config(cfg: AppConfig) -> OOFPipelineConfig:
    return OOFPipelineConfig(
        n_folds=cfg.validation.n_folds,
        min_train_rows=cfg.validation.min_train_rows,
        min_test_rows=cfg.validation.min_test_rows,
        horizon_bars=cfg.target.horizon_bars,  # purge size == target.horizon_bars, always
        embargo_bars=cfg.validation.embargo_bars,
        min_class_count=cfg.validation.min_class_count,
        correlation_threshold=cfg.features.correlation_threshold,
        importance_top_k=cfg.features.importance_top_k,
        internal_validation_fraction=cfg.validation.internal_validation_fraction,
        lgbm_early_stopping_rounds=cfg.models.lightgbm.early_stopping_rounds,
        random_state=cfg.models.random_state,
        lgbm_params=_lgbm_params(cfg),
    )


def _threshold_kwargs(cfg: AppConfig) -> dict:
    bt = cfg.backtest
    return {
        "threshold_sweep_min": bt.threshold_sweep_min, "threshold_sweep_max": bt.threshold_sweep_max,
        "threshold_sweep_step": bt.threshold_sweep_step, "min_trades_for_tuning": bt.min_trades_for_tuning,
        "default_buy_threshold": bt.default_buy_threshold, "default_sell_threshold": bt.default_sell_threshold,
    }


def _backtest_kwargs(cfg: AppConfig) -> dict:
    bt = cfg.backtest
    return {
        "initial_equity": bt.initial_equity, "position_fraction": bt.position_fraction,
        "fee_bps": bt.fee_bps, "slippage_bps": bt.slippage_bps, "latency_bps": bt.latency_bps,
        "funding_bps_per_bar": bt.funding_bps_per_bar,
    }


def train_backtest(cfg: AppConfig, output_dir: str = None) -> None:
    LOGGER.info("=" * 80)
    LOGGER.info("TRAIN & BACKTEST (canonical Stage 1-2O pipeline)")
    LOGGER.info("=" * 80)

    collector = build_collector(cfg)
    collector.create_version("stage2p_canonical_pipeline")

    LOGGER.info("Collecting %d days of %s data (paginated)...", cfg.data.target_days, cfg.data.interval)
    try:
        result = collector.collect_historical_paginated()
        LOGGER.info("Data collection result: %s", result)
    except Exception as exc:
        LOGGER.warning(
            "Data collection failed (%s) -- proceeding with whatever OHLCV is already stored locally.", exc
        )

    if cfg.data.integrity_check:
        integrity = collector.run_integrity_check()
        if integrity["gap_events"] > 0:
            LOGGER.warning(
                "Data has %d gap event(s), %d missing bar(s) total -- NOT auto-filled "
                "(this pipeline never forward-fills OHLCV gaps).",
                integrity["gap_events"], integrity["missing_bars"],
            )

    ohlcv = collector.load_ohlcv_dataframe()
    if len(ohlcv) < cfg.validation.min_train_rows:
        LOGGER.error(
            "Insufficient data: %d rows stored, need >= %d (validation.min_train_rows). "
            "Increase data.target_days or wait for more history to accumulate.",
            len(ohlcv), cfg.validation.min_train_rows,
        )
        return

    LOGGER.info(
        "Building causal features + target (horizon_bars=%d, neutral_atr_mult=%.3f, atr_period=%d)...",
        cfg.target.horizon_bars, cfg.target.neutral_atr_mult, cfg.target.atr_period,
    )
    dataset = build_dataset(
        ohlcv, horizon_bars=cfg.target.horizon_bars, neutral_atr_mult=cfg.target.neutral_atr_mult,
        atr_period=cfg.target.atr_period, lag_periods=cfg.features.lag_periods,
    )
    candidate_features = feature_columns(dataset)
    LOGGER.info("Dataset: %d rows | %d candidate features", len(dataset), len(candidate_features))

    oof_config = _oof_config_from_app_config(cfg)
    out_dir = output_dir or str(Path(cfg.artifacts.root_dir) / cfg.experiment.name)

    exp_result = run_experiment(
        dataset, candidate_features, symbol=cfg.data.symbol, interval=cfg.data.interval,
        run_id=cfg.experiment.name, oof_config=oof_config, output_dir=out_dir,
        min_fold_stability=cfg.features.min_fold_stability,
        threshold_kwargs=_threshold_kwargs(cfg), backtest_kwargs=_backtest_kwargs(cfg),
        production_lgbm_params=_lgbm_params(cfg),
    )

    # Production/live thresholds (Stage 2K's separate "computed once from
    # full historical OOF" rule) -- persisted alongside the rest of the
    # experiment artifacts, NEVER used to recompute the metrics.json values
    # already written by run_experiment above.
    production_threshold = select_production_threshold(
        exp_result.oof_df, MODEL_LIGHTGBM, **_threshold_kwargs(cfg),
    )
    Path(out_dir, "production_thresholds.json").write_text(json.dumps({
        "model": MODEL_LIGHTGBM,
        "buy_threshold": production_threshold.buy_threshold,
        "sell_threshold": production_threshold.sell_threshold,
        "buy_used_fallback": production_threshold.buy_used_fallback,
        "sell_used_fallback": production_threshold.sell_used_fallback,
    }, indent=2))

    LOGGER.info("=" * 80)
    LOGGER.info("ARTIFACTS SAVED -> %s", out_dir)
    for model in ALL_MODELS:
        tm = exp_result.trading_metrics_by_model[model]
        LOGGER.info(
            "  %-24s | total_return=%.4f sharpe=%s max_dd=%.4f trades=%d",
            model, tm.total_return,
            ("%.2f" % tm.sharpe_daily_compounded) if tm.sharpe_daily_compounded == tm.sharpe_daily_compounded else "nan",
            tm.max_drawdown, tm.trade_count,
        )
    bm = exp_result.benchmark_trading_metrics
    LOGGER.info("  %-24s | total_return=%.4f max_dd=%.4f (passive-long benchmark)", "benchmark", bm.total_return, bm.max_drawdown)
    LOGGER.info(
        "  production threshold (lightgbm) | buy=%.3f sell=%.3f",
        production_threshold.buy_threshold, production_threshold.sell_threshold,
    )
    LOGGER.info("=" * 80)


def live(cfg: AppConfig) -> None:
    """Production inference loop (Stage 1 §2's "Production Inference
    Pipeline"). Loads the most recently trained production model artifact
    and polls the exchange for fresh bars, computing a signal with the
    SAME frozen SIGNAL RULE used everywhere else in this codebase
    (threshold_selection.compute_signal) -- never a separate
    reimplementation. Uses the production_thresholds.json artifact written
    by train_backtest (computed once from full historical OOF); falls back
    to config.yaml's default_buy_threshold/default_sell_threshold if that
    artifact is not present.
    """
    LOGGER.info("=" * 80)
    LOGGER.info("LIVE / INFERENCE MODE (canonical pipeline)")
    LOGGER.info("=" * 80)

    out_dir = Path(cfg.artifacts.root_dir) / cfg.experiment.name
    model_path = out_dir / "production_model.joblib"
    selected_features_path = out_dir / "selected_features.json"
    if not model_path.exists() or not selected_features_path.exists():
        LOGGER.error("No production model found at %s. Run train_backtest first.", out_dir)
        return

    import joblib
    model = joblib.load(model_path)
    selected_features = json.loads(selected_features_path.read_text())
    production = ProductionModelResult(
        model=model, selected_features=selected_features, n_estimators=getattr(model, "n_estimators", 0),
        lgbm_params={}, n_training_rows=0, class_counts={},
    )

    thresholds_path = out_dir / "production_thresholds.json"
    if thresholds_path.exists():
        thresholds = json.loads(thresholds_path.read_text())
        buy_threshold, sell_threshold = thresholds["buy_threshold"], thresholds["sell_threshold"]
    else:
        LOGGER.warning("No production_thresholds.json found -- falling back to config.yaml default thresholds.")
        buy_threshold, sell_threshold = cfg.backtest.default_buy_threshold, cfg.backtest.default_sell_threshold

    collector = build_collector(cfg)
    max_iterations = cfg.live.max_iterations or 1_000_000_000

    for step in range(max_iterations):
        collector.collect_rest_once()

        if (step + 1) % cfg.live.emit_every_iterations != 0:
            time.sleep(cfg.live.poll_seconds)
            continue

        ohlcv = collector.load_ohlcv_dataframe()
        min_rows_needed = cfg.target.atr_period + cfg.features.lag_periods + cfg.target.horizon_bars + 5
        if len(ohlcv) < min_rows_needed:
            time.sleep(cfg.live.poll_seconds)
            continue

        dataset = build_dataset(
            ohlcv, horizon_bars=cfg.target.horizon_bars, neutral_atr_mult=cfg.target.neutral_atr_mult,
            atr_period=cfg.target.atr_period, lag_periods=cfg.features.lag_periods,
        )
        latest = dataset.iloc[[-1]]
        missing = [f for f in selected_features if f not in latest.columns]
        if missing:
            LOGGER.warning("Latest row missing features %s -- skipping this iteration.", missing)
            time.sleep(cfg.live.poll_seconds)
            continue

        probs = predict_proba_production(production, latest)
        signal = compute_signal(probs[:, 0], probs[:, 1], probs[:, 2], buy_threshold, sell_threshold)[0]

        LOGGER.info(
            "[%d] signal=%s prob_down=%.3f prob_neutral=%.3f prob_up=%.3f (buy_thr=%.3f sell_thr=%.3f)",
            step + 1, signal, probs[0, 0], probs[0, 1], probs[0, 2], buy_threshold, sell_threshold,
        )
        time.sleep(cfg.live.poll_seconds)

    LOGGER.info("Live loop ended.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTCUSDT Perpetual Futures 3-Class Direction Model (canonical Stage 1-2O pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train_backtest --config config.yaml
  python main.py --mode live           --config config.yaml
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["train_backtest", "live"], default="train_backtest")
    parser.add_argument("--output-dir", default=None, help="Override artifacts output directory for train_backtest.")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    if args.mode == "train_backtest":
        train_backtest(cfg, output_dir=args.output_dir)
    else:
        live(cfg)


if __name__ == "__main__":
    main()
